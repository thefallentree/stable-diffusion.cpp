#!/usr/bin/env python3
"""Small fan-out gateway for multiple sd-server async workers.

The gateway preserves the native ``/sdcpp/v1`` API for one-copy requests and
adds an optional ``copies`` request field.  Jobs are namespaced at the gateway,
so identical worker-local job IDs cannot collide.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class GatewayError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class WorkerJob:
    worker: int
    id: str
    seed: int

    def as_json(self) -> dict[str, Any]:
        return {"worker": self.worker, "id": self.id, "seed": self.seed}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "WorkerJob":
        return cls(worker=int(value["worker"]), id=str(value["id"]), seed=int(value["seed"]))


@dataclass
class PoolJob:
    id: str
    kind: str
    created: int
    jobs: list[WorkerJob] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "created": self.created,
            "jobs": [job.as_json() for job in self.jobs],
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "PoolJob":
        return cls(
            id=str(value["id"]),
            kind=str(value["kind"]),
            created=int(value["created"]),
            jobs=[WorkerJob.from_json(item) for item in value.get("jobs", [])],
        )


class WorkerPool:
    def __init__(self, workers: list[str], state_dir: Path, timeout: float = 15.0) -> None:
        if not workers:
            raise ValueError("at least one worker is required")
        self.workers = [worker.rstrip("/") for worker in workers]
        self.state_dir = state_dir
        self.timeout = timeout
        self.lock = threading.RLock()
        self.jobs: dict[str, PoolJob] = {}
        self.next_worker = 0
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def _load_jobs(self) -> None:
        for path in self.state_dir.glob("pool_*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                job = PoolJob.from_json(value)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if job.jobs and all(0 <= item.worker < len(self.workers) for item in job.jobs):
                self.jobs[job.id] = job

    def _persist(self, job: PoolJob) -> None:
        destination = self.state_dir / f"{job.id}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(job.as_json(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    def _request(
        self,
        worker: int,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.workers[worker]}{path}",
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise GatewayError(error.code, detail or error.reason) from error
        except (OSError, urllib.error.URLError) as error:
            raise GatewayError(HTTPStatus.BAD_GATEWAY, f"worker {worker} unavailable: {error}") from error
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GatewayError(HTTPStatus.BAD_GATEWAY, f"worker {worker} returned invalid JSON") from error

    def capabilities(self) -> dict[str, Any]:
        available, errors = self._available_workers()
        if not available:
            raise GatewayError(HTTPStatus.SERVICE_UNAVAILABLE, "; ".join(errors) or "no workers online")
        response = available[0][1]
        online_workers = {worker for worker, _response in available}
        online: list[bool] = []
        for worker in range(len(self.workers)):
            online.append(worker in online_workers)
        result = copy.deepcopy(response)
        limits = result.setdefault("limits", {})
        limits["max_parallel_copies"] = sum(online)
        for mode in ("img_gen", "vid_gen"):
            features = result.setdefault("features_by_mode", {}).setdefault(mode, {})
            features["parallel_copies"] = sum(online) > 1
        result["pool"] = {
            "worker_count": len(self.workers),
            "online_workers": sum(online),
            "workers": [
                {"index": index, "online": is_online}
                for index, is_online in enumerate(online)
            ],
        }
        return result

    def _available_workers(self) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
        available: list[tuple[int, dict[str, Any]]] = []
        errors: list[str] = []
        for worker in range(len(self.workers)):
            try:
                response = self._request(worker, "GET", "/sdcpp/v1/capabilities")
                available.append((worker, response))
            except GatewayError as error:
                errors.append(str(error))
        return available, errors

    def _worker_order(self, copies: int, workers: list[int]) -> list[int]:
        with self.lock:
            start = self.next_worker % len(workers)
            self.next_worker = (self.next_worker + copies) % len(workers)
        return [workers[(start + index) % len(workers)] for index in range(copies)]

    def submit(self, kind: str, request: dict[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(request)
        raw_copies = request.pop("copies", 1)
        raw_stride = request.pop("seed_stride", 1)
        try:
            copies = int(raw_copies)
            seed_stride = int(raw_stride)
        except (TypeError, ValueError) as error:
            raise GatewayError(HTTPStatus.BAD_REQUEST, "copies and seed_stride must be integers") from error
        available, errors = self._available_workers()
        online_workers = [worker for worker, _response in available]
        if not online_workers:
            raise GatewayError(HTTPStatus.SERVICE_UNAVAILABLE, "; ".join(errors) or "no workers online")
        if copies < 1 or copies > len(online_workers):
            raise GatewayError(
                HTTPStatus.BAD_REQUEST,
                f"copies must be between 1 and {len(online_workers)} online workers",
            )
        try:
            base_seed = int(request.get("seed", -1))
        except (TypeError, ValueError) as error:
            raise GatewayError(HTTPStatus.BAD_REQUEST, "seed must be an integer") from error
        assignments = self._worker_order(copies, online_workers)
        submitted: list[WorkerJob] = []

        def submit_one(copy_index: int, worker: int) -> WorkerJob:
            body = copy.deepcopy(request)
            seed = base_seed if base_seed < 0 else base_seed + copy_index * seed_stride
            body["seed"] = seed
            result = self._request(worker, "POST", f"/sdcpp/v1/{kind}", body)
            backend_id = result.get("id")
            if not isinstance(backend_id, str) or not backend_id:
                raise GatewayError(HTTPStatus.BAD_GATEWAY, f"worker {worker} omitted job id")
            return WorkerJob(worker=worker, id=backend_id, seed=seed)

        failure: Exception | None = None
        with ThreadPoolExecutor(max_workers=copies) as executor:
            futures = [
                executor.submit(submit_one, index, worker)
                for index, worker in enumerate(assignments)
            ]
            for future in as_completed(futures):
                try:
                    submitted.append(future.result())
                except Exception as error:
                    failure = failure or error
        if failure is not None:
            for item in submitted:
                try:
                    self._request(item.worker, "POST", f"/sdcpp/v1/jobs/{item.id}/cancel")
                except GatewayError:
                    pass
            raise failure

        by_worker = {item.worker: item for item in submitted}
        ordered = [by_worker[worker] for worker in assignments]
        pool_job = PoolJob(
            id=f"pool_{uuid.uuid4().hex}",
            kind=kind,
            created=int(time.time()),
            jobs=ordered,
        )
        with self.lock:
            self.jobs[pool_job.id] = pool_job
            self._persist(pool_job)
        return {
            "id": pool_job.id,
            "kind": kind,
            "status": "queued",
            "created": pool_job.created,
            "copies": len(pool_job.jobs),
            "poll_url": f"/sdcpp/v1/jobs/{pool_job.id}",
        }

    def _get_pool_job(self, job_id: str) -> PoolJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise GatewayError(HTTPStatus.NOT_FOUND, "unknown pool job")
        return job

    def poll(self, job_id: str) -> dict[str, Any]:
        pool_job = self._get_pool_job(job_id)
        responses: list[dict[str, Any] | None] = [None] * len(pool_job.jobs)

        def poll_one(index: int, item: WorkerJob) -> tuple[int, dict[str, Any]]:
            return index, self._request(item.worker, "GET", f"/sdcpp/v1/jobs/{item.id}")

        with ThreadPoolExecutor(max_workers=len(pool_job.jobs)) as executor:
            futures = [
                executor.submit(poll_one, index, item)
                for index, item in enumerate(pool_job.jobs)
            ]
            for future in as_completed(futures):
                index, response = future.result()
                responses[index] = response

        concrete = [response for response in responses if response is not None]
        statuses = [str(response.get("status", "failed")) for response in concrete]
        if statuses and all(status == "completed" for status in statuses):
            status = "completed"
        elif any(status == "failed" for status in statuses):
            status = "failed"
        elif statuses and all(status == "cancelled" for status in statuses):
            status = "cancelled"
        elif any(status == "generating" for status in statuses):
            status = "generating"
        else:
            status = "queued"

        starts = [int(response["started"]) for response in concrete if response.get("started")]
        completions = [int(response["completed"]) for response in concrete if response.get("completed")]
        result: dict[str, Any] = {
            "id": pool_job.id,
            "kind": pool_job.kind,
            "status": status,
            "created": pool_job.created,
            "started": min(starts) if starts else None,
            "completed": max(completions) if len(completions) == len(concrete) else None,
            "queue_position": max((int(response.get("queue_position", 0)) for response in concrete), default=0),
            "copies": len(pool_job.jobs),
            "result": None,
            "error": None,
        }
        if status == "completed":
            variants: list[dict[str, Any]] = []
            for index, (item, response) in enumerate(zip(pool_job.jobs, concrete)):
                variant = copy.deepcopy(response.get("result") or {})
                variant.update(
                    {
                        "index": index,
                        "worker": item.worker,
                        "worker_job_id": item.id,
                        "seed": item.seed,
                    }
                )
                variants.append(variant)
            if len(variants) == 1:
                result["result"] = variants[0]
            else:
                result["result"] = {"variants": variants}
        elif status in {"failed", "cancelled"}:
            errors = [response.get("error") for response in concrete if response.get("error")]
            result["error"] = {
                "code": status,
                "message": "; ".join(str(error.get("message", error)) for error in errors)
                or f"pool job {status}",
            }
        return result

    def cancel(self, job_id: str) -> dict[str, Any]:
        pool_job = self._get_pool_job(job_id)
        for item in pool_job.jobs:
            try:
                self._request(item.worker, "POST", f"/sdcpp/v1/jobs/{item.id}/cancel")
            except GatewayError as error:
                if error.status not in {HTTPStatus.CONFLICT, HTTPStatus.GONE}:
                    raise
        return self.poll(job_id)


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "SDCppWorkerPool/1.0"

    @property
    def pool(self) -> WorkerPool:
        return self.server.pool  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}", flush=True)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise GatewayError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from error
        if length <= 0:
            raise GatewayError(HTTPStatus.BAD_REQUEST, "empty body")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GatewayError(HTTPStatus.BAD_REQUEST, "invalid JSON") from error
        if not isinstance(value, dict):
            raise GatewayError(HTTPStatus.BAD_REQUEST, "request body must be an object")
        return value

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/health":
                capabilities = self.pool.capabilities()
                self._json(HTTPStatus.OK, {"status": "ok", "pool": capabilities["pool"]})
                return
            if path == "/sdcpp/v1/capabilities":
                self._json(HTTPStatus.OK, self.pool.capabilities())
                return
            prefix = "/sdcpp/v1/jobs/"
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                self._json(HTTPStatus.OK, self.pool.poll(path[len(prefix):]))
                return
            raise GatewayError(HTTPStatus.NOT_FOUND, "not found")
        except GatewayError as error:
            self._json(error.status, {"error": str(error)})
        except Exception as error:  # keep the gateway alive on malformed worker responses
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path in {"/sdcpp/v1/img_gen", "/sdcpp/v1/vid_gen"}:
                kind = "img_gen" if path.endswith("img_gen") else "vid_gen"
                self._json(HTTPStatus.ACCEPTED, self.pool.submit(kind, self._body()))
                return
            prefix = "/sdcpp/v1/jobs/"
            suffix = "/cancel"
            if path.startswith(prefix) and path.endswith(suffix):
                job_id = path[len(prefix):-len(suffix)]
                if job_id and "/" not in job_id:
                    self._json(HTTPStatus.OK, self.pool.cancel(job_id))
                    return
            raise GatewayError(HTTPStatus.NOT_FOUND, "not found")
        except GatewayError as error:
            self._json(error.status, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fan out native sd-server async jobs")
    parser.add_argument("--listen-ip", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument("--worker", action="append", required=True, help="worker base URL; repeat per worker")
    parser.add_argument("--state-dir", type=Path, default=Path("./pool-state"))
    parser.add_argument("--worker-timeout", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = WorkerPool(args.worker, args.state_dir.resolve(), timeout=args.worker_timeout)
    server = ThreadingHTTPServer((args.listen_ip, args.listen_port), GatewayHandler)
    server.pool = pool  # type: ignore[attr-defined]
    print(
        f"listening on http://{args.listen_ip}:{args.listen_port} with {len(args.worker)} workers",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

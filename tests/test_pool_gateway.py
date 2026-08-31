#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1] / "examples" / "server"
sys.path.insert(0, str(SERVER_DIR))

from pool_gateway import WorkerPool  # noqa: E402


class MockWorker:
    def __init__(self, index: int) -> None:
        self.index = index
        self.online = True
        self.requests: list[dict] = []
        self.jobs: dict[str, dict] = {}


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        pass

    @property
    def state(self) -> MockWorker:
        return self.server.state  # type: ignore[attr-defined]

    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/sdcpp/v1/capabilities":
            if not self.state.online:
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "offline"})
                return
            self.send_json(
                HTTPStatus.OK,
                {
                    "supported_modes": ["vid_gen"],
                    "limits": {"max_queue_size": 64},
                    "features_by_mode": {"vid_gen": {}},
                },
            )
            return
        prefix = "/sdcpp/v1/jobs/"
        if self.path.startswith(prefix):
            job_id = self.path[len(prefix):]
            job = self.state.jobs.get(job_id)
            if job is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "missing"})
                return
            self.send_json(HTTPStatus.OK, job)
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "missing"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/sdcpp/v1/vid_gen":
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            self.state.requests.append(request)
            # Deliberately collide across workers; the pool must namespace IDs.
            job_id = "job_same_id"
            self.state.jobs[job_id] = {
                "id": job_id,
                "kind": "vid_gen",
                "status": "completed",
                "created": 100,
                "started": 101 + self.state.index,
                "completed": 110 + self.state.index,
                "queue_position": 0,
                "result": {
                    "output_format": "webm",
                    "mime_type": "video/webm",
                    "fps": 24,
                    "frame_count": 124,
                    "b64_json": f"payload-{self.state.index}",
                },
                "error": None,
            }
            self.send_json(
                HTTPStatus.ACCEPTED,
                {"id": job_id, "kind": "vid_gen", "status": "queued", "created": 100},
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "missing"})


class PoolGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = [MockWorker(0), MockWorker(1)]
        self.servers: list[ThreadingHTTPServer] = []
        self.threads: list[threading.Thread] = []
        for state in self.states:
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
            server.state = state  # type: ignore[attr-defined]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.servers.append(server)
            self.threads.append(thread)
        self.temporary = tempfile.TemporaryDirectory()
        workers = [f"http://127.0.0.1:{server.server_port}" for server in self.servers]
        self.pool = WorkerPool(workers, Path(self.temporary.name))

    def tearDown(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)
        self.temporary.cleanup()

    def test_capabilities_advertise_online_parallel_workers(self) -> None:
        capabilities = self.pool.capabilities()
        self.assertEqual(capabilities["pool"]["worker_count"], 2)
        self.assertEqual(capabilities["pool"]["online_workers"], 2)
        self.assertEqual(capabilities["limits"]["max_parallel_copies"], 2)
        self.assertTrue(capabilities["features_by_mode"]["vid_gen"]["parallel_copies"])

    def test_two_copies_use_both_workers_and_increment_seed(self) -> None:
        submitted = self.pool.submit(
            "vid_gen",
            {"prompt": "test", "seed": 900, "copies": 2, "seed_stride": 7},
        )
        self.assertTrue(submitted["id"].startswith("pool_"))
        self.assertEqual(submitted["copies"], 2)
        self.assertEqual(self.states[0].requests[0]["seed"], 900)
        self.assertEqual(self.states[1].requests[0]["seed"], 907)
        self.assertNotIn("copies", self.states[0].requests[0])

        result = self.pool.poll(submitted["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["started"], 101)
        self.assertEqual(result["completed"], 111)
        variants = result["result"]["variants"]
        self.assertEqual([item["worker"] for item in variants], [0, 1])
        self.assertEqual([item["seed"] for item in variants], [900, 907])
        self.assertEqual([item["b64_json"] for item in variants], ["payload-0", "payload-1"])

    def test_single_copy_preserves_native_result_shape(self) -> None:
        submitted = self.pool.submit("vid_gen", {"prompt": "test", "seed": 12})
        result = self.pool.poll(submitted["id"])
        self.assertEqual(result["status"], "completed")
        self.assertIn("b64_json", result["result"])
        self.assertNotIn("variants", result["result"])

    def test_single_copy_skips_an_offline_worker(self) -> None:
        self.states[1].online = False
        capabilities = self.pool.capabilities()
        self.assertEqual(capabilities["limits"]["max_parallel_copies"], 1)
        self.assertFalse(capabilities["features_by_mode"]["vid_gen"]["parallel_copies"])

        submitted = self.pool.submit("vid_gen", {"prompt": "test", "seed": 77})
        result = self.pool.poll(submitted["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["worker"], 0)
        self.assertEqual(len(self.states[0].requests), 1)
        self.assertEqual(len(self.states[1].requests), 0)


if __name__ == "__main__":
    unittest.main()

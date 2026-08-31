# Example

The following example starts `sd-server` with a standalone diffusion model, VAE, and LLM text encoder:

```
.\bin\Release\sd-server.exe --diffusion-model  ..\models\diffusion_models\z_image_turbo_bf16.safetensors --vae ..\models\vae\ae.sft  --llm ..\models\text_encoders\qwen_3_4b.safetensors --diffusion-fa --offload-to-cpu -v --cfg-scale 1.0
```

What this example does:

* `--diffusion-model` selects the standalone diffusion model
* `--vae` selects the VAE decoder
* `--llm` selects the text encoder / language model used by this pipeline
* `--diffusion-fa` enables flash attention in the diffusion model
* `--offload-to-cpu` reduces VRAM pressure by keeping weights in RAM when possible
* `-v` enables verbose logging
* `--cfg-scale 1.0` sets the default CFG scale for generation

After the server starts successfully:

* the web UI is available at `http://127.0.0.1:1234/`
* the native async API is available under `/sdcpp/v1/...`
* the compatibility APIs are available under `/v1/...` and `/sdapi/v1/...`

If you want to use a different host or port, pass:

```bash
--listen-ip <ip> --listen-port <port>
```

## Multiple async workers

`pool_gateway.py` can preserve the native `/sdcpp/v1` API while dispatching
independent copies to several `sd-server` processes. This is useful when a
model fits on one accelerator and aggregate throughput matters more than
splitting a single generation across devices.

Start one server per device on private ports, then run:

```bash
python3 examples/server/pool_gateway.py \
  --listen-ip 127.0.0.1 \
  --listen-port 8080 \
  --worker http://127.0.0.1:18083 \
  --worker http://127.0.0.1:18084 \
  --state-dir ./pool-state
```

Normal requests remain one-copy requests. To fan out a native image or video
request, add:

```json
{
  "copies": 2,
  "seed_stride": 1
}
```

The gateway returns a namespaced pool job ID. Completed multi-copy jobs expose
their native worker results under `result.variants[]`, including the effective
seed and worker index. Worker-local IDs may collide and are therefore never
used as public pool IDs.

### Compose worker pool

`compose.worker-pool.yaml` manages two GPU-pinned workers and the gateway as a
single stack. The workers are reachable only on the Compose network; the
gateway is the only service published on the host. Set the required image,
model-root, component-path, gateway-path, and state-directory variables, then
run:

```bash
docker compose -f examples/server/compose.worker-pool.yaml config
docker compose -f examples/server/compose.worker-pool.yaml up -d
docker compose -f examples/server/compose.worker-pool.yaml down
```

The Compose stack deliberately does not serve a browser application. Build
`examples/server/frontend`, serve its `dist/index.html` from a separate web
server, and proxy that application's `/sdcpp/` path to the gateway. This keeps
frontend deployment independent of model loading and worker restarts.

# Frontend

## Build with Frontend

The server can optionally build the web frontend and embed it into the binary as `gen_index_html.h`.

### Requirements

Install the following tools:

* **Node.js** ≥ 20
  https://nodejs.org/

* **pnpm** ≥ 10
  Install via npm:

```bash
npm install -g pnpm
```

Verify installation:

```bash
node -v
pnpm -v
```

### Install frontend dependencies

Go to the frontend directory and install dependencies:

```bash
cd examples/server/frontend
pnpm install
```

### Build the server with CMake

Enable the frontend build option when configuring CMake:

```bash
cmake -B build -DSD_SERVER_BUILD_FRONTEND=ON
cmake --build build --config Release
```

If `pnpm` is available, the build system will automatically run:

```
pnpm run build
pnpm run build:header
```

and embed the generated frontend into the server binary.

## Frontend Repository

The web frontend is maintained in a **separate repository**, https://github.com/leejet/sdcpp-webui.

If you want to modify the UI or frontend logic, please submit pull requests to the **frontend repository**.

This repository (`stable-diffusion.cpp`) only vendors the frontend periodically. Changes from the frontend repo are synchronized:

* approximately **every 1–2 weeks**, or
* when there are **major frontend updates**

Because of this, frontend changes will **not appear here immediately** after being merged upstream.

## Using an external frontend

By default, the server uses the **embedded frontend** generated during the build (`gen_index_html.h`).

You can also serve a custom frontend file instead of the embedded one by using:

```bash
--serve-html-path <path-to-index.html>
```

For example:

```bash
sd-server --serve-html-path ./index.html
```

In this case, the server will load and serve the specified `index.html` file instead of the embedded frontend. This is useful when:

* developing or testing frontend changes
* using a custom UI
* avoiding rebuilding the binary after frontend modifications

# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-server -h
```

# core58-w2a8-msvc

A minimal Windows-native inference framework for 1.58-bit ternary LLMs (BitNet).

The implementation focuses on a dense `W2A8` (Weight 2-bit, Activation 8-bit) execution path that bypasses PyTorch's native `BF16` bottlenecks on Windows. The CPU route is automated around `llama.cpp`/GGUF, while the GPU route stays as a separate Windows-native experimental path built around PyTorch, CUDA graphs, and a custom DLL.

## Features
- **Automated CPU Build Pipeline:** The `setup_env.py` script manages HuggingFace weights, generates the selected CPU kernels, and compiles the `llama-cli`, `llama-server`, and quantization binaries via CMake.
- **Universal CUDA Support (Fatbin):** The GPU kernel natively targets Ampere (`sm_80`, `sm_86`), Lovelace (`sm_89`), and Hopper (`sm_90`) simultaneously without requiring manual reconfiguration.
- **PyTorch CUDAGraphs + FFI Integration:** Python execution relies on PyTorch `CUDAGraphs` to statically allocate memory arrays in VRAM, routing execution directly into the unrolled C++ NVCC kernel via `ctypes` FFI to minimize kernel-launch overhead.

## Installation

Ensure you have Python 3.8+, Git, and Visual Studio C++ build tools installed with the LLVM/Clang toolchain enabled. For GPU builds, install the CUDA toolkit so `nvcc` is available on `PATH`.
This repository does not ship model weights or prepared GPU checkpoints.

```bash
git clone https://github.com/syn-999/core58-w2a8-msvc.git
cd core58-w2a8-msvc
git submodule update --init --recursive

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use the activated repo venv for the Python commands below. The examples are written from the repository root to avoid `cd`/relative-path mistakes. The CPU wrappers can sometimes work with a global Python, but the GPU entrypoints expect the repo environment with `torch`, `fire`, `fastapi`, and `uvicorn` installed.

The `3rdparty/llama.cpp` submodule is pinned intentionally through [`.gitmodules`](./.gitmodules). Treat that fork and revision as part of the build surface unless you are deliberately revalidating the Windows-native toolchain.

## Setup & Compilation

The environment script automates the CPU download, conversion, and native compilation process.
The CPU path is the primary automated flow. The GPU path remains a separate Windows-native experimental route on purpose and is not folded into `setup_env.py`.

**For CPU Inference (default `i2_s` GGUF):**
```bash
python setup_env.py --hf-repo tiiuae/Falcon3-10B-Instruct-1.58bit
```

**For CPU Inference (`tl2` on x86_64):**
```bash
python setup_env.py --hf-repo tiiuae/Falcon3-10B-Instruct-1.58bit --quant-type tl2
```

**For GPU Inference (CUDA):**
The GPU runtime does not ship checkpoints or compiled CUDA binaries. Prepare a checkpoint directory that contains `model_state_fp16.pt` and `model_state_int2.pt`, then build `src/cuda/bitnet_kernels/libbitnet.dll` locally with `src/cuda/bitnet_kernels/compile.bat`.
The examples below assume you place those artifacts under `models/gpu/bitnet-b1.58-2B-4T-bf16`.
The default GPU decode backend is `int2`, which uses the packed CUDA kernel. If you need a slower reference fallback for debugging, switch to `--decode_backend=fp16`.
Upstream BitNet also uses `xformers` attention for its fastest Linux/A100 path. On Windows this is optional and only works if your `xformers` wheel matches the exact local PyTorch, CUDA, and Python build.
If you want to validate that stack explicitly, run `python scripts/check_gpu_env.py`. A working `xformers` path requires the local CUDA toolkit version to match `torch.version.cuda`.

## Quick Start

Most users only need the three commands below.

**CPU terminal chat:**
Starts an interactive conversation in the terminal with the Falcon GGUF model.
```bash
python inference/cpu_inference.py -m models/cpu/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf -p "You are a concise, accurate assistant. Stay on topic and stop when the answer is complete." -cnv -t 8 -c 4096 -n 512
```

**CPU browser chat:**
Starts the local `llama-server.exe` web UI on `http://127.0.0.1:8080`.
```bash
python inference/cpu_server.py -m models/cpu/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf -p "You are a concise, accurate assistant. Stay on topic and stop when the answer is complete." -t 8 -c 4096 --host 127.0.0.1 --port 8080
```

**GPU terminal chat:**
Runs the Windows-native CUDA path from the activated repo venv.
```bash
python inference/gpu_generate.py models/gpu/bitnet-b1.58-2B-4T-bf16 --interactive=True --chat_format=True --sampling=True --max_new_tokens=256
```

**GPU browser chat:**
Starts a simple local browser UI backed by the FastAPI/OpenAI-compatible GPU server on `http://127.0.0.1:8000`.
The prompt budget below is the tested long-form setting that kept browser/API responses coherent without changing the code defaults globally.
```powershell
$env:BITNET_CKPT_DIR = "models/gpu/bitnet-b1.58-2B-4T-bf16"
$env:BITNET_PROMPT_LENGTH = "512"
$env:BITNET_MAX_TOKENS = "768"
python inference/gpu_server.py
```

## Additional Commands

**CPU one-shot generation:**
Routes directly via the C++ `llama-cli.exe` engine.
```bash
python inference/cpu_inference.py -m models/cpu/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf -p "A complete structural breakdown of a cell is" -n 200
```

**CPU interactive chat server:**
Launches the local `llama-server.exe` web UI. The wrapper resolves the common Falcon model filename even if your local `models/cpu` tree is nested one level deeper. Continuous batching is left off by default for stability on Windows; add `--continuous-batching` if you want to experiment with it.
```bash
python inference/cpu_server.py -m models/cpu/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf -p "You are a concise, accurate assistant. Stay on topic and stop when the answer is complete." -t 8 -c 4096 --host 127.0.0.1 --port 8080
```

**GPU execution:**
Routes via the native PyTorch/NVCC wrapper from the activated repo venv. If you keep separate local environments instead of the single `venv` shown above, use your GPU-specific interpreter such as `venv_gpu\Scripts\python.exe`.
```bash
python inference/gpu_generate.py models/gpu/bitnet-b1.58-2B-4T-bf16 --interactive=True --chat_format=True --sampling=True --max_new_tokens=256
```

**GPU browser/API server:**
Serves both a small local chat UI at `http://127.0.0.1:8000` and the OpenAI-style API/docs at `/v1/chat/completions` and `/docs`.
```powershell
$env:BITNET_CKPT_DIR = "models/gpu/bitnet-b1.58-2B-4T-bf16"
$env:BITNET_PROMPT_LENGTH = "512"
$env:BITNET_MAX_TOKENS = "768"
python inference/gpu_server.py
```

**GPU server tuning:**
The server defaults to a larger prompt budget than the CLI. Increase these before startup if you want longer conversations or larger generations.
```powershell
$env:BITNET_PROMPT_LENGTH = "512"
$env:BITNET_MAX_TOKENS = "1024"
$env:BITNET_TEMPERATURE = "0.2"
$env:BITNET_TOP_P = "0.9"
python inference/gpu_server.py
```

**Recommended browser capability test prompts:**
- CPU browser: `Explain quantum entanglement with one analogy and one real-world use case.`
- CPU browser follow-up: `Now summarize that in three bullet points.`
- GPU browser: `Write a concise but detailed explanation of how quantum entanglement works, with one practical analogy and one real-world use case.`
- GPU browser follow-up: `Now give me a shorter version in four sentences.`

**Reference BF16 decode:**
Uses the slower BF16 fallback path instead of the packed CUDA kernel.
```bash
python inference/gpu_generate.py models/gpu/bitnet-b1.58-2B-4T-bf16 --interactive=True --chat_format=True --sampling=True --max_new_tokens=256 --decode_backend=fp16
```

**Preparing a New GPU Checkpoint:**
```bash
python utils/gpu/convert_safetensors.py --safetensors_file models/gpu/bitnet-b1.58-2B-4T-bf16/model.safetensors --output models/gpu/bitnet-b1.58-2B-4T-bf16/model_state.pt --model_name 2B
python utils/gpu/convert_checkpoint.py --input models/gpu/bitnet-b1.58-2B-4T-bf16/model_state.pt
cd src/cuda/bitnet_kernels
.\compile.bat
```

## Runtime Notes

- `cpu_inference.py` exits when generation finishes. With `-cnv`, it remains attached to your terminal session until you stop it.
- `cpu_server.py` keeps a `llama-server.exe` process running until you press `Ctrl+C`.
- `gpu_generate.py --interactive=True` keeps the Python process alive until you exit the prompt or press `Ctrl+C`.
- `gpu_server.py` serves a browser UI at `/`, API docs at `/docs`, and an OpenAI-style chat route at `/v1/chat/completions`.
- Seeing one active model process is normal. Seeing multiple `llama-cli.exe` or `llama-server.exe` entries usually means you started more than one session or left an older server open.
- The GPU path now stops cleanly on both `<|eot_id|>` and `<|end_of_text|>`, which improves browser/API chat termination and reduces repetitive trailing output.

To inspect or clean up lingering CPU runtime processes on Windows:

```powershell
Get-Process llama* -ErrorAction SilentlyContinue
Stop-Process -Name llama-cli,llama-server -Force
```

## Smoke Test

For a quick Windows release check:

```powershell
.\scripts\smoke_test.ps1
```

To also verify the local CUDA helper build:

```powershell
.\scripts\smoke_test.ps1 -CheckGpu
```

For low-level CUDA kernel debugging on Windows:

```powershell
venv_gpu\Scripts\python.exe .\scripts\gpu_kernel_selftest.py
```

To inspect whether your GPU environment is ready for optional `xformers` attention:

```powershell
venv_gpu\Scripts\python.exe .\scripts\check_gpu_env.py
```

## License

This project is released under the MIT License. It includes work derived from Microsoft BitNet and `llama.cpp`; see [`LICENSE`](./LICENSE).

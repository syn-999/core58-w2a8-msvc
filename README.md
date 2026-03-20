# core58-w2a8-msvc

A minimal Windows-native inference framework for 1.58-bit ternary LLMs.

The repo keeps the CPU and GPU paths separate on purpose:
- the CPU path is the primary automated flow, built around `llama.cpp` and GGUF
- the GPU path is a Windows-native experimental runtime built around PyTorch, CUDA graphs, and a custom DLL

This implementation focuses on a dense `W2A8` execution path. It does not implement Sparse-BitNet-style structured sparsity.

## Features

- Automated CPU build pipeline through `setup_env.py`
- Native Windows CPU wrappers for `llama-cli` and `llama-server`
- Native Windows GPU runtime with packed `int2` decode and an optional `fp16` fallback
- Fatbin CUDA helper build targeting `sm_80`, `sm_86`, `sm_89`, and `sm_90`

## Prerequisites

- Windows
- Python 3.10 or later
- Git
- Visual Studio Build Tools with C++ support and the LLVM/Clang toolchain enabled
- CUDA toolkit on `PATH` if you plan to build or run the GPU path

This repository does not ship model weights, prepared GPU checkpoints, or prebuilt binaries.

## Installation

All commands below assume your current working directory is the repository root.

```powershell
git clone https://github.com/syn-999/core58-w2a8-msvc.git
cd core58-w2a8-msvc
git submodule update --init --recursive
```

Create the CPU environment:

```powershell
python -m venv venv_cpu
.\venv_cpu\Scripts\python.exe -m pip install -r .\requirements.txt
```

Create the GPU environment:

```powershell
python -m venv venv_gpu
.\venv_gpu\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\venv_gpu\Scripts\python.exe -m pip install -r .\requirements.txt
```

The GPU instructions above match the tested local stack for this repo: Python `3.11`, PyTorch `2.5.1+cu121`, and Windows CUDA execution. If you choose a different PyTorch or CUDA stack, use the official PyTorch selector or previous-version guide and revalidate the GPU path afterward.

Sources:
- https://pytorch.org/
- https://pytorch.org/get-started/previous-versions/

The `3rdparty/llama.cpp` submodule is pinned intentionally through [`.gitmodules`](./.gitmodules). Treat that fork and revision as part of the build surface unless you are deliberately revalidating the Windows toolchain.

## Setup

### CPU path

`setup_env.py` automates the CPU download, GGUF conversion, kernel codegen, and native build.

If you use `--model-dir` instead of `--hf-repo`, that directory must contain the original Hugging Face checkpoint files (`.safetensors` or `.bin`). A directory that only contains already-converted GGUF files is not a valid input for `setup_env.py`.

Default `i2_s` build:

```powershell
.\venv_cpu\Scripts\python.exe .\setup_env.py --hf-repo tiiuae/Falcon3-10B-Instruct-1.58bit
```

`tl2` build on `x86_64`:

```powershell
.\venv_cpu\Scripts\python.exe .\setup_env.py --hf-repo tiiuae/Falcon3-10B-Instruct-1.58bit --quant-type tl2
```

### GPU path

The GPU runtime is not folded into `setup_env.py`.
The current GPU runtime is validated against the `BitNet-b1.58-2B-4T` checkpoint layout used by `models\gpu\bitnet-b1.58-2B-4T-bf16`. Other GPU model shapes are not plug-and-play and would require code changes.

Prepare a checkpoint directory that contains:
- `model_state_fp16.pt`
- `model_state_int2.pt`

Then build the CUDA helper DLL:

```powershell
cmd /c .\src\cuda\bitnet_kernels\compile.bat
```

The examples below assume the GPU artifacts live under `models\gpu\bitnet-b1.58-2B-4T-bf16`.

The default GPU decode backend is `int2`. If you need a slower reference path for debugging, use `--decode_backend=fp16`.
If `libbitnet.dll` or either checkpoint file is missing, the GPU entrypoints now fail with a direct path-level error instead of a raw loader exception.

Optional `xformers` attention is not required for the current Windows runtime. If you want to validate that stack explicitly, run:

```powershell
.\venv_gpu\Scripts\python.exe .\scripts\check_gpu_env.py
```

A working `xformers` path requires the local CUDA toolkit version to match `torch.version.cuda`.

## Quick Start

### CPU terminal chat

```powershell
.\venv_cpu\Scripts\python.exe .\inference\cpu_inference.py -m .\models\cpu\Falcon3-10B-Instruct-1.58bit\ggml-model-i2_s.gguf -p "You are a concise, accurate assistant. Stay on topic and stop when the answer is complete." -cnv -t 8 -c 4096 -temp 0.2 -n 512
```

### CPU browser chat

```powershell
.\venv_cpu\Scripts\python.exe .\inference\cpu_server.py -m .\models\cpu\Falcon3-10B-Instruct-1.58bit\ggml-model-i2_s.gguf -p "You are a concise, accurate assistant. Stay on topic and stop when the answer is complete." -t 8 -c 4096 --temperature 0.2 --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

### GPU terminal chat

```powershell
.\venv_gpu\Scripts\python.exe .\inference\gpu_generate.py .\models\gpu\bitnet-b1.58-2B-4T-bf16 --interactive=True --chat_format=True --sampling=True --prompt_length=1024 --max_new_tokens=512
```

### GPU browser chat

```powershell
$env:BITNET_CKPT_DIR = ".\models\gpu\bitnet-b1.58-2B-4T-bf16"
$env:BITNET_PROMPT_LENGTH = "1024"
$env:BITNET_MAX_TOKENS = "512"
.\venv_gpu\Scripts\python.exe .\inference\gpu_server.py
```

Open `http://127.0.0.1:8000`.

## Reference Commands

### CPU instruct one-shot

```powershell
.\venv_cpu\Scripts\python.exe .\inference\cpu_inference.py -m .\models\cpu\Falcon3-10B-Instruct-1.58bit\ggml-model-i2_s.gguf -p "You are a concise, accurate assistant. Explain the structure of a biological cell in one clear paragraph." -cnv -t 8 -c 4096 -temp 0.2 -n 256
```

### GPU browser and API server

```powershell
$env:BITNET_CKPT_DIR = ".\models\gpu\bitnet-b1.58-2B-4T-bf16"
$env:BITNET_PROMPT_LENGTH = "1024"
$env:BITNET_MAX_TOKENS = "512"
.\venv_gpu\Scripts\python.exe .\inference\gpu_server.py
```

This serves:
- browser UI at `/`
- API docs at `/docs`
- OpenAI-style chat route at `/v1/chat/completions`

### GPU longer-context server profile

```powershell
$env:BITNET_PROMPT_LENGTH = "1536"
$env:BITNET_MAX_TOKENS = "768"
$env:BITNET_TEMPERATURE = "0.2"
$env:BITNET_TOP_P = "0.9"
.\venv_gpu\Scripts\python.exe .\inference\gpu_server.py
```

Use this profile when you want longer multi-turn GPU chats and have enough VRAM headroom. The default browser quick start above is the safer general-purpose setting.

### BF16 decode fallback

```powershell
.\venv_gpu\Scripts\python.exe .\inference\gpu_generate.py .\models\gpu\bitnet-b1.58-2B-4T-bf16 --interactive=True --chat_format=True --sampling=True --prompt_length=1024 --max_new_tokens=512 --decode_backend=fp16
```

### Preparing a new GPU checkpoint

```powershell
.\venv_gpu\Scripts\python.exe .\utils\gpu\convert_safetensors.py --safetensors_file .\models\gpu\bitnet-b1.58-2B-4T-bf16\model.safetensors --output .\models\gpu\bitnet-b1.58-2B-4T-bf16\model_state.pt --model_name 2B
.\venv_gpu\Scripts\python.exe .\utils\gpu\convert_checkpoint.py --input .\models\gpu\bitnet-b1.58-2B-4T-bf16\model_state.pt
cmd /c .\src\cuda\bitnet_kernels\compile.bat
```

## Runtime Notes

- `cpu_inference.py` exits when generation finishes. With `-cnv`, it stays attached to your terminal session until you stop it.
- `cpu_server.py` keeps one `llama-server.exe` child alive until you press `Ctrl+C`.
- `gpu_generate.py --interactive=True` keeps one Python process alive until you exit the prompt or press `Ctrl+C`.
- `gpu_server.py` serves a browser UI at `/`, API docs at `/docs`, and an OpenAI-style chat route at `/v1/chat/completions`.
- The GPU browser UI is self-contained and does not require loading frontend libraries from the public internet.
- Seeing one active model process is normal. Seeing multiple `llama-cli.exe` or `llama-server.exe` entries usually means you started more than one session or left an older one running.
- The CPU browser route uses the vendored `llama.cpp` web UI, so the browser tab title is still upstream by default and the browser page may retain local UI state across reloads.
- The GPU browser route uses this repo's own FastAPI frontend and identifies as `core58 GPU Chat`. Its conversation state lives in page memory and resets on refresh or server restart.
- The GPU backend now gracefully truncates the oldest chat history if the conversation exceeds `BITNET_PROMPT_LENGTH`. 
- GPU benchmarking should be warmed once before you record throughput. Cold first runs can underreport tokens per second.

To inspect or clean up lingering CPU runtime processes on Windows:

```powershell
Get-Process llama* -ErrorAction SilentlyContinue
Stop-Process -Name llama-cli,llama-server -Force
```

## Smoke Test

Quick Windows release check. This is a clean rebuild of the CPU wrapper binaries, not a sub-second smoke probe:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1
```

On a normal Windows dev box, expect this to take a few minutes.

Also verify the local CUDA helper build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1 -CheckGpu
```

If you run the GPU variant, stop any live `gpu_generate.py` or `gpu_server.py` session first so `libbitnet.dll` is not locked.

If you want the smoke test output to refresh the runtime binaries under `build\bin\Release` for packaging, keep the build directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1 -BuildDir build -KeepBuildDir
```

Low-level CUDA kernel self-test:

```powershell
.\venv_gpu\Scripts\python.exe .\scripts\gpu_kernel_selftest.py
```

## Release Packaging

After you have a validated `build\bin\Release` directory, create a publishable zip with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1
```

The packaging script expects a clean git working tree by default so the archive name matches the commit it contains.

That archive contains:
- tracked source files, including the pinned `llama.cpp` submodule contents
- `build\bin\Release\llama-cli.exe`
- `build\bin\Release\llama-server.exe`
- the required `ggml.dll` and `llama.dll`
- `src\cuda\bitnet_kernels\libbitnet.dll` if it exists locally

It intentionally excludes:
- local model weights
- virtual environments
- `.git` metadata
- transient build, profile, and log artifacts

## License

This project is released under the MIT License. It includes work derived from Microsoft BitNet and `llama.cpp`; see [`LICENSE`](./LICENSE).

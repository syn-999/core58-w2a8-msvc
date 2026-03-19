# core58-w2a8-msvc

A minimal inferencing framework for 1.58-bit Ternary LLMs (BitNet), natively optimized for Windows MSVC environments.

The implementation focuses on bypassing PyTorch's native `BF16` computational bottlenecks utilizing custom `W2A8` (Weight 2-bit, Activation 8-bit) quantization kernels. It provides a clean, automated build pipeline that avoids common CMake failures on Windows and strips out upstream bloat, reducing the compiled footprint to ~25MB.

## Features
- **Automated MSVC/Clang Build Pipeline:** The `setup_env.py` script automatically manages HuggingFace weights, compiles the required `libbitnet.dll` and `llama-cli` binaries via CMake, and quantizes the model.
- **Universal CUDA Support (Fatbin):** The GPU kernel natively targets Ampere (`sm_80`, `sm_86`), Lovelace (`sm_89`), and Hopper (`sm_90`) simultaneously without requiring manual reconfiguration.
- **PyTorch CUDAGraphs + FFI Integration:** Python execution relies on PyTorch `CUDAGraphs` to statically allocate memory arrays in VRAM, routing execution directly into the unrolled C++ NVCC kernel via `ctypes` FFI to minimize kernel-launch overhead.

## Installation

Ensure you have Python 3.8+ and MSVC visual tools installed.

```bash
git clone https://github.com/your-username/core58-w2a8-msvc.git
cd core58-w2a8-msvc

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Setup & Compilation

The environment script automates the download, conversion, and native compilation process. 

**For CPU Inference (AVX2 / NEON):**
```bash
python setup_env.py --hf-repo tiiuae/Falcon3-10B-Instruct-1.58bit
```

**For GPU Inference (CUDA):**
Use the `-p` flag to compile with natively tuned Block Memory tensors (BM/BK blocking) specific to your hardware architecture for maximum throughput.
```bash
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T -p
```

## Inference

**CPU Execution:**
Routes directly via the C++ `llama-cli.exe` engine.
```bash
cd inference
python cpu_inference.py -m ../models/cpu/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf -p "A complete structural breakdown of a cell is" -n 200
```

**GPU Execution:**
Routes via the native PyTorch/NVCC wrapper.
```bash
cd inference
python gpu_generate.py ../models/gpu/bitnet-b1.58-2B-4T-bf16 --interactive
```

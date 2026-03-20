# Contributing

Thanks for your interest in improving `core58-w2a8-msvc`.

## Scope

This repo is intentionally split into two lanes:

- CPU: the primary automated path, built around `llama.cpp` and GGUF
- GPU: a separate experimental Windows-native runtime

Changes should preserve that separation unless the work is explicitly about unifying or redesigning the architecture.

## Before You Start

- Search existing issues and release notes before opening a new bug or feature request.
- For larger changes, open an issue first so scope and fit can be discussed before implementation.
- Keep user-facing docs aligned with any behavior change.

## Development Expectations

- Prefer small, focused pull requests over broad unrelated refactors.
- Keep Windows as the primary supported platform.
- Do not commit model weights, virtual environments, local logs, or generated build artifacts.
- If you change packaging or release behavior, keep `scripts/smoke_test.ps1` and `scripts/package_release.ps1` aligned.

## Validation

Run the checks that match your change:

```powershell
python -m py_compile .\inference\cpu_inference.py .\inference\cpu_server.py .\inference\gpu_generate.py .\inference\gpu_server.py
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1
```

If your change touches the GPU helper build or GPU runtime setup, also run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1 -CheckGpu
.\venv_gpu\Scripts\python.exe .\scripts\gpu_kernel_selftest.py
```

If your change affects release packaging, verify:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1
```

## Performance Changes

For performance claims, include:

- hardware details
- model and quantization used
- exact command line
- before/after numbers
- commit or branch tested

## Pull Requests

Pull requests should clearly state:

- what changed
- why it changed
- what was validated
- any remaining risks or limits

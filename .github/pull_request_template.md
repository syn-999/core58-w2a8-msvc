## Summary

- What changed?
- Why was it needed?

## Validation

- [ ] `python -m py_compile` on the touched Python entrypoints
- [ ] `powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1`
- [ ] `powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1 -CheckGpu` if GPU build/runtime behavior changed
- [ ] Docs updated if user-facing behavior changed

## Notes

- Hardware used for testing:
- Commands used:
- Remaining risks or limitations:

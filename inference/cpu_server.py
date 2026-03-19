import os
import sys
import signal
import platform
import argparse
import subprocess
import ctypes
from pathlib import Path

ACTIVE_PROCESS = None
ACTIVE_JOB = None

if platform.system() == "Windows":
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    KERNEL32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    KERNEL32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    KERNEL32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    KERNEL32.CloseHandle.restype = wintypes.BOOL

def assign_kill_on_close_job(process: subprocess.Popen):
    global ACTIVE_JOB
    if platform.system() != "Windows":
        return

    job = KERNEL32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not KERNEL32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        KERNEL32.CloseHandle(job)
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    if not KERNEL32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle)):
        KERNEL32.CloseHandle(job)
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    ACTIVE_JOB = job

def close_active_job():
    global ACTIVE_JOB
    if platform.system() != "Windows" or not ACTIVE_JOB:
        return
    KERNEL32.CloseHandle(ACTIVE_JOB)
    ACTIVE_JOB = None

def terminate_active_process():
    global ACTIVE_PROCESS
    if ACTIVE_PROCESS is None or ACTIVE_PROCESS.poll() is not None:
        return
    ACTIVE_PROCESS.terminate()
    try:
        ACTIVE_PROCESS.wait(timeout=5)
    except subprocess.TimeoutExpired:
        ACTIVE_PROCESS.kill()

def run_command(command, shell=False):
    """Run a system command and keep track of the child for clean shutdown."""
    global ACTIVE_PROCESS
    try:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        ACTIVE_PROCESS = subprocess.Popen(command, shell=shell, creationflags=creationflags)
        assign_kill_on_close_job(ACTIVE_PROCESS)
        returncode = ACTIVE_PROCESS.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    except KeyboardInterrupt:
        terminate_active_process()
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)
    finally:
        ACTIVE_PROCESS = None
        close_active_job()

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

def resolve_binary(name: str) -> str:
    candidates = []
    build_dir = REPO_ROOT / "build" / "bin"
    if platform.system() == "Windows":
        candidates.extend([
            build_dir / "Release" / f"{name}.exe",
            build_dir / f"{name}.exe",
        ])
    else:
        candidates.append(build_dir / name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    print(
        f"Unable to find {name}. Build artifacts are missing under {build_dir}. "
        f"Rebuild with `powershell -ExecutionPolicy Bypass -File .\\scripts\\smoke_test.ps1 -BuildDir build -KeepBuildDir` from the repo root."
    )
    sys.exit(1)

def resolve_model_path(model_arg: str) -> str:
    requested = Path(model_arg)
    candidates = []

    if requested.exists():
        return str(requested)

    rel_candidate = (THIS_DIR / requested).resolve()
    if rel_candidate.exists():
        return str(rel_candidate)

    if requested.name:
        matches = list((REPO_ROOT / "models" / "cpu").rglob(requested.name))
        if len(matches) == 1:
            return str(matches[0])
        candidates.extend(matches)

    print(f"Unable to find model: {model_arg}")
    if candidates:
        print("Matching candidates:")
        for candidate in candidates:
            print(f"  {candidate}")
    sys.exit(1)

def run_server():
    server_path = resolve_binary("llama-server")
    model_path = resolve_model_path(args.model)
    
    command = [
        f'{server_path}',
        '-m', model_path,
        '--alias', 'core58-cpu',
        '-c', str(args.ctx_size),
        '-t', str(args.threads),
        '-n', str(args.n_predict),
        '-ngl', '0',
        '--temp', str(args.temperature),
        '--host', args.host,
        '--port', str(args.port),
    ]

    if args.continuous_batching:
        command.append('-cb')
    
    if args.prompt:
        command.extend(['-p', args.prompt])
    
    # Note: -cnv flag is removed as it's not supported by the server
    
    print(f"Starting server on {args.host}:{args.port}")
    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, shutting down server...")
    terminate_active_process()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    parser = argparse.ArgumentParser(description='Run the core58 CPU server')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="../models/cpu/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str, help="System prompt for the model", required=False)
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", required=False, default=4096)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the context window", required=False, default=2048)
    parser.add_argument("--temperature", type=float, help="Temperature for sampling", required=False, default=0.8)
    parser.add_argument("--host", type=str, help="IP address to listen on", required=False, default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Port to listen on", required=False, default=8080)
    parser.add_argument("--continuous-batching", action="store_true", help="Enable llama-server continuous batching")
    
    args = parser.parse_args()
    run_server()

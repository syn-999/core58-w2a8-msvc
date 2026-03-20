param(
    [string]$BuildDir = "build",
    [switch]$CheckGpu,
    [switch]$KeepBuildDir,
    [switch]$CleanBuildDir
)

$ErrorActionPreference = "Stop"

if ($KeepBuildDir -and $CleanBuildDir) {
    throw "Use either -KeepBuildDir or -CleanBuildDir, not both."
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    Write-Host "==> $Label"
    & $Action
}

function Resolve-CMakeCommand {
    param([string]$RepoRoot)

    if (Get-Command "cmake" -ErrorAction SilentlyContinue) {
        return [pscustomobject]@{
            Executable = "cmake"
            UsePythonModule = $false
        }
    }

    foreach ($repoPython in @(
        (Join-Path $RepoRoot "venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "venv_cpu\Scripts\python.exe")
    )) {
        if (Test-Path $repoPython) {
            & $repoPython -m cmake --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{
                    Executable = $repoPython
                    UsePythonModule = $true
                }
            }
        }
    }

    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        & python -m cmake --version *> $null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{
                Executable = "python"
                UsePythonModule = $true
            }
        }
    }

    throw "Unable to find CMake on PATH or via python -m cmake."
}

function Resolve-ClangToolchain {
    $clangCmd = Get-Command "clang" -ErrorAction SilentlyContinue
    $clangxxCmd = Get-Command "clang++" -ErrorAction SilentlyContinue
    if ($clangCmd -and $clangxxCmd) {
        return [pscustomobject]@{
            CCompiler = $clangCmd.Source
            CxxCompiler = $clangxxCmd.Source
        }
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $vsPath) {
            foreach ($llvmDir in @(
                (Join-Path $vsPath "VC\Tools\Llvm\x64\bin"),
                (Join-Path $vsPath "VC\Tools\Llvm\bin")
            )) {
                $clangPath = Join-Path $llvmDir "clang.exe"
                $clangxxPath = Join-Path $llvmDir "clang++.exe"
                if ((Test-Path $clangPath) -and (Test-Path $clangxxPath)) {
                    return [pscustomobject]@{
                        CCompiler = $clangPath
                        CxxCompiler = $clangxxPath
                    }
                }
            }
        }
    }

    throw "Unable to find clang/clang++. Install the Visual Studio LLVM/Clang toolchain."
}

function Invoke-CMake {
    param(
        [psobject]$Command,
        [string[]]$CommandArgs
    )

    if ($Command.UsePythonModule) {
        & $Command.Executable -m cmake @CommandArgs
    } else {
        & $Command.Executable @CommandArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "CMake command failed."
    }
}

function Find-Binary {
    param(
        [string]$Root,
        [string]$Name
    )

    $candidate = Get-ChildItem -Path $Root -Recurse -File -Filter $Name -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $candidate) {
        throw "Expected binary '$Name' was not produced under '$Root'."
    }
    return $candidate.FullName
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir
} else {
    Join-Path $RepoRoot $BuildDir
}
$ResolvedBuildDir = [System.IO.Path]::GetFullPath($ResolvedBuildDir)
$DefaultBuildDir = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "build"))
$PreserveBuildDir = if ($KeepBuildDir) {
    $true
} elseif ($CleanBuildDir) {
    $false
} else {
    $ResolvedBuildDir -eq $DefaultBuildDir
}

$CMakeCommand = Resolve-CMakeCommand -RepoRoot $RepoRoot
$ClangToolchain = Resolve-ClangToolchain
$cmakeLabel = if ($CMakeCommand.UsePythonModule) {
    "$($CMakeCommand.Executable) -m cmake"
} else {
    $CMakeCommand.Executable
}
Write-Host "Using CMake via $cmakeLabel"
Write-Host "Using clang from $($ClangToolchain.CCompiler)"

Push-Location $RepoRoot
try {
    if (Test-Path $ResolvedBuildDir) {
        Remove-Item -Recurse -Force $ResolvedBuildDir
    }

    Invoke-Step "Configuring CMake" {
        $configureArgs = @(
            "-S", ".",
            "-B", $ResolvedBuildDir,
            "-T", "ClangCL",
            "-DBITNET_X86_TL2=ON",
            "-DBITNET_BUILD_SERVER=ON",
            "-DCMAKE_C_COMPILER=$($ClangToolchain.CCompiler)",
            "-DCMAKE_CXX_COMPILER=$($ClangToolchain.CxxCompiler)"
        )
        Invoke-CMake -Command $CMakeCommand -CommandArgs $configureArgs
    }

    Invoke-Step "Building llama-cli and llama-server" {
        Invoke-CMake -Command $CMakeCommand -CommandArgs @(
            "--build", $ResolvedBuildDir,
            "--config", "Release",
            "--target", "llama-cli", "llama-server"
        )
    }

    $cliPath = Find-Binary -Root $ResolvedBuildDir -Name "llama-cli.exe"
    $serverPath = Find-Binary -Root $ResolvedBuildDir -Name "llama-server.exe"
    Write-Host "Found $cliPath"
    Write-Host "Found $serverPath"

    if ($CheckGpu) {
        Invoke-Step "Building libbitnet.dll" {
            & cmd /c src\cuda\bitnet_kernels\compile.bat
            if ($LASTEXITCODE -ne 0) {
                throw "GPU helper build failed."
            }
        }

        $dllPath = Join-Path $RepoRoot "src\cuda\bitnet_kernels\libbitnet.dll"
        if (-not (Test-Path $dllPath)) {
            throw "Expected GPU helper DLL was not produced."
        }

        Write-Host "Found $dllPath"

        foreach ($artifact in "libbitnet.exp", "libbitnet.lib") {
            $artifactPath = Join-Path $RepoRoot "src\cuda\bitnet_kernels\$artifact"
            if (Test-Path $artifactPath) {
                Remove-Item -Force $artifactPath
            }
        }
    }
}
finally {
    Pop-Location
    if ((-not $PreserveBuildDir) -and (Test-Path $ResolvedBuildDir)) {
        Remove-Item -Recurse -Force $ResolvedBuildDir
    }

    $submoduleBuildInfo = Join-Path $RepoRoot "3rdparty\llama.cpp\common\build-info.cpp"
    if (Test-Path $submoduleBuildInfo) {
        & git -C 3rdparty/llama.cpp restore common/build-info.cpp
    }
}

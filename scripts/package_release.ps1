param(
    [string]$OutputDir = "..\release-artifacts",
    [string]$BuildDir = "build\bin\Release",
    [string]$NamePrefix = "core58-w2a8-msvc-windows",
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

function Copy-TrackedFiles {
    param(
        [string]$RepoRoot,
        [string]$DestinationRoot
    )

    $trackedFiles = & git -C $RepoRoot ls-files --recurse-submodules
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to enumerate tracked files."
    }

    foreach ($relativePath in $trackedFiles) {
        if ([string]::IsNullOrWhiteSpace($relativePath)) {
            continue
        }

        $sourcePath = Join-Path $RepoRoot $relativePath
        if (-not (Test-Path $sourcePath -PathType Leaf)) {
            continue
        }

        $destinationPath = Join-Path $DestinationRoot $relativePath
        $destinationDir = Split-Path -Parent $destinationPath
        if ($destinationDir -and -not (Test-Path $destinationDir)) {
            New-Item -ItemType Directory -Path $destinationDir | Out-Null
        }

        Copy-Item -Path $sourcePath -Destination $destinationPath -Force
    }
}

function Copy-RuntimeArtifacts {
    param(
        [string]$RepoRoot,
        [string]$DestinationRoot,
        [string]$ResolvedBuildDir
    )

    $requiredBuildArtifacts = @(
        "ggml.dll",
        "llama.dll",
        "llama-cli.exe",
        "llama-server.exe"
    )

    foreach ($artifact in $requiredBuildArtifacts) {
        $artifactPath = Join-Path $ResolvedBuildDir $artifact
        if (-not (Test-Path $artifactPath -PathType Leaf)) {
            throw "Missing runtime artifact: $artifactPath"
        }
    }

    $releaseBuildDir = Join-Path $DestinationRoot "build\bin\Release"
    if (-not (Test-Path $releaseBuildDir)) {
        New-Item -ItemType Directory -Path $releaseBuildDir | Out-Null
    }

    foreach ($artifact in $requiredBuildArtifacts) {
        Copy-Item -Path (Join-Path $ResolvedBuildDir $artifact) -Destination (Join-Path $releaseBuildDir $artifact) -Force
    }

    $gpuDllPath = Join-Path $RepoRoot "src\cuda\bitnet_kernels\libbitnet.dll"
    if (Test-Path $gpuDllPath -PathType Leaf) {
        $gpuDllDestination = Join-Path $DestinationRoot "src\cuda\bitnet_kernels"
        if (-not (Test-Path $gpuDllDestination)) {
            New-Item -ItemType Directory -Path $gpuDllDestination | Out-Null
        }
        Copy-Item -Path $gpuDllPath -Destination (Join-Path $gpuDllDestination "libbitnet.dll") -Force
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $RepoRoot $OutputDir
}
$ResolvedBuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir
} else {
    Join-Path $RepoRoot $BuildDir
}

if (-not (Test-Path $ResolvedBuildDir -PathType Container)) {
    throw "Build directory not found: $ResolvedBuildDir. Rebuild with `powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1` first."
}

if (-not $AllowDirty) {
    $statusLines = & git -C $RepoRoot status --short
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect repository status."
    }
    if ($statusLines) {
        throw "Refusing to package a dirty working tree. Commit or stash changes first, or rerun with -AllowDirty."
    }
}

$head = (& git -C $RepoRoot rev-parse --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $head) {
    throw "Unable to resolve git HEAD."
}

$dateStamp = Get-Date -Format "yyyyMMdd"
$archiveBaseName = "$NamePrefix-$dateStamp-$head"
$stagingRoot = Join-Path $env:TEMP ("$archiveBaseName-staging")
$archivePath = Join-Path $ResolvedOutputDir "$archiveBaseName.zip"

if (Test-Path $stagingRoot) {
    Remove-Item -Recurse -Force $stagingRoot
}
if (-not (Test-Path $ResolvedOutputDir)) {
    New-Item -ItemType Directory -Path $ResolvedOutputDir | Out-Null
}
if (Test-Path $archivePath) {
    Remove-Item -Force $archivePath
}

try {
    $packageRoot = Join-Path $stagingRoot "core58-w2a8-msvc"
    New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null

    Copy-TrackedFiles -RepoRoot $RepoRoot -DestinationRoot $packageRoot
    Copy-RuntimeArtifacts -RepoRoot $RepoRoot -DestinationRoot $packageRoot -ResolvedBuildDir $ResolvedBuildDir

    Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $archivePath -CompressionLevel Optimal
    Write-Host $archivePath
}
finally {
    if (Test-Path $stagingRoot) {
        Remove-Item -Recurse -Force $stagingRoot
    }
}

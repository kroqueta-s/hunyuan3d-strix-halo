# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    Hunyuan3D 2.1 (shape) on Strix Halo (gfx1151 / Windows / ROCm) - one-command install.

.DESCRIPTION
    Creates a dedicated virtual environment, installs ROCm PyTorch, clones the
    upstream Hunyuan3D-2.1 repository at a pinned commit, downloads the weights and
    writes a .env file.

    **No CUDA-only package is installed.** the attention path is replaced at launch
    time by a pure-torch shim in runners/hunyuan3d/. Upstream code is never patched.

.PARAMETER Root
    Where the virtual environment, the upstream clone and the weights go.
    Defaults to the parent of this repository.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Root D:\models\hunyuan3d
#>
[CmdletBinding()]
param(
    # Where the virtual environment, the upstream clone and the weights go.
    # Empty means: next to this repository, in hunyuan3d-strix-halo-data.
    [string]$Root = "",
    [string]$Python = "py -3.12"
)

$ErrorActionPreference = "Stop"
# $PSScriptRoot can be empty while param defaults are evaluated under
# Windows PowerShell 5.1, so the paths are resolved here instead.
$repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $repo) "hunyuan3d-strix-halo-data" }

# Pinned versions. Do not float these: the ROCm wheels and the upstream commit
# are the two things that decide whether this works at all.
$TorchIndex = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
$TorchVersion = "2.9.1+rocm7.2.1"
$TorchvisionVersion = "0.24.1+rocm7.2.1"
$UpstreamUrl = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
$UpstreamCommit = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
$WeightsRepo = "tencent/Hunyuan3D-2.1"

$venv = Join-Path $Root ".venv"
$upstream = Join-Path $Root "Hunyuan3D-2.1"
$weights = Join-Path $Root "weights"
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "==> Root: $Root"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1. Virtual environment ------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "==> Creating virtual environment"
    & cmd /c "$Python -m venv `"$venv`""
}
& $py -m pip install --upgrade pip

# 2. ROCm PyTorch -------------------------------------------------------------
# torch requires the `rocm` meta-package, which lives on the same index.
# Passing the wheel URL directly fails with "No matching distribution for rocm".
Write-Host "==> Installing ROCm PyTorch"
& $py -m pip install --no-cache-dir --find-links $TorchIndex `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion"

# 3. Upstream repository (never forked, never patched) ------------------------
# git reports progress on stderr. Under output redirection, PowerShell 5.1
# turns native stderr into error records, and ErrorActionPreference=Stop would
# kill the script on the first progress line - so git runs with it relaxed and
# its exit code is checked instead.
$ErrorActionPreference = "Continue"
if (-not (Test-Path $upstream)) {
    Write-Host "==> Cloning upstream Hunyuan3D-2.1"
    # A shallow clone: a full history (TRELLIS in particular) can stall for
    # minutes in server-side pack preparation. The pinned commit is fetched
    # right below, also shallow.
    git clone --depth 1 $UpstreamUrl $upstream 2>&1 | Out-Host
    if ($LASTEXITCODE) { throw "git clone failed ($LASTEXITCODE)" }
}
Push-Location $upstream
git fetch --depth 1 origin $UpstreamCommit 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git fetch failed ($LASTEXITCODE)" }
git checkout $UpstreamCommit 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git checkout failed ($LASTEXITCODE)" }
git submodule update --init --recursive 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git submodule update failed ($LASTEXITCODE)" }
Pop-Location
$ErrorActionPreference = "Stop"

# 4. Pure-python dependencies -------------------------------------------------
Write-Host "==> Installing dependencies"
& $py -m pip install --no-cache-dir -r (Join-Path $repo "requirements.txt")

# 5. Weights ------------------------------------------------------------------
# Upstream resolves weights as HY3DGEN_MODELS/<model id>/<subfolder>, so the
# files must land under weights\tencent\Hunyuan3D-2.1. Only the shape stage is
# downloaded (dit + vae, about 7.5 GB); the texture stage is unused here.
Write-Host "==> Downloading weights (shape stage only, about 7.5 GB)"
$weightsDir = Join-Path $weights $WeightsRepo.Replace("/", "\")
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('$WeightsRepo', local_dir=r'$weightsDir', allow_patterns=['hunyuan3d-dit-v2-1/*', 'hunyuan3d-vae-v2-1/*', 'LICENSE', 'Notice.txt'])"

# 6. .env ---------------------------------------------------------------------
$envPath = Join-Path $repo ".env"
if (-not (Test-Path $envPath)) {
    Write-Host "==> Writing .env"
    (Get-Content (Join-Path $repo ".env.example") -Raw).
        Replace("__REPO__", (Join-Path $upstream "hy3dshape")).
        Replace("__WEIGHTS__", $weights) | Set-Content -Path $envPath -Encoding utf8
}

# 7. Smoke test: the runner starts and answers without loading weights --------
Write-Host "==> Checking that the runner starts (capabilities round-trip)"
Push-Location $repo
$reply = '{"id":1,"method":"capabilities"}', '{"id":2,"method":"shutdown"}' | & $py -m runners.hunyuan3d
Pop-Location
if (-not ($reply -match '"image_to_mesh": true')) {
    throw "the runner did not answer capabilities: $reply"
}
Write-Host "    capabilities OK"

Write-Host ""
Write-Host "Done. Generate a first mesh with:"
Write-Host "  $py $repo\tools\run_single.py --image $repo\assets\sample.png --out $Root\out"
Write-Host ""
Write-Host "Or point hearth at this checkout:"
Write-Host "  HEARTH_RUNNER_HUNYUAN3D_PYTHON=$py"
Write-Host "  HEARTH_RUNNER_HUNYUAN3D_MODULE=runners.hunyuan3d"
Write-Host "  HEARTH_RUNNER_HUNYUAN3D_CWD=$repo"

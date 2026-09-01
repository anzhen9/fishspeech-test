<#
.SYNOPSIS
    统一语音处理服务（USS）—— 跨机器环境初始化脚本

.DESCRIPTION
    在一台「干净的新机器」上从零搭建运行环境。适用于：
      - 从 GitHub 克隆本仓库后，目标机没有任何 Python / 依赖 / 模型
      - 仓库的 .tools/ 目录被 .gitignore 排除，不会随 clone 下发，本脚本负责重建它

    脚本会依次完成：
      1. 下载并搭建 Windows 嵌入版 Python 3.11.9（落于 .tools/python311）
      2. 固化国内 pip 源（清华 TUNA）并安装 pip
      3. 安装 PyTorch 2.6.0+cu124（RTX 3060 用，走清华 PyTorch wheel 镜像）
      4. 安装 requirements 依赖（含 faster-whisper / speechbrain / fastapi 等）
      5. 校验 torch CUDA 与关键依赖可导入
      6. （可选 -DownloadModels）预拉 FishSpeech 1.5 权重（其余模型运行时自动下载）

.PARAMETER PythonVersion
    嵌入版 Python 版本，默认 3.11.9（与 README 要求一致）

.PARAMETER PipMirror
    pip 主索引，默认清华 TUNA 镜像（国内加速）

.PARAMETER PytorchIndex
    PyTorch 专用 wheel 索引（cu124 构建），默认清华 PyTorch 镜像

.PARAMETER HFEndpoint
    HuggingFace 镜像端点，默认 hf-mirror.com（国内直连不通）

.PARAMETER DownloadModels
    开关。带此参数会额外预下载 FishSpeech 1.5 权重到 models/fish-speech-1.5。
    注意：faster-whisper 与 speechbrain 模型由各自库在首次调用时自动下载，
    无需在此手动拉取。

.PARAMETER Force
    开关。带此参数会强制重建 .tools/python311（先删除再下载），用于环境损坏时救急。

.EXAMPLE
    # 最简：用默认国内源一键初始化（不下载模型）
    powershell -ExecutionPolicy Bypass -File setup_env.ps1

.EXAMPLE
    # 初始化并预拉 FishSpeech 权重
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 -DownloadModels

.EXAMPLE
    # 环境损坏，强制重建 Python + 重装依赖 + 拉模型
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Force -DownloadModels
#>

[CmdletBinding()]
param(
    [string] $PythonVersion = "3.11.9",
    [string] $PipMirror     = "https://pypi.tuna.tsinghua.edu.cn/simple/",
    [string] $PytorchIndex  = "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124/",
    [string] $HFEndpoint    = "https://hf-mirror.com",
    [switch] $DownloadModels,
    [switch] $Force
)

# ----------------------------- 基础设置 -----------------------------
$ErrorActionPreference = "Stop"

# 仓库根目录 = 脚本所在目录（setup_env.ps1 应放在仓库根）
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$ToolsDir = Join-Path $Root ".tools"
$PyDir    = Join-Path $ToolsDir "python311"
$PyExe    = Join-Path $PyDir "python.exe"
$PipIni   = Join-Path $PyDir "pip.ini"

function Write-Step($msg) {
    Write-Host ""
    Write-Host ("=" * 64) -ForegroundColor Cyan
    Write-Host ("  $msg") -ForegroundColor Cyan
    Write-Host ("=" * 64) -ForegroundColor Cyan
}

function Invoke-Native($exe, $args) {
    Write-Host ">> $exe $args" -ForegroundColor DarkGray
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "命令失败 (exit=$LASTEXITCODE): $exe $args"
    }
}

# Windows 专用：嵌入版 Python 仅 Windows 可用
$isWin = ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT)
if (-not $isWin) {
    Write-Error "本脚本使用 Windows 嵌入版 Python，仅支持 Windows。请用 Windows + RTX 3060 目标机运行。"
    exit 1
}

Write-Step "环境初始化 · 统一语音处理服务 (USS)"
Write-Host "仓库根目录 : $Root"
Write-Host "Python 目标: $PythonVersion (嵌入版, 落于 $PyDir)"
Write-Host "pip 镜像   : $PipMirror"
Write-Host "torch 索引 : $PytorchIndex"
Write-Host "HF 镜像    : $HFEndpoint"
if ($DownloadModels) { Write-Host "模型预拉   : 启用 (FishSpeech 1.5)" }
if ($Force)          { Write-Host "强制重建   : 启用" -ForegroundColor Yellow }

# ----------------------------- 1. 搭建本地 Python -----------------------------
Write-Step "1/6  搭建本地嵌入版 Python $PythonVersion"

$majorMinor = ($PythonVersion -replace '\.', '').Substring(0, 3)   # 3.11.9 -> 311
$pyZipName  = "python-$PythonVersion-embed-amd64.zip"
$pyUrl      = "https://www.python.org/ftp/python/$PythonVersion/$pyZipName"
$getPipUrl  = "https://bootstrap.pypa.io/get-pip.py"

$pyReady = (Test-Path $PyExe) -and (-not $Force)
if ($pyReady) {
    $ver = & $PyExe --version 2>&1
    Write-Host "[跳过] 已存在 $PyExe -> $ver"
} else {
    if ($Force -and (Test-Path $PyDir)) {
        Write-Host "[强制] 删除旧 .tools/python311 ..."
        Remove-Item $PyDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

    $pyZip = Join-Path $ToolsDir $pyZipName
    if (-not (Test-Path $pyZip)) {
        Write-Host "[下载] $pyUrl"
        Invoke-WebRequest -Uri $pyUrl -OutFile $pyZip -UseBasicParsing
    }
    Write-Host "[解压] -> $PyDir"
    Expand-Archive -Path $pyZip -DestinationPath $PyDir -Force

    # 修正 _pth：开启 import site 并加入 Lib\site-packages，否则 pip 装了也 import 不到
    $pthFile = Get-ChildItem $PyDir -Filter "python3*._pth" | Select-Object -First 1
    if (-not $pthFile) { $pthFile = Get-ChildItem $PyDir -Filter "*.pth" | Select-Object -First 1 }
    if ($pthFile) {
        $fixed = @(
            "python$majorMinor.zip",
            ".",
            "Lib\site-packages",
            "import site"
        ) -join "`n"
        Set-Content -Path $pthFile.FullName -Value $fixed -Encoding ASCII
        Write-Host "[修正] $($pthFile.Name): 已启用 import site"
    } else {
        Write-Warning "未找到 _pth 文件，pip 可能不可用，请检查 Python 嵌入包。"
    }

    # 安装 pip（下载 get-pip.py 并运行）
    $getPip = Join-Path $ToolsDir "get-pip.py"
    if (-not (Test-Path $getPip)) {
        Write-Host "[下载] $getPipUrl"
        Invoke-WebRequest -Uri $getPipUrl -OutFile $getPip -UseBasicParsing
    }
    Write-Host "[安装] pip ..."
    Invoke-Native $PyExe @($getPip)
}

# ----------------------------- 2. 固化 pip 国内源 -----------------------------
Write-Step "2/6  固化 pip 国内源 (TUNA)"
$ini = @"
[global]
index-url = $PipMirror
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
retries = 5
disable-pip-version-check = true
no-cache-dir = true

[install]
no-warn-script-location = true
"@
Set-Content -Path $PipIni -Value $ini -Encoding UTF8
Write-Host "[写入] $PipIni"
# 校验 pip 可用
Invoke-Native $PyExe @("-m", "pip", "--version")

# ----------------------------- 3. 安装 PyTorch (cu124) -----------------------------
Write-Step "3/6  安装 PyTorch 2.6.0+cu124 (RTX 3060)"
# 走清华 PyTorch wheel 镜像；依赖（sympy/networkx/filelock/jinja2 等）由 pip.ini 的 TUNA 索引解析
Invoke-Native $PyExe @("-m", "pip", "install",
    "torch==2.6.0+cu124", "torchaudio==2.6.0+cu124",
    "-f", $PytorchIndex, "--no-cache-dir")

# ----------------------------- 4. 安装其余依赖 -----------------------------
Write-Step "4/6  安装项目依赖 (requirements)"
$req = if (Test-Path (Join-Path $Root "requirements.lock.txt")) {
    Write-Host "[使用] requirements.lock.txt (锁定版本, 推荐)"
    "requirements.lock.txt"
} else {
    Write-Host "[使用] requirements.txt"
    "requirements.txt"
}
# -f 同样传入，避免 lock 中的 torch==...+cu124 回退到 PyPI 时报错
Invoke-Native $PyExe @("-m", "pip", "install", "-r", $req,
    "-f", $PytorchIndex, "--no-cache-dir")

# ----------------------------- 5. 校验 -----------------------------
Write-Step "5/6  校验环境"
# 把校验脚本写成 .py 文件再执行，避免 python -c 传递含引号的多行代码时引号被吞
$checkPy = Join-Path $ToolsDir "check_env.py"
@'
import importlib, sys, os
# fish_speech 是项目本地包，需把源码根目录加入 sys.path 才能导入
# (服务运行时由 service/engines/clone.py 注入，这里校验时手动注入)
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fish-speech-1.5.0")
if _src not in sys.path:
    sys.path.insert(0, _src)
mods = ["torch", "torchaudio", "fastapi", "uvicorn", "faster_whisper",
        "speechbrain", "fish_speech", "transformers", "hydra", "pydub",
        "imageio", "pypinyin", "loguru", "soundfile"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, repr(e)))
print("Python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
if bad:
    print("!! 以下模块导入失败:")
    for m, e in bad:
        print("   -", m, e)
    sys.exit(2)
print("所有关键依赖导入 OK")
'@ | Set-Content -Path $checkPy -Encoding UTF8
Invoke-Native $PyExe @($checkPy)

# ----------------------------- 6. (可选) 预拉模型 -----------------------------
if ($DownloadModels) {
    Write-Step "6/6  预拉 FishSpeech 1.5 权重"
    $env:HF_ENDPOINT = $HFEndpoint
    $env:USS_ROOT = $Root
    Write-Host "HF_ENDPOINT=$HFEndpoint"
    $dlPy = Join-Path $ToolsDir "download_models.py"
    @'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["USS_ROOT"])
local = root / "models" / "fish-speech-1.5"
local.mkdir(parents=True, exist_ok=True)
print("下载 FishSpeech 1.5 ->", local)
snap = snapshot_download(
    "fishaudio/fish-speech-1.5",
    local_dir=str(local),
    allow_patterns=[
        "**/config.json",
        "**/special_tokens.json",
        "**/tokenizer.tiktoken",
        "**/model.pth",
        "**/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    ],
)
print("完成:", snap)
'@ | Set-Content -Path $dlPy -Encoding UTF8
    Invoke-Native $PyExe @($dlPy)
    Write-Host "提示: faster-whisper 与 speechbrain 模型将在首次调用时自动下载，无需手动预拉。" -ForegroundColor DarkGray
} else {
    Write-Step "6/6  跳过模型预拉 (未指定 -DownloadModels)"
    Write-Host "提示: 首次启动服务时，faster-whisper / speechbrain 会自动下载模型；" -ForegroundColor DarkGray
    Write-Host "      FishSpeech 1.5 权重需手动放置到 models/fish-speech-1.5/，" -ForegroundColor DarkGray
    Write-Host "      或重新运行本脚本并加 -DownloadModels。" -ForegroundColor DarkGray
}

# ----------------------------- 完成 -----------------------------
Write-Step "初始化完成"
Write-Host "启动服务（任选其一）:"
Write-Host "  方式一: .\.tools\python311\python.exe -m service.server" -ForegroundColor Green
Write-Host "  方式二: .\.tools\python311\python.exe -m uvicorn service.server:app --host 127.0.0.1 --port 8080" -ForegroundColor Green
Write-Host ""
Write-Host "若需预拉模型，重新运行: powershell -ExecutionPolicy Bypass -File setup_env.ps1 -DownloadModels" -ForegroundColor DarkGray

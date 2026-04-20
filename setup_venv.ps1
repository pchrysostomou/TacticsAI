$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  TacticsAI -- Environment Setup" -ForegroundColor Cyan
Write-Host "  Python 3.11 + PyTorch CUDA 12.1 + YOLOv8" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Print Python version (3.10+ all work with YOLOv8)
$pyVersion = python --version 2>&1
Write-Host "  Python: $pyVersion" -ForegroundColor Green

# Step 1: Create venv
Write-Host ""
Write-Host "[1/5] Creating virtual environment (.venv)..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "      .venv already exists -- skipping." -ForegroundColor DarkYellow
} else {
    python -m venv .venv
    Write-Host "      OK" -ForegroundColor Green
}

# Step 2: Activate
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"
Write-Host "      OK" -ForegroundColor Green

# Step 3: Upgrade pip
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "      OK" -ForegroundColor Green

# Step 4: PyTorch with CUDA 12.1
Write-Host "[4/5] Installing PyTorch 2.x with CUDA 12.1 (2GB download)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
Write-Host "      OK" -ForegroundColor Green

# Step 5: Remaining requirements
Write-Host "[5/5] Installing project requirements..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
Write-Host "      OK" -ForegroundColor Green

# Verify
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
python -c "import torch, ultralytics, supervision, cv2, streamlit; print('  PyTorch     :', torch.__version__); print('  CUDA        :', torch.cuda.is_available(), '(' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A') + ')'); print('  Ultralytics :', ultralytics.__version__); print('  Supervision :', supervision.__version__); print('  OpenCV      :', cv2.__version__); print('  Streamlit   :', streamlit.__version__)"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate venv:      .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  Run dashboard:      streamlit run app.py" -ForegroundColor White
Write-Host "  Run CLI:            python main.py data\test_clip.mp4" -ForegroundColor White
Write-Host ""

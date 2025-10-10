# 🚀 J.A.R.V.I.S. Mega Training System

Write-Host "Starting JARVIS Mega Training System" -ForegroundColor Cyan
Write-Host "====================================`n"

# Setup Python environment
Write-Host "[INFO] Setting up Python environment..." -ForegroundColor Blue
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

# Activate virtual environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Blue
.\.venv\Scripts\Activate

# Install requirements
Write-Host "[INFO] Installing required packages..." -ForegroundColor Blue
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements-train.txt

# Create optimized training config for RTX 5050
$config = @"
{
    "model_path": "jarvis-model",
    "train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "save_steps": 5,
    "max_grad_norm": 0.5,
    "fp16": true,
    "gpu_settings": {
        "device": "cuda",
        "precision": "float16",
        "memory_efficient": true
    }
}
"@

$config | Out-File "train_config.json"
Write-Host "[INFO] Created optimized training configuration for RTX 5050" -ForegroundColor Blue

# Start enhanced training
Write-Host "`n[INFO] Starting enhanced training process..." -ForegroundColor Blue
python train_jarvis.py --config train_config.json

# Convert to optimized format
Write-Host "`n[INFO] Converting to optimized format..." -ForegroundColor Blue
python convert_to_gguf.py

Write-Host "`n[SUCCESS] JARVIS is now ready! 🎉" -ForegroundColor Green
# 🚀 J.A.R.V.I.S. Mega Training System

Write-Host "Starting JARVIS Mega Training System" -ForegroundColor Cyan
Write-Host "====================================`n"

Write-Host "Just A Rather Very Intelligent System" -ForegroundColor Cyan
Write-Host "Production-Grade AI Training & Deployment System`n" -ForegroundColor Magenta
Write-Host "============================================================"
}
}

# Check Python and create virtual environment
Write-Host "`n[INFO] Setting up Python environment..." -ForegroundColor Blue
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "[✓] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[✓] Virtual environment exists" -ForegroundColor Green
}
}

# Activate virtual environment
Write-Host "`n[INFO] Activating virtual environment..." -ForegroundColor Blue
.\.venv\Scripts\Activate
Write-Host "[✓] Virtual environment activated" -ForegroundColor Green

# Install requirements
Write-Host "`n[INFO] Installing required packages..." -ForegroundColor Blue
pip install -r requirements.txt
pip install -r requirements-train.txt

# Create optimized training config for RTX 5050
$config = @{
    model_path = "jarvis-model"
    train_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-5
    num_train_epochs = 10
    save_steps = 5
    max_grad_norm = 0.5
    fp16 = $true
} | ConvertTo-Json

$config | Out-File "train_config.json"
Write-Host "[✓] Training configuration saved" -ForegroundColor Green

# Start training
Write-Host "`n[INFO] Starting enhanced training process..." -ForegroundColor Blue
python train_jarvis.py --config train_config.json

Write-Host "`n[INFO] Converting model to optimized format..." -ForegroundColor Blue
python convert_to_gguf.py

Write-Host "`n[✓] JARVIS is now ready! 🎉" -ForegroundColor Green

# Activate virtual environment
Print-Status "Activating virtual environment..."
.\.venv\Scripts\Activate
Print-Success "Virtual environment activated"

# Install requirements
Print-Status "Installing required packages..."
pip install -r requirements.txt
pip install -r requirements-train.txt
Print-Success "Packages installed"

# Configure training settings for RTX 5050
$config = @{
    model_path = "jarvis-model"
    train_batch_size = 4  # Adjust based on GPU memory
    gradient_accumulation_steps = 4
    learning_rate = 2e-5
    num_train_epochs = 10  # Increased epochs for better learning
    save_steps = 5
    max_grad_norm = 0.5
    fp16 = $true  # Enable mixed precision training
}

# Save configuration
$config | ConvertTo-Json | Out-File "train_config.json"
Print-Success "Training configuration saved"

# Start training
Print-Status "Starting enhanced training process..."
python train_jarvis.py --config train_config.json

Print-Status "Converting model to optimized format..."
python convert_to_gguf.py

Print-Success "JARVIS is now ready! 🎉"
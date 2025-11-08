# Setup script for backend virtual environment
Write-Host "Setting up Python virtual environment..." -ForegroundColor Green

# Navigate to backend directory
Set-Location $PSScriptRoot

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Initialize database
Write-Host "`nInitializing database..." -ForegroundColor Yellow
python init_db.py

Write-Host "`n====================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "====================" -ForegroundColor Green
Write-Host "`nTo start the Flask server:" -ForegroundColor Cyan
Write-Host "1. Make sure you are in the backend directory" -ForegroundColor Yellow
Write-Host "   cd C:\Users\Sudhanshu\Desktop\Projects\Eudia\backend" -ForegroundColor White
Write-Host "`n2. Run the server:" -ForegroundColor Yellow
Write-Host "   .\.venv\Scripts\python.exe -m flask run --port=8000" -ForegroundColor White
Write-Host "`nNote: Always run Flask from the backend directory!" -ForegroundColor Red

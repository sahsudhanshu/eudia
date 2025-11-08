# Start Flask backend server
Write-Host "Starting Flask backend server..." -ForegroundColor Green

# Navigate to backend directory
$BackendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $BackendDir

# Check if database exists
if (!(Test-Path ".\instance\eudia.db")) {
    Write-Host "Database not found. Initializing..." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe init_db.py
}

# Set Flask environment variables
$env:FLASK_APP = "app:create_app()"
$env:FLASK_ENV = "development"

# Start Flask server
Write-Host "`nStarting server on http://127.0.0.1:8000..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m flask run --port=8000

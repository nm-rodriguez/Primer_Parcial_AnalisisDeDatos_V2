# Script para iniciar MLflow UI en PowerShell
$venvMlflow = "..\Parcial 1 - Analisis Moscow Real Estate\.venv\Scripts\mlflow.exe"

if (Test-Path $venvMlflow) {
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host "     Iniciando MLflow UI (Parcial 1 V2)" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host "[OK] Entorno virtual detectado." -ForegroundColor Green
    Write-Host "[INFO] Tracking Store: sqlite:///mlflow.db" -ForegroundColor Yellow
    Write-Host "[INFO] Abriendo navegador en http://127.0.0.1:5000" -ForegroundColor Green
    Write-Host "Presiona Ctrl + C para detener el servidor." -ForegroundColor Gray
    Write-Host ""
    
    Start-Process "http://127.0.0.1:5000"
    & $venvMlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
} else {
    Write-Host "[ERROR] No se encontro el entorno virtual en: $venvMlflow" -ForegroundColor Red
    Write-Host "Asegurate de activar tu entorno virtual y ejecutar:" -ForegroundColor Yellow
    Write-Host "mlflow ui --backend-store-uri sqlite:///mlflow.db" -ForegroundColor White
}

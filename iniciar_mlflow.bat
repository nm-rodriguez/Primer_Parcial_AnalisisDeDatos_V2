@echo off
setlocal
title Servidor MLflow UI - Parcial 1 V2
echo =======================================================
echo      Iniciando MLflow UI (Parcial 1 V2)
echo =======================================================
echo.

set "VENV_MLFLOW=..\Parcial 1 - Analisis Moscow Real Estate\.venv\Scripts\mlflow.exe"

if exist "%VENV_MLFLOW%" (
    echo [OK] Entorno virtual detectado con exito.
    echo [INFO] Conectando con la base de datos: sqlite:///mlflow.db
    echo [INFO] Abriendo MLflow en el navegador (http://127.0.0.1:5000)...
    echo.
    echo Para detener el servidor, presiona Ctrl + C en esta ventana.
    echo.
    start http://127.0.0.1:5000
    "%VENV_MLFLOW%" ui --backend-store-uri sqlite:///mlflow.db --port 5000
) else (
    echo [ERROR] No se encontro el entorno virtual en la ruta esperada:
    echo %VENV_MLFLOW%
    echo.
    echo Ejecuta manualmente en tu entorno:
    echo mlflow ui --backend-store-uri sqlite:///mlflow.db
    echo.
    pause
)

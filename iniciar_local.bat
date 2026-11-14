@echo off
chcp 65001 > nul
title Studio Video AI - Painel Profissional Local
color 0B

echo ================================================================
echo       STUDIO VIDEO AI - PAINEL PROFISSIONAL DE VIDEO IA
echo ================================================================
echo.
echo [1/2] Iniciando o servidor local na porta 5000...
echo [2/2] Abrindo http://localhost:5000 no seu navegador...
echo.

cd /d "%~dp0"

start "" http://localhost:5000

python app.py

pause

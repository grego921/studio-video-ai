@echo off
chcp 65001 > nul
title OpenHiggsfield AI - Geração de Vídeos Open-Source
color 0B

echo ================================================================
echo       OPEN HIGGSFIELD AI - GERAÇÃO DE VÍDEOS OPEN-SOURCE
echo ================================================================
echo.
echo [1/2] Iniciando o servidor Next.js na porta 3000...
echo [2/2] Abrindo http://localhost:3000 no navegador...
echo.

cd /d "C:\Open sorce geração de vídeos"

start "" http://localhost:3000

call pnpm dev

pause

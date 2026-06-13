@echo off
title Czech Liga Pro — Scheduler
cd /d C:\Users\drag_\Documents\prediccion_tenis_mesa
call venv\Scripts\activate
echo.
echo ============================================
echo   CZECH LIGA PRO — EJECUTANDO SCHEDULER
echo ============================================
echo.
python scheduler.py
echo.
deactivate
echo ============================================
echo   Listo. Revisa tu Telegram.
echo ============================================
pause

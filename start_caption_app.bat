@echo off
title Evoastra Image Captioning Server

REM 1️⃣ Navigate to project folder
cd /d "E:\Evoastra Internship\MAJOR PROJECT"

REM 2️⃣ Activate virtual environment
call venv\Scripts\activate

REM 3️⃣ Start FastAPI server in background
start cmd /k "uvicorn backend.caption_api:app --reload"

REM 4️⃣ Wait a few seconds to let API start
timeout /t 5 /nobreak >nul

REM 5️⃣ Open your web interface
start "" "E:\Evoastra Internship\MAJOR PROJECT\frontend\index.html"

exit

@echo off
echo === AI CharacherHub ===
echo.
echo [1/2] Installing dependencies...
pip install -r requirements.txt -q
echo [2/2] Starting server...
echo.
echo   Open: http://localhost:8000
echo   API:  http://localhost:8000/docs
echo.
cd backend && python main.py

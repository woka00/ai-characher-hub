@echo off
chcp 65001 > nul
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║       AI CharacherHub — v2.0             ║
echo  ║       Платформа оценки ИИ-моделей        ║
echo  ╚══════════════════════════════════════════╝
echo.

echo [1/2] Установка зависимостей...
python -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo.
    echo ОШИБКА: pip не сработал. Попробуй: py -3.11 -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [2/2] Запуск сервера...
echo.
echo  Браузер откроется автоматически через 1.5 сек.
echo  Для остановки нажми Ctrl+C
echo.
cd backend
python main.py

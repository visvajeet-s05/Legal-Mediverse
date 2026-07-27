@echo off
echo Starting Legal Mediverse Backend...
cd backend
set PYTHONPATH=..
python -m uvicorn app.main:app --reload --port 8000
pause

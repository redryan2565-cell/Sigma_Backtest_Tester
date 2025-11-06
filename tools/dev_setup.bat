@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
pip install pre-commit
pre-commit install
echo Setup complete. Virtual env and hooks ready!
pause

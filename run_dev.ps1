# Developer Mode 실행 스크립트
# PowerShell에서 실행: .\run_dev.ps1

# 환경 변수 설정 (문자열 "true"로 설정)
$env:DEVELOPER_MODE = "true"
$env:DEBUG_MODE = "true"
$env:DEBUG_SETTINGS = "true"

Write-Host "환경 변수 설정 완료:" -ForegroundColor Green
Write-Host "  DEVELOPER_MODE = $env:DEVELOPER_MODE"
Write-Host "  DEBUG_MODE = $env:DEBUG_MODE"
Write-Host "  DEBUG_SETTINGS = $env:DEBUG_SETTINGS"
Write-Host ""

# Python으로 환경 변수 확인
Write-Host "환경 변수 확인 중..." -ForegroundColor Yellow
python -c "import os; print('DEVELOPER_MODE:', os.getenv('DEVELOPER_MODE')); print('DEBUG_MODE:', os.getenv('DEBUG_MODE'))"
Write-Host ""

Write-Host "Streamlit 실행 중..." -ForegroundColor Cyan
streamlit run app/main.py


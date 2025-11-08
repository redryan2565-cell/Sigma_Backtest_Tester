# Developer Mode 실행 스크립트
# PowerShell에서 실행: .\run_dev.ps1

$env:DEVELOPER_MODE = "true"
$env:DEBUG_MODE = "true"
$env:DEBUG_SETTINGS = "true"

Write-Host "환경 변수 설정 완료:" -ForegroundColor Green
Write-Host "  DEVELOPER_MODE = $env:DEVELOPER_MODE"
Write-Host "  DEBUG_MODE = $env:DEBUG_MODE"
Write-Host ""

streamlit run app/main.py


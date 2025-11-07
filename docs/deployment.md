# 배포 가이드

Normal Dip Backtest를 Streamlit Cloud에 배포하는 방법입니다.

## Streamlit Cloud 배포

### 1. GitHub 저장소 준비

코드를 GitHub 저장소에 푸시합니다:

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### 2. Streamlit Cloud에서 앱 생성

1. [Streamlit Cloud](https://streamlit.io/cloud)에 로그인
2. "New app" 클릭
3. GitHub 저장소 선택
4. 앱 설정:
   - **Main file path**: `app/main.py` (권장) 또는 루트의 `streamlit_app.py`
   - **Python version**: 3.10 이상 선택 (권장: 3.11)
   - **Branch**: `main` (또는 원하는 브랜치)

**진입점 파일**:
- `app/main.py`: 메인 Streamlit 앱 파일 (권장)
- `streamlit_app.py`: 루트 진입점 파일 (선택사항, `app/main.py`를 호출)

**설정 파일 위치**:
- `app/.streamlit/config.toml`: Streamlit 설정 파일 (자동 인식)
- `app/.streamlit/secrets.toml.example`: Secrets 템플릿 파일

### 3. 환경 변수 설정

Streamlit Cloud 대시보드에서 "Advanced settings" → "Secrets"에 다음을 추가:

```toml
# Streamlit Cloud Secrets (paste directly into Secrets editor)
DEVELOPER_MODE = "false"
DEBUG_MODE = "false"
CACHE_ENABLED = "true"
CACHE_TTL_HOURS = "24"
```

**중요**: 프로덕션 배포에서는 반드시 `DEVELOPER_MODE=false`와 `DEBUG_MODE=false`로 설정하세요.

환경 변수 설명:

- `DEVELOPER_MODE`: 개발자 모드 활성화 (기본값: false)
  - `true`: Optimization 및 Leverage Mode 탭 표시
  - `false`: 기본 Backtest 기능만 표시 (프로덕션 권장)
- `DEBUG_MODE`: 디버그 모드 활성화 (기본값: false)
  - `true`: 상세한 에러 트레이스 표시 (보안 위험)
  - `false`: 일반적인 에러 메시지만 표시 (프로덕션 권장)
- `CACHE_ENABLED`: 캐시 활성화 (기본값: true)
- `CACHE_TTL_HOURS`: 캐시 TTL 시간 (기본값: 24)
- `ALPHA_VANTAGE_KEY`: Alpha Vantage API 키 (선택사항)

### 4. 배포 확인

배포가 완료되면 Streamlit Cloud에서 제공하는 URL로 앱에 접근할 수 있습니다.

## 로컬 배포

### 개발 모드 실행

로컬에서 개발 모드로 실행:

```bash
# Windows PowerShell
$env:DEVELOPER_MODE="true"
streamlit run app/main.py

# Linux/Mac
export DEVELOPER_MODE=true
streamlit run app/main.py
```

### 프로덕션 모드 실행

기본 설정으로 실행 (Optimization/Leverage Mode 숨김):

```bash
streamlit run app/main.py
```

## 설정 파일

### Streamlit 설정

`app/.streamlit/config.toml`에서 테마 및 서버 설정을 변경할 수 있습니다.
이 파일은 프로젝트에 포함되어 있으며, 보안 및 성능 최적화 설정이 포함되어 있습니다.

### Secrets 관리

민감한 정보는 Streamlit Cloud의 Secrets 기능을 사용하거나 환경 변수로 설정하세요.

**로컬 개발용**: `app/.streamlit/secrets.toml.example` 파일을 참고하여 `secrets.toml`을 생성할 수 있습니다.

**주의**: `secrets.toml` 파일은 절대 버전 관리에 포함하지 마세요! `.gitignore`에 이미 포함되어 있습니다.

## 문제 해결

### 배포 실패

- Python 버전 확인 (3.10 이상 필요)
- 의존성 설치 확인 (`requirements.txt` 또는 `pyproject.toml` 확인)
- 로그 확인 (Streamlit Cloud 대시보드)
- `app/.streamlit/config.toml` 파일이 올바르게 설정되었는지 확인

### 환경 변수 미적용

- Streamlit Cloud의 Secrets 설정 확인
- 환경 변수 이름 확인 (대소문자 구분)
- 앱 재배포
- `DEVELOPER_MODE`와 `DEBUG_MODE`가 `"false"`로 설정되었는지 확인 (문자열)

### 성능 문제

- 캐시가 활성화되어 있는지 확인 (`CACHE_ENABLED=true`)
- 대용량 데이터셋 사용 시 날짜 범위를 제한 (최대 10년)
- Streamlit Cloud의 리소스 제한 확인

### 보안 문제

- `DEBUG_MODE=false`로 설정되어 있는지 확인
- `DEVELOPER_MODE=false`로 설정되어 있는지 확인 (프로덕션)
- 에러 메시지에 민감한 정보가 노출되지 않는지 확인
- API 키가 코드에 하드코딩되지 않았는지 확인
- Git에 추적 중인 CSV 파일이나 로그 파일이 없는지 확인

### 빌드 크기 문제

- Git에 추적 중인 대용량 파일 확인:
  ```bash
  git ls-files | grep -E '\.(csv|log|db)$'
  ```
- `.gitignore`에 결과 파일이 제외되어 있는지 확인:
  - `results/*.csv`
  - `backtest_*.csv`
  - `*.log`, `*.db`
- 불필요한 파일 제거:
  ```bash
  git rm --cached <file>
  ```

## 추가 리소스

- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [프로젝트 README](../README.md)


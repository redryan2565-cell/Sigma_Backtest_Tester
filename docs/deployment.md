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
   - **Main file path**: `app/main.py`
   - **Python version**: 3.10 이상 선택
   - **Branch**: `main` (또는 원하는 브랜치)

### 3. 환경 변수 설정

Streamlit Cloud 대시보드에서 "Advanced settings" → "Secrets"에 다음을 추가:

```toml
# app/.streamlit/secrets.toml
DEVELOPER_MODE = "false"
DEBUG_MODE = "false"
CACHE_ENABLED = "true"
CACHE_TTL_HOURS = "24"
```

또는 환경 변수로 설정:

- `DEVELOPER_MODE`: 개발자 모드 활성화 (기본값: false)
- `DEBUG_MODE`: 디버그 모드 활성화 (기본값: false)
- `CACHE_ENABLED`: 캐시 활성화 (기본값: true)
- `CACHE_TTL_HOURS`: 캐시 TTL 시간 (기본값: 24)

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

### Secrets 관리

민감한 정보는 `app/.streamlit/secrets.toml`에 저장하거나 환경 변수로 설정하세요.

**주의**: `secrets.toml` 파일은 절대 버전 관리에 포함하지 마세요!

## 문제 해결

### 배포 실패

- Python 버전 확인 (3.10 이상 필요)
- 의존성 설치 확인 (`pyproject.toml` 확인)
- 로그 확인 (Streamlit Cloud 대시보드)

### 환경 변수 미적용

- Streamlit Cloud의 Secrets 설정 확인
- 환경 변수 이름 확인 (대소문자 구분)
- 앱 재배포

## 추가 리소스

- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [프로젝트 README](../README.md)


# Getting Started

Normal Dip Backtest (yAOIL)를 시작하는 빠른 가이드입니다.

## 설치

### 필수 요구사항

- Python 3.10 이상 (3.10, 3.11, 3.12 지원)
- pip (Python 패키지 관리자)

### 설치 방법

#### Windows

```bash
# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화
.venv\Scripts\activate

# 개발 모드로 설치 (테스트/린트 도구 포함)
pip install -e .[dev,gui]
```

#### Linux/Mac

```bash
# 가상 환경 생성
python3 -m venv .venv

# 가상 환경 활성화
source .venv/bin/activate

# 개발 모드로 설치
pip install -e .[dev,gui]
```

또는 `tools/setup.sh` 스크립트를 사용할 수 있습니다:

```bash
chmod +x tools/setup.sh
./tools/setup.sh
```

## 빠른 시작

### Streamlit GUI 사용

가장 쉬운 방법은 Streamlit 웹 인터페이스를 사용하는 것입니다:

```bash
streamlit run app/main.py
```

브라우저가 자동으로 열리며 웹 인터페이스가 표시됩니다.

### CLI 사용

명령줄에서 직접 백테스트를 실행할 수 있습니다:

```bash
ndbt run TQQQ --start 2023-01-01 --end 2023-12-31 --threshold -0.041 --shares-per-signal 10
```

## 환경 변수 설정 (선택)

`.env.example` 파일을 `.env`로 복사하고 필요한 설정을 추가하세요:

```bash
cp .env.example .env
```

주요 환경 변수:

- `DEVELOPER_MODE`: Optimization/Leverage Mode 활성화 (기본값: false)
- `DEBUG_MODE`: 상세한 에러 메시지 표시 (기본값: false)
- `CACHE_ENABLED`: 데이터 캐싱 활성화 (기본값: true)
- `CACHE_TTL_HOURS`: 캐시 TTL 시간 (기본값: 24)

## 다음 단계

- [배포 가이드](deployment.md) - Streamlit Cloud 배포 방법
- [README.md](../README.md) - 전체 기능 설명
- [SECURITY.md](../SECURITY.md) - 보안 가이드


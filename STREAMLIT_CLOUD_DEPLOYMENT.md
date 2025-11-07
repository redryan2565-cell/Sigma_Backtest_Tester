# Streamlit Cloud 배포 완전 가이드

## 1단계: GitHub 저장소 Public으로 변경

**주의**: 이 작업은 GitHub 웹 인터페이스에서만 가능합니다.

1. https://github.com/redryan2565-cell/pratice 접속
2. 저장소 페이지에서 **Settings** 탭 클릭
3. 스크롤 다운하여 **Danger Zone** 섹션 찾기
4. **Change visibility** 클릭
5. **Make public** 선택
6. 저장소 이름 입력하여 확인
7. **I understand, change repository visibility** 클릭

**완료되면**: 저장소가 Public으로 변경되어 Streamlit Cloud에서 무료 배포 가능합니다.

---

## 2단계: Streamlit Cloud에서 앱 생성

1. [Streamlit Cloud](https://share.streamlit.io/) 접속 및 로그인
2. **"New app"** 버튼 클릭
3. GitHub 저장소 선택: `redryan2565-cell/pratice`
4. 앱 설정:
   - **Main file path**: `app/main.py`
   - **Python version**: 3.11 (권장)
   - **Branch**: `feature/tp-sl-logic-improvement` 또는 `main`

---

## 3단계: Secrets 설정 (중요!)

앱 생성 후 **"Advanced settings"** → **"Secrets"** 클릭

### 필수 Secrets (프로덕션)

아래 내용을 그대로 복사하여 Secrets 편집기에 붙여넣으세요:

```toml
DEVELOPER_MODE = "false"
DEBUG_MODE = "false"
CACHE_ENABLED = "true"
CACHE_TTL_HOURS = "24"
```

### 선택적 Secrets (Alpha Vantage 사용 시)

Alpha Vantage API를 사용하는 경우에만 추가:

```toml
ALPHA_VANTAGE_KEY = "your_alpha_vantage_api_key_here"
```

**중요 사항**:
- 모든 값은 **문자열**로 입력 (`"false"` 형식)
- `DEVELOPER_MODE`와 `DEBUG_MODE`는 프로덕션에서 반드시 `"false"`로 설정
- API 키는 절대 코드에 하드코딩하지 않음

---

## 4단계: 배포 및 확인

1. **"Deploy!"** 버튼 클릭
2. 배포 완료 대기 (1-2분)
3. 제공된 URL로 앱 접근 테스트

### 배포 후 확인 사항

- [ ] 앱이 정상적으로 로드되는지 확인
- [ ] Backtest 탭에서 기본 기능 테스트
- [ ] Optimization/Leverage Mode 탭이 숨겨져 있는지 확인 (프로덕션 모드)
- [ ] 에러 발생 시 일반적인 메시지만 표시되는지 확인 (상세 트레이스 없음)
- [ ] CSV 다운로드 기능 테스트

---

## Secrets 관리 체크리스트

### ✅ 보안 확인 사항

- [ ] `DEVELOPER_MODE = "false"` 설정됨
- [ ] `DEBUG_MODE = "false"` 설정됨
- [ ] API 키가 코드에 하드코딩되지 않음
- [ ] API 키는 Streamlit Cloud Secrets에만 저장됨
- [ ] `.env` 파일이 `.gitignore`에 포함됨
- [ ] `secrets.toml` 파일이 Git에 추적되지 않음

### 🔒 Secrets 보안 모범 사례

1. **절대 코드에 하드코딩하지 않기**
   ```python
   # ❌ 나쁜 예
   API_KEY = "sk-1234567890"
   
   # ✅ 좋은 예
   API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
   ```

2. **환경 변수로만 관리**
   - 로컬 개발: `.env` 파일 사용
   - 프로덕션: Streamlit Cloud Secrets 사용

3. **민감 정보 노출 방지**
   - 에러 메시지에서 API 키 제거
   - 로그에 민감 정보 기록하지 않기

---

## 문제 해결

### 배포 실패 시
- Python 버전 확인 (3.10 이상)
- `app/main.py` 파일 경로 확인
- Streamlit Cloud 로그 확인
- `requirements.txt` 의존성 확인

### Secrets 미적용 시
- Secrets에 값이 문자열로 설정되었는지 확인 (`"false"` 형식)
- 환경 변수 이름 확인 (대소문자 구분)
- 앱 재배포

### 성능 문제
- 캐시 활성화 확인 (`CACHE_ENABLED=true`)
- 날짜 범위를 10년 이하로 제한

---

## 추가 리소스

- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [Secrets 관리 가이드](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [프로젝트 README](../README.md)


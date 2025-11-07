# 보안 정책 및 가이드라인

## 개요

이 문서는 프로덕션 배포 전 보안 검토 결과와 권장사항을 정리한 것입니다.

## 보안 검토 결과

### ✅ 잘 구현된 보안 기능

1. **환경변수 관리**
   - API 키가 `.env` 파일과 환경변수에서 안전하게 로드됨
   - `.gitignore`에 `.env`, `auth.json`, `.streamlit/secrets.toml` 제외됨

2. **입력 검증**
   - 티커 입력: 정규식 검증 (`^[A-Z0-9.\-]+$`), 길이 제한 (최대 15자)
   - 날짜 입력: 미래 날짜 제한 (`max_value=date.today()`)
   - 숫자 입력: Streamlit의 `st.number_input`으로 범위 검증 자동 적용

3. **의존성 관리**
   - `pyproject.toml`과 `requirements.txt`로 명확한 버전 관리
   - Python 버전 범위 지정 (`>=3.10,<3.13`)

### 🔒 구현된 보안 강화 사항

#### 1. 에러 메시지 정보 노출 방지 (HIGH PRIORITY)

**구현 내용**:
- 프로덕션 모드에서는 상세한 스택 트레이스를 숨김
- `DEBUG_MODE` 환경변수로 개발/프로덕션 모드 구분
- 에러 메시지에서 파일 경로 제거 (정규식으로 마스킹)

**사용 방법**:
```bash
# 프로덕션 모드 (기본값)
export DEBUG_MODE=false

# 개발 모드 (상세 에러 표시)
export DEBUG_MODE=true
```

**파일 위치**: `src/config.py`, `app/main.py`

#### 2. API 키 노출 방지 (MEDIUM PRIORITY)

**구현 내용**:
- API 키가 에러 메시지에 포함되지 않도록 수정
- 일반적인 에러 메시지만 표시

**파일 위치**: `src/data/alpha_vantage.py`

#### 3. 입력 검증 강화 (MEDIUM PRIORITY)

**현재 상태**:
- 티커 입력: ✅ 정규식 및 길이 제한 적용
- 날짜 입력: ✅ 미래 날짜 제한
- 숫자 입력: ✅ Streamlit 자동 범위 검증

**추가 권장사항**:
- 모든 사용자 입력에 대해 서버 측 검증 추가 고려
- 파일 업로드 기능이 있다면 파일 타입 및 크기 제한

## 보안 체크리스트

### 배포 전 필수 확인 사항

- [x] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [x] API 키가 코드에 하드코딩되지 않았는지 확인
- [x] `DEBUG_MODE=false`로 설정되어 있는지 확인
- [x] `DEVELOPER_MODE=false`로 설정되어 있는지 확인 (프로덕션)
- [x] 의존성 보안 취약점 검사 실행 (`pip-audit` 또는 `safety`)
- [x] 모든 사용자 입력에 검증이 적용되어 있는지 확인
- [x] CSV 파일 업로드 크기 제한 확인 (10MB)
- [x] 날짜 범위 제한 확인 (최대 10년)
- [x] 에러 메시지에서 파일 경로 제거 확인
- [x] Streamlit 설정 파일 (`app/.streamlit/config.toml`) 확인

### 정기적인 보안 점검

1. **의존성 취약점 검사** (월 1회 권장)
   ```bash
   pip-audit
   # 또는
   safety check
   ```

2. **의존성 업데이트** (분기 1회 권장)
   ```bash
   pip list --outdated
   pip install --upgrade <package>
   ```

3. **보안 로그 검토** (주 1회 권장)
   - 에러 로그에서 민감 정보 노출 여부 확인
   - 비정상적인 접근 패턴 확인

## 환경변수 설정

### 필수 환경변수

- `ALPHA_VANTAGE_KEY`: Alpha Vantage API 키 (선택사항)

### 선택적 환경변수

- `DEBUG_MODE`: 디버그 모드 활성화 (`true`/`false`, 기본값: `false`)
- `DEVELOPER_MODE`: 개발자 모드 활성화 (`true`/`false`, 기본값: `false`)
- `CACHE_ENABLED`: 캐시 활성화 (`true`/`false`, 기본값: `true`)
- `CACHE_TTL_HOURS`: 캐시 TTL 시간 (기본값: 24)
- `LOG_LEVEL`: 로깅 레벨 (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, 기본값: `INFO`)

### 설정 예시

`.env` 파일:
```env
ALPHA_VANTAGE_KEY=your_api_key_here
DEBUG_MODE=false
DEVELOPER_MODE=false
CACHE_ENABLED=true
CACHE_TTL_HOURS=24
LOG_LEVEL=INFO
```

## 알려진 제한사항

1. **캐시 파일 보안**
   - 캐시 파일이 pickle 형식으로 저장됨
   - 로컬 파일 시스템이므로 낮은 위험도
   - 향후 JSON 형식으로 변경 고려

2. **세션 상태**
   - Streamlit session_state에 사용자 입력 저장
   - 민감한 정보는 없지만 세션 타임아웃 고려 필요

## 보안 취약점 신고

보안 취약점을 발견한 경우, 다음 방법으로 신고해주세요:
1. GitHub Issues에 보안 취약점 라벨로 이슈 생성
2. 또는 프로젝트 관리자에게 직접 연락

**중요**: 공개 이슈에 민감한 정보를 포함하지 마세요.

## 참고 자료

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security.html)
- [Streamlit Security](https://docs.streamlit.io/knowledge-base/using-streamlit/security)

## 변경 이력

- 2025-01-XX: 초기 보안 검토 완료
- 2025-01-XX: 에러 메시지 정보 노출 방지 구현
- 2025-01-XX: API 키 노출 방지 강화
- 2025-11-07: 프로젝트 구조 재정리 후 파일 경로 업데이트
- 2025-11-07: Streamlit Cloud 배포 최적화 완료
  - 입력 검증 강화 (CSV 파일 크기 제한, 날짜 범위 제한)
  - 에러 메시지 보안 강화 (파일 경로 제거, 민감 정보 마스킹)
  - Streamlit 설정 파일 추가 (`app/.streamlit/config.toml`)
  - 로깅 시스템 추가
  - 성능 최적화 (Streamlit 캐싱)


# Streamlit Cloud 배포 가이드

## 1단계: GitHub 저장소 확인 ✅

코드가 성공적으로 푸시되었습니다:
- 브랜치: `feature/tp-sl-logic-improvement`
- 커밋: `9df5ea1` - Streamlit Cloud 배포 최적화 완료

## 2단계: Streamlit Cloud에서 앱 생성

1. [Streamlit Cloud](https://share.streamlit.io/)에 로그인
2. "New app" 버튼 클릭
3. GitHub 저장소 선택: `redryan2565-cell/pratice`
4. 앱 설정:
   - **Main file path**: `app/main.py`
   - **Python version**: 3.10 이상 선택 (권장: 3.11)
   - **Branch**: `feature/tp-sl-logic-improvement` 또는 `main` 선택

## 3단계: Secrets 설정 (필수!)

앱 생성 후 "Advanced settings" → "Secrets" 클릭하고 다음을 추가:

```toml
DEVELOPER_MODE = "false"
DEBUG_MODE = "false"
CACHE_ENABLED = "true"
CACHE_TTL_HOURS = "24"
```

**중요**: 
- `DEVELOPER_MODE`와 `DEBUG_MODE`는 반드시 `"false"`로 설정하세요 (문자열)
- 프로덕션 모드에서 Optimization/Leverage Mode는 자동으로 숨겨집니다

## 4단계: 배포 확인

1. "Deploy!" 버튼 클릭
2. 배포가 완료될 때까지 대기 (보통 1-2분)
3. 제공된 URL로 앱 접근 테스트

## 5단계: 배포 후 확인 사항

- [ ] 앱이 정상적으로 로드되는지 확인
- [ ] Backtest 탭에서 기본 기능 테스트
- [ ] Optimization/Leverage Mode 탭이 숨겨져 있는지 확인 (프로덕션 모드)
- [ ] 에러 발생 시 일반적인 메시지만 표시되는지 확인 (상세 트레이스 없음)
- [ ] CSV 다운로드 기능 테스트

## 문제 해결

### 배포 실패 시
- Python 버전 확인 (3.10 이상)
- `app/main.py` 파일 경로 확인
- Streamlit Cloud 로그 확인

### 환경 변수 미적용 시
- Secrets에 값이 문자열로 설정되었는지 확인 (`"false"` 형식)
- 앱 재배포

### 성능 문제
- 캐시가 활성화되어 있는지 확인
- 날짜 범위를 10년 이하로 제한

## 다음 단계 (선택사항)

### main 브랜치에 머지 (권장)

배포가 성공적으로 완료되면 main 브랜치에 머지하는 것을 권장합니다:

```bash
git checkout main
git merge feature/tp-sl-logic-improvement
git push origin main
```

그 후 Streamlit Cloud에서 브랜치를 `main`으로 변경하세요.

## 배포 URL

배포가 완료되면 다음과 같은 URL이 제공됩니다:
`https://[your-app-name].streamlit.app`

이 URL을 README.md에 추가하거나 공유할 수 있습니다.


# 확장 기능 추천 (Extensions Recommendations)

## 필수 확장 (Essential Extensions)

### 1. **Plotly** - 인터랙티브 차트
```toml
plotly>=5.18.0
```
- **장점**: 줌, 팬, 데이터 포인트 호버, 범례 클릭 등 인터랙티브 기능
- **사용**: NAV 차트를 클릭 가능하게 만들어 더 상세한 분석 가능
- **적용 예**: `st.plotly_chart()` 사용

### 2. **Streamlit-aggrid** - 고급 데이터 테이블
```toml
streamlit-aggrid>=0.3.4
```
- **장점**: 정렬, 필터링, 그룹핑, 페이징, CSV 내보내기
- **사용**: 일별 데이터 테이블에 필터링 및 정렬 기능 추가
- **적용 예**: `AgGrid()` 사용

### 3. **Pandas-profiling** - 자동 데이터 분석
```toml
ydata-profiling>=4.5.0  # pandas-profiling의 후속작
```
- **장점**: 데이터 품질 보고서 자동 생성, 통계 분석
- **사용**: 백테스트 결과에 대한 자동 리포트 생성

## 권장 확장 (Recommended Extensions)

### 4. **Streamlit-option-menu** - 메뉴 네비게이션
```toml
streamlit-option-menu>=0.3.12
```
- **장점**: 더 나은 UI/UX, 탭 형태의 메뉴
- **사용**: "Backtest", "Compare Strategies", "Settings" 등 탭 구조

### 5. **Streamlit-authenticator** - 사용자 인증 (선택)
```toml
streamlit-authenticator>=0.2.3
```
- **장점**: 멀티 유저 지원, 세션 관리
- **사용**: 여러 사용자가 각자의 백테스트 결과 저장/관리

### 6. **Streamlit-cache** - 성능 최적화
- **장점**: 데이터 페치 결과 캐싱
- **사용**: 같은 티커/날짜 범위 재요청 시 빠른 응답

### 7. **Altair** - 고급 시각화
```toml
altair>=5.2.0  # 이미 streamlit의 의존성에 포함될 수 있음
```
- **장점**: 선언적 시각화, 복잡한 차트 조합
- **사용**: 여러 지표를 한 차트에 표시, 브러싱 등

## 고급 기능 확장 (Advanced Features)

### 8. **Ta-lib** 또는 **pandas-ta** - 기술적 지표
```toml
pandas-ta>=0.3.14b  # Ta-lib보다 설치가 쉬움
```
- **장점**: RSI, MACD, 볼린저 밴드 등 기술적 지표 추가 가능
- **사용**: 신호 생성 조건 확장 (단순 등락률 외에 기술적 지표 결합)

### 9. **Riskfolio-lib** - 리스크 분석
```toml
Riskfolio-Lib>=5.0.0
```
- **장점**: 포트폴리오 리스크 메트릭, 최적화
- **사용**: Sharpe ratio, Sortino ratio, VaR 등 고급 리스크 지표

### 10. **QuantStats** - 퀀트 분석
```toml
quantstats>=0.0.62
```
- **장점**: 전문적인 성과 분석 리포트 (HTML 생성)
- **사용**: 벤치마크 대비 성과, 드로다운 분석 등

### 11. **Streamlit-elements** - 고급 UI 컴포넌트
```toml
streamlit-elements>=0.1.0
```
- **장점**: 드래그 앤 드롭, 모달, 알림 등
- **사용**: 대시보드 레이아웃 커스터마이징

## 데이터 소스 확장

### 12. **yfinance-cache** - 데이터 캐싱
```toml
yfinance-cache>=0.1.5
```
- **장점**: yfinance 결과를 로컬 캐시하여 속도 향상
- **사용**: 반복 실행 시 네트워크 요청 감소

### 13. **polygon-api-client** - 실시간 데이터 (유료 API)
```toml
polygon-api-client>=1.13.0
```
- **장점**: 더 상세한 주가 데이터, 분봉 데이터
- **사용**: 고급 백테스트 (시간대별 매수/매도)

## 배포 및 운영

### 14. **Streamlit-cloud** 또는 **Docker**
- **장점**: 클라우드 배포, 공유 가능한 웹앱
- **사용**: 팀원들과 결과 공유

### 15. **Loguru** - 로깅
```toml
loguru>=0.7.2
```
- **장점**: 구조화된 로깅, 파일/콘솔 출력
- **사용**: 디버깅 및 모니터링

## 우선순위별 구현 권장사항

### Phase 1 (즉시 적용 가능)
1. ✅ **Plotly** - 차트 개선 (가장 큰 사용자 경험 향상)
2. ✅ **Streamlit-aggrid** - 테이블 기능 강화
3. ✅ **yfinance-cache** - 성능 개선

### Phase 2 (중기 개선)
4. **pandas-ta** - 기술적 지표 추가
5. **Riskfolio-lib** - 리스크 분석 강화
6. **Streamlit-option-menu** - UI 구조 개선

### Phase 3 (장기 확장)
7. **QuantStats** - 전문 리포트 생성
8. **Streamlit-authenticator** - 멀티 유저 지원
9. **polygon-api-client** - 고급 데이터 소스


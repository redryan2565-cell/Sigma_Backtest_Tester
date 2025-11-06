# Normal Dip Backtest (yAOIL)

백테스트 엔진 + Streamlit GUI를 활용한 딥 구매 전략 백테스트 도구.

## 주요 기능

- **Budget-based 모드**: 주간 예산 기반 매수
- **Shares-based 모드**: 신호당 고정 주식 수 매수
- **Take-Profit / Stop-Loss (TP/SL)**: 포트폴리오 수익률 기반 자동 매도
  - Baseline Reset: TP/SL 트리거 후 기준선 리셋으로 연속 트리거 방지
  - Hysteresis: 작은 변동으로 인한 재트리거 방지
  - Cooldown: 트리거 후 일정 기간 재발동 금지
- **파라미터 최적화**: Grid Search / Random Search로 최적 파라미터 탐색
  - IS/OS 분리: 과적합 방지를 위한 In-Sample / Out-of-Sample 평가
  - 자동 랭킹: CAGR → Sortino → Sharpe → Cumulative Return 기준
- **프리셋 저장/불러오기**: 백테스트 설정 저장 및 재사용
- **Hysteresis/Cooldown 프리셋**: Conservative, Moderate, Aggressive 프리셋 제공
- **데이터 캐싱**: Yahoo Finance 데이터 자동 캐싱 (TTL 24시간)
- **고급 시각화**: 
  - NAV 차트 (TP/SL 마커 포함)
  - Drawdown 차트
  - 월별 수익률 히트맵
- **Streamlit GUI**: 인터랙티브 웹 인터페이스
- **CLI**: 명령줄 인터페이스
- **CSV 내보내기**: 백테스트 결과 저장

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -e .[dev]  # 개발 모드로 설치 (테스트/린트 도구 포함)
# 또는 GUI 사용 시: pip install -e .[gui,dev]
```

## 사용법

### CLI 사용

```bash
ndbt run TQQQ --start 2023-01-01 --end 2023-12-31 --threshold -0.041 --weekly-budget 500 --mode split
```

### Streamlit GUI 사용

```bash
streamlit run src/gui_streamlit.py
```

## 프로젝트 구조

```
src/
├── backtest/         # 백테스트 엔진
│   ├── engine.py     # 백테스트 실행 및 레저 계산
│   └── metrics.py    # 성능 지표 (CAGR, MDD, Sharpe, Sortino, XIRR)
├── data/             # 데이터 소스
│   ├── yfin.py       # Yahoo Finance 데이터 피드
│   ├── cache.py      # 데이터 캐싱 시스템
│   └── base.py       # 데이터 피드 인터페이스
├── strategy/         # 전략 (딥 구매)
├── optimization/     # 파라미터 최적화
│   └── grid_search.py  # Grid Search / Random Search 엔진
├── storage/          # 설정 저장
│   └── presets.py    # 프리셋 관리
├── visualization/    # 시각화 유틸리티
│   └── heatmap.py    # 월별 수익률 히트맵
├── cli.py            # CLI 인터페이스
├── gui_streamlit.py  # Streamlit GUI
└── config.py         # 설정 관리
```

## 주요 기능 상세 설명

### Take-Profit / Stop-Loss (TP/SL)

포트폴리오 수익률 기반 자동 매도 시스템:

- **기준**: 포트폴리오 평균 매수가 대비 현재 가치 수익률
- **Baseline Reset**: TP/SL 트리거 후 기준선을 현재 NAV로 리셋하여 연속 트리거 방지
- **Hysteresis**: 트리거 후 작은 변동으로 인한 재트리거 방지
  - TP: 트리거 후 수익률이 (threshold - hysteresis) 아래로 떨어져야 재무장
  - SL: 트리거 후 수익률이 (threshold + hysteresis) 위로 올라가야 재무장
- **Cooldown**: 트리거 후 일정 기간 동안 재발동 금지
- **매도 비율**: TP/SL 각각 별도 설정 가능 (25%, 50%, 75%, 100%)
- **반올림 규칙**: 0.5 이상 올림, 최소 1주 보장

#### Baseline Reset 동작 원리

TP/SL 트리거 시 기준선 리셋 메커니즘:

1. **초기 상태**: 첫 매수 시 `ROI_base = NAV`
2. **트리거 판정**: `NAVReturn_baselined = (NAV - ROI_base) / ROI_base` 계산
3. **TP/SL 체결**: 조건 충족 시 지정된 비율로 매도 실행
4. **기준선 리셋**: 체결 직후 `ROI_base = NAV_post_trade`로 업데이트
5. **재평가**: 다음 트리거는 새로운 기준선 기준으로 평가

이를 통해 연속적인 트리거를 방지하고, 각 매도 후 새로운 수익 실행 기회를 제공합니다.

### 파라미터 최적화

IS/OS 분리 기반 파라미터 최적화:

- **IS/OS 분리**: 
  - IS (In-Sample): 최적 파라미터 선택용 학습 기간 (예: 2014-01-01 ~ 2022-12-31)
  - OS (Out-of-Sample): 선택된 파라미터 검증용 기간 (예: 2023-01-01 ~ 2025-11-06)
- **과적합 방지**: 
  - IS에서만 최적화 수행
  - OS 성과 확인으로 과튜닝 검증
  - 한 번만 최적화 (OS에서 재튜닝 금지)
  - 데이터 캐시 공유로 일관성 유지
- **제약 조건**: 
  - MDD ≥ -60%
  - Trades ≥ 15
  - HitDays ≥ 15
- **랭킹 순서**: CAGR → Sortino → Sharpe → Cumulative Return
- **탐색 방식**: 
  - Grid Search: 모든 조합 탐색
  - Random Search: 무작위 샘플링 (큰 범위용)
- **결과 저장**: IS/OS 각각 CSV로 저장

### 데이터 캐싱

- **자동 캐싱**: Yahoo Finance 데이터 자동 캐싱
- **TTL**: 24시간 (설정 가능)
- **캐시 키**: (ticker, start_date, end_date)
- **설정**: `.env` 파일 또는 환경 변수로 제어 가능
  - `CACHE_ENABLED`: 캐시 활성화 여부 (기본: True)
  - `CACHE_TTL_HOURS`: TTL 시간 (기본: 24)
  - `CACHE_DIR`: 캐시 디렉토리 경로

### 프리셋

- **Hysteresis/Cooldown 프리셋**: 
  - Conservative: TP/SL Hysteresis 5%, TP Cooldown 5일, SL Cooldown 3일
  - Moderate: TP/SL Hysteresis 3%, TP Cooldown 3일, SL Cooldown 2일
  - Aggressive: TP/SL Hysteresis 1%, Cooldown 1일
- **백테스트 설정 저장**: 현재 설정을 프리셋으로 저장
- **프리셋 불러오기**: 저장된 프리셋으로 설정 복원
- **프리셋 관리**: 저장/불러오기/삭제 기능

### 고급 시각화

- **NAV 차트**: 
  - TP/SL 트리거 마커 표시
  - Interactive zoom, pan, hover
- **Drawdown 차트**: 포트폴리오 하락폭 시각화
- **월별 수익률 히트맵**: 연도별/월별 수익률 패턴 분석

## 테스트

프로젝트에는 네트워크 의존성 없는 단위 테스트가 포함되어 있습니다:

```bash
# 모든 테스트 실행
pytest

# 간단한 출력으로 실행
pytest -q

# 특정 테스트 파일 실행
pytest src/tests/test_engine_minicase.py

# 커버리지 확인 (선택)
pytest --cov=src --cov-report=term-missing
```

### 테스트 파일

- `test_engine_minicase.py`: 기본 백테스트 로직 검증 (5일 토이 시계열)
- `test_tp_sl_baseline.py`: TP/SL 기준선 리셋 기능 검증
- `test_invariants.py`: 회계 항등식 및 모노토닉 검증 (NAV=Equity+CumCF 등)
- `test_metrics_mdd.py`: MDD 계산 검증 (범위, Edge case)
- `test_baseline_reset.py`: Baseline reset 종합 테스트

모든 테스트는 네트워크 없이 로컬 DataFrame을 사용하여 실행됩니다.

## 린트 및 타입 체크

```bash
# Ruff 린트
ruff check .

# 자동 수정
ruff check --fix .

# Mypy 타입 체크 (선택)
mypy src/
```

## 개발 환경

- **Python**: 3.10, 3.11, 3.12 지원
- **의존성**: `pyproject.toml`에서 관리
- **CI/CD**: GitHub Actions에서 Python 3.10/3.11/3.12 매트릭스 테스트 실행

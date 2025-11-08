# 앱 기능 및 계산 로직 상세 문서

## 목차
1. [앱 개요](#앱-개요)
2. [모드별 기능](#모드별-기능)
3. [백테스트 모드 상세](#백테스트-모드-상세)
4. [최적화 모드 상세](#최적화-모드-상세)
5. [레버리지 모드 상세](#레버리지-모드-상세)
6. [모든 파라미터 설명](#모든-파라미터-설명)
7. [TP/SL 기능 상세](#tpsl-기능-상세)
8. [Preset 시스템](#preset-시스템)
9. [차트 및 시각화](#차트-및-시각화)
10. [데이터 Export/Import](#데이터-exportimport)

---

## 앱 개요

**앱 이름**: Normal Dip Backtest (Sigma Backtest Tester)

**목적**: Dip-Buy 전략 백테스트 및 파라미터 최적화

**핵심 전략**: 
- 일일 수익률이 임계값(threshold) 이하로 하락하면 매수
- Shares-based 모드만 지원 (고정 주식 수 매수)
- TP/SL 기능으로 수익 실현 및 손실 제한

**배포 모드**:
- **Developer Mode**: 모든 기능 접근 가능 (Optimization, Leverage Mode)
- **Deployment Mode**: 백테스트 모드만 접근 가능 (일반 사용자용)

---

## 모드별 기능

### 1. Backtest 모드 (기본 모드)

**기능**: 단일 파라미터 세트로 백테스트 실행

**주요 기능**:
- Ticker 입력 및 검증
- 날짜 범위 설정 (최대 10년)
- 매수 파라미터 설정 (Threshold, Shares per Signal)
- TP/SL 독립 설정 (각각 활성화/비활성화 가능)
- Hysteresis/Cooldown 프리셋 (Conservative, Moderate, Aggressive)
- Preset 저장/불러오기 (세션 기반 + JSON Export/Import)
- Quick Presets (TQQQ, SOXL, QLD 전용)
- 차트 및 메트릭 표시
- 일일 데이터 CSV 다운로드

**데이터 제한**:
- 날짜 범위: 최대 10년 (3,650일)
- JSON Import: 최대 5개 preset, 파일 크기 1MB 이하

### 2. Optimization 모드 (Developer Only)

**기능**: Grid Search 또는 Random Search로 최적 파라미터 탐색

**탐색 방식**:
- **Grid Search**: 모든 파라미터 조합 탐색
- **Random Search**: 랜덤 샘플링 (10~1,000개)

**IS/OS Split 방법론**:
- **IS (In-Sample)**: 학습 기간 (파라미터 선택용)
- **OS (Out-of-Sample)**: 검증 기간 (과적합 방지)

**제약 조건**:
- MDD ≥ -60% (최대 손실 60% 이하)
- Trades ≥ 15 (최소 거래 15회 이상)
- HitDays ≥ 15 (최소 신호 발생일 15일 이상)

**랭킹 기준** (우선순위 순):
1. CAGR (연평균 성장률)
2. Sortino Ratio
3. Sharpe Ratio
4. Cumulative Return

**탐색 파라미터**:
- Threshold: -10% ~ -1%
- TP Threshold: 10% ~ 50%
- SL Threshold: -50% ~ -10%
- TP Sell %: 25%, 50%, 75%, 100%
- SL Sell %: 25%, 50%, 75%, 100%

### 3. Leverage Mode (Developer Only)

**기능**: 레버리지 ETF 전용 TP/SL 파라미터 최적화

**특징**:
- Threshold 고정 (사용자 입력)
- TP/SL 파라미터만 탐색 (fine-tuning)
- 레버리지 ETF (TQQQ, SOXL, QLD 등)에 최적화

**탐색 범위 설정**:
- TP Threshold: Min/Max/Step (예: 15%~50%, Step 5%)
- SL Threshold: Min/Max/Step (예: -50%~-10%, Step 5%)
- TP Sell Options: 25%, 50%, 75%, 100% (다중 선택)
- SL Sell Options: 25%, 50%, 75%, 100% (다중 선택)

**기타 설정**:
- Shares per Signal
- Fee Rate (운영보수)
- Slippage Rate
- Baseline Reset 사용 여부

---

## 백테스트 모드 상세

### UI 구성

#### 1. Ticker 입력
- **입력**: 텍스트 (예: TQQQ, AAPL, SPY)
- **검증**: Yahoo Finance API로 실시간 검증
- **캐싱**: 검증 결과 24시간 캐싱
- **보안**: 입력값 sanitization (XSS 방지)

#### 2. Quick Presets (범용 프리셋)
- **옵션**: None, TQQQ, SOXL, QLD
- **기능**: 
  - Ticker 자동 입력
  - 모든 파라미터 자동 설정
  - 날짜 범위 자동 설정
- **자동 선택**: 
  - Ticker 입력 시 TQQQ/SOXL/QLD 중 하나면 해당 프리셋 자동 선택
  - 다른 Ticker 입력 시 "None"으로 자동 변경
- **파라미터 유지**: Preset 해제 시에도 파라미터 값은 유지 (재설정 안 됨)

#### 3. Date Range
- **Start Date**: 백테스트 시작일 (최대 10년 전)
- **End Date**: 백테스트 종료일 (오늘까지)
- **검증**: 
  - Start ≤ End
  - (End - Start) ≤ 10년 (3,650일)

#### 4. Threshold (%)
- **의미**: 일일 수익률 임계값 (매수 신호 발생 조건)
- **범위**: -100% ~ 100%
- **기본값**: -4.1%
- **예시**: -4.1% 설정 시 → 일일 수익률 -4.1% 이하 하락 시 매수

#### 5. Shares per Signal
- **의미**: 신호 발생 시 매수할 주식 수
- **범위**: 0.01 ~ 1,000,000
- **기본값**: 10주
- **모드**: Shares-based 모드만 지원 (Budget-based 모드 제거됨)

#### 6. Fee Rate (%) (운영보수)
- **의미**: 거래 시 발생하는 수수료 비율
- **범위**: 0% ~ 10%
- **기본값**: 0.05%
- **적용**: 매수/매도 시 각각 적용

#### 7. Slippage Rate (%)
- **의미**: 주문 실행 시 예상 가격과 실제 체결 가격의 차이
- **범위**: -10% ~ 10%
- **기본값**: 0.25%
- **적용**: 
  - 매수 시: 실행가 = 종가 × (1 + Slippage Rate)
  - 매도 시: 실행가 = 종가 × (1 - Slippage Rate)

---

## TP/SL 기능 상세

### 1. Enable Take-Profit (TP)

**토글**: 체크박스 (독립 활성화)

**활성화 시 입력**:
- **TP Threshold (%)**: 수익 실현 임계값 (예: 30% 상승 시 매도)
  - 범위: 1% ~ 100%
  - 기본값: 30%
  - 계산 기준: 포지션 수익률 (진입가 대비)

- **TP Sell Percentage**: 매도 비율
  - 옵션: 25%, 50%, 75%, 100%
  - 기본값: 100% (전량 매도)
  - 예시: 50% 선택 시 → 보유 주식의 50%만 매도

### 2. Enable Stop-Loss (SL)

**토글**: 체크박스 (독립 활성화)

**활성화 시 입력**:
- **SL Threshold (%)**: 손실 제한 임계값 (예: -25% 하락 시 매도)
  - 범위: -100% ~ -1%
  - 기본값: -20%
  - 계산 기준: 포지션 수익률 (진입가 대비)

- **SL Sell Percentage**: 매도 비율
  - 옵션: 25%, 50%, 75%, 100%
  - 기본값: 100% (전량 매도)
  - 예시: 75% 선택 시 → 보유 주식의 75% 매도

### 3. Baseline Reset After TP/SL

**토글**: 체크박스

**의미**: TP/SL 트리거 후 남은 포지션의 진입가를 현재가로 리셋

**동작**:
- **활성화 (기본값)**: 
  - TP/SL 후 남은 포지션의 진입가 = 매도 실행가
  - 수익률 0%에서 다시 시작 (즉각 재트리거 방지)
- **비활성화**: 
  - 원래 진입가 유지
  - 즉각 재트리거 가능

### 4. Hysteresis (재트리거 방지)

**정의**: TP/SL 트리거 후 일정 수익률 이하/이상으로 이동해야 재활성화

**설정 방식**: Preset 선택 (Conservative, Moderate, Aggressive)

**TP Hysteresis**:
- **의미**: TP 트리거 후 수익률이 (TP Threshold - Hysteresis) 이하로 떨어져야 재활성화
- **예시**: TP 30%, Hysteresis 10% → 트리거 후 20% 이하로 떨어져야 다시 TP 가능
- **범위**: 0% ~ 50%

**SL Hysteresis**:
- **의미**: SL 트리거 후 수익률이 (SL Threshold + Hysteresis) 이상으로 올라가야 재활성화
- **예시**: SL -25%, Hysteresis 10% → 트리거 후 -15% 이상으로 올라가야 다시 SL 가능
- **범위**: 0% ~ 50%

### 5. Cooldown (대기 기간)

**정의**: TP/SL 트리거 후 일정 기간(일) 동안 재발동 금지

**설정 방식**: Preset 선택 (Conservative, Moderate, Aggressive)

**TP Cooldown Days**:
- **의미**: TP 트리거 후 N일 동안 TP 재발동 불가
- **범위**: 0 ~ 365일

**SL Cooldown Days**:
- **의미**: SL 트리거 후 N일 동안 SL 재발동 불가
- **범위**: 0 ~ 365일

### 6. Hysteresis/Cooldown Presets

**Conservative (보수적)**:
- TP Hysteresis: 15%
- SL Hysteresis: 15%
- TP Cooldown: 10일
- SL Cooldown: 10일
- **목적**: 빈번한 재트리거 방지, 안정적 운용

**Moderate (중도적)**:
- TP Hysteresis: 10%
- SL Hysteresis: 10%
- TP Cooldown: 5일
- SL Cooldown: 5일
- **목적**: 균형잡힌 재트리거 방지

**Aggressive (공격적)**:
- TP Hysteresis: 5%
- SL Hysteresis: 5%
- TP Cooldown: 2일
- SL Cooldown: 2일
- **목적**: 신속한 재진입 가능, 적극적 운용

**Custom (사용자 정의)**:
- 모든 값 수동 입력 가능

---

## TP/SL 계산 로직

### 1. 트리거 조건 확인 시점

**조건**:
- 오늘 매수하지 않은 경우만 (당일 매수한 주식은 제외)
- 최소 1일 이상 보유한 포지션만 대상
- `cum_shares > 0` (보유 주식 있음)

### 2. 포지션 수익률 계산

```
# 1일 이상 보유한 포지션만 선택
eligible_positions = [pos for pos in positions if (today - pos.entry_date) >= 1]

# 가중 평균 진입가 계산
weighted_avg_entry_price = sum(pos.shares * pos.entry_price) / sum(pos.shares)

# 포지션 수익률
position_return = (current_price / weighted_avg_entry_price) - 1.0
```

### 3. TP 트리거 확인

```
if position_return >= tp_threshold:
    if not tp_cooldown_active:  # Cooldown 확인
        if not tp_hysteresis_active:  # Hysteresis 확인
            tp_trigger = True
```

**Hysteresis 활성화**:
```
# TP 트리거 후
if tp_hysteresis > 0:
    tp_hysteresis_active = True

# 재활성화 조건
if position_return < (tp_threshold - tp_hysteresis):
    tp_hysteresis_active = False
```

**Cooldown 확인**:
```
# TP 트리거 후
last_tp_date = today

# 다음 날부터
days_since_tp = today - last_tp_date
if days_since_tp < tp_cooldown_days:
    tp_cooldown_active = True
```

### 4. SL 트리거 확인

**조건**: TP가 트리거되지 않은 경우만

```
if not tp_triggered and position_return <= sl_threshold:
    if not sl_cooldown_active:
        if not sl_hysteresis_active:
            sl_trigger = True
```

### 5. 매도 실행

```
# 매도 주식 수 계산
shares_to_sell = cum_shares * sell_percentage

# 실행가 (슬리피지 포함)
execution_sell_price = current_price * (1 - slippage_rate)

# 매도 대금
gross_proceeds = shares_to_sell * execution_sell_price
sell_fee = gross_proceeds * fee_rate
net_proceeds = gross_proceeds - sell_fee

# 상태 업데이트
cum_shares -= shares_to_sell
cum_cash_flow += net_proceeds

# Baseline Reset (활성화된 경우)
if reset_baseline_after_tp_sl:
    # 남은 포지션의 진입가를 현재 매도가로 리셋
    for pos in remaining_positions:
        pos.entry_price = execution_sell_price * (1 + fee_rate)
        pos.entry_date = today
```

---

## Preset 시스템

### 1. 세션 Presets (Session-based)

**저장 위치**: `st.session_state` (메모리)

**기능**:
- **Save Current Settings**: 현재 파라미터를 Preset으로 저장
  - Preset 이름 입력 필요
  - 세션 내에서만 유지 (브라우저 닫으면 삭제)
  - 최대 5개 제한 없음
- **Load Preset**: 저장된 Preset 불러오기
  - 드롭다운에서 선택
  - 모든 파라미터 자동 입력
- **Delete Preset**: 선택한 Preset 삭제

**저장 내용**:
- Ticker
- Date Range (Start, End)
- Threshold
- Shares per Signal
- Fee Rate, Slippage Rate
- TP/SL 파라미터 (활성화 여부, Threshold, Sell %)
- Baseline Reset 설정
- Hysteresis/Cooldown 값

### 2. Quick Presets (범용 프리셋)

**고정 Preset**: TQQQ, SOXL, QLD

**자동 설정 내용**:
- Ticker: 해당 심볼
- Date Range: 2016-01-01 ~ Today
- Threshold: -4.1%
- Shares per Signal: 10
- Fee Rate: 0.05%
- Slippage Rate: 0.25%
- TP: 30% (100% 매도)
- SL: -20% (100% 매도)
- Baseline Reset: On
- Hysteresis/Cooldown: Moderate

**자동 선택 로직**:
- Ticker 입력란에 "TQQQ" 입력 → TQQQ Preset 자동 선택
- Ticker를 "TMF"로 변경 → "None"으로 자동 변경
- 파라미터 값은 유지 (재설정 안 됨)

### 3. JSON Export/Import

**Export**:
- **기능**: 세션의 모든 Preset을 JSON 파일로 다운로드
- **파일명 형식**: `presets_YYYYMMDD_HHMMSS.json`
- **내용**: 모든 저장된 Preset의 파라미터

**Import**:
- **기능**: JSON 파일에서 Preset 불러오기
- **보안 제한**:
  - 파일 크기: 최대 1MB
  - Preset 개수: 최대 5개
  - Preset 이름: 최대 100자, 위험 문자 검사 (`..`, `/`, `\`, null byte)
- **중복 방지**: 같은 파일 재업로드 시 중복 import 방지
- **검증**: JSON 형식, 파라미터 유효성 검사

---

## 차트 및 시각화

### 1. NAV Chart

**기능**: Net Asset Value (순자산가치) 시계열 차트

**요소**:
- **NAV 라인** (파란색): Equity + CashBalance
- **Signal Days 세로선** (회색): 매수 신호 발생일 (BuyAmt > 0)
  - 투명도 0.4, 가늘게 (1px)
  - 배경 레이어 (TP/SL 마커 아래)
- **TP 마커** (녹색 삼각형 위): Take-Profit 트리거 날짜
- **SL 마커** (빨간색 삼각형 아래): Stop-Loss 트리거 날짜

**차트 라이브러리**:
- Plotly (기본): 인터랙티브 차트, 호버 기능
- Matplotlib (fallback): 정적 차트

### 2. Equity vs NAV Chart

**기능**: Equity와 NAV 비교 차트

**요소**:
- **Equity 라인** (녹색): 보유 주식의 시가 평가금액
- **NAV 라인** (파란색): Equity + CashBalance

**용도**: 
- TP/SL 매도 후 Equity/NAV 변화 확인
- 현금 잔액 vs 주식 가치 비율 확인

### 3. Drawdown Chart

**기능**: 포트폴리오 손실률 시계열 차트

**계산**:
```
running_max = NAV.expanding().max()
drawdown = (NAV / running_max - 1.0) * 100
```

**의미**: 최고점 대비 현재 손실률 (%)

### 4. Monthly Returns Heatmap

**기능**: 월별 수익률 히트맵

**계산 기준**: Portfolio Return Ratio (Equity / PositionCost)
- NAV가 아닌 포지션 수익률 기반
- 신규 투자 효과 제외

**색상**:
- 녹색: 수익
- 빨간색: 손실

---

## 메트릭 설명

### 1. Total Invested
- **의미**: 총 투자 금액 (매수 금액 + 수수료 누적)
- **계산**: `cum_invested`

### 2. Ending Equity
- **의미**: 최종 보유 주식의 시가 평가금액
- **계산**: `cum_shares * final_price`

### 3. Ending NAV
- **의미**: 최종 순자산가치
- **계산**: `Ending Equity + CashBalance`

### 4. Profit
- **의미**: 순 손익 (현금 흐름 + 보유 주식 가치)
- **계산**: `Equity + CumCashFlow`

### 5. NAV (including invested)
- **의미**: 투입 자본 + 수익
- **계산**: `CumInvested + Profit`

### 6. Cumulative Return
- **의미**: 누적 수익률
- **계산**: `(Ending NAV / Total Invested) - 1.0`

### 7. CAGR (연평균 성장률)
- **의미**: 연평균 복리 성장률
- **계산**: `((Ending NAV / Total Invested) ^ (1 / years)) - 1.0`

### 8. Benchmark CAGR
- **의미**: Buy & Hold 전략의 CAGR
- **계산**: `((Final Price / Initial Price) ^ (1 / years)) - 1.0`

### 9. MDD (Maximum Drawdown)
- **의미**: 최대 손실률
- **계산**: Portfolio Return Ratio 기반 (Equity / PositionCost)

### 10. Trades
- **의미**: 총 거래 횟수
- **계산**: `매수 횟수 + TP 트리거 횟수 + SL 트리거 횟수`

### 11. Signal Days (HitDays)
- **의미**: 매수 신호 발생 일수
- **계산**: `(DailyReturn <= Threshold).sum()`

### 12. XIRR
- **의미**: 내부 수익률 (현금 흐름 기반)
- **계산**: Newton-Raphson 방법

### 13. Num Take-Profits
- **의미**: TP 트리거 횟수

### 14. Num Stop-Losses
- **의미**: SL 트리거 횟수

### 15. Baseline Reset Count
- **의미**: Baseline Reset 발생 횟수
- **계산**: `Num TP + Num SL` (Baseline Reset 활성화 시)

### 16. Total Realized Gain
- **의미**: 실현 이익 합계 (매도 시 이익 발생분만)

### 17. Total Realized Loss
- **의미**: 실현 손실 합계 (매도 시 손실 발생분만)

### 18. Net Realized P/L
- **의미**: 순 실현 손익
- **계산**: `Total Realized Gain - Total Realized Loss`

### 19. Ending Position Cost
- **의미**: 최종 보유 주식의 매입 원가

---

## 데이터 Export/Import

### CSV Download
- **기능**: 일일 백테스트 결과 다운로드
- **파일명 형식**: `daily_data_{TICKER}_{STARTDATE}_{ENDDATE}.csv`
- **내용**: 
  - Date, DailyRet, Signal, AdjClose, PxExec
  - BuyAmt, Fee, SharesBought, CashFlow
  - CumShares, Equity, CashBalance, NAV
  - TP_triggered, SL_triggered, SharesSold
  - 기타 모든 일일 컬럼

### JSON Export/Import
- **기능**: Preset 백업/복원
- **보안**: 파일 크기/개수 제한, 이름 검증
- 위 Preset 시스템 섹션 참고

---

## 토글 방식 기능 요약

### 1. 모드 선택 (Radio Buttons)
- **옵션**: Run Backtest, Optimization (Dev), Leverage Mode (Dev), About
- **동작**: 모드 전환 시 화면 전체 변경

### 2. Enable TP (Checkbox)
- **동작**: 체크 시 TP Threshold, TP Sell % 입력란 표시
- **계산**: TP 활성화 시 포지션 수익률 확인하여 트리거

### 3. Enable SL (Checkbox)
- **동작**: 체크 시 SL Threshold, SL Sell % 입력란 표시
- **계산**: SL 활성화 시 포지션 수익률 확인하여 트리거

### 4. Baseline Reset After TP/SL (Checkbox)
- **동작**: 체크 시 TP/SL 후 남은 포지션 진입가 리셋
- **계산**: 진입가 = 매도 실행가로 변경

### 5. Hysteresis/Cooldown Preset (Selectbox)
- **옵션**: Conservative, Moderate, Aggressive, Custom
- **동작**: Preset 선택 시 Hysteresis/Cooldown 값 자동 입력
- **Custom**: 모든 값 수동 입력

### 6. Search Mode (Radio Buttons) (Optimization/Leverage)
- **옵션**: Grid, Random
- **동작**: 
  - Grid: 모든 조합 탐색
  - Random: 랜덤 샘플링 (Random Samples 입력 필요)

### 7. Quick Preset (Radio Buttons)
- **옵션**: None, TQQQ, SOXL, QLD
- **동작**: Preset 선택 시 모든 파라미터 자동 입력
- **자동 선택**: Ticker 입력 시 자동 매칭

---

## 계산 로직 흐름도 (Backtest 모드)

```
1. Ticker 입력 → Yahoo Finance 검증 → 데이터 가져오기 (캐싱)

2. 파라미터 설정 → BacktestParams 객체 생성

3. 백테스트 실행 (일일 루프):
   A. 오늘 일일 수익률 계산
   B. Threshold 확인 → 매수 신호 발생 여부
   C. 매수 실행 (신호 발생 시)
      - 실행가 계산 (슬리피지 포함)
      - 매수 금액, 수수료 계산
      - 상태 업데이트 (cum_shares, position_cost, cum_invested, cum_cash_flow)
      - 포지션 추적 리스트 추가
   D. TP/SL 확인 (오늘 매수하지 않은 경우만)
      - 포지션 수익률 계산
      - TP 조건 확인 (Hysteresis/Cooldown 포함)
      - SL 조건 확인 (Hysteresis/Cooldown 포함)
   E. 매도 실행 (TP/SL 트리거 시)
      - 매도 주식 수 계산 (Sell %)
      - 매도 금액, 수수료 계산
      - 실현 손익 계산
      - 상태 업데이트 (cum_shares, position_cost, cum_cash_flow)
      - FIFO 포지션 제거
      - Baseline Reset (활성화 시)
   F. NAV 계산
      - Equity = cum_shares * current_price
      - CashBalance = cum_invested + cum_cash_flow
      - NAV = Equity + CashBalance
   G. 메트릭 업데이트 (Portfolio Return, Drawdown, 등)

4. 백테스트 완료 → 최종 메트릭 계산 (CAGR, MDD, XIRR, 등)

5. 검증 로직 실행 (NAV = Equity + CashBalance, 등)

6. 결과 표시 (차트, 메트릭, 테이블)
```

---

## 주요 제한사항

1. **날짜 범위**: 최대 10년 (3,650일)
2. **JSON Import**: 최대 5개 preset, 파일 크기 1MB
3. **Preset 이름**: 최대 100자
4. **Shares per Signal**: 0.01 ~ 1,000,000
5. **Developer Mode**: 환경 변수 `DEVELOPER_MODE=true` 필요

---

## 보안 기능

1. **Ticker 입력 Sanitization**: XSS 방지
2. **JSON Import 검증**: 
   - 파일 크기 제한
   - Preset 개수 제한
   - 이름 검증 (위험 문자 차단)
3. **날짜 범위 제한**: DoS 방지
4. **캐싱**: 불필요한 API 호출 방지

---

## 결론

이 앱은 다음과 같은 기능을 제공합니다:
1. **단일 백테스트**: 파라미터 설정 → 실행 → 결과 확인
2. **파라미터 최적화**: Grid/Random Search로 최적 파라미터 탐색
3. **레버리지 ETF 전용 최적화**: TP/SL fine-tuning
4. **고급 TP/SL 기능**: 독립 설정, Baseline Reset, Hysteresis, Cooldown
5. **Preset 시스템**: 세션 기반 + JSON Export/Import
6. **Quick Presets**: TQQQ, SOXL, QLD 전용 원클릭 설정
7. **시각화**: NAV, Equity, Drawdown, Monthly Heatmap 차트
8. **데이터 Export**: CSV 다운로드

모든 계산은 회계적 정합성을 유지하며, 검증 로직으로 정확성을 보장합니다.


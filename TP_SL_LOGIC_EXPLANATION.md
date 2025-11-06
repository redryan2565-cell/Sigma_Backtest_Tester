# TP/SL 트리거 로직 및 자금 추적 시스템 설명서

## 📌 개요

이 백테스트 엔진은 **entry_price 기반 TP/SL 트리거 시스템**을 사용하여 실제 보유 주식의 매수 단가 대비 수익률을 기준으로 Take-Profit(익절) 및 Stop-Loss(손절)를 실행합니다.

---

## 🔄 주요 변경 사항 (이전 vs 현재)

### 이전 로직 (문제점)
- **기준**: `NAVReturn_baselined = (NAV / roi_base) - 1.0`
- **문제**: NAV는 새로운 투자금 유입이나 TP/SL 후 리셋으로 인해 실제 보유 주식의 수익률과 불일치
- **결과**: 실제로는 손실인데 TP가 트리거되거나, 실제 수익인데 SL이 트리거되는 오작동 발생

### 현재 로직 (수정 후)
- **기준**: `position_return = (현재가격 / 가중평균_entry_price) - 1.0`
- **장점**: 실제 보유 주식의 매수 단가 대비 수익률을 정확히 반영
- **결과**: 실제 수익률과 일치하는 TP/SL 트리거

---

## 💰 자금 추적 시스템

### 1. 포지션 추적 (FIFO 방식)

각 매수 거래마다 다음 정보를 추적합니다:
```python
positions: list[tuple[int, float, float]] = [
    (entry_date_idx, shares, entry_price),
    ...
]
```

- **entry_date_idx**: 매수일 인덱스 (날짜 순서)
- **shares**: 매수한 주식 수
- **entry_price**: 매수 단가 (슬리피지 + 수수료 포함)
  - 계산식: `entry_price = 실행가격 × (1 + fee_rate)`
  - 실행가격: `AdjClose × (1 + slippage_rate)` (매수 시)

### 2. 가중 평균 Entry Price 계산

TP/SL 트리거 판단을 위해 **1일 이상 보유한 포지션**들의 가중 평균 entry_price를 계산:

```python
# 1일 이상 보유한 포지션만 필터링
total_eligible_shares = 0.0
total_eligible_cost = 0.0

for entry_date_idx, pos_shares, pos_entry_price in positions:
    holding_days = 현재일 - entry_date_idx
    if holding_days >= 1 and pos_entry_price > 0:
        total_eligible_shares += pos_shares
        total_eligible_cost += pos_shares * pos_entry_price

# 가중 평균 계산
weighted_avg_entry_price = total_eligible_cost / total_eligible_shares
```

**중요**: 매수 당일은 TP/SL 체크를 하지 않음 (최소 보유 기간 1일)

### 3. 포지션 수익률 계산

```python
position_return = (현재_AdjClose / weighted_avg_entry_price) - 1.0
```

**예시**:
- 매수 단가: $100
- 현재 가격: $150
- 수익률: (150 / 100) - 1 = **+50%**

---

## 🎯 TP/SL 트리거 메커니즘

### 트리거 조건

```python
# TP 트리거 조건
if position_return >= tp_threshold:  # 예: +30%
    TP 트리거 발생

# SL 트리거 조건  
if position_return <= sl_threshold:  # 예: -20%
    SL 트리거 발생
```

### 트리거 순서

1. **매수 로직 먼저 실행**
   - 신호 발생 시 주식 매수
   - `positions` 리스트에 추가: `(i, shares_bought, entry_price)`

2. **TP/SL 체크 (매수 당일 제외)**
   - `shares_bought_today == 0`인 경우만 체크
   - 1일 이상 보유한 포지션만 평가
   - 가중 평균 entry_price 기준으로 수익률 계산

3. **트리거 시 매도 실행**
   - 설정된 비율만큼 매도 (예: TP 50%, SL 100%)
   - FIFO 방식으로 포지션 제거

### 매도 실행 상세

```python
# 매도 주식 수 계산
shares_to_sell = round(cum_shares * sell_percentage)  # 반올림

# 매도 가격 (슬리피지 적용)
sell_price = AdjClose × (1 - slippage_rate)

# 매도 수수료
sell_fee = gross_proceeds × fee_rate

# 순수익
net_proceeds = gross_proceeds - sell_fee
```

---

## 🔄 TP/SL 트리거 후 처리 (핵심 개선 사항)

### 문제 상황
TP 트리거 후 **부분 매도**(예: 50% 매도)를 하면, 남은 포지션의 entry_price가 여전히 낮은 값으로 유지되어 다음날 즉시 다시 TP가 트리거될 수 있음.

### 해결 방법: Entry Price 재설정

TP/SL 트리거 후 **남은 포지션의 entry_price를 현재 매도 가격으로 재설정**:

```python
if positions and (tp_triggered or sl_triggered):
    # 현재 매도 실행 가격 (슬리피지 포함)
    current_sell_price = AdjClose × (1 - slippage_rate)
    
    # 새로운 entry_price = 매도가격 + 수수료
    new_entry_price = current_sell_price × (1 + fee_rate)
    
    # 모든 남은 포지션 업데이트
    positions = [
        (현재일, pos_shares, new_entry_price) 
        for _, pos_shares, _ in positions
    ]
```

**효과**:
- 남은 포지션의 수익률이 **0%로 리셋**됨
- 다음날 즉시 TP/SL이 재트리거되지 않음
- `entry_date`도 현재일로 업데이트되어 최소 보유 기간 체크 적용

---

## 💵 자금 추적 (Accounting)

### 주요 변수

1. **cum_shares**: 누적 보유 주식 수
2. **position_cost**: 현재 보유 주식의 총 매수 원가
3. **equity**: 현재 주식 가치 = `cum_shares × AdjClose`
4. **cum_invested**: 총 투자금 (매수 시 누적)
5. **cum_cash_flow**: 누적 현금 흐름 (매수: 음수, 매도: 양수)
6. **NAV**: 순자산가치 = `equity + cash_balance`
   - `cash_balance = cum_invested + cum_cash_flow`

### 매수 시

```python
buy_amount = shares × 실행가격
fee = buy_amount × fee_rate
total_cost = buy_amount + fee

# 업데이트
cum_shares += shares
position_cost += total_cost
cum_invested += total_cost
cum_cash_flow -= total_cost  # 현금 유출
```

### 매도 시 (TP/SL)

```python
gross_proceeds = shares_sold × sell_price
sell_fee = gross_proceeds × fee_rate
net_proceeds = gross_proceeds - sell_fee

# 원가 계산 (평균 원가법)
avg_cost_per_share = position_cost / cum_shares
cost_of_shares_sold = avg_cost_per_share × shares_sold

# 실현 손익
realized_pnl = net_proceeds - cost_of_shares_sold

# 업데이트
cum_shares -= shares_sold
position_cost -= cost_of_shares_sold
cum_cash_flow += net_proceeds  # 현금 유입
```

---

## 📊 ROI 및 수익률 계산

### 1. Portfolio Return (포트폴리오 수익률)

```python
portfolio_return = (equity / position_cost) - 1.0
```

- **의미**: 현재 보유 주식의 매수 원가 대비 수익률
- **용도**: Drawdown 및 MDD 계산 기준

### 2. NAV Return Global (전역 수익률)

```python
nav_return_global = (NAV / cum_invested) - 1.0
```

- **의미**: 초기 투자금 대비 전체 기간 누적 수익률
- **특징**: TP/SL 리셋과 무관하게 전체 기간 성과 반영

### 3. NAV Return Baselined (기준선 리셋 수익률)

```python
if reset_baseline_after_tp_sl:
    roi_base = NAV  # TP/SL 트리거 시 리셋

nav_return_baselined = (NAV / roi_base) - 1.0
```

- **의미**: 마지막 TP/SL 트리거 시점 이후의 수익률
- **용도**: 구간별 성과 평가

### 4. Position Return (포지션 수익률) - TP/SL 트리거 기준

```python
position_return = (AdjClose / weighted_avg_entry_price) - 1.0
```

- **의미**: 실제 보유 주식의 매수 단가 대비 수익률
- **용도**: TP/SL 트리거 판단

---

## 🛡️ 안전장치 (Guard Clauses)

### 1. 매수 당일 TP/SL 방지

```python
# 매수 로직 먼저 실행
if shares_bought_today > 0:
    # TP/SL 체크 스킵
    pass

# 최종 가드
if shares_bought_today > 1e-9:
    tp_triggered[i] = False
    sl_triggered[i] = False
```

### 2. 최소 보유 기간 체크

```python
min_holding_days = 1
for entry_date_idx, pos_shares, pos_entry_price in positions:
    holding_days = i - entry_date_idx
    if holding_days >= min_holding_days:
        # TP/SL 평가 대상
```

### 3. Cooldown 기간

```python
if tp_cooldown_days > 0:
    days_since_tp = 현재일 - last_tp_date_idx
    if days_since_tp < tp_cooldown_days:
        tp_cooldown_active = True  # TP 트리거 비활성화
```

### 4. Hysteresis (히스테리시스)

```python
# TP 트리거 후
if tp_hysteresis_active:
    # 재무장 조건: 수익률이 (threshold - hysteresis) 아래로 떨어져야 함
    if position_return < tp_threshold - tp_hysteresis:
        tp_hysteresis_active = False  # 재무장
```

---

## 📈 Drawdown 및 MDD 계산

### Drawdown 계산 기준

**Portfolio Return Ratio**를 기준으로 계산:

```python
portfolio_return_ratio = equity / position_cost

# Drawdown 계산
if position_cost > 1e-9:
    drawdown = portfolio_return_ratio - peak_ratio
else:
    drawdown = 0.0  # 포지션이 없으면 drawdown = 0
```

**이유**:
- NAV는 TP/SL 리셋이나 새로운 투자로 인해 불연속적
- Portfolio Return Ratio는 연속적이며 실제 포트폴리오 손실을 정확히 반영

### MDD (Maximum Drawdown)

```python
MDD = min(drawdown)  # 전체 기간 중 최대 하락폭
```

---

## 🔍 검증 로직

### 1. Accounting 검증

```python
# 현금 흐름 검증
expected_cash_flow = -(buy_amt + fee) + net_proceeds
assert abs(actual_cash_flow - expected_cash_flow) < 1e-6

# 주식 수 검증
total_bought - total_sold = final_shares
```

### 2. TP/SL 검증

```python
# TP/SL과 매수는 같은 날 발생 불가
assert not (tp_triggered and shares_bought > 0)

# Cooldown 검증
assert days_since_tp >= tp_cooldown_days
```

---

## 📝 요약

1. **TP/SL 트리거 기준**: `position_return = (현재가격 / 가중평균_entry_price) - 1.0`
2. **포지션 추적**: FIFO 방식으로 `(entry_date, shares, entry_price)` 추적
3. **트리거 후 처리**: 남은 포지션의 entry_price를 현재 매도 가격으로 재설정
4. **자금 추적**: `position_cost`, `equity`, `NAV`, `cum_cash_flow` 등으로 정확히 추적
5. **안전장치**: 매수 당일 방지, 최소 보유 기간, Cooldown, Hysteresis

이 시스템은 실제 보유 주식의 매수 단가 대비 수익률을 정확히 반영하여 TP/SL을 트리거하며, 부분 매도 후에도 연속 트리거를 방지합니다.


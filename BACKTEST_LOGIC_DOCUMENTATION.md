# 백테스트 로직 상세 문서

## 목차
1. [개요](#개요)
2. [핵심 변수 정의](#핵심-변수-정의)
3. [매수 로직](#매수-로직)
4. [TP/SL 트리거 로직](#tpsl-트리거-로직)
5. [매도 로직](#매도-로직)
6. [NAV 계산 로직](#nav-계산-로직)
7. [수익률 계산](#수익률-계산)
8. [일일 처리 순서](#일일-처리-순서)
9. [검증 로직](#검증-로직)

---

## 개요

이 백테스트 시스템은 Dip-Buy 전략을 기반으로 하며, shares-based 모드만 지원합니다.
- 일일 수익률이 threshold 이하로 떨어지면 매수 신호 발생
- TP/SL 기능으로 수익 실현 및 손실 제한
- 포지션별 진입가 추적 (FIFO)

---

## 핵심 변수 정의

### 1. 투자 추적 변수

#### `cum_invested` (Cumulative Invested, ActualInvested)
- **의미**: 사용자가 실제로 투입한 누적 자본 금액
- **초기값**: 0
- **업데이트**:
  - 매수 시: `cum_invested += (buy_amt + fee)`
  - 매도 시: 변경 없음 (누적 투자액은 유지)
- **용도**: 전체 수익률 계산의 기준점

#### `position_cost` (Position Cost Basis)
- **의미**: 현재 보유 주식의 매입 원가 (cost basis)
- **초기값**: 0
- **업데이트**:
  - 매수 시: `position_cost += (buy_amt + fee)`
  - 매도 시: `position_cost -= cost_of_shares_sold`
- **용도**: 포지션 수익률 계산, TP/SL 트리거 판단

### 2. 현금 흐름 변수

#### `cum_cash_flow` (Cumulative Cash Flow)
- **의미**: 누적 순 현금 흐름 (매수는 음수, 매도는 양수)
- **초기값**: 0
- **업데이트**:
  - 매수 시: `cum_cash_flow -= (buy_amt + fee)`
  - 매도 시: `cum_cash_flow += net_proceeds`
- **계산**:
  - 매수 시: `cash_flow = -(buy_amt + fee)`
  - 매도 시: `cash_flow = net_proceeds`

#### `CashBalance` (현금 잔액)
- **의미**: 현재 보유한 현금 잔액
- **계산식**: `CashBalance = cum_invested + cum_cash_flow`
- **해석**:
  - `cum_invested`: 총 투입 금액
  - `cum_cash_flow`: 순 현금 변화 (음수 = 지출, 양수 = 수입)
  - 예시 1 (매수 후): cum_invested=$1,000, cum_cash_flow=-$1,000 → CashBalance=$0
  - 예시 2 (매도 후): cum_invested=$1,000, cum_cash_flow=+$500 → CashBalance=$1,500

### 3. 자산 평가 변수

#### `cum_shares` (Cumulative Shares)
- **의미**: 현재 보유한 총 주식 수
- **초기값**: 0
- **업데이트**:
  - 매수 시: `cum_shares += shares_bought`
  - 매도 시: `cum_shares -= shares_sold`

#### `Equity` (보유 자산 평가금액)
- **의미**: 현재 보유 중인 주식의 시가 평가 금액
- **계산식**: `Equity = cum_shares * current_price`
- **특징**: 주가 변동에 따라 실시간 변동

#### `NAV` (Net Asset Value, 순자산가치)
- **의미**: 투자자가 현재 보유한 전체 자산의 총합
- **계산식**: `NAV = Equity + CashBalance`
- **확장**: `NAV = (cum_shares * price) + (cum_invested + cum_cash_flow)`

### 4. 포지션 추적

#### `positions` (리스트)
- **의미**: FIFO 방식으로 관리되는 포지션 목록
- **구조**: `[(entry_date_idx, shares, entry_price_per_share), ...]`
- **entry_price_per_share**: 진입가 = 실행가 * (1 + fee_rate) (수수료 포함)
- **용도**: TP/SL 트리거 판단용 평균 진입가 계산

---

## 매수 로직

### 1. 신호 생성 (Dip Signal)

```python
# 일일 수익률 계산
daily_return = (today_price - yesterday_price) / yesterday_price

# 매수 신호 발생 조건
if daily_return <= threshold:
    signal = True  # 예: threshold = -0.041 (즉, -4.1% 이하 하락 시)
```

### 2. 매수 실행

```python
if signal:
    # 매수할 주식 수 (고정)
    shares_to_buy = shares_per_signal  # 예: 10주
    
    # 실행가 (슬리피지 포함)
    execution_price = adj_close * (1 + slippage_rate)  # 예: 0.25% 슬리피지
    
    # 매수 금액
    buy_amt = shares_to_buy * execution_price
    
    # 수수료
    fee = buy_amt * fee_rate  # 예: 0.05%
    
    # 총 비용
    total_cost = buy_amt + fee
```

### 3. 상태 업데이트

```python
# 주식 수 증가
cum_shares += shares_to_buy

# 포지션 원가 증가
position_cost += total_cost

# 투자액 증가 (원금 기준선)
cum_invested += total_cost

# 현금 흐름 (지출)
cash_flow = -total_cost
cum_cash_flow += cash_flow

# 포지션 추적에 추가
entry_price_per_share = execution_price * (1 + fee_rate)
positions.append((current_date_index, shares_to_buy, entry_price_per_share))
```

---

## TP/SL 트리거 로직

### 1. 트리거 조건 확인 시점

- **시점**: 매수 후 (오늘 매수한 경우 제외, 최소 1일 보유 필요)
- **조건**: `cum_shares > 0` (보유 주식 있음)

### 2. 포지션 수익률 계산

```python
# 1일 이상 보유한 포지션만 대상
eligible_shares = 0
eligible_cost = 0

for (entry_date, pos_shares, pos_entry_price) in positions:
    holding_days = current_date - entry_date
    if holding_days >= 1:
        eligible_shares += pos_shares
        eligible_cost += pos_shares * pos_entry_price

# 가중 평균 진입가
if eligible_shares > 0:
    weighted_avg_entry_price = eligible_cost / eligible_shares
    
    # 포지션 수익률 계산
    position_return = (current_price / weighted_avg_entry_price) - 1.0
```

### 3. Take-Profit (TP) 트리거

```python
# TP 조건 확인
if position_return >= tp_threshold:  # 예: 0.30 (30%)
    tp_triggered = True
    
    # 매도 비율 적용
    shares_to_sell = cum_shares * tp_sell_percentage  # 예: 100% (전량 매도)
```

**Hysteresis (재트리거 방지):**
```python
if tp_hysteresis > 0:  # 예: 0.10 (10%)
    # TP 트리거 후, 수익률이 (threshold - hysteresis) 이하로 떨어져야 재활성화
    # 예: 30% TP 후 → 20% 이하로 떨어져야 다시 TP 가능
    if position_return < (tp_threshold - tp_hysteresis):
        tp_hysteresis_active = False
```

**Cooldown (대기 기간):**
```python
if tp_cooldown_days > 0:  # 예: 5일
    # 마지막 TP 트리거 후 5일 이내에는 TP 재발동 불가
    days_since_last_tp = current_date - last_tp_date
    if days_since_last_tp < tp_cooldown_days:
        tp_trigger_disabled = True
```

### 4. Stop-Loss (SL) 트리거

```python
# SL 조건 확인 (TP가 트리거되지 않은 경우만)
if not tp_triggered and position_return <= sl_threshold:  # 예: -0.25 (-25%)
    sl_triggered = True
    
    # 매도 비율 적용
    shares_to_sell = cum_shares * sl_sell_percentage  # 예: 100%
```

**Hysteresis & Cooldown도 동일하게 적용**

---

## 매도 로직

### 1. 매도 주식 수 계산

```python
# 매도할 주식 수 (반올림)
shares_to_sell_float = cum_shares * sell_percentage
shares_to_sell = int(shares_to_sell_float + 0.5)  # 표준 반올림

# 최소 1주 (비율이 너무 작아 0이 되는 경우 방지)
if shares_to_sell == 0 and cum_shares > 0 and sell_percentage > 0.01:
    shares_to_sell = 1

# 보유 주식 수 초과 방지
shares_to_sell = min(shares_to_sell, int(cum_shares))
```

### 2. 매도 금액 계산

```python
# 실행가 (슬리피지 포함, 매도는 마이너스)
execution_sell_price = adj_close * (1 - slippage_rate)

# 총 매도 대금
gross_proceeds = shares_to_sell * execution_sell_price

# 수수료
sell_fee = gross_proceeds * fee_rate

# 순 매도 대금 (수수료 차감)
net_proceeds = gross_proceeds - sell_fee
```

### 3. 실현 손익 계산

```python
# 매도한 주식의 원가 계산 (평균 원가법)
avg_cost_per_share = position_cost / cum_shares
cost_of_shares_sold = avg_cost_per_share * shares_to_sell

# 실현 손익
realized_pnl = net_proceeds - cost_of_shares_sold
```

### 4. 상태 업데이트

```python
# 주식 수 감소
cum_shares -= shares_to_sell

# 포지션 원가 감소
position_cost -= cost_of_shares_sold

# 현금 흐름 (수입)
cash_flow = net_proceeds
cum_cash_flow += net_proceeds

# 투자액은 변경 없음 (누적 투자액 유지)
# cum_invested: 변경 없음
```

### 5. 포지션 추적 업데이트 (FIFO)

```python
# FIFO 방식으로 포지션 제거
remaining_to_sell = shares_to_sell

while remaining_to_sell > 0 and positions:
    (entry_date, entry_shares, entry_price) = positions[0]
    
    if entry_shares <= remaining_to_sell:
        # 전체 포지션 제거
        remaining_to_sell -= entry_shares
        positions.pop(0)
    else:
        # 부분 제거
        positions[0] = (entry_date, entry_shares - remaining_to_sell, entry_price)
        remaining_to_sell = 0
```

### 6. Baseline Reset (TP/SL 후)

```python
# TP/SL 트리거 후 남은 포지션의 진입가를 현재가로 리셋
if positions and (tp_triggered or sl_triggered):
    new_entry_price = execution_sell_price * (1 + fee_rate)
    
    # 모든 남은 포지션의 진입가와 날짜를 리셋
    positions = [(current_date, pos_shares, new_entry_price) 
                 for (_, pos_shares, _) in positions]
```

**의미**: TP/SL 후 남은 포지션은 0% 수익률에서 다시 시작 (즉각 재트리거 방지)

---

## NAV 계산 로직

### 매 일자 NAV 계산 (매수/매도 후)

```python
# Step 1: Equity 계산
Equity = cum_shares * current_adj_close

# Step 2: CashBalance 계산
CashBalance = cum_invested + cum_cash_flow

# Step 3: NAV 계산
NAV = Equity + CashBalance
```

### 예시 1: 초기 매수

```
초기 상태:
- cum_invested = 0
- cum_cash_flow = 0
- cum_shares = 0
- Equity = 0
- CashBalance = 0
- NAV = 0

매수 ($1,000 투입, 10주 매수):
- buy_amt = $1,000
- fee = $0.50
- total_cost = $1,000.50

업데이트:
- cum_invested = 0 + $1,000.50 = $1,000.50
- cum_cash_flow = 0 - $1,000.50 = -$1,000.50
- cum_shares = 0 + 10 = 10
- Equity = 10 * $100 = $1,000
- CashBalance = $1,000.50 - $1,000.50 = $0
- NAV = $1,000 + $0 = $1,000
```

### 예시 2: 주가 상승 후 전량 매도

```
주가 상승 (10% → $110):
- cum_shares = 10
- Equity = 10 * $110 = $1,100
- CashBalance = $0 (변경 없음)
- NAV = $1,100 + $0 = $1,100

TP 트리거 (30% 수익률) - 전량 매도:
- execution_sell_price = $110 * 0.9975 = $109.725 (슬리피지)
- gross_proceeds = 10 * $109.725 = $1,097.25
- sell_fee = $1,097.25 * 0.0005 = $0.55
- net_proceeds = $1,097.25 - $0.55 = $1,096.70

업데이트:
- cum_invested = $1,000.50 (변경 없음)
- cum_cash_flow = -$1,000.50 + $1,096.70 = $96.20
- cum_shares = 10 - 10 = 0
- Equity = 0 * $110 = $0
- CashBalance = $1,000.50 + $96.20 = $1,096.70
- NAV = $0 + $1,096.70 = $1,096.70

실현 손익: $1,096.70 - $1,000.50 = $96.20 (9.6% 수익)
```

### 예시 3: 부분 매도 (50%)

```
주가 상승 ($110):
- cum_shares = 10
- Equity = $1,100
- NAV = $1,100

TP 트리거 (50% 매도):
- shares_to_sell = 5
- net_proceeds = $548.35 (계산 생략)

업데이트:
- cum_invested = $1,000.50 (변경 없음)
- cum_cash_flow = -$1,000.50 + $548.35 = -$452.15
- cum_shares = 10 - 5 = 5
- Equity = 5 * $110 = $550
- CashBalance = $1,000.50 - $452.15 = $548.35
- NAV = $550 + $548.35 = $1,098.35
```

---

## 수익률 계산

### 1. Portfolio Return (포지션 수익률)

```python
# 현재 포지션의 수익률 (TP/SL 판단용)
if position_cost > 0 and cum_shares > 0:
    portfolio_return = (Equity / position_cost) - 1.0
else:
    portfolio_return = 0.0
```

**의미**: 현재 보유 주식의 시가 대비 원가 수익률

### 2. NAV Return (전체 수익률)

```python
# 투자 원금 대비 NAV 수익률
if cum_invested > 0:
    nav_return = (NAV / cum_invested) - 1.0
else:
    nav_return = 0.0
```

**의미**: 투입 자본 대비 전체 자산 수익률

### 3. Cumulative Return (누적 수익률)

```python
# 최종 수익률
cumulative_return = (ending_nav / total_invested) - 1.0
```

### 4. CAGR (연평균 성장률)

```python
# 연평균 성장률
years = (end_date - start_date).days / 365.25
if years > 0 and ending_nav > 0 and total_invested > 0:
    cagr = ((ending_nav / total_invested) ** (1.0 / years)) - 1.0
```

---

## 일일 처리 순서

각 거래일마다 다음 순서로 처리:

### Phase 1: 초기화
```python
# 오늘의 변수 초기화
shares_bought_today = 0
tp_triggered = False
sl_triggered = False
```

### Phase 2: 매수 처리
```python
# Step 1: 신호 확인
if daily_return <= threshold:
    signal = True

# Step 2: 매수 실행
if signal:
    # 매수 금액 계산
    # 상태 업데이트 (cum_shares, position_cost, cum_invested, cum_cash_flow)
    # 포지션 추적 리스트에 추가
    shares_bought_today = shares_per_signal
```

### Phase 3: TP/SL 확인 (오늘 매수하지 않은 경우만)
```python
if shares_bought_today == 0 and cum_shares > 0:
    # Step 1: 포지션 수익률 계산
    position_return = calculate_position_return()
    
    # Step 2: TP 확인
    if position_return >= tp_threshold and not tp_cooldown_active:
        tp_triggered = True
    
    # Step 3: SL 확인 (TP가 트리거되지 않은 경우)
    elif position_return <= sl_threshold and not sl_cooldown_active:
        sl_triggered = True
```

### Phase 4: 매도 실행 (TP/SL 트리거된 경우)
```python
if tp_triggered or sl_triggered:
    # Step 1: 매도 주식 수 계산
    # Step 2: 매도 금액 계산
    # Step 3: 실현 손익 계산
    # Step 4: 상태 업데이트 (cum_shares, position_cost, cum_cash_flow)
    # Step 5: 포지션 추적 리스트 업데이트 (FIFO)
    # Step 6: Baseline Reset (남은 포지션 진입가 리셋)
    
    # Equity와 NAV 재계산
    Equity = cum_shares * current_price
    CashBalance = cum_invested + cum_cash_flow
    NAV = Equity + CashBalance
```

### Phase 5: 최종 NAV 계산 (매수만 하거나 아무것도 안 한 경우)
```python
# 현재 상태로 NAV 계산
Equity = cum_shares * current_price
CashBalance = cum_invested + cum_cash_flow
NAV = Equity + CashBalance
```

### Phase 6: 메트릭 계산
```python
# Portfolio Return, NAV Return, Unrealized P/L 등 계산
# Drawdown 계산
# 배열에 저장
```

---

## 검증 로직

백테스트 완료 후 다음 검증 수행:

### 1. 매수 금액 검증
```python
# BuyAmt = SharesBought * PxExec
expected_buy_amt = shares_bought * execution_price
assert buy_amt == expected_buy_amt
```

### 2. 수수료 검증
```python
# Fee = BuyAmt * fee_rate (매수)
# SellFee = GrossProceeds * fee_rate (매도)
expected_fee = buy_amt * fee_rate
assert fee == expected_fee
```

### 3. 현금 흐름 검증
```python
# 매수: CashFlow = -(BuyAmt + Fee)
# 매도: CashFlow = NetProceeds (+ 매수가 있으면 - buy_cost)
expected_cash_flow = -buy_cost + net_proceeds
assert cash_flow == expected_cash_flow
```

### 4. CumInvested 검증
```python
# CumInvested = 모든 매수의 누적 합계
expected_cum_invested = (BuyAmt + Fee).cumsum()
assert cum_invested == expected_cum_invested
```

### 5. CashBalance 검증
```python
# CashBalance = CumInvested + CumCashFlow
expected_cash_balance = cum_invested + cum_cash_flow
assert cash_balance == expected_cash_balance
```

### 6. NAV 검증
```python
# NAV = Equity + CashBalance
expected_nav = equity + cash_balance
assert nav == expected_nav
```

### 7. 포지션 원가 검증
```python
# PositionCost는 항상 >= 0
assert position_cost >= 0

# 주식이 없으면 PositionCost도 0
if cum_shares == 0:
    assert position_cost == 0
```

---

## 주요 불변식 (Invariants)

다음 조건들은 항상 성립해야 함:

1. **NAV 보존**: `NAV = Equity + CashBalance = (cum_shares * price) + (cum_invested + cum_cash_flow)`

2. **현금 보존**: `CashBalance = cum_invested + cum_cash_flow`
   - cum_invested: 투입 자본 (항상 증가)
   - cum_cash_flow: 순 현금 변화 (매수는 음수, 매도는 양수)

3. **주식 수 비음수**: `cum_shares >= 0`

4. **포지션 원가 비음수**: `position_cost >= 0`

5. **포지션 원가와 주식 수 일관성**:
   - `cum_shares == 0` ⇒ `position_cost == 0`
   - `cum_shares > 0` ⇒ `position_cost > 0`

6. **투자액 단조 증가**: `cum_invested[i] >= cum_invested[i-1]` (매수만 발생, 매도 시 불변)

---

## 잠재적 버그 체크리스트

1. **NAV 중복 계산**: 
   - CashBalance에 cum_invested를 포함하는 것이 맞는가?
   - ✅ 맞음: CashBalance = (초기 자본 0) + 투입 자본 + 현금 변화

2. **매도 후 NAV 급등**:
   - 전량 매도 시 Equity = 0, CashBalance = cum_invested + (매도대금 - 투입금액)
   - NAV = 0 + CashBalance = 매도대금 (실현수익 포함)
   - ✅ 정상: 수익이 실현되면 NAV는 증가

3. **Baseline Reset 후 즉각 재트리거**:
   - TP/SL 후 남은 포지션의 진입가를 현재가로 리셋
   - ✅ 방지됨: 수익률이 0%에서 다시 시작

4. **슬리피지/수수료 이중 적용**:
   - 진입가에 수수료 포함: `entry_price * (1 + fee_rate)`
   - ✅ 정확: 실제 주당 비용 반영

5. **FIFO 포지션 추적 오류**:
   - 매도 시 가장 오래된 포지션부터 제거
   - ✅ 구현됨: positions.pop(0)

---

## 결론

이 백테스트 시스템은:
- **회계적 정합성**: NAV = Equity + CashBalance 보존
- **포지션 추적**: FIFO 방식으로 진입가 관리
- **TP/SL 정확성**: 포지션 수익률 기반, Baseline Reset, Hysteresis, Cooldown 지원
- **검증 완료**: 모든 계산이 검증 로직을 통과

**주요 설계 원칙**:
1. cum_invested는 누적 투자액 (원금 기준선)
2. cum_cash_flow는 순 현금 변화 (매수 음수, 매도 양수)
3. CashBalance = cum_invested + cum_cash_flow (현재 보유 현금)
4. NAV = Equity + CashBalance (전체 자산)

이 문서를 ChatGPT 또는 다른 LLM에게 제공하여 논리적 오류를 검증받을 수 있습니다.


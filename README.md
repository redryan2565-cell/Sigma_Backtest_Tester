# Normal Dip Backtest (yAOIL)

백테스트 엔진 + Streamlit GUI를 활용한 딥 구매 전략 백테스트 도구.

## 주요 기능

- **Budget-based 모드**: 주간 예산 기반 매수
- **Shares-based 모드**: 신호당 고정 주식 수 매수
- **Streamlit GUI**: 인터랙티브 웹 인터페이스
- **CLI**: 명령줄 인터페이스
- **CSV 내보내기**: 백테스트 결과 저장

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
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
├── backtest/      # 백테스트 엔진
├── data/          # 데이터 소스 (Yahoo Finance, Alpha Vantage)
├── strategy/      # 전략 (딥 구매)
├── cli.py         # CLI 인터페이스
├── gui_streamlit.py  # Streamlit GUI
└── config.py      # 설정 관리
```

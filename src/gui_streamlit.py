from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Optional imports for enhanced features
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGrid_AVAILABLE = True
except ImportError:
    AGGrid_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except ImportError:
    OPTION_MENU_AVAILABLE = False

# Fix import path for Streamlit execution
if Path(__file__).parent.parent.parent.exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtest.engine import BacktestParams, run_backtest
from src.data.yfin import YFinanceFeed


def main() -> None:
    st.set_page_config(page_title="normal-dip-bt", layout="wide", page_icon="📈")
    
    # Navigation menu
    if OPTION_MENU_AVAILABLE:
        selected = option_menu(
            menu_title=None,
            options=["📊 Backtest", "📁 Load CSV", "ℹ️ About"],
            icons=["graph-up", "folder", "info-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        view_mode = "Run Backtest" if selected == "📊 Backtest" else ("Load CSV" if selected == "📁 Load CSV" else "About")
    else:
        st.title("📈 Normal Dip Backtest")
        view_mode = st.radio("Mode", options=["Run Backtest", "Load CSV", "About"], horizontal=True)

    # About page
    if view_mode == "About":
        st.header("📖 About Normal Dip Backtest")
        st.markdown("""
        ### Budget-based Mode
        - Set a weekly budget amount
        - Choose allocation mode (split or first_hit)
        - Enable/disable carryover of unused budget
        
        ### Shares-based Mode
        - Enter number of shares to buy per signal
        - Each signal day will buy the specified number of shares at market price
        
        ### Parameters
        - **Ticker**: Stock symbol (e.g., TQQQ, AAPL)
        - **Date Range**: Start and end dates for backtest
        - **Threshold**: Daily return threshold (must be negative, e.g., -0.041 for -4.1%)
        - **Fee Rate**: Trading fee per transaction
        - **Slippage Rate**: Price impact assumption
        """)
        
        st.header("🚀 Enhanced Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Plotly Charts**\n\nInteractive zoom, pan, and hover")
        with col2:
            st.info("**AgGrid Tables**\n\nSort, filter, and export data")
        with col3:
            st.info("**Auto CSV Save**\n\nResults automatically saved")
        
        return

    with st.sidebar:
        if view_mode == "Run Backtest":
            st.header("Purchase Mode")
            purchase_mode = st.radio(
                "Purchase Mode",
                options=["Budget-based", "Shares-based"],
                index=0,
                help="Budget-based: Buy with fixed weekly budget. Shares-based: Buy fixed number of shares per signal."
            )

            st.header("Parameters")
            
            # Ticker input with validation
            ticker = st.text_input(
                "Ticker",
                value="TQQQ",
                help="Stock ticker symbol (e.g., TQQQ, AAPL, SPY)"
            ).strip().upper()
            
            # Date inputs
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input(
                    "Start Date",
                    value=date(2016, 1, 1),
                    help="Backtest start date"
                )
            with col2:
                end = st.date_input(
                    "End Date",
                    value=date.today(),
                    help="Backtest end date"
                )
            
            # Threshold input
            threshold = st.number_input(
                "Threshold (daily return)",
                value=-0.041,
                step=0.001,
                format="%0.6f",
                help="Daily return threshold (must be negative, e.g., -0.041 for -4.1% drop)"
            )
            
            # Conditional inputs based on purchase mode
            if purchase_mode == "Budget-based":
                weekly_budget = st.number_input(
                    "Weekly Budget",
                    value=500.0,
                    step=50.0,
                    min_value=0.01,
                    help="Amount to invest per week (in currency units)"
                )
                mode = st.selectbox(
                    "Allocation Mode",
                    options=["split", "first_hit"],
                    index=0,
                    help="split: Divide budget equally among signal days. first_hit: Spend entire budget on first signal."
                )
                carryover = st.checkbox(
                    "Carryover",
                    value=True,
                    help="Carry unused weekly budget to next week"
                )
                shares_per_signal = None
            else:  # Shares-based
                shares_per_signal = st.number_input(
                    "Shares per Signal",
                    value=10.0,
                    step=1.0,
                    min_value=0.01,
                    help="Number of shares to buy each time a signal occurs"
                )
                weekly_budget = None
                mode = None
                carryover = None
            
            # Fee and slippage
            col3, col4 = st.columns(2)
            with col3:
                fee_rate = st.number_input(
                    "Fee Rate",
                    value=0.0005,
                    step=0.0001,
                    format="%0.6f",
                    help="Trading fee rate (e.g., 0.0005 = 0.05%)"
                )
            with col4:
                slippage_rate = st.number_input(
                    "Slippage Rate",
                    value=0.0005,
                    step=0.0001,
                    format="%0.6f",
                    help="Price slippage assumption (e.g., 0.0005 = 0.05%)"
                )
            
            run_btn = st.button("🚀 Run Backtest", type="primary", width='stretch')
            
        else:  # Load CSV mode
            st.header("CSV Settings")
            csv_path = st.text_input("CSV path", value="daily.csv")
            uploaded = st.file_uploader(
                "...or upload CSV",
                type=["csv"],
                accept_multiple_files=False,
                help="Upload a CSV file to view results"
            )
            run_btn = st.button("📂 Load CSV", type="primary", width='stretch')

    if run_btn:
        if view_mode == "Load CSV":
            try:
                if uploaded is not None:
                    daily = pd.read_csv(uploaded, index_col=0, parse_dates=True, encoding="utf-8-sig")
                    st.success(f"✅ Loaded CSV from upload ({uploaded.name})")
                else:
                    path_obj = Path(csv_path)
                    if not path_obj.is_absolute():
                        base = Path.cwd()
                        candidates = [base / csv_path, base / "daily.csv", Path(csv_path).expanduser()]
                        for candidate in candidates:
                            if candidate.exists():
                                csv_path = str(candidate)
                                break
                        else:
                            csv_path = str(path_obj)
                    
                    if not Path(csv_path).exists():
                        st.error(f"❌ CSV file not found: {csv_path}")
                        st.info(f"Current working directory: {Path.cwd()}")
                        csv_files = list(Path.cwd().glob("*.csv"))
                        if csv_files:
                            st.info(f"Available CSV files: {[str(f.name) for f in csv_files]}")
                        return
                    
                    daily = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
                    st.success(f"✅ Loaded CSV from: {csv_path}")
            except Exception as exc:
                st.error(f"❌ CSV load failed: {exc}")
                with st.expander("Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())
                return

            # Display CSV results
            st.subheader("📈 NAV Chart (from CSV)")
            
            if "NAV" in daily.columns:
                nav_series = daily["NAV"]
                chart_title = "Net Asset Value Over Time"
            else:
                st.warning("⚠️ 'NAV' column not found in CSV; showing first numeric column.")
                nav_series = daily.select_dtypes(include=[float, int]).iloc[:, 0]
                chart_title = f"{nav_series.name} Over Time"
            
            # Use Plotly if available, else matplotlib
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=nav_series.index,
                    y=nav_series.values,
                    mode='lines',
                    name='NAV',
                    line=dict(width=2, color='steelblue'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>NAV:</b> $%{y:,.2f}<extra></extra>'
                ))
                fig.update_layout(
                    title=dict(text=chart_title, font=dict(size=16, color='black')),
                    xaxis_title="Date",
                    yaxis_title="NAV ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                nav_series.plot(ax=ax, linewidth=2, color='steelblue')
                ax.set_title(chart_title, fontsize=14, fontweight="bold")
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Value", fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

            st.subheader("📊 Daily Data")
            if AGGrid_AVAILABLE:
                # Configure AgGrid
                gb = GridOptionsBuilder.from_dataframe(daily)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, sortable=True, filterable=True)
                gb.configure_selection('single')
                grid_options = gb.build()
                
                AgGrid(
                    daily,
                    gridOptions=grid_options,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=True,
                    theme='streamlit',
                    height=400,
                )
            else:
                st.dataframe(daily, width='stretch', height=400)
            return

        # Run backtest path
        # Input validation
        errors = []
        
        if not ticker:
            errors.append("❌ Ticker symbol is required")
        
        if start > end:
            errors.append(f"❌ End date ({end}) must be after start date ({start})")
        
        if threshold >= 0:
            errors.append("❌ Threshold must be negative (e.g., -0.041 for -4.1% drop)")
        
        if purchase_mode == "Budget-based":
            if weekly_budget is None or weekly_budget <= 0:
                errors.append("❌ Weekly budget must be positive")
            if mode is None:
                errors.append("❌ Allocation mode must be selected")
        else:  # Shares-based
            if shares_per_signal is None or shares_per_signal <= 0:
                errors.append("❌ Shares per signal must be positive")
        
        if fee_rate < 0:
            errors.append("❌ Fee rate must be non-negative")
        
        if slippage_rate < 0:
            errors.append("❌ Slippage rate must be non-negative")
        
        if errors:
            for error in errors:
                st.error(error)
            return
        
        # Fetch data with spinner
        with st.spinner("📡 Fetching stock data..."):
            try:
                feed = YFinanceFeed()
                prices = feed.get_daily(ticker, start, end)
                st.success(f"✅ Fetched {len(prices)} days of data for {ticker}")
            except Exception as exc:
                st.error(f"❌ Data fetch failed: {exc}")
                with st.expander("Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())
                return
        
        # Run backtest with spinner
        with st.spinner("🔄 Running backtest..."):
            try:
                params = BacktestParams(
                    threshold=float(threshold),
                    weekly_budget=float(weekly_budget) if weekly_budget else None,
                    mode=mode,  # type: ignore[arg-type]
                    carryover=carryover if carryover is not None else None,
                    shares_per_signal=float(shares_per_signal) if shares_per_signal else None,
                    fee_rate=float(fee_rate),
                    slippage_rate=float(slippage_rate),
                )
                
                daily, metrics = run_backtest(prices, params)
                st.success("✅ Backtest completed successfully!")
            except Exception as exc:
                st.error(f"❌ Backtest failed: {exc}")
                with st.expander("Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())
                return
        
        # Save CSV automatically
        csv_filename = f"backtest_{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        csv_path_full = Path.cwd() / csv_filename
        try:
            daily.to_csv(csv_path_full, encoding="utf-8-sig")
            st.success(f"💾 Results saved to: `{csv_filename}`")
            
            # Download button
            with open(csv_path_full, "rb") as f:
                st.download_button(
                    label="📥 Download CSV",
                    data=f.read(),
                    file_name=csv_filename,
                    mime="text/csv",
                )
        except Exception as exc:
            st.warning(f"⚠️ Could not save CSV: {exc}")

        # Display metrics in columns
        st.subheader("📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Invested", f"${metrics['TotalInvested']:,.2f}")
            st.metric("Ending NAV", f"${metrics['EndingNAV']:,.2f}")
            st.metric("Cumulative Return", f"{metrics['CumulativeReturn']*100:.2f}%")
        
        with col2:
            st.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
            st.metric("Maximum Drawdown", f"{metrics['MDD']*100:.2f}%")
            st.metric("XIRR", f"{metrics['XIRR']*100:.2f}%")
        
        with col3:
            st.metric("Total Trades", f"{int(metrics['Trades'])}")
            st.metric("Signal Days", f"{int(metrics['HitDays'])}")
            st.metric("Ending Equity", f"${metrics['EndingEquity']:,.2f}")
        
        # Full metrics JSON (collapsible)
        with st.expander("📋 Full Metrics (JSON)"):
            st.json(json.loads(json.dumps(metrics, default=float)))

        # NAV Chart with multiple views
        st.subheader("📈 NAV Chart")
        
        # Create tabs for different chart views
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 NAV", "💰 Equity vs NAV", "📉 Drawdown"])
        
        with chart_tab1:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily.index,
                    y=daily["NAV"].values,
                    mode='lines',
                    name='NAV',
                    line=dict(width=2, color='steelblue'),
                    fill='tozeroy',
                    fillcolor='rgba(70, 130, 180, 0.1)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>NAV:</b> $%{y:,.2f}<extra></extra>'
                ))
                fig.update_layout(
                    title=dict(text=f"Net Asset Value - {ticker}", font=dict(size=16)),
                    xaxis_title="Date",
                    yaxis_title="NAV ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                daily["NAV"].plot(ax=ax, linewidth=2, color="steelblue")
                ax.set_title(f"Net Asset Value - {ticker}", fontsize=16, fontweight="bold")
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("NAV ($)", fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)
        
        with chart_tab2:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily.index,
                    y=daily["Equity"].values,
                    mode='lines',
                    name='Equity',
                    line=dict(width=2, color='green'),
                    hovertemplate='<b>Equity:</b> $%{y:,.2f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=daily.index,
                    y=daily["NAV"].values,
                    mode='lines',
                    name='NAV',
                    line=dict(width=2, color='steelblue'),
                    hovertemplate='<b>NAV:</b> $%{y:,.2f}<extra></extra>'
                ))
                fig.update_layout(
                    title=dict(text=f"Equity vs NAV - {ticker}", font=dict(size=16)),
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(fig, width='stretch')
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                daily["Equity"].plot(ax=ax, label="Equity", linewidth=2, color="green")
                daily["NAV"].plot(ax=ax, label="NAV", linewidth=2, color="steelblue")
                ax.set_title(f"Equity vs NAV - {ticker}", fontsize=16, fontweight="bold")
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Value ($)", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)
        
        with chart_tab3:
            # Calculate drawdown
            running_max = daily["NAV"].expanding().max()
            drawdown = (daily["NAV"] / running_max - 1.0) * 100
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(width=2, color='red'),
                    hovertemplate='<b>Drawdown:</b> %{y:.2f}%<extra></extra>'
                ))
                fig.update_layout(
                    title=dict(text=f"Drawdown Analysis - {ticker}", font=dict(size=16)),
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    showlegend=False
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, width='stretch')
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                drawdown.plot(ax=ax, linewidth=2, color="red", kind="area", alpha=0.3)
                ax.set_title(f"Drawdown Analysis - {ticker}", fontsize=16, fontweight="bold")
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Drawdown (%)", fontsize=12)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

        # Daily data table
        st.subheader("📋 Daily Data")
        if AGGrid_AVAILABLE:
            # Configure AgGrid with better defaults
            gb = GridOptionsBuilder.from_dataframe(daily)
            gb.configure_pagination(paginationAutoPageSize=True, paginationPageSize=20)
            gb.configure_side_bar()
            gb.configure_default_column(
                groupable=True,
                sortable=True,
                filterable=True,
                resizable=True,
                editable=False
            )
            # Format numeric columns
            for col in daily.select_dtypes(include=[float, int]).columns:
                gb.configure_column(col, type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                                  precision=2)
            gb.configure_selection('single')
            grid_options = gb.build()
            
            grid_response = AgGrid(
                daily,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                allow_unsafe_jscode=True,
                theme='streamlit',
                height=500,
            )
            
            # Show selected row info
            if grid_response['selected_rows']:
                st.info(f"Selected: {grid_response['selected_rows'][0]}")
        else:
            st.dataframe(daily, width='stretch', height=400)


if __name__ == "__main__":  # pragma: no cover
    main()

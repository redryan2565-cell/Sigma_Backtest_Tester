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
try:
    from src.optimization.grid_search import generate_param_space, run_search
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    OPTIMIZATION_ERROR = str(e)
    generate_param_space = None
    run_search = None

try:
    from src.storage.presets import ALL_PRESETS, get_preset_manager
    PRESETS_AVAILABLE = True
except ImportError as e:
    PRESETS_AVAILABLE = False
    PRESETS_ERROR = str(e)
    ALL_PRESETS = {}
    get_preset_manager = None

try:
    from src.visualization.heatmap import create_monthly_returns_heatmap
    HEATMAP_AVAILABLE = True
except ImportError as e:
    HEATMAP_AVAILABLE = False
    HEATMAP_ERROR = str(e)
    create_monthly_returns_heatmap = None


def main() -> None:
    st.set_page_config(page_title="normal-dip-bt", layout="wide", page_icon="📈")
    
    # Navigation menu
    if OPTION_MENU_AVAILABLE:
        selected = option_menu(
            menu_title=None,
            options=["📊 Backtest", "🔍 Optimization", "📁 Load CSV", "ℹ️ About"],
            icons=["graph-up", "search", "folder", "info-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        view_mode = (
            "Run Backtest" if selected == "📊 Backtest"
            else "Optimization" if selected == "🔍 Optimization"
            else "Load CSV" if selected == "📁 Load CSV"
            else "About"
        )
    else:
        st.title("📈 Normal Dip Backtest")
        view_mode = st.radio("Mode", options=["Run Backtest", "Optimization", "Load CSV", "About"], horizontal=True)

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
        - **Threshold**: Daily return threshold as percentage (must be negative, e.g., -4.1 for -4.1%)
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

    # Optimization page
    if view_mode == "Optimization":
        if not OPTIMIZATION_AVAILABLE:
            st.error(f"❌ Optimization module not available: {OPTIMIZATION_ERROR}")
            st.info("Please check that all optimization dependencies are installed.")
            return
        
        st.header("🔍 Parameter Optimization")
        st.info("""
        **Grid Search / Random Search Optimization**
        
        This tool helps find optimal parameter combinations using IS/OS split methodology.
        - **IS (In-Sample)**: Training period for parameter selection
        - **OS (Out-of-Sample)**: Validation period to test robustness
        
        **Constraints**: MDD ≥ -60%, Trades ≥ 15, HitDays ≥ 15
        **Ranking**: CAGR → Sortino → Sharpe → Cumulative Return
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            opt_ticker = st.text_input("Ticker", value="TQQQ", key="opt_ticker").strip().upper()
            search_mode = st.radio("Search Mode", options=["Grid", "Random"], index=0, key="search_mode")
            random_samples = st.number_input("Random Samples", value=100, min_value=10, max_value=1000, step=10, disabled=(search_mode == "Grid"), key="random_samples")
        with col2:
            st.subheader("Date Ranges")
            is_start = st.date_input("IS Start", value=date(2014, 1, 1), key="is_start")
            is_end = st.date_input("IS End", value=date(2022, 12, 31), key="is_end")
            os_start = st.date_input("OS Start", value=date(2023, 1, 1), key="os_start")
            os_end = st.date_input("OS End", value=date.today(), key="os_end")
        
        use_baseline_reset = st.checkbox("Use baseline reset TP/SL", value=True, key="use_baseline_reset")
        shares_per_signal = st.number_input("Shares per Signal", value=10.0, min_value=0.01, step=1.0, key="opt_shares")
        
        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("📡 Fetching data..."):
                try:
                    feed = YFinanceFeed()
                    # Fetch full range
                    full_start = min(is_start, os_start)
                    full_end = max(is_end, os_end)
                    prices = feed.get_daily(opt_ticker, full_start, full_end)
                    st.success(f"✅ Loaded {len(prices)} days of data")
                except Exception as exc:
                    st.error(f"❌ Data fetch failed: {exc}")
                    return
            
            with st.spinner("🔍 Running optimization..."):
                try:
                    base_params = BacktestParams(
                        threshold=-0.04,
                        shares_per_signal=shares_per_signal,
                        fee_rate=0.0005,
                        slippage_rate=0.0005,
                        enable_tp_sl=True,
                        reset_baseline_after_tp_sl=use_baseline_reset,
                    )
                    
                    param_space = generate_param_space(
                        mode=search_mode.lower(),
                        seed=42,
                        budget_n=random_samples,
                        base_params=base_params,
                    )
                    
                    split = {
                        "is": (is_start, is_end),
                        "os": (os_start, os_end),
                    }
                    
                    summary_df, best_params = run_search(param_space, prices, split)
                    
                    if best_params is None:
                        st.warning("⚠️ No valid parameters found (all failed constraints)")
                        return
                    
                    st.success("✅ Optimization completed!")
                    
                    # Show top 10 IS results
                    st.subheader("📊 Top 10 IS Results")
                    top_is = summary_df.nlargest(10, "IS_CAGR")
                    display_cols = ["threshold", "tp_threshold", "sl_threshold", "tp_sell", "sl_sell",
                                   "IS_CAGR", "IS_MDD", "IS_Sortino", "IS_Trades", "IS_HitDays"]
                    st.dataframe(top_is[display_cols], use_container_width=True)
                    
                    # Show best params OS results
                    st.subheader("🎯 Best Parameters - OS Performance")
                    best_row = summary_df[
                        (summary_df["threshold"] == best_params.threshold * 100) &
                        (summary_df["tp_threshold"] == (best_params.tp_threshold * 100 if best_params.tp_threshold else None)) &
                        (summary_df["sl_threshold"] == (best_params.sl_threshold * 100 if best_params.sl_threshold else None))
                    ].iloc[0] if len(summary_df) > 0 else None
                    
                    if best_row is not None:
                        os_cols = ["OS_CAGR", "OS_MDD", "OS_Sortino", "OS_Sharpe", "OS_CumulativeReturn", "OS_Trades", "OS_HitDays"]
                        st.dataframe(best_row[os_cols].to_frame().T, use_container_width=True)
                    
                    # Use Best Params button
                    if st.button("📋 Use Best Parameters"):
                        st.session_state['best_params'] = best_params
                        st.success("✅ Best parameters saved. Go to Backtest tab to use them.")
                        
                except Exception as exc:
                    st.error(f"❌ Optimization failed: {exc}")
                    with st.expander("Technical Details"):
                        import traceback
                        st.code(traceback.format_exc())
        
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
                "Threshold (%)",
                value=None,
                step=0.1,
                format="%0.1f",
                help="Daily return threshold as percentage (must be negative, e.g., -4.1 for -4.1% drop)"
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
                    "Fee Rate (%)",
                    value=None,
                    step=0.01,
                    format="%0.2f",
                    help="Trading fee rate as percentage (e.g., 0.05 for 0.05%)"
                )
            with col4:
                slippage_rate = st.number_input(
                    "Slippage Rate (%)",
                    value=None,
                    step=0.01,
                    format="%0.2f",
                    help="Price slippage assumption as percentage (e.g., 0.05 for 0.05%)"
                )
            
            # Take-Profit / Stop-Loss section
            st.header("Take-Profit / Stop-Loss")
            enable_tp_sl = st.checkbox(
                "Enable TP/SL",
                value=False,
                help="Enable portfolio-level take-profit and stop-loss triggers"
            )
            
            tp_threshold = None
            sl_threshold = None
            tp_sell_percentage = 1.0
            sl_sell_percentage = 1.0
            reset_baseline_after_tp_sl = True
            tp_hysteresis = 0.0
            sl_hysteresis = 0.0
            tp_cooldown_days = 0
            sl_cooldown_days = 0
            
            if enable_tp_sl:
                tp_threshold = st.number_input(
                    "Take-Profit Threshold (%)",
                    value=None,
                    step=1.0,
                    format="%0.1f",
                    help="Trigger take-profit at this gain percentage (e.g., 30 for 30%)"
                )
                sl_threshold = st.number_input(
                    "Stop-Loss Threshold (%)",
                    value=None,
                    step=1.0,
                    format="%0.1f",
                    help="Trigger stop-loss at this loss percentage (e.g., -25 for -25%)"
                )
                tp_sell_percentage = st.selectbox(
                    "TP Sell Percentage",
                    options=[25, 50, 75, 100],
                    index=3,
                    format_func=lambda x: f"{x}%",
                    help="Percentage of shares to sell when Take-Profit triggers. Rounding: 0.5 and above rounds up, minimum 1 share if rounding yields 0."
                ) / 100.0
                sl_sell_percentage = st.selectbox(
                    "SL Sell Percentage",
                    options=[25, 50, 75, 100],
                    index=3,
                    format_func=lambda x: f"{x}%",
                    help="Percentage of shares to sell when Stop-Loss triggers. Rounding: 0.5 and above rounds up, minimum 1 share if rounding yields 0."
                ) / 100.0
                
                # Baseline reset and advanced options
                st.subheader("Baseline Reset Options")
                reset_baseline_after_tp_sl = st.checkbox(
                    "Reset baseline after TP/SL",
                    value=True,
                    help="Reset ROI baseline after TP/SL trigger to prevent consecutive triggers. Recommended: ON."
                )
                
                # Hysteresis/Cooldown Presets
                st.subheader("Hysteresis & Cooldown Presets")
                use_preset = st.checkbox(
                    "Use Preset Values",
                    value=False,
                    help="Apply preset values for hysteresis and cooldown parameters"
                )
                
                preset_type = None
                if use_preset:
                    if PRESETS_AVAILABLE and ALL_PRESETS:
                        preset_type = st.selectbox(
                            "Preset Type",
                            options=list(ALL_PRESETS.keys()),
                            index=1,  # Default to Moderate
                            help="Select a preset configuration for hysteresis and cooldown values"
                        )
                        preset = ALL_PRESETS[preset_type]
                        # Auto-fill preset values (user can still override)
                        preset_tp_hysteresis = preset.tp_hysteresis * 100  # Convert to percentage
                        preset_sl_hysteresis = preset.sl_hysteresis * 100
                        preset_tp_cooldown = preset.tp_cooldown_days
                        preset_sl_cooldown = preset.sl_cooldown_days
                    else:
                        st.warning("⚠️ Presets not available")
                        preset_tp_hysteresis = None
                        preset_sl_hysteresis = None
                        preset_tp_cooldown = None
                        preset_sl_cooldown = None
                else:
                    preset_tp_hysteresis = None
                    preset_sl_hysteresis = None
                    preset_tp_cooldown = None
                    preset_sl_cooldown = None
                
                # Hysteresis options
                st.subheader("Hysteresis (Optional)")
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    tp_hysteresis = st.number_input(
                        "TP Hysteresis (%)",
                        value=preset_tp_hysteresis if use_preset and preset_tp_hysteresis is not None else 0.0,
                        min_value=0.0,
                        step=1.0,
                        format="%0.1f",
                        help="TP hysteresis percentage. After TP triggers, require return to drop below (threshold - hysteresis) before re-arming. Default: 0 (disabled)."
                    )
                with col_h2:
                    sl_hysteresis = st.number_input(
                        "SL Hysteresis (%)",
                        value=preset_sl_hysteresis if use_preset and preset_sl_hysteresis is not None else 0.0,
                        min_value=0.0,
                        step=1.0,
                        format="%0.1f",
                        help="SL hysteresis percentage. After SL triggers, require return to rise above (threshold + hysteresis) before re-arming. Default: 0 (disabled)."
                    )
                
                # Convert percentage to decimal for BacktestParams
                tp_hysteresis = tp_hysteresis / 100.0
                sl_hysteresis = sl_hysteresis / 100.0
                
                # Cooldown options
                st.subheader("Cooldown (Optional)")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    tp_cooldown_days = st.number_input(
                        "TP Cooldown (days)",
                        value=preset_tp_cooldown if use_preset and preset_tp_cooldown is not None else 0,
                        min_value=0,
                        step=1,
                        format="%d",
                        help="Number of days to wait before allowing another TP trigger. Default: 0 (disabled)."
                    )
                with col_c2:
                    sl_cooldown_days = st.number_input(
                        "SL Cooldown (days)",
                        value=preset_sl_cooldown if use_preset and preset_sl_cooldown is not None else 0,
                        min_value=0,
                        step=1,
                        format="%d",
                        help="Number of days to wait before allowing another SL trigger. Default: 0 (disabled)."
                    )
            
            # Presets section
            st.header("Presets")
            if not PRESETS_AVAILABLE or get_preset_manager is None:
                st.warning("⚠️ Presets functionality not available")
                saved_presets = []
            else:
                preset_manager = get_preset_manager()
                saved_presets = preset_manager.list_presets()
            
            # Load preset dropdown
            if saved_presets:
                selected_preset_name = st.selectbox(
                    "Load Preset",
                    options=[""] + saved_presets,
                    index=0,
                    help="Load a saved preset configuration"
                )
                
                if selected_preset_name:
                    if PRESETS_AVAILABLE and get_preset_manager:
                        loaded_params = preset_manager.load(selected_preset_name)
                        if loaded_params:
                            st.info(f"📂 Loaded preset: {selected_preset_name}")
                            st.session_state['loaded_preset'] = loaded_params
                            st.session_state['loaded_preset_name'] = selected_preset_name
                        else:
                            st.error(f"❌ Failed to load preset: {selected_preset_name}")
                    else:
                        st.error("❌ Presets functionality not available")
                
                # Delete preset button
                if st.button("🗑️ Delete Selected Preset", disabled=not selected_preset_name):
                    if PRESETS_AVAILABLE and get_preset_manager:
                        preset_manager = get_preset_manager()
                        if preset_manager.delete(selected_preset_name):
                            st.success(f"✅ Deleted preset: {selected_preset_name}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete preset: {selected_preset_name}")
                    else:
                        st.error("❌ Presets functionality not available")
            else:
                st.info("No saved presets. Save current settings to create one.")
            
            # Save preset
            st.subheader("Save Current Settings")
            new_preset_name = st.text_input(
                "Preset Name",
                value="",
                help="Enter a name for this preset configuration"
            )
            
            if st.button("💾 Save Current Settings", disabled=not new_preset_name):
                try:
                    # Build BacktestParams from current inputs
                    save_params = BacktestParams(
                        threshold=float(threshold) / 100.0 if threshold is not None else 0.0,
                        weekly_budget=float(weekly_budget) if weekly_budget else None,
                        mode=mode,  # type: ignore[arg-type]
                        carryover=carryover,
                        shares_per_signal=float(shares_per_signal) if shares_per_signal else None,
                        fee_rate=float(fee_rate) / 100.0 if fee_rate is not None else 0.0005,
                        slippage_rate=float(slippage_rate) / 100.0 if slippage_rate is not None else 0.0005,
                        enable_tp_sl=enable_tp_sl,
                        tp_threshold=float(tp_threshold) / 100.0 if enable_tp_sl and tp_threshold is not None else None,
                        sl_threshold=float(sl_threshold) / 100.0 if enable_tp_sl and sl_threshold is not None else None,
                        tp_sell_percentage=tp_sell_percentage,
                        sl_sell_percentage=sl_sell_percentage,
                        reset_baseline_after_tp_sl=reset_baseline_after_tp_sl,
                        tp_hysteresis=tp_hysteresis,
                        sl_hysteresis=sl_hysteresis,
                        tp_cooldown_days=tp_cooldown_days,
                        sl_cooldown_days=sl_cooldown_days,
                    )
                    if PRESETS_AVAILABLE and get_preset_manager:
                        preset_manager = get_preset_manager()
                        preset_manager.save(new_preset_name, save_params)
                        st.success(f"✅ Saved preset: {new_preset_name}")
                        st.rerun()
                    else:
                        st.error("❌ Presets functionality not available")
                except Exception as exc:
                    st.error(f"❌ Failed to save preset: {exc}")
            
            # Load preset values into form if preset was selected
            if 'loaded_preset' in st.session_state and st.session_state.get('loaded_preset'):
                loaded_params = st.session_state['loaded_preset']
                st.info(f"💡 Preset '{st.session_state.get('loaded_preset_name', 'Unknown')}' loaded. Fill form fields manually or use values from preset.")
                # Note: Streamlit doesn't easily allow programmatic form updates, so we show info
                # Users need to manually apply values or we implement a more complex state management
            
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
        
        if threshold is None:
            errors.append("❌ Threshold must be provided and negative (e.g., -4.1 for -4.1%)")
        elif threshold >= 0:
            errors.append("❌ Threshold must be negative (e.g., -4.1 for -4.1% drop)")
        
        if purchase_mode == "Budget-based":
            if weekly_budget is None or weekly_budget <= 0:
                errors.append("❌ Weekly budget must be positive")
            if mode is None:
                errors.append("❌ Allocation mode must be selected")
        else:  # Shares-based
            if shares_per_signal is None or shares_per_signal <= 0:
                errors.append("❌ Shares per signal must be positive")
        
        if fee_rate is None:
            errors.append("❌ Fee rate must be provided and non-negative (e.g., 0.05 for 0.05%)")
        elif fee_rate < 0:
            errors.append("❌ Fee rate must be non-negative (e.g., 0.05 for 0.05%)")
        
        if slippage_rate is None:
            errors.append("❌ Slippage rate must be provided and non-negative (e.g., 0.05 for 0.05%)")
        elif slippage_rate < 0:
            errors.append("❌ Slippage rate must be non-negative (e.g., 0.05 for 0.05%)")
        
        # TP/SL validation
        if enable_tp_sl:
            if tp_threshold is None:
                errors.append("❌ Take-profit threshold must be provided when TP/SL is enabled")
            elif tp_threshold <= 0:
                errors.append("❌ Take-profit threshold must be positive (e.g., 30 for 30%)")
            
            if sl_threshold is None:
                errors.append("❌ Stop-loss threshold must be provided when TP/SL is enabled")
            elif sl_threshold >= 0:
                errors.append("❌ Stop-loss threshold must be negative (e.g., -25 for -25%)")
        
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
                    threshold=float(threshold) / 100.0 if threshold is not None else 0.0,
                    weekly_budget=float(weekly_budget) if weekly_budget else None,
                    mode=mode,  # type: ignore[arg-type]
                    carryover=carryover if carryover is not None else None,
                    shares_per_signal=float(shares_per_signal) if shares_per_signal else None,
                    fee_rate=float(fee_rate) / 100.0 if fee_rate is not None else 0.0,
                    slippage_rate=float(slippage_rate) / 100.0 if slippage_rate is not None else 0.0,
                    enable_tp_sl=enable_tp_sl,
                    tp_threshold=float(tp_threshold) / 100.0 if tp_threshold is not None else None,
                    sl_threshold=float(sl_threshold) / 100.0 if sl_threshold is not None else None,
                    tp_sell_percentage=tp_sell_percentage,
                    sl_sell_percentage=sl_sell_percentage,
                    reset_baseline_after_tp_sl=reset_baseline_after_tp_sl,
                    tp_hysteresis=float(tp_hysteresis) / 100.0 if tp_hysteresis is not None else 0.0,
                    sl_hysteresis=float(sl_hysteresis) / 100.0 if sl_hysteresis is not None else 0.0,
                    tp_cooldown_days=int(tp_cooldown_days) if tp_cooldown_days is not None else 0,
                    sl_cooldown_days=int(sl_cooldown_days) if sl_cooldown_days is not None else 0,
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
            st.metric("Profit", f"${metrics.get('Profit', 0.0):,.2f}", help="Profit = Equity + CumCashFlow (net profit/loss)")
            st.metric("NAV (including invested)", f"${metrics.get('NAV_including_invested', 0.0):,.2f}", help="NAV_including_invested = CumInvested + Profit")
        
        with col2:
            st.metric("Cumulative Return", f"{metrics['CumulativeReturn']*100:.2f}%")
            st.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
            st.metric("Maximum Drawdown", f"{metrics['MDD']*100:.2f}%")
            st.metric("XIRR", f"{metrics['XIRR']*100:.2f}%")
        
        with col3:
            st.metric("Total Trades", f"{int(metrics['Trades'])}")
            st.metric("Signal Days", f"{int(metrics['HitDays'])}")
            st.metric("Ending Equity", f"${metrics['EndingEquity']:,.2f}")
        
        # TP/SL metrics (if enabled)
        if enable_tp_sl and "NumTakeProfits" in metrics:
            st.subheader("🎯 Take-Profit / Stop-Loss Metrics")
            tp_col1, tp_col2, tp_col3, tp_col4 = st.columns(4)
            with tp_col1:
                st.metric("Take-Profits", f"{int(metrics['NumTakeProfits'])}")
            with tp_col2:
                st.metric("Stop-Losses", f"{int(metrics['NumStopLosses'])}")
            with tp_col3:
                st.metric("Realized Gain", f"${metrics['TotalRealizedGain']:,.2f}")
            with tp_col4:
                st.metric("Realized Loss", f"${metrics['TotalRealizedLoss']:,.2f}")
            st.metric("Net Realized P/L", f"${metrics['NetRealizedPnl']:,.2f}")
        
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
                
                # Add TP/SL markers if enabled
                if enable_tp_sl and "TP_triggered" in daily.columns:
                    tp_mask = daily["TP_triggered"]
                    sl_mask = daily["SL_triggered"]
                    
                    if tp_mask.any():
                        tp_dates = daily.index[tp_mask]
                        tp_navs = daily.loc[tp_mask, "NAV"]
                        fig.add_trace(go.Scatter(
                            x=tp_dates,
                            y=tp_navs.values,
                            mode='markers',
                            name='Take-Profit',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='green',
                                line=dict(width=1, color='darkgreen')
                            ),
                            hovertemplate='<b>Date:</b> %{x}<br><b>NAV:</b> $%{y:,.2f}<br><b>Type:</b> Take-Profit<extra></extra>'
                        ))
                    
                    if sl_mask.any():
                        sl_dates = daily.index[sl_mask]
                        sl_navs = daily.loc[sl_mask, "NAV"]
                        fig.add_trace(go.Scatter(
                            x=sl_dates,
                            y=sl_navs.values,
                            mode='markers',
                            name='Stop-Loss',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='red',
                                line=dict(width=1, color='darkred')
                            ),
                            hovertemplate='<b>Date:</b> %{x}<br><b>NAV:</b> $%{y:,.2f}<br><b>Type:</b> Stop-Loss<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=dict(text=f"Net Asset Value - {ticker}", font=dict(size=16)),
                    xaxis_title="Date",
                    yaxis_title="NAV ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    showlegend=True if enable_tp_sl and "TP_triggered" in daily.columns and (daily["TP_triggered"].any() or daily["SL_triggered"].any()) else False
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

        # Monthly Returns Heatmap
        st.subheader("📊 Advanced Charts")
        if HEATMAP_AVAILABLE and create_monthly_returns_heatmap:
            heatmap_fig = create_monthly_returns_heatmap(
                daily["NAV"],
                title=f"Monthly Returns Heatmap - {ticker}"
            )
            if heatmap_fig is not None:
                st.plotly_chart(heatmap_fig, width='stretch')
            else:
                st.info("Monthly returns heatmap requires Plotly. Install with: pip install plotly")
        else:
            st.info("Monthly returns heatmap not available. Check that visualization module is installed.")

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

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Optional imports for enhanced features
# Initialize all variables first to prevent NameError in Streamlit Cloud
PLOTLY_AVAILABLE = False
AGGrid_AVAILABLE = False
OPTION_MENU_AVAILABLE = False
option_menu = None

# Import optional packages
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    import matplotlib.pyplot as plt

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGrid_AVAILABLE = True
except ImportError:
    pass

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except (ImportError, Exception):
    # Log error in debug mode, but don't fail
    OPTION_MENU_AVAILABLE = False
    option_menu = None

# Fix import path for Streamlit execution
if Path(__file__).parent.parent.parent.exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtest.engine import BacktestParams, run_backtest
from src.config import get_settings
from src.data.providers.yfin import YFinanceFeed

# Try to import setup_logging (may not be available in all environments)
try:
    from src.config import setup_logging
    _setup_logging_available = True
except ImportError:
    _setup_logging_available = False
    import logging
    # Fallback: basic logging setup
    logging.basicConfig(level=logging.INFO)

# Initialize settings at module level for security configuration
# This ensures settings are loaded once and available throughout the app
try:
    _settings = get_settings()
    # Safe attribute access with fallback defaults
    DEVELOPER_MODE = getattr(_settings, 'developer_mode', False)
    debug_mode = getattr(_settings, 'debug_mode', False)
except Exception as e:
    # Fallback to safe defaults if settings loading fails
    import logging
    logging.warning(f"Failed to load settings: {e}, using defaults")
    DEVELOPER_MODE = False
    debug_mode = False
    _settings = None

# Setup logging (only log errors in production)
if _setup_logging_available and _settings is not None:
    setup_logging(_settings)
else:
    # Fallback logging setup
    import logging
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not debug_mode:
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

import logging

logger = logging.getLogger(__name__)
try:
    from src.optimization.grid_search import (
        generate_leverage_param_space,
        generate_param_space,
        run_search,
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    OPTIMIZATION_ERROR = str(e)
    generate_param_space = None
    run_search = None

try:
    from src.storage.presets import ALL_PRESETS, get_preset_manager, reset_preset_manager, get_universal_preset, UNIVERSAL_PRESETS
    PRESETS_AVAILABLE = True
except ImportError as e:
    PRESETS_AVAILABLE = False
    reset_preset_manager = None
    PRESETS_ERROR = str(e)
    ALL_PRESETS = {}
    get_preset_manager = None
    get_universal_preset = None
    UNIVERSAL_PRESETS = {}

try:
    from src.visualization.heatmap import create_monthly_returns_heatmap
    HEATMAP_AVAILABLE = True
except ImportError as e:
    HEATMAP_AVAILABLE = False
    HEATMAP_ERROR = str(e)
    create_monthly_returns_heatmap = None


# Performance: Cached functions for data fetching and validation
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_price_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch price data with caching."""
    feed = YFinanceFeed()
    return feed.get_daily(ticker, start_date, end_date)


@st.cache_data(ttl=86400)  # Cache for 24 hours (ticker validity doesn't change often)
def validate_ticker_cached(ticker: str) -> bool:
    """Validate ticker with caching."""
    feed = YFinanceFeed()
    return feed.validate_ticker(ticker)


def main() -> None:
    st.set_page_config(page_title="normal-dip-bt", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

    # Settings are already loaded at module level (DEVELOPER_MODE, debug_mode)
    # No need to reload here for security and performance

    # Navigation menu (hide Optimization/Leverage Mode in deployment mode)
    # Ensure all optional imports are available - handle Streamlit Cloud module loading issues
    use_option_menu = False
    option_menu_func = None

    try:
        # Try to use module-level variable first
        if 'OPTION_MENU_AVAILABLE' in globals() and globals().get('OPTION_MENU_AVAILABLE', False):
            use_option_menu = True
            option_menu_func = globals().get('option_menu')
        elif 'option_menu' in globals() and globals().get('option_menu') is not None:
            use_option_menu = True
            option_menu_func = globals().get('option_menu')
    except (NameError, AttributeError, KeyError):
        pass

    # If not available, try to import directly
    if not use_option_menu:
        try:
            from streamlit_option_menu import option_menu as option_menu_func
            use_option_menu = True
        except ImportError:
            use_option_menu = False
            option_menu_func = None

    if use_option_menu and option_menu_func:
        if DEVELOPER_MODE:
            # Developer mode: show all tabs
            options = ["📊 Backtest", "🔍 Optimization", "⚡ Leverage Mode", "ℹ️ About"]
            icons = ["graph-up", "search", "zap", "info-circle"]
        else:
            # Deployment mode: hide Optimization and Leverage Mode
            options = ["📊 Backtest", "ℹ️ About"]
            icons = ["graph-up", "info-circle"]

        selected = option_menu_func(
            menu_title=None,
            options=options,
            icons=icons,
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        view_mode = (
            "Run Backtest" if selected == "📊 Backtest"
            else "Optimization" if selected == "🔍 Optimization"
            else "Leverage Mode" if selected == "⚡ Leverage Mode"
            else "About"
        )
    else:
        st.title("📈 Normal Dip Backtest")
        if DEVELOPER_MODE:
            view_mode = st.radio("Mode", options=["Run Backtest", "Optimization", "Leverage Mode", "About"], horizontal=True)
        else:
            view_mode = st.radio("Mode", options=["Run Backtest", "About"], horizontal=True)

    # AI 생성 앱 경고 문구 (모든 페이지 상단에 고정 표시)
    st.info("⚠️ 이 백테스트 앱은 AI로 만든 앱이여서 오류가 있을수 있습니다. 유의하여주세요.")

    # About page
    if view_mode == "About":
        st.header("📖 About Normal Dip Backtest")
        st.markdown("""
        ### Shares-based Mode
        - Enter number of shares to buy per signal
        - Each signal day will buy the specified number of shares at market price

        ### Parameters
        - **Ticker**: Stock symbol (e.g., TQQQ, AAPL)
        - **Date Range**: Start and end dates for backtest
        - **Threshold**: Daily return threshold as percentage (must be negative, e.g., -4.1 for -4.1%)
        - **Shares per Signal**: Number of shares to buy each time a dip signal occurs
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

    # Optimization page (only available in developer mode)
    if view_mode == "Optimization":
        if not DEVELOPER_MODE:
            st.error("❌ Optimization mode is only available in developer mode.")
            st.info("💡 To enable developer mode, set `DEVELOPER_MODE=true` in your environment variables.")
            return
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

            # Ticker validation for Optimization tab
            opt_ticker_valid = True
            opt_ticker_error_message = None

            if opt_ticker:
                # Initialize validation state in session_state if not present
                if 'ticker_validation_cache' not in st.session_state:
                    st.session_state['ticker_validation_cache'] = {}

                # Check cache first (sanitize ticker for cache key to prevent injection)
                # Only use alphanumeric characters for cache key
                safe_ticker = ''.join(c for c in opt_ticker if c.isalnum() or c in '.-')
                cache_key = f"ticker_valid_{safe_ticker}"
                if cache_key in st.session_state['ticker_validation_cache']:
                    opt_ticker_valid = st.session_state['ticker_validation_cache'][cache_key]
                else:
                    # Validate ticker using cached function
                    try:
                        opt_ticker_valid = validate_ticker_cached(opt_ticker)
                        # Cache the result in session_state for UI state management
                        st.session_state['ticker_validation_cache'][cache_key] = opt_ticker_valid
                    except Exception as exc:
                        # Network error or other exception
                        opt_ticker_valid = False
                        opt_ticker_error_message = f"⚠️ 티커 검증 중 오류가 발생했습니다: {exc}"

                if not opt_ticker_valid:
                    if opt_ticker_error_message:
                        st.warning(opt_ticker_error_message)
                    else:
                        # Sanitize ticker for display (prevent XSS-like issues)
                        safe_display_ticker = opt_ticker[:20]  # Limit length
                        st.error(f"❌ 입력한 Ticker는 존재하지 않거나 지원되지 않습니다. (예: '{safe_display_ticker}')\n실제 존재하는 티커를 입력하세요. 예: 'TQQQ', 'AAPL', 'SPY'")
                else:
                    st.success(f"✅ 티커 '{opt_ticker}'가 유효합니다.")
            else:
                opt_ticker_valid = False

            search_mode = st.radio("Search Mode", options=["Grid", "Random"], index=0, key="search_mode")
            random_samples = st.number_input("Random Samples", value=100, min_value=10, max_value=1000, step=10, disabled=(search_mode == "Grid"), key="random_samples")
        with col2:
            st.subheader("Date Ranges")
            is_start = st.date_input("IS Start", value=date(2014, 1, 1), key="is_start", max_value=date.today())
            is_end = st.date_input("IS End", value=date(2022, 12, 31), key="is_end", max_value=date.today())
            os_start = st.date_input("OS Start", value=date(2023, 1, 1), key="os_start", max_value=date.today())
            os_end = st.date_input("OS End", value=date.today(), key="os_end", max_value=date.today())

            # Validate date ranges (show warning only, don't modify session_state after widget creation)
            if is_start > is_end:
                st.warning(f"⚠️ IS Start date ({is_start}) is after IS End date ({is_end}). Please adjust the dates.")

            if os_start > os_end:
                st.warning(f"⚠️ OS Start date ({os_start}) is after OS End date ({os_end}). Please adjust the dates.")

        use_baseline_reset = st.checkbox("Use baseline reset TP/SL", value=True, key="use_baseline_reset")
        shares_per_signal = st.number_input("Shares per Signal", value=10.0, min_value=0.01, step=1.0, key="opt_shares")

        # Disable button if ticker is invalid
        opt_run_btn = st.button(
            "🚀 Run Optimization",
            type="primary",
            disabled=not opt_ticker_valid if opt_ticker else False
        )

        if opt_run_btn:
            # Validate ticker before proceeding
            if not opt_ticker:
                st.error("❌ Ticker symbol is required")
                return
            if not opt_ticker_valid:
                st.error("❌ 유효하지 않은 티커입니다. 올바른 티커를 입력하세요.")
                return

            with st.spinner("📡 Fetching data..."):
                try:
                    # Fetch full range using cached function
                    full_start = min(is_start, os_start)
                    full_end = max(is_end, os_end)
                    prices = fetch_price_data(opt_ticker, full_start, full_end)
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
                        tp_threshold=0.30,  # Default TP threshold (30%)
                        sl_threshold=-0.20,  # Default SL threshold (-20%)
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

                    summary_df, best_params, constraint_stats = run_search(param_space, prices, split)

                    # Display constraint statistics
                    total = constraint_stats.get("total", 0)
                    passed = constraint_stats.get("passed", 0)
                    failed_mdd = constraint_stats.get("failed_mdd", 0)
                    failed_trades = constraint_stats.get("failed_trades", 0)
                    failed_hitdays = constraint_stats.get("failed_hitdays", 0)

                    # Check if constraints were passed
                    constraints_passed = passed > 0

                    if not constraints_passed:
                        st.warning("⚠️ No valid parameters found (all failed constraints)")
                        st.info(f"""
                        **Constraint Statistics:**
                        - Total parameters tested: {total}
                        - Parameters passed: {passed}
                        - Failed MDD constraint (< -60%): {failed_mdd}
                        - Failed Trades constraint (< 15): {failed_trades}
                        - Failed HitDays constraint (< 15): {failed_hitdays}

                        **Constraints:**
                        - MDD must be >= -60%
                        - Trades must be >= 15
                        - HitDays must be >= 15
                        """)

                        # Show top 10 results anyway
                        if len(summary_df) > 0:
                            st.subheader("📊 Top 10 IS Results (Constraints Failed)")
                            top_is = summary_df.nlargest(10, "IS_CAGR")
                            display_cols = ["threshold", "tp_threshold", "sl_threshold", "tp_sell", "sl_sell",
                                           "IS_CAGR", "IS_MDD", "IS_Sortino", "IS_Trades", "IS_HitDays"]
                            st.dataframe(top_is[display_cols], width='stretch')
                            st.caption("⚠️ These parameters did not pass all constraints but are shown for reference.")
                        return
                    elif passed < total:
                        # Some passed, but show warning
                        st.warning(f"⚠️ Only {passed} out of {total} parameters passed constraints")
                        st.info(f"""
                        **Constraint Statistics:**
                        - Total parameters tested: {total}
                        - Parameters passed: {passed} ({passed/total*100:.1f}%)
                        - Failed MDD constraint (< -60%): {failed_mdd}
                        - Failed Trades constraint (< 15): {failed_trades}
                        - Failed HitDays constraint (< 15): {failed_hitdays}
                        """)

                    st.success("✅ Optimization completed!")

                    # Show top 10 IS results
                    st.subheader("📊 Top 10 IS Results")
                    top_is = summary_df.nlargest(10, "IS_CAGR")
                    display_cols = ["threshold", "tp_threshold", "sl_threshold", "tp_sell", "sl_sell",
                                   "IS_CAGR", "IS_MDD", "IS_Sortino", "IS_Trades", "IS_HitDays"]
                    st.dataframe(top_is[display_cols], width='stretch')

                    # Show best params OS results
                    if best_params is not None:
                        st.subheader("🎯 Best Parameters - OS Performance")
                        best_row = summary_df[
                            (summary_df["threshold"] == best_params.threshold * 100) &
                            (summary_df["tp_threshold"] == (best_params.tp_threshold * 100 if best_params.tp_threshold else None)) &
                            (summary_df["sl_threshold"] == (best_params.sl_threshold * 100 if best_params.sl_threshold else None))
                        ].iloc[0] if len(summary_df) > 0 else None

                        if best_row is not None:
                            os_cols = ["OS_CAGR", "OS_MDD", "OS_Sortino", "OS_Sharpe", "OS_CumulativeReturn", "OS_Trades", "OS_HitDays"]
                            st.dataframe(best_row[os_cols].to_frame().T, width='stretch')

                        # Use Best Params button
                        if st.button("📋 Use Best Parameters"):
                            st.session_state['best_params'] = best_params
                            st.success("✅ Best parameters saved. Go to Backtest tab to use them.")
                    else:
                        st.warning("⚠️ Could not determine best parameters")

                except Exception as exc:
                    st.error(f"❌ Optimization failed: {exc}")
                    # Only show detailed traceback in debug mode
                    if debug_mode:
                        with st.expander("Technical Details"):
                            import traceback
                            # Sanitize error message to prevent information leakage
                            tb_str = traceback.format_exc()
                            # Remove file paths and replace with generic paths
                            import re
                            tb_str = re.sub(r'File "[^"]+[/\\]', 'File "', tb_str)
                            st.code(tb_str)
                    else:
                        st.info("💡 For detailed error information, enable DEBUG_MODE in environment variables.")

        return

    # Leverage Mode page (only available in developer mode)
    if view_mode == "Leverage Mode":
        if not DEVELOPER_MODE:
            st.error("❌ Leverage Mode is only available in developer mode.")
            st.info("💡 To enable developer mode, set `DEVELOPER_MODE=true` in your environment variables.")
            return
        if not OPTIMIZATION_AVAILABLE:
            st.error(f"❌ Optimization module not available: {OPTIMIZATION_ERROR}")
            st.info("Please check that all optimization dependencies are installed.")
            return

        st.header("⚡ Leverage Mode - TP/SL 조합 최적화")
        st.info("""
        **Threshold 고정, TP/SL 조합 집중 탐색**

        Threshold는 사용자가 직접 입력하여 고정하고, Take-Profit/Stop-Loss 비율 조합만 탐색하여 CAGR 최대화를 목표로 합니다.

        **목표 지표**: CAGR 우선순위 → Sortino → Sharpe
        """)

        col1, col2 = st.columns(2)
        with col1:
            lev_ticker = st.text_input("Ticker", value="TQQQ", key="lev_ticker").strip().upper()

            # Ticker validation for Leverage Mode tab
            lev_ticker_valid = True
            lev_ticker_error_message = None

            if lev_ticker:
                if 'ticker_validation_cache' not in st.session_state:
                    st.session_state['ticker_validation_cache'] = {}

                cache_key = f"ticker_valid_{lev_ticker}"
                if cache_key in st.session_state['ticker_validation_cache']:
                    lev_ticker_valid = st.session_state['ticker_validation_cache'][cache_key]
                else:
                    try:
                        # Validate ticker using cached function
                        lev_ticker_valid = validate_ticker_cached(lev_ticker)
                        st.session_state['ticker_validation_cache'][cache_key] = lev_ticker_valid
                    except Exception as exc:
                        lev_ticker_valid = False
                        lev_ticker_error_message = f"⚠️ 티커 검증 중 오류가 발생했습니다: {exc}"

                if not lev_ticker_valid:
                    if lev_ticker_error_message:
                        st.warning(lev_ticker_error_message)
                    else:
                        safe_display_ticker = lev_ticker[:20]
                        st.error(f"❌ 입력한 Ticker는 존재하지 않거나 지원되지 않습니다. (예: '{safe_display_ticker}')\n실제 존재하는 티커를 입력하세요. 예: 'TQQQ', 'AAPL', 'SPY'")
                else:
                    st.success(f"✅ 티커 '{lev_ticker}'가 유효합니다.")
            else:
                lev_ticker_valid = False

            # Threshold input (fixed)
            st.subheader("Threshold (고정값)")
            threshold_input = st.number_input(
                "Threshold (%)",
                value=-4.1,
                step=0.1,
                format="%0.1f",
                help="Dip threshold as percentage (must be negative, e.g., -4.1 for -4.1% drop). This value is fixed for all optimization runs.",
                key="lev_threshold"
            )

            if threshold_input >= 0:
                st.error("❌ Threshold must be negative (e.g., -4.1 for -4.1% drop)")
                threshold_valid = False
            else:
                threshold_valid = True

            search_mode_lev = st.radio("Search Mode", options=["Grid", "Random"], index=0, key="search_mode_lev")
            random_samples_lev = st.number_input(
                "Random Samples",
                value=100,
                min_value=10,
                max_value=1000,
                step=10,
                disabled=(search_mode_lev == "Grid"),
                key="random_samples_lev"
            )

        with col2:
            st.subheader("Date Ranges")
            lev_is_start = st.date_input("IS Start", value=date(2014, 1, 1), key="lev_is_start", max_value=date.today())
            lev_is_end = st.date_input("IS End", value=date(2022, 12, 31), key="lev_is_end", max_value=date.today())
            lev_os_start = st.date_input("OS Start", value=date(2023, 1, 1), key="lev_os_start", max_value=date.today())
            lev_os_end = st.date_input("OS End", value=date.today(), key="lev_os_end", max_value=date.today())

            if lev_is_start > lev_is_end:
                st.warning(f"⚠️ IS Start date ({lev_is_start}) is after IS End date ({lev_is_end}). Please adjust the dates.")

            if lev_os_start > lev_os_end:
                st.warning(f"⚠️ OS Start date ({lev_os_start}) is after OS End date ({lev_os_end}). Please adjust the dates.")

        # TP/SL Range Settings
        st.subheader("TP/SL 탐색 범위 설정")
        col_tp1, col_tp2, col_sl1, col_sl2 = st.columns(4)

        with col_tp1:
            tp_min = st.number_input("TP Threshold Min (%)", value=15.0, min_value=5.0, max_value=100.0, step=5.0, key="lev_tp_min")
            tp_max = st.number_input("TP Threshold Max (%)", value=50.0, min_value=15.0, max_value=100.0, step=5.0, key="lev_tp_max")
            if tp_min >= tp_max:
                st.error("❌ TP Min must be less than TP Max")

        with col_tp2:
            tp_step = st.number_input("TP Step (%)", value=5.0, min_value=1.0, max_value=20.0, step=1.0, key="lev_tp_step", disabled=(search_mode_lev == "Random"))
            tp_sell_options_input = st.multiselect(
                "TP Sell Options (%)",
                options=[25, 50, 75, 100],
                default=[25, 50, 75, 100],
                key="lev_tp_sell_options"
            )
            if not tp_sell_options_input:
                st.warning("⚠️ At least one TP Sell option must be selected")

        with col_sl1:
            sl_min = st.number_input("SL Threshold Min (%)", value=-50.0, min_value=-100.0, max_value=-10.0, step=5.0, key="lev_sl_min")
            sl_max = st.number_input("SL Threshold Max (%)", value=-10.0, min_value=-50.0, max_value=-5.0, step=5.0, key="lev_sl_max")
            if sl_min >= sl_max:
                st.error("❌ SL Min must be less than SL Max (both negative)")

        with col_sl2:
            sl_step = st.number_input("SL Step (%)", value=5.0, min_value=1.0, max_value=20.0, step=1.0, key="lev_sl_step", disabled=(search_mode_lev == "Random"))
            sl_sell_options_input = st.multiselect(
                "SL Sell Options (%)",
                options=[25, 50, 75, 100],
                default=[25, 50, 75, 100],
                key="lev_sl_sell_options"
            )
            if not sl_sell_options_input:
                st.warning("⚠️ At least one SL Sell option must be selected")

        # Other settings
        st.subheader("기타 설정")
        col_other1, col_other2 = st.columns(2)
        with col_other1:
            shares_per_signal_lev = st.number_input("Shares per Signal", value=10.0, min_value=0.01, step=1.0, key="lev_shares")
        with col_other2:
            use_baseline_reset_lev = st.checkbox("Use baseline reset TP/SL", value=True, key="lev_baseline_reset")

        col_fee1, col_fee2 = st.columns(2)
        with col_fee1:
            fee_rate_lev = st.number_input("Fee Rate (%) (운영보수)", value=0.05, min_value=0.0, step=0.01, format="%0.2f", key="lev_fee_rate", help="운영보수: 거래 시 발생하는 수수료 비율")
        with col_fee2:
            slippage_rate_lev = st.number_input("Slippage Rate (%) (괴리율)", value=0.05, min_value=0.0, step=0.01, format="%0.2f", key="lev_slippage_rate", help="괴리율: 주문 체결 시 발생하는 가격 차이 비율")

        # Run button
        can_run = (
            lev_ticker_valid and
            threshold_valid and
            tp_min < tp_max and
            sl_min < sl_max and
            tp_sell_options_input and
            sl_sell_options_input
        )

        lev_run_btn = st.button(
            "🚀 Run Leverage Optimization",
            type="primary",
            disabled=not can_run
        )

        if lev_run_btn:
            if not lev_ticker:
                st.error("❌ Ticker symbol is required")
                return
            if not lev_ticker_valid:
                st.error("❌ 유효하지 않은 티커입니다. 올바른 티커를 입력하세요.")
                return
            if not threshold_valid:
                st.error("❌ Threshold must be negative")
                return

            with st.spinner("📡 Fetching data..."):
                try:
                    # Fetch full range using cached function
                    full_start = min(lev_is_start, lev_os_start)
                    full_end = max(lev_is_end, lev_os_end)
                    prices = fetch_price_data(lev_ticker, full_start, full_end)
                    st.success(f"✅ Loaded {len(prices)} days of data")
                except Exception as exc:
                    st.error(f"❌ Data fetch failed: {exc}")
                    return

            with st.spinner("🔍 Running Leverage Mode optimization..."):
                try:
                    # Convert sell options from percentage to decimal
                    tp_sell_decimal = [x / 100.0 for x in tp_sell_options_input]
                    sl_sell_decimal = [x / 100.0 for x in sl_sell_options_input]

                    # Generate parameter space
                    threshold_decimal = threshold_input / 100.0
                    param_space = generate_leverage_param_space(
                        threshold=threshold_decimal,
                        tp_range=(tp_min, tp_max),
                        sl_range=(sl_min, sl_max),
                        tp_sell_options=tp_sell_decimal,
                        sl_sell_options=sl_sell_decimal,
                        tp_step=tp_step,
                        sl_step=sl_step,
                        mode="grid" if search_mode_lev == "Grid" else "random",
                        budget_n=random_samples_lev if search_mode_lev == "Random" else 100,
                        base_params=BacktestParams(
                            threshold=threshold_decimal,
                            shares_per_signal=shares_per_signal_lev,
                            fee_rate=fee_rate_lev / 100.0,
                            slippage_rate=slippage_rate_lev / 100.0,
                            reset_baseline_after_tp_sl=use_baseline_reset_lev,
                            tp_hysteresis=0.0,
                            sl_hysteresis=0.0,
                            tp_cooldown_days=0,
                            sl_cooldown_days=0,
                        ),
                    )

                    st.info(f"📊 Testing {len(param_space)} parameter combinations...")

                    # Run search
                    split = {
                        "is": (lev_is_start, lev_is_end),
                        "os": (lev_os_start, lev_os_end),
                    }

                    summary_df, best_params, constraint_stats = run_search(
                        param_space=param_space,
                        prices=prices,
                        split=split,
                        save_every=None,  # Don't save intermediate results
                    )

                    if summary_df is None or summary_df.empty:
                        st.warning("⚠️ 조건 내 유효한 결과 없음")
                        return

                    # Display results
                    st.success("✅ Optimization completed!")

                    # Constraint statistics
                    st.subheader("📊 Constraint Statistics")
                    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                    with col_stat1:
                        st.metric("Total", constraint_stats.get("total", 0))
                    with col_stat2:
                        st.metric("Passed", constraint_stats.get("passed", 0))
                    with col_stat3:
                        st.metric("Failed MDD", constraint_stats.get("failed_mdd", 0))
                    with col_stat4:
                        st.metric("Failed Trades", constraint_stats.get("failed_trades", 0))
                    with col_stat5:
                        st.metric("Failed HitDays", constraint_stats.get("failed_hitdays", 0))

                    # Top results table
                    st.subheader("🏆 Top Results (CAGR 기준)")

                    # Sort by CAGR descending
                    top_results = summary_df.nlargest(10, "IS_CAGR")

                    # Format display columns (summary_df has: tp_threshold, sl_threshold, tp_sell, sl_sell, IS_CAGR, etc.)
                    display_cols = [
                        "tp_threshold", "sl_threshold", "tp_sell", "sl_sell",
                        "IS_CAGR", "IS_Sortino", "IS_Sharpe", "IS_MDD"
                    ]

                    # Filter to available columns only
                    available_cols = summary_df.columns.tolist()
                    display_cols = [col for col in display_cols if col in available_cols]

                    if not display_cols:
                        st.warning("⚠️ No display columns found in results")
                        return

                    # Convert to percentage for display
                    display_df = top_results[display_cols].copy()

                    # Format percentage columns (these are already in percentage form from summary_df)
                    if "tp_threshold" in display_df.columns:
                        display_df["tp_threshold"] = display_df["tp_threshold"].round(1).astype(str) + "%"
                    if "sl_threshold" in display_df.columns:
                        display_df["sl_threshold"] = display_df["sl_threshold"].round(1).astype(str) + "%"
                    if "tp_sell" in display_df.columns:
                        display_df["tp_sell"] = display_df["tp_sell"].round(0).astype(int).astype(str) + "%"
                    if "sl_sell" in display_df.columns:
                        display_df["sl_sell"] = display_df["sl_sell"].round(0).astype(int).astype(str) + "%"
                    if "IS_CAGR" in display_df.columns:
                        display_df["IS_CAGR"] = (display_df["IS_CAGR"] * 100).round(2).astype(str) + "%"
                    if "IS_Sortino" in display_df.columns:
                        display_df["IS_Sortino"] = display_df["IS_Sortino"].round(2)
                    if "IS_Sharpe" in display_df.columns:
                        display_df["IS_Sharpe"] = display_df["IS_Sharpe"].round(2)
                    if "IS_MDD" in display_df.columns:
                        display_df["IS_MDD"] = (display_df["IS_MDD"] * 100).round(2).astype(str) + "%"

                    # Rename columns for display
                    display_df = display_df.rename(columns={
                        "tp_threshold": "TP Threshold",
                        "sl_threshold": "SL Threshold",
                        "tp_sell": "TP Sell",
                        "sl_sell": "SL Sell",
                        "IS_CAGR": "CAGR",
                        "IS_Sortino": "Sortino",
                        "IS_Sharpe": "Sharpe",
                        "IS_MDD": "MDD",
                    })

                    st.dataframe(display_df, width='stretch', height=400)

                    # Best parameters
                    if best_params:
                        st.subheader("🎯 Best Parameters")
                        st.json({
                            "threshold": f"{threshold_input}%",
                            "tp_threshold": f"{best_params.tp_threshold * 100:.1f}%",
                            "sl_threshold": f"{best_params.sl_threshold * 100:.1f}%",
                            "tp_sell_percentage": f"{best_params.tp_sell_percentage * 100:.0f}%",
                            "sl_sell_percentage": f"{best_params.sl_sell_percentage * 100:.0f}%",
                        })

                        if st.button("📋 Use Best Parameters"):
                            st.session_state['best_params'] = best_params
                            st.success("✅ Best parameters saved. Go to Backtest tab to use them.")
                    else:
                        st.warning("⚠️ Could not determine best parameters")

                except Exception as exc:
                    st.error(f"❌ Leverage Mode optimization failed: {exc}")
                    # Only show detailed traceback in debug mode
                    if debug_mode:
                        with st.expander("Technical Details"):
                            import traceback
                            # Sanitize error message to prevent information leakage
                            tb_str = traceback.format_exc()
                            # Remove file paths and replace with generic paths
                            import re
                            tb_str = re.sub(r'File "[^"]+[/\\]', 'File "', tb_str)
                            st.code(tb_str)
                    else:
                        st.info("💡 For detailed error information, enable DEBUG_MODE in environment variables.")

        return

    with st.sidebar:
        if view_mode == "Run Backtest":
            st.header("Parameters")

            # Initialize preset variables early (before ticker input)
            loaded_params = None
            universal_preset = None
            if 'universal_preset' in st.session_state and st.session_state.get('universal_preset'):
                universal_preset = st.session_state['universal_preset']
            elif 'loaded_preset' in st.session_state and st.session_state.get('loaded_preset'):
                loaded_params = st.session_state['loaded_preset']

            # Ticker input with validation
            ticker_default = "TQQQ"
            if universal_preset:
                ticker_default = universal_preset.ticker
            elif 'ticker_input' in st.session_state:
                ticker_default = st.session_state['ticker_input']
            
            ticker = st.text_input(
                "Ticker",
                value=ticker_default,
                help="Stock ticker symbol (e.g., TQQQ, AAPL, SPY)",
                key="ticker_input"
            ).strip().upper()

            # Ticker validation
            ticker_valid = True
            ticker_error_message = None

            # Universal preset tickers are always valid
            universal_preset_tickers = ['TQQQ', 'SOXL', 'QLD']
            
            # Skip validation if universal preset is loaded (preset tickers are always valid)
            # Also skip validation for known universal preset tickers
            if universal_preset:
                ticker_valid = True
            elif ticker and ticker.upper() in [t.upper() for t in universal_preset_tickers]:
                # Known universal preset tickers are always valid
                ticker_valid = True
            elif not ticker:
                ticker_valid = False
            else:
                # Initialize validation state in session_state if not present
                if 'ticker_validation_cache' not in st.session_state:
                    st.session_state['ticker_validation_cache'] = {}

                # Check cache first (sanitize ticker for cache key to prevent injection)
                # Only use alphanumeric characters for cache key
                safe_ticker = ''.join(c for c in ticker if c.isalnum() or c in '.-')
                cache_key = f"ticker_valid_{safe_ticker}"
                if cache_key in st.session_state['ticker_validation_cache']:
                    ticker_valid = st.session_state['ticker_validation_cache'][cache_key]
                else:
                    # Validate ticker using cached function
                    try:
                        ticker_valid = validate_ticker_cached(ticker)
                        # Cache the result in session_state for UI state management
                        st.session_state['ticker_validation_cache'][cache_key] = ticker_valid
                    except Exception as exc:
                        # Network error or other exception - don't fail validation on network errors
                        # Just show warning, but allow the ticker to be used
                        ticker_valid = True  # Allow ticker on network errors
                        ticker_error_message = f"⚠️ 티커 검증 중 오류가 발생했습니다. 네트워크 문제일 수 있습니다: {exc}"

                if not ticker_valid:
                    if ticker_error_message:
                        st.warning(ticker_error_message)
                    else:
                        # Sanitize ticker for display (prevent XSS-like issues)
                        safe_display_ticker = ticker[:20]  # Limit length
                        st.error(f"❌ 입력한 Ticker는 존재하지 않거나 지원되지 않습니다. (예: '{safe_display_ticker}')\n실제 존재하는 티커를 입력하세요. 예: 'TQQQ', 'AAPL', 'SPY'")
                else:
                    # Only show success if validation actually ran (not skipped)
                    if ticker and not universal_preset and ticker.upper() not in [t.upper() for t in universal_preset_tickers]:
                        st.success(f"✅ 티커 '{ticker}'가 유효합니다.")

            # Presets section - Quick Presets first, then user presets
            if PRESETS_AVAILABLE and get_preset_manager:
                preset_manager = get_preset_manager()
                saved_presets = preset_manager.list_presets()
                
                # Quick Presets (Universal Presets) - Always shown
                st.subheader("📌 Quick Presets")
                universal_preset_options = list(UNIVERSAL_PRESETS.keys()) if UNIVERSAL_PRESETS else []
                
                if universal_preset_options:
                    # Get current selection from session_state to maintain state after rerun
                    current_selection = st.session_state.get('universal_preset_loaded', "")
                    if current_selection not in universal_preset_options:
                        current_selection = ""
                    
                    # Determine index for radio button
                    radio_index = 0  # Default to "None"
                    if current_selection and current_selection in universal_preset_options:
                        radio_index = universal_preset_options.index(current_selection) + 1
                    
                    selected_universal = st.radio(
                        "Select Quick Preset",
                        options=["None"] + universal_preset_options,
                        index=radio_index,
                        horizontal=True,
                        help="범용 preset을 선택하면 모든 설정이 자동으로 채워집니다",
                        key="universal_preset_selector"
                    )
                    
                    # Handle universal preset selection
                    if selected_universal and selected_universal != "None":
                        # Check if this is a new selection (not already loaded)
                        current_loaded = st.session_state.get('universal_preset_loaded', "")
                        if selected_universal != current_loaded and get_universal_preset:
                            universal_preset = get_universal_preset(selected_universal)
                            if universal_preset:
                                # Clear user preset if universal preset is selected
                                if 'loaded_preset' in st.session_state:
                                    del st.session_state['loaded_preset']
                                if 'loaded_preset_name' in st.session_state:
                                    del st.session_state['loaded_preset_name']
                                
                                # Store universal preset in session_state
                                st.session_state['universal_preset_loaded'] = selected_universal
                                st.session_state['universal_preset'] = universal_preset
                                
                                # Apply preset values immediately by updating session_state
                                st.session_state['start_date_input'] = universal_preset.start_date
                                st.session_state['end_date_input'] = universal_preset.end_date if universal_preset.end_date else date.today()
                                st.session_state['ticker_input'] = universal_preset.ticker
                                
                                # Trigger rerun to apply values
                                st.rerun()
                    elif selected_universal == "None" and 'universal_preset_loaded' in st.session_state and st.session_state['universal_preset_loaded']:
                        # Clear universal preset if "None" is selected
                        del st.session_state['universal_preset_loaded']
                        if 'universal_preset' in st.session_state:
                            del st.session_state['universal_preset']
                        # Reset ticker if it was set by universal preset
                        if 'ticker_input' in st.session_state and st.session_state['ticker_input'] in ['TQQQ', 'SOXL', 'QLD']:
                            st.session_state['ticker_input'] = 'TQQQ'
                        st.rerun()
                    
                    # Show current universal preset status
                    if 'universal_preset_loaded' in st.session_state:
                        preset_name = st.session_state['universal_preset_loaded']
                        st.caption(f"✓ Using: **{preset_name}**")
                else:
                    st.caption("No quick presets available")
                
                st.divider()

                # Load preset UI (프리셋이 있을 때만 표시)
                if saved_presets:
                    st.subheader("📁 My Presets")
                    
                    col_load1, col_load2 = st.columns([3, 1])
                    with col_load1:
                        selected_preset_name = st.selectbox(
                            "Select Preset",
                            options=[""] + saved_presets,
                            index=0,
                            help="세션에 저장된 preset을 불러옵니다",
                            key="preset_loader"
                        )

                    with col_load2:
                        if st.button("Load", disabled=not selected_preset_name, key="load_preset_btn"):
                            loaded_params, loaded_start_date, loaded_end_date = preset_manager.load(selected_preset_name)
                            if loaded_params:
                                # Clear universal preset if user preset is loaded
                                if 'universal_preset_loaded' in st.session_state:
                                    del st.session_state['universal_preset_loaded']
                                if 'universal_preset' in st.session_state:
                                    del st.session_state['universal_preset']
                                
                                st.session_state['loaded_preset'] = loaded_params
                                st.session_state['loaded_preset_name'] = selected_preset_name
                                # Store dates in session_state and update date inputs
                                if loaded_start_date:
                                    st.session_state['loaded_start_date'] = loaded_start_date
                                    st.session_state['start_date_input'] = loaded_start_date
                                if loaded_end_date:
                                    st.session_state['loaded_end_date'] = loaded_end_date
                                    st.session_state['end_date_input'] = loaded_end_date
                                st.success(f"✅ Loaded: {selected_preset_name}")
                                st.rerun()
                            else:
                                st.error("❌ Load failed")

                    # Delete button and status
                    if selected_preset_name:
                        col_del1, col_del2 = st.columns([3, 1])
                        with col_del2:
                            if st.button("🗑️", disabled=not selected_preset_name, key="delete_preset_btn", help="Delete preset"):
                                if preset_manager.delete(selected_preset_name):
                                    st.success(f"✅ Deleted")
                                    st.rerun()

                    # Show loaded preset status (compact)
                    if 'loaded_preset' in st.session_state and st.session_state.get('loaded_preset'):
                        preset_name = st.session_state.get('loaded_preset_name', 'Unknown')
                        st.caption(f"✓ Using: **{preset_name}**")
                        if st.button("Clear", key="clear_preset_btn", use_container_width=True):
                            if 'loaded_preset' in st.session_state:
                                del st.session_state['loaded_preset']
                            if 'loaded_preset_name' in st.session_state:
                                del st.session_state['loaded_preset_name']
                            if 'loaded_start_date' in st.session_state:
                                del st.session_state['loaded_start_date']
                            if 'loaded_end_date' in st.session_state:
                                del st.session_state['loaded_end_date']
                            # Reset dates to defaults
                            st.session_state['start_date_input'] = date(2016, 1, 1)
                            st.session_state['end_date_input'] = date.today()
                            st.rerun()

                    st.divider()

                # Save Current Settings UI (항상 표시 - 프리셋이 없어도 저장 가능)
                st.subheader("💾 Save Current Settings")
                
                new_preset_name = st.text_input(
                    "Preset Name",
                    value="",
                    help="현재 설정을 세션에 저장합니다",
                    key="save_preset_name"
                )

                if st.button("💾 Save Current Settings", disabled=not new_preset_name, key="save_preset_btn"):
                    # This will be handled after all inputs are collected
                    # Set a flag in session_state to trigger save after BacktestParams is created
                    st.session_state['trigger_save_preset'] = True
                    st.session_state['preset_name_to_save'] = new_preset_name

                st.divider()

                # Import/Export JSON File section
                st.subheader("📤 Import/Export JSON File")
                
                # Export (Download) button
                if saved_presets:
                    try:
                        from datetime import datetime
                        all_presets = preset_manager.export_all()
                        json_data = json.dumps(all_presets, indent=2, ensure_ascii=False)
                        st.download_button(
                            "📥 Export",
                            data=json_data,
                            file_name=f"presets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="세션의 모든 preset을 JSON으로 다운로드",
                            key="download_presets_btn"
                        )
                    except AttributeError:
                        # If export_all is not available (old PresetManager), skip download button
                        pass
                else:
                    st.caption("No presets to export")
                
                # Import (Upload) button
                uploaded_file = st.file_uploader(
                    "Import JSON",
                    type=["json"],
                    help="JSON 파일을 업로드하여 preset 불러오기",
                    key="upload_presets_file"
                )
                if uploaded_file is not None:
                    # Initialize processed files tracking in session_state
                    if 'processed_import_files' not in st.session_state:
                        st.session_state['processed_import_files'] = set()
                    
                    # Initialize last imported file tracking
                    if 'last_imported_file_id' not in st.session_state:
                        st.session_state['last_imported_file_id'] = None
                    if 'last_imported_count' not in st.session_state:
                        st.session_state['last_imported_count'] = 0
                    
                    # Create unique file identifier (name + size)
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    
                    # Check if this is the file we just imported (success case)
                    if file_id == st.session_state.get('last_imported_file_id'):
                        # This is the file we just successfully imported, show success message
                        # Note: We can't read the file again after rerun, so use stored count
                        imported_count = st.session_state.get('last_imported_count', 0)
                        if imported_count > 0:
                            st.success(f"✅ Imported {imported_count} preset(s)")
                        else:
                            st.success("✅ Import completed successfully")
                    # Check if this file has already been processed before (duplicate upload)
                    elif file_id in st.session_state['processed_import_files']:
                        # File already processed in a previous session/attempt, show info message
                        st.info("ℹ️ This file has already been imported. Upload a different file to import again.")
                    else:
                        # New file, process it
                        try:
                            presets_data = json.load(uploaded_file)
                            if isinstance(presets_data, dict):
                                # Check if preset_manager has import_all method
                                if hasattr(preset_manager, 'import_all'):
                                    preset_manager.import_all(presets_data)
                                    # Mark this file as processed
                                    st.session_state['processed_import_files'].add(file_id)
                                    # Store as last imported file and count
                                    st.session_state['last_imported_file_id'] = file_id
                                    st.session_state['last_imported_count'] = len(presets_data)
                                    st.success(f"✅ Imported {len(presets_data)} preset(s)")
                                    st.rerun()
                                else:
                                    st.error("❌ Import not supported with current preset manager")
                            else:
                                st.error("❌ Invalid preset file format. Expected a JSON object.")
                        except json.JSONDecodeError as exc:
                            st.error(f"❌ Failed to parse JSON file: {exc}")
                        except Exception as exc:
                            st.error(f"❌ Failed to import presets: {exc}")

                st.divider()

            # Date inputs (with loaded preset dates if available)
            # Initialize session_state for date inputs if not present
            if universal_preset:
                # Universal preset dates are already set in selection handler
                # Just ensure they exist
                if 'start_date_input' not in st.session_state:
                    st.session_state['start_date_input'] = universal_preset.start_date
                if 'end_date_input' not in st.session_state:
                    st.session_state['end_date_input'] = universal_preset.end_date if universal_preset.end_date else date.today()
            else:
                if 'start_date_input' not in st.session_state:
                    st.session_state['start_date_input'] = date(2016, 1, 1)
                if 'end_date_input' not in st.session_state:
                    st.session_state['end_date_input'] = date.today()

            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input(
                    "Start Date",
                    help="Backtest start date (max 10 years from end date)",
                    key="start_date_input",
                    max_value=date.today()
                )
            with col2:
                end = st.date_input(
                    "End Date",
                    help="Backtest end date",
                    key="end_date_input",
                    max_value=date.today()
                )

            # Security: Validate date range (max 10 years)
            max_date_range_days = 365 * 10  # 10 years
            if start and end:
                date_range_days = (end - start).days
                if date_range_days < 0:
                    st.error("❌ Start date must be before end date.")
                elif date_range_days > max_date_range_days:
                    st.error(f"❌ Date range too large: {date_range_days} days. Maximum allowed range is 10 years ({max_date_range_days} days).")
                elif date_range_days == 0:
                    st.warning("⚠️ Start and end dates are the same. No data will be available.")

            # Threshold input
            threshold_value = None
            if universal_preset:
                threshold_value = universal_preset.params.threshold * 100.0 if universal_preset.params.threshold else None
            elif loaded_params:
                threshold_value = loaded_params.threshold * 100.0 if loaded_params.threshold else None

            threshold = st.number_input(
                "Threshold (%)",
                value=threshold_value,
                step=0.1,
                format="%0.1f",
                min_value=-100.0,
                max_value=100.0,
                help="Daily return threshold as percentage (must be negative, e.g., -4.1 for -4.1% drop)"
            )

            # Shares per Signal input (always use shares-based mode)
            shares_per_signal_value = None
            if universal_preset:
                shares_per_signal_value = universal_preset.params.shares_per_signal
            elif loaded_params:
                shares_per_signal_value = loaded_params.shares_per_signal

            shares_per_signal = st.number_input(
                "Shares per Signal",
                value=shares_per_signal_value if shares_per_signal_value is not None else 10.0,
                step=1.0,
                min_value=0.01,
                max_value=1000000.0,
                help="Number of shares to buy each time a signal occurs"
            )

            # Fee and slippage
            fee_rate_value = None
            slippage_rate_value = None
            if universal_preset:
                fee_rate_value = universal_preset.params.fee_rate * 100.0 if universal_preset.params.fee_rate else None
                slippage_rate_value = universal_preset.params.slippage_rate * 100.0 if universal_preset.params.slippage_rate else None
            elif loaded_params:
                fee_rate_value = loaded_params.fee_rate * 100.0 if loaded_params.fee_rate else None
                slippage_rate_value = loaded_params.slippage_rate * 100.0 if loaded_params.slippage_rate else None

            col3, col4 = st.columns(2)
            with col3:
                fee_rate = st.number_input(
                    "Fee Rate (%) (운영보수)",
                    value=fee_rate_value,
                    step=0.01,
                    format="%0.2f",
                    min_value=0.0,
                    max_value=10.0,
                    help="운영보수: 거래 시 발생하는 수수료 비율 (e.g., 0.05 for 0.05%)"
                )
            with col4:
                slippage_rate = st.number_input(
                    "Slippage Rate (%) (괴리율)",
                    value=slippage_rate_value,
                    step=0.01,
                    format="%0.2f",
                    min_value=0.0,
                    max_value=10.0,
                    help="괴리율: 주문 체결 시 발생하는 가격 차이 비율 (e.g., 0.05 for 0.05%)"
                )

            # Take-Profit / Stop-Loss section
            st.header("Take-Profit / Stop-Loss")

            # Initialize values from loaded preset if available
            tp_threshold = None
            sl_threshold = None
            tp_sell_percentage = 1.0
            sl_sell_percentage = 1.0
            reset_baseline_after_tp_sl = True
            tp_hysteresis = 0.0
            sl_hysteresis = 0.0
            tp_cooldown_days = 0
            sl_cooldown_days = 0

            if universal_preset:
                tp_threshold = (universal_preset.params.tp_threshold * 100.0) if universal_preset.params.tp_threshold else None
                sl_threshold = (universal_preset.params.sl_threshold * 100.0) if universal_preset.params.sl_threshold else None
                tp_sell_percentage = universal_preset.params.tp_sell_percentage
                sl_sell_percentage = universal_preset.params.sl_sell_percentage
                reset_baseline_after_tp_sl = universal_preset.params.reset_baseline_after_tp_sl
                tp_hysteresis = universal_preset.params.tp_hysteresis * 100.0
                sl_hysteresis = universal_preset.params.sl_hysteresis * 100.0
                tp_cooldown_days = universal_preset.params.tp_cooldown_days
                sl_cooldown_days = universal_preset.params.sl_cooldown_days
            elif loaded_params:
                tp_threshold = (loaded_params.tp_threshold * 100.0) if loaded_params.tp_threshold else None
                sl_threshold = (loaded_params.sl_threshold * 100.0) if loaded_params.sl_threshold else None
                tp_sell_percentage = loaded_params.tp_sell_percentage
                sl_sell_percentage = loaded_params.sl_sell_percentage
                reset_baseline_after_tp_sl = loaded_params.reset_baseline_after_tp_sl
                tp_hysteresis = loaded_params.tp_hysteresis * 100.0
                sl_hysteresis = loaded_params.sl_hysteresis * 100.0
                tp_cooldown_days = loaded_params.tp_cooldown_days
                sl_cooldown_days = loaded_params.sl_cooldown_days

            # Enable TP/SL independently
            enable_tp = st.checkbox(
                "Enable Take-Profit (TP)",
                value=(tp_threshold is not None),
                disabled=(universal_preset is not None),  # Disable if universal preset is loaded
                help="Enable take-profit trigger (can be used independently from stop-loss)"
            )

            enable_sl = st.checkbox(
                "Enable Stop-Loss (SL)",
                value=(sl_threshold is not None),
                disabled=(universal_preset is not None),  # Disable if universal preset is loaded
                help="Enable stop-loss trigger (can be used independently from take-profit)"
            )

            if enable_tp:
                tp_threshold = st.number_input(
                    "Take-Profit Threshold (%)",
                    value=tp_threshold if tp_threshold is not None else 30.0,
                    step=1.0,
                    format="%0.1f",
                    help="Trigger take-profit at this gain percentage (e.g., 30 for 30%)"
                )
            else:
                tp_threshold = None

            if enable_sl:
                sl_threshold = st.number_input(
                    "Stop-Loss Threshold (%)",
                    value=sl_threshold if sl_threshold is not None else -20.0,
                    step=1.0,
                    format="%0.1f",
                    help="Trigger stop-loss at this loss percentage (e.g., -25 for -25%)"
                )
            else:
                sl_threshold = None

            # Show TP/SL options only if at least one is enabled
            if enable_tp or enable_sl:

                # TP Sell Percentage (only if TP is enabled)
                if enable_tp:
                    tp_sell_percentage_index = 3  # Default to 100%
                    if loaded_params and loaded_params.tp_sell_percentage:
                        tp_pct_100 = int(loaded_params.tp_sell_percentage * 100)
                        if tp_pct_100 in [25, 50, 75, 100]:
                            tp_sell_percentage_index = [25, 50, 75, 100].index(tp_pct_100)

                    tp_sell_percentage = st.selectbox(
                        "TP Sell Percentage",
                        options=[25, 50, 75, 100],
                        index=tp_sell_percentage_index,
                        format_func=lambda x: f"{x}%",
                        help="Percentage of shares to sell when Take-Profit triggers. Rounding: 0.5 and above rounds up, minimum 1 share if rounding yields 0."
                    ) / 100.0
                else:
                    tp_sell_percentage = 1.0  # Default value when TP is disabled

                # SL Sell Percentage (only if SL is enabled)
                if enable_sl:
                    sl_sell_percentage_index = 3  # Default to 100%
                    if loaded_params and loaded_params.sl_sell_percentage:
                        sl_pct_100 = int(loaded_params.sl_sell_percentage * 100)
                        if sl_pct_100 in [25, 50, 75, 100]:
                            sl_sell_percentage_index = [25, 50, 75, 100].index(sl_pct_100)

                    sl_sell_percentage = st.selectbox(
                        "SL Sell Percentage",
                        options=[25, 50, 75, 100],
                        index=sl_sell_percentage_index,
                        format_func=lambda x: f"{x}%",
                        help="Percentage of shares to sell when Stop-Loss triggers. Rounding: 0.5 and above rounds up, minimum 1 share if rounding yields 0."
                    ) / 100.0
                else:
                    sl_sell_percentage = 1.0  # Default value when SL is disabled

                # Baseline reset and advanced options
                st.subheader("Baseline Reset Options")
                reset_baseline_after_tp_sl = st.checkbox(
                    "Reset baseline after TP/SL",
                    value=reset_baseline_after_tp_sl,
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

                # Add expandable info section
                with st.expander("ℹ️ Hysteresis란 무엇인가요?", expanded=False):
                    st.markdown("""
                    **Hysteresis(히스테리시스)**는 TP/SL 트리거 후 즉시 재트리거되는 것을 방지하는 기능입니다.

                    **TP Hysteresis 예시:**
                    - TP 임계값이 +30%이고 Hysteresis가 2.5%인 경우
                    - TP 트리거 후, 수익률이 (30% - 2.5%) = 27.5% 이하로 떨어져야 다시 TP가 활성화됩니다
                    - 이렇게 하면 작은 변동으로 인한 반복적인 TP 트리거를 방지할 수 있습니다

                    **SL Hysteresis 예시:**
                    - SL 임계값이 -20%이고 Hysteresis가 1.5%인 경우
                    - SL 트리거 후, 수익률이 (-20% + 1.5%) = -18.5% 이상으로 올라가야 다시 SL이 활성화됩니다
                    - 이렇게 하면 작은 반등으로 인한 반복적인 SL 트리거를 방지할 수 있습니다
                    """)

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    tp_hysteresis_input_value = preset_tp_hysteresis if use_preset and preset_tp_hysteresis is not None else tp_hysteresis
                    tp_hysteresis = st.number_input(
                        "TP Hysteresis (%)",
                        value=tp_hysteresis_input_value,
                        min_value=0.0,
                        step=0.5,
                        format="%0.1f",
                        help="TP 트리거 후 재활성화를 위해 수익률이 (TP 임계값 - Hysteresis) 이하로 떨어져야 합니다. 예: TP 30%, Hysteresis 2.5% → 27.5% 이하로 떨어져야 재활성화. 기본값: 0 (비활성화)"
                    )
                with col_h2:
                    sl_hysteresis_input_value = preset_sl_hysteresis if use_preset and preset_sl_hysteresis is not None else sl_hysteresis
                    sl_hysteresis = st.number_input(
                        "SL Hysteresis (%)",
                        value=sl_hysteresis_input_value,
                        min_value=0.0,
                        step=0.5,
                        format="%0.1f",
                        help="SL 트리거 후 재활성화를 위해 수익률이 (SL 임계값 + Hysteresis) 이상으로 올라가야 합니다. 예: SL -20%, Hysteresis 1.5% → -18.5% 이상으로 올라가야 재활성화. 기본값: 0 (비활성화)"
                    )

                # Convert percentage to decimal for BacktestParams
                tp_hysteresis = tp_hysteresis / 100.0
                sl_hysteresis = sl_hysteresis / 100.0

                # Cooldown options
                st.subheader("Cooldown (Optional)")

                # Add expandable info section
                with st.expander("ℹ️ Cooldown이란 무엇인가요?", expanded=False):
                    st.markdown("""
                    **Cooldown(쿨다운)**은 TP/SL 트리거 후 일정 기간 동안 같은 종류의 트리거를 비활성화하는 기능입니다.

                    **TP Cooldown 예시:**
                    - TP Cooldown이 3일인 경우
                    - TP 트리거 후 3일 동안은 TP 조건을 만족해도 트리거되지 않습니다
                    - 이렇게 하면 짧은 시간 내 반복적인 TP 트리거를 방지할 수 있습니다

                    **SL Cooldown 예시:**
                    - SL Cooldown이 5일인 경우
                    - SL 트리거 후 5일 동안은 SL 조건을 만족해도 트리거되지 않습니다
                    - 이렇게 하면 급격한 하락 후 즉시 다시 SL이 트리거되는 것을 방지할 수 있습니다

                    **Hysteresis vs Cooldown:**
                    - **Hysteresis**: 수익률 기준으로 재활성화 조건을 설정 (예: 2.5% 더 떨어져야 재활성화)
                    - **Cooldown**: 시간 기준으로 재활성화 조건을 설정 (예: 3일 후에만 재활성화)
                    - 두 기능을 함께 사용하면 더욱 안정적인 트리거 제어가 가능합니다
                    """)

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    tp_cooldown_input_value = preset_tp_cooldown if use_preset and preset_tp_cooldown is not None else tp_cooldown_days
                    tp_cooldown_days = st.number_input(
                        "TP Cooldown (days)",
                        value=tp_cooldown_input_value,
                        min_value=0,
                        step=1,
                        format="%d",
                        help="TP 트리거 후 이 기간 동안은 TP 조건을 만족해도 트리거되지 않습니다. 예: 3일 설정 시 TP 트리거 후 3일간 TP 비활성화. 기본값: 0 (비활성화)"
                    )
                with col_c2:
                    sl_cooldown_input_value = preset_sl_cooldown if use_preset and preset_sl_cooldown is not None else sl_cooldown_days
                    sl_cooldown_days = st.number_input(
                        "SL Cooldown (days)",
                        value=sl_cooldown_input_value,
                        min_value=0,
                        step=1,
                        format="%d",
                        help="SL 트리거 후 이 기간 동안은 SL 조건을 만족해도 트리거되지 않습니다. 예: 5일 설정 시 SL 트리거 후 5일간 SL 비활성화. 기본값: 0 (비활성화)"
                    )

            # Disable button if ticker is invalid
            run_btn = st.button(
                "🚀 Run Backtest",
                type="primary",
                width='stretch',
                disabled=not ticker_valid if ticker else False
            )

            # Handle Save Current Settings button (if clicked)
            # This needs to happen after all inputs are collected but before backtest runs
            if st.session_state.get('trigger_save_preset', False) and 'preset_name_to_save' in st.session_state:
                try:
                    # Build BacktestParams from current inputs
                    save_params = BacktestParams(
                        threshold=float(threshold) / 100.0 if threshold is not None else 0.0,
                        shares_per_signal=float(shares_per_signal) if shares_per_signal else None,
                        fee_rate=float(fee_rate) / 100.0 if fee_rate is not None else 0.0005,
                        slippage_rate=float(slippage_rate) / 100.0 if slippage_rate is not None else 0.0005,
                        enable_tp_sl=(tp_threshold is not None or sl_threshold is not None),  # Auto-set based on thresholds
                        tp_threshold=float(tp_threshold) / 100.0 if tp_threshold is not None else None,
                        sl_threshold=float(sl_threshold) / 100.0 if sl_threshold is not None else None,
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
                        preset_manager.save(
                            st.session_state['preset_name_to_save'],
                            save_params,
                            start_date=start,
                            end_date=end
                        )
                        # SessionPresetManager uses session_state directly, so no cache reset needed
                        st.success(f"✅ Saved preset: {st.session_state['preset_name_to_save']}")
                        # Clear the trigger flags
                        del st.session_state['trigger_save_preset']
                        del st.session_state['preset_name_to_save']
                        st.rerun()
                    else:
                        st.error("❌ Presets functionality not available")
                except Exception as exc:
                    st.error(f"❌ Failed to save preset: {exc}")
                    # Clear the trigger flags even on error
                    if 'trigger_save_preset' in st.session_state:
                        del st.session_state['trigger_save_preset']
                    if 'preset_name_to_save' in st.session_state:
                        del st.session_state['preset_name_to_save']


    if run_btn:
        # Run backtest path
        # Input validation
        errors = []

        if not ticker:
            errors.append("❌ Ticker symbol is required")
        elif not ticker_valid:
            errors.append("❌ 유효하지 않은 티커입니다. 올바른 티커를 입력하세요.")

        if start > end:
            errors.append(f"❌ End date ({end}) must be after start date ({start})")

        if threshold is None:
            errors.append("❌ Threshold must be provided and negative (e.g., -4.1 for -4.1%)")
        elif threshold >= 0:
            errors.append("❌ Threshold must be negative (e.g., -4.1 for -4.1% drop)")

        # Shares per signal validation (always required)
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

        # TP/SL validation (each validated independently)
        if tp_threshold is not None:
            if tp_threshold <= 0:
                errors.append("❌ Take-profit threshold must be positive (e.g., 30 for 30%)")

        if sl_threshold is not None:
            if sl_threshold >= 0:
                errors.append("❌ Stop-loss threshold must be negative (e.g., -25 for -25%)")

        if errors:
            for error in errors:
                st.error(error)
            return

        # Fetch data with spinner (using cached function)
        with st.spinner("📡 Fetching stock data..."):
            try:
                prices = fetch_price_data(ticker, start, end)
                st.success(f"✅ Fetched {len(prices)} days of data for {ticker}")
            except Exception as exc:
                # Sanitize error message: don't expose system details
                error_msg = str(exc)
                import re
                error_msg = re.sub(r'[A-Z]:[\\/][^\\s]+', '[path removed]', error_msg)
                error_msg = re.sub(r'/home/[^\\s]+', '[path removed]', error_msg)
                st.error(f"❌ Data fetch failed: {error_msg}")
                # Only show detailed traceback in debug mode
                if debug_mode:
                    with st.expander("Technical Details"):
                        import traceback
                        # Sanitize error message to prevent information leakage
                        tb_str = traceback.format_exc()
                        # Remove file paths and replace with generic paths
                        tb_str = re.sub(r'File "[^"]+[/\\]', 'File "', tb_str)
                        st.code(tb_str)
                else:
                    st.info("💡 For detailed error information, enable DEBUG_MODE in environment variables.")
                return

        # Run backtest with spinner
        with st.spinner("🔄 Running backtest..."):
            try:
                params = BacktestParams(
                    threshold=float(threshold) / 100.0 if threshold is not None else 0.0,
                    shares_per_signal=float(shares_per_signal) if shares_per_signal else None,
                    fee_rate=float(fee_rate) / 100.0 if fee_rate is not None else 0.0,
                    slippage_rate=float(slippage_rate) / 100.0 if slippage_rate is not None else 0.0,
                    enable_tp_sl=(tp_threshold is not None or sl_threshold is not None),  # Auto-set based on thresholds
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
                # Sanitize error message: don't expose system details
                error_msg = str(exc)
                import re
                error_msg = re.sub(r'[A-Z]:[\\/][^\\s]+', '[path removed]', error_msg)
                error_msg = re.sub(r'/home/[^\\s]+', '[path removed]', error_msg)
                st.error(f"❌ Backtest failed: {error_msg}")
                # Only show detailed traceback in debug mode
                if debug_mode:
                    with st.expander("Technical Details"):
                        import traceback
                        # Sanitize error message to prevent information leakage
                        tb_str = traceback.format_exc()
                        # Remove file paths and replace with generic paths
                        tb_str = re.sub(r'File "[^"]+[/\\]', 'File "', tb_str)
                        st.code(tb_str)
                else:
                    st.info("💡 For detailed error information, enable DEBUG_MODE in environment variables.")
                return

        # Save CSV automatically - use memory buffer for Streamlit Cloud compatibility
        csv_filename = f"backtest_{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        try:
            import io
            csv_buffer = io.StringIO()
            daily.to_csv(csv_buffer, encoding="utf-8-sig")
            csv_data = csv_buffer.getvalue().encode("utf-8-sig")

            st.success(f"💾 Results ready for download: `{csv_filename}`")

            # Download button (always available, no file system dependency)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
            )

            # Try to save to file system if possible (for local development)
            if debug_mode:
                try:
                    csv_path_full = Path.cwd() / csv_filename
                    daily.to_csv(csv_path_full, encoding="utf-8-sig")
                    st.info(f"💾 Also saved to: `{csv_filename}`")
                except Exception:
                    pass  # Ignore file system errors in production
        except Exception as exc:
            # Sanitize error message
            error_msg = str(exc)
            import re
            error_msg = re.sub(r'[A-Z]:[\\/][^\\s]+', '[path removed]', error_msg)
            error_msg = re.sub(r'/home/[^\\s]+', '[path removed]', error_msg)
            st.warning(f"⚠️ Could not prepare CSV download: {error_msg}")

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
            st.metric("Strategy CAGR", f"{metrics['CAGR']*100:.2f}%")
            st.metric("Benchmark CAGR (Buy & Hold)", f"{metrics.get('BenchmarkCAGR', 0.0)*100:.2f}%")
            st.metric("Maximum Drawdown", f"{metrics['MDD']*100:.2f}%")
            st.metric("XIRR", f"{metrics['XIRR']*100:.2f}%")

        with col3:
            st.metric("Total Trades", f"{int(metrics['Trades'])}")
            st.metric("Signal Days", f"{int(metrics['HitDays'])}")
            st.metric("Ending Equity", f"${metrics['EndingEquity']:,.2f}")

        # TP/SL metrics (if enabled)
        if (tp_threshold is not None or sl_threshold is not None) and "NumTakeProfits" in metrics:
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
                if (tp_threshold is not None or sl_threshold is not None) and "TP_triggered" in daily.columns:
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
                    showlegend=True if (tp_threshold is not None or sl_threshold is not None) and "TP_triggered" in daily.columns and (daily["TP_triggered"].any() or daily["SL_triggered"].any()) else False
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

        # Monthly Returns Heatmap (based on Portfolio Return Ratio)
        st.subheader("📊 Advanced Charts")
        if HEATMAP_AVAILABLE and create_monthly_returns_heatmap:
            # Use Portfolio Return Ratio (Equity / PositionCost) instead of NAV
            # This excludes the effect of new investments and shows actual portfolio performance
            if "Equity" in daily.columns and "PositionCost" in daily.columns:
                heatmap_fig = create_monthly_returns_heatmap(
                    daily["Equity"],
                    daily["PositionCost"],
                    title=f"Monthly Returns Heatmap - {ticker} (Portfolio Return Ratio)"
                )
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, width='stretch')
                    st.caption("💡 Based on Portfolio Return Ratio (Equity / PositionCost), excluding new investment effects")
                else:
                    st.info("Monthly returns heatmap requires valid Equity and PositionCost data.")
            else:
                st.info("Monthly returns heatmap requires Equity and PositionCost columns in daily data.")
        else:
            st.info("Monthly returns heatmap not available. Check that visualization module is installed.")

        # Daily data table
        st.subheader("📋 Daily Data")
        
        # CSV download button
        csv_buffer = io.StringIO()
        daily.to_csv(csv_buffer, encoding="utf-8-sig")
        csv_data = csv_buffer.getvalue().encode("utf-8-sig")
        
        st.download_button(
            label="📥 Download Daily Data CSV",
            data=csv_data,
            file_name=f"daily_data_{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        
        # Display table with horizontal scroll
        st.dataframe(daily, use_container_width=True, height=500)


if __name__ == "__main__":  # pragma: no cover
    main()

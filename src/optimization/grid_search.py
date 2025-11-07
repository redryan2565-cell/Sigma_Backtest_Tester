from __future__ import annotations

import itertools
import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestParams, run_backtest
from ..backtest.metrics import sharpe_ratio, sortino_ratio


def generate_param_space(
    mode: str = "grid",
    seed: int | None = None,
    budget_n: int = 100,
    base_params: BacktestParams | None = None,
) -> list[BacktestParams]:
    """Generate parameter space for optimization.
    
    Args:
        mode: "grid" or "random"
        seed: Random seed for random mode
        budget_n: Number of samples for random mode
        base_params: Base parameters to inherit from (allocation mode, fee, etc.)
        
    Returns:
        List of BacktestParams to test
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Default base params if not provided
    if base_params is None:
        base_params = BacktestParams(
            threshold=-0.04,  # Default threshold
            shares_per_signal=10.0,
            fee_rate=0.0005,
            slippage_rate=0.0005,
            enable_tp_sl=True,
            tp_threshold=0.30,  # 30%
            sl_threshold=-0.20,  # -20%
            tp_sell_percentage=0.5,  # 50%
            sl_sell_percentage=1.0,  # 100%
            reset_baseline_after_tp_sl=True,
            tp_hysteresis=0.0,
            sl_hysteresis=0.0,
            tp_cooldown_days=0,
            sl_cooldown_days=0,
        )

    # Parameter ranges
    threshold_values = [-3.0, -3.5, -4.0, -4.5, -5.0]  # As percentages
    tp_threshold_values = [20, 25, 30, 35]  # As percentages
    sl_threshold_values = [-15, -20, -25, -30, -40, -50]  # As percentages
    tp_sell_values = [0.25, 0.50, 0.75, 1.0]
    sl_sell_values = [0.25, 0.50, 0.75, 1.0]

    if mode == "grid":
        # Grid search: all combinations
        param_list = []
        for thresh, tp, sl, tp_sell, sl_sell in itertools.product(
            threshold_values, tp_threshold_values, sl_threshold_values,
            tp_sell_values, sl_sell_values
        ):
            params = BacktestParams(
                threshold=thresh / 100.0,
                shares_per_signal=base_params.shares_per_signal,
                fee_rate=base_params.fee_rate,
                slippage_rate=base_params.slippage_rate,
                enable_tp_sl=True,
                tp_threshold=tp / 100.0,
                sl_threshold=sl / 100.0,
                tp_sell_percentage=tp_sell,
                sl_sell_percentage=sl_sell,
                reset_baseline_after_tp_sl=base_params.reset_baseline_after_tp_sl,
                tp_hysteresis=base_params.tp_hysteresis,
                sl_hysteresis=base_params.sl_hysteresis,
                tp_cooldown_days=base_params.tp_cooldown_days,
                sl_cooldown_days=base_params.sl_cooldown_days,
            )
            param_list.append(params)
        return param_list

    elif mode == "random":
        # Random search: sample randomly
        param_list = []
        for _ in range(budget_n):
            thresh = random.choice(threshold_values)
            tp = random.choice(tp_threshold_values)
            sl = random.choice(sl_threshold_values)
            tp_sell = random.choice(tp_sell_values)
            sl_sell = random.choice(sl_sell_values)

            params = BacktestParams(
                threshold=thresh / 100.0,
                shares_per_signal=base_params.shares_per_signal,
                fee_rate=base_params.fee_rate,
                slippage_rate=base_params.slippage_rate,
                enable_tp_sl=True,
                tp_threshold=tp / 100.0,
                sl_threshold=sl / 100.0,
                tp_sell_percentage=tp_sell,
                sl_sell_percentage=sl_sell,
                reset_baseline_after_tp_sl=base_params.reset_baseline_after_tp_sl,
                tp_hysteresis=base_params.tp_hysteresis,
                sl_hysteresis=base_params.sl_hysteresis,
                tp_cooldown_days=base_params.tp_cooldown_days,
                sl_cooldown_days=base_params.sl_cooldown_days,
            )
            param_list.append(params)
        return param_list

    else:
        raise ValueError(f"Unknown mode: {mode}")


def generate_leverage_param_space(
    threshold: float,
    tp_range: tuple[float, float] = (15.0, 50.0),
    sl_range: tuple[float, float] = (-50.0, -10.0),
    tp_sell_options: list[float] = [0.25, 0.50, 0.75, 1.0],
    sl_sell_options: list[float] = [0.25, 0.50, 0.75, 1.0],
    tp_step: float = 5.0,
    sl_step: float = 5.0,
    mode: str = "grid",
    budget_n: int = 100,
    seed: int | None = None,
    base_params: BacktestParams | None = None,
) -> list[BacktestParams]:
    """Generate parameter space for Leverage Mode optimization.
    
    Threshold is fixed, only TP/SL combinations are explored.
    
    Args:
        threshold: Fixed threshold value (as decimal, e.g., -0.041 for -4.1%)
        tp_range: Tuple of (min, max) TP threshold percentages (e.g., (15.0, 50.0))
        sl_range: Tuple of (min, max) SL threshold percentages (e.g., (-50.0, -10.0))
        tp_sell_options: List of TP sell percentages (e.g., [0.25, 0.50, 0.75, 1.0])
        sl_sell_options: List of SL sell percentages (e.g., [0.25, 0.50, 0.75, 1.0])
        tp_step: Step size for TP threshold in grid mode (as percentage)
        sl_step: Step size for SL threshold in grid mode (as percentage)
        mode: "grid" or "random"
        budget_n: Number of samples for random mode
        seed: Random seed for random mode
        base_params: Base parameters to inherit from (fee, slippage, etc.)
        
    Returns:
        List of BacktestParams with fixed threshold and varying TP/SL combinations
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Default base params if not provided
    if base_params is None:
        base_params = BacktestParams(
            threshold=threshold,  # Use provided threshold
            shares_per_signal=10.0,
            fee_rate=0.0005,
            slippage_rate=0.0005,
            tp_threshold=0.30,
            sl_threshold=-0.20,
            tp_sell_percentage=0.5,
            sl_sell_percentage=1.0,
            reset_baseline_after_tp_sl=True,
            tp_hysteresis=0.0,
            sl_hysteresis=0.0,
            tp_cooldown_days=0,
            sl_cooldown_days=0,
        )

    # Generate TP/SL threshold ranges
    if mode == "grid":
        # Grid search: generate all combinations
        tp_min, tp_max = tp_range
        sl_min, sl_max = sl_range

        # Generate TP threshold values
        tp_values = []
        current = tp_min
        while current <= tp_max:
            tp_values.append(current)
            current += tp_step

        # Generate SL threshold values
        sl_values = []
        current = sl_min
        while current <= sl_max:
            sl_values.append(current)
            current += sl_step

        # Generate all combinations
        param_list = []
        for tp, sl, tp_sell, sl_sell in itertools.product(
            tp_values, sl_values, tp_sell_options, sl_sell_options
        ):
            params = BacktestParams(
                threshold=threshold,  # Fixed threshold
                shares_per_signal=base_params.shares_per_signal,
                fee_rate=base_params.fee_rate,
                slippage_rate=base_params.slippage_rate,
                tp_threshold=tp / 100.0,  # Convert percentage to decimal
                sl_threshold=sl / 100.0,  # Convert percentage to decimal
                tp_sell_percentage=tp_sell,
                sl_sell_percentage=sl_sell,
                reset_baseline_after_tp_sl=base_params.reset_baseline_after_tp_sl,
                tp_hysteresis=base_params.tp_hysteresis,
                sl_hysteresis=base_params.sl_hysteresis,
                tp_cooldown_days=base_params.tp_cooldown_days,
                sl_cooldown_days=base_params.sl_cooldown_days,
            )
            param_list.append(params)
        return param_list

    elif mode == "random":
        # Random search: sample randomly
        tp_min, tp_max = tp_range
        sl_min, sl_max = sl_range

        param_list = []
        for _ in range(budget_n):
            tp = random.uniform(tp_min, tp_max)
            sl = random.uniform(sl_min, sl_max)
            tp_sell = random.choice(tp_sell_options)
            sl_sell = random.choice(sl_sell_options)

            params = BacktestParams(
                threshold=threshold,  # Fixed threshold
                shares_per_signal=base_params.shares_per_signal,
                fee_rate=base_params.fee_rate,
                slippage_rate=base_params.slippage_rate,
                tp_threshold=tp / 100.0,  # Convert percentage to decimal
                sl_threshold=sl / 100.0,  # Convert percentage to decimal
                tp_sell_percentage=tp_sell,
                sl_sell_percentage=sl_sell,
                reset_baseline_after_tp_sl=base_params.reset_baseline_after_tp_sl,
                tp_hysteresis=base_params.tp_hysteresis,
                sl_hysteresis=base_params.sl_hysteresis,
                tp_cooldown_days=base_params.tp_cooldown_days,
                sl_cooldown_days=base_params.sl_cooldown_days,
            )
            param_list.append(params)
        return param_list

    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_params(
    params: BacktestParams,
    prices: pd.DataFrame,
    split: dict[str, tuple[date, date]],
) -> dict[str, dict[str, float]]:
    """Evaluate parameters on IS and OS splits.
    
    Optimized version: filters prices once and avoids unnecessary copies.
    
    Args:
        params: BacktestParams to evaluate
        prices: Full price DataFrame
        split: Dict with "is" and "os" keys, each containing (start_date, end_date)
        
    Returns:
        Dict with "is" and "os" keys, each containing metrics dict
    """
    results = {}

    for split_name, (start_date, end_date) in split.items():
        # Filter prices to split range (use view instead of copy when possible)
        mask = (prices.index.date >= start_date) & (prices.index.date <= end_date)
        split_prices = prices.loc[mask]

        if split_prices.empty:
            results[split_name] = {
                "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
                "CumulativeReturn": 0.0, "Trades": 0.0, "HitDays": 0.0,
                "NumTP": 0.0, "NumSL": 0.0,
            }
            continue

        try:
            ledger, metrics = run_backtest(split_prices, params)

            # Early exit: skip if constraints fail badly (performance optimization)
            trades = metrics.get("Trades", 0.0)
            hit_days = metrics.get("HitDays", 0.0)
            mdd = metrics.get("MDD", 0.0)

            # Quick constraint check - skip detailed calculations if clearly bad
            if trades < 5 or hit_days < 5 or mdd < -0.80:
                results[split_name] = {
                    "CAGR": metrics.get("CAGR", 0.0),
                    "MDD": mdd,
                    "Sharpe": 0.0,
                    "Sortino": 0.0,
                    "CumulativeReturn": metrics.get("CumulativeReturn", 0.0),
                    "Trades": trades,
                    "HitDays": hit_days,
                    "NumTP": 0.0,
                    "NumSL": 0.0,
                }
                continue

            # Calculate Sharpe/Sortino from NAV series
            nav_series = ledger["NAV"]
            sharpe = sharpe_ratio(nav_series)
            sortino = sortino_ratio(nav_series)

            # Get TP/SL counts from ledger if available (optimized)
            num_tp = 0.0
            num_sl = 0.0
            if "TP_triggered" in ledger.columns:
                num_tp = float(ledger["TP_triggered"].sum())
            if "SL_triggered" in ledger.columns:
                num_sl = float(ledger["SL_triggered"].sum())

            results[split_name] = {
                "CAGR": metrics.get("CAGR", 0.0),
                "MDD": mdd,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "CumulativeReturn": metrics.get("CumulativeReturn", 0.0),
                "Trades": trades,
                "HitDays": hit_days,
                "NumTP": num_tp,
                "NumSL": num_sl,
            }
        except Exception:
            # On error, return zeros
            results[split_name] = {
                "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
                "CumulativeReturn": 0.0, "Trades": 0.0, "HitDays": 0.0,
                "NumTP": 0.0, "NumSL": 0.0,
            }

    return results


def evaluate_params_optimized(
    params: BacktestParams,
    split_prices_cache: dict[str, pd.DataFrame],
    split: dict[str, tuple[date, date]],
) -> dict[str, dict[str, float]]:
    """Optimized version of evaluate_params that uses pre-filtered prices.
    
    Args:
        params: BacktestParams to evaluate
        split_prices_cache: Pre-filtered prices for each split
        split: Dict with "is" and "os" keys (for reference)
        
    Returns:
        Dict with "is" and "os" keys, each containing metrics dict
    """
    results = {}

    for split_name in split.keys():
        split_prices = split_prices_cache.get(split_name)

        if split_prices is None or split_prices.empty:
            results[split_name] = {
                "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
                "CumulativeReturn": 0.0, "Trades": 0.0, "HitDays": 0.0,
                "NumTP": 0.0, "NumSL": 0.0,
            }
            continue

        try:
            ledger, metrics = run_backtest(split_prices, params)

            # Early exit: skip if constraints fail badly (performance optimization)
            trades = metrics.get("Trades", 0.0)
            hit_days = metrics.get("HitDays", 0.0)
            mdd = metrics.get("MDD", 0.0)

            # Quick constraint check - skip detailed calculations if clearly bad
            if trades < 5 or hit_days < 5 or mdd < -0.80:
                results[split_name] = {
                    "CAGR": metrics.get("CAGR", 0.0),
                    "MDD": mdd,
                    "Sharpe": 0.0,
                    "Sortino": 0.0,
                    "CumulativeReturn": metrics.get("CumulativeReturn", 0.0),
                    "Trades": trades,
                    "HitDays": hit_days,
                    "NumTP": 0.0,
                    "NumSL": 0.0,
                }
                continue

            # Calculate Sharpe/Sortino from NAV series
            nav_series = ledger["NAV"]
            sharpe = sharpe_ratio(nav_series)
            sortino = sortino_ratio(nav_series)

            # Get TP/SL counts from ledger if available (optimized)
            num_tp = 0.0
            num_sl = 0.0
            if "TP_triggered" in ledger.columns:
                num_tp = float(ledger["TP_triggered"].sum())
            if "SL_triggered" in ledger.columns:
                num_sl = float(ledger["SL_triggered"].sum())

            results[split_name] = {
                "CAGR": metrics.get("CAGR", 0.0),
                "MDD": mdd,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "CumulativeReturn": metrics.get("CumulativeReturn", 0.0),
                "Trades": trades,
                "HitDays": hit_days,
                "NumTP": num_tp,
                "NumSL": num_sl,
            }
        except Exception:
            # On error, return zeros
            results[split_name] = {
                "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
                "CumulativeReturn": 0.0, "Trades": 0.0, "HitDays": 0.0,
                "NumTP": 0.0, "NumSL": 0.0,
            }

    return results


def rank_key(row: dict[str, float]) -> tuple[float, ...]:
    """Ranking function for parameter selection.
    
    Returns tuple for lexicographic ordering:
    - (-1e9,) if constraints fail
    - (CAGR, Sortino, Sharpe, CumulativeReturn) if constraints pass
    """
    mdd = row.get("MDD", 0.0)
    trades = row.get("Trades", 0.0)
    hit_days = row.get("HitDays", 0.0)

    # Check constraints
    if mdd < -0.60:  # MDD worse than -60%
        return (-1e9,)
    if trades < 15 or hit_days < 15:
        return (-1e9,)

    # Return ranking tuple
    return (
        row.get("CAGR", 0.0),
        row.get("Sortino", 0.0),
        row.get("Sharpe", 0.0),
        row.get("CumulativeReturn", 0.0),
    )


def rank_results(is_results: list[dict[str, any]]) -> tuple[BacktestParams | None, dict[str, int]]:
    """Rank IS results and return best parameter with constraint statistics.
    
    Args:
        is_results: List of dicts with "params" and "metrics" keys
        
    Returns:
        Tuple of (best BacktestParams or None, constraint_stats dict)
        constraint_stats contains: total, passed, failed_mdd, failed_trades, failed_hitdays
    """
    if not is_results:
        return None, {"total": 0, "passed": 0, "failed_mdd": 0, "failed_trades": 0, "failed_hitdays": 0}

    # Count constraint failures
    total = len(is_results)
    passed = 0
    failed_mdd = 0
    failed_trades = 0
    failed_hitdays = 0

    # Rank each result
    ranked = []
    for result in is_results:
        metrics = result.get("metrics", {}).get("is", {})
        rank_val = rank_key(metrics)
        ranked.append((rank_val, result["params"], metrics))

        # Count constraint failures (check all constraints regardless of rank_key result)
        mdd = metrics.get("MDD", 0.0)
        trades = metrics.get("Trades", 0.0)
        hit_days = metrics.get("HitDays", 0.0)

        # Check each constraint independently
        if mdd < -0.60:
            failed_mdd += 1
        if trades < 15:
            failed_trades += 1
        if hit_days < 15:
            failed_hitdays += 1

        # Check if all constraints passed
        if not (mdd < -0.60 or trades < 15 or hit_days < 15):
            passed += 1

    # Sort (highest rank first)
    ranked.sort(key=lambda x: x[0], reverse=True)

    # Build constraint statistics
    constraint_stats = {
        "total": total,
        "passed": passed,
        "failed_mdd": failed_mdd,
        "failed_trades": failed_trades,
        "failed_hitdays": failed_hitdays,
    }

    # Return best params (even if constraints failed, return top one)
    if ranked:
        best_params = ranked[0][1]
        # Return None only if all constraints failed
        if ranked[0][0][0] > -1e8:  # Valid result
            return best_params, constraint_stats
        else:
            # Return best params anyway, but with warning stats
            return best_params, constraint_stats

    return None, constraint_stats


def run_search(
    param_space: list[BacktestParams],
    prices: pd.DataFrame,
    split: dict[str, tuple[date, date]],
    output_dir: Path | None = None,
    save_every: int | None = None,
) -> tuple[pd.DataFrame, BacktestParams | None, dict[str, int]]:
    """Run parameter search.
    
    Args:
        param_space: List of BacktestParams to test
        prices: Full price DataFrame
        split: Dict with "is" and "os" keys
        output_dir: Directory to save CSV results
        save_every: Save intermediate results every N samples (None to disable)
        
    Returns:
        Tuple of (summary DataFrame, best_params, constraint_stats)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-filter prices for each split (performance optimization)
    split_prices_cache: dict[str, pd.DataFrame] = {}
    for split_name, (start_date, end_date) in split.items():
        mask = (prices.index.date >= start_date) & (prices.index.date <= end_date)
        split_prices_cache[split_name] = prices.loc[mask]

    # Results storage
    is_results_list = []
    os_results_list = []
    memo_cache: dict[tuple, dict[str, dict[str, float]]] = {}

    # Process each parameter combination
    for i, params in enumerate(param_space):
        # Memoization key (include shares_per_signal for better cache hit rate)
        cache_key = (
            params.threshold,
            params.tp_threshold,
            params.sl_threshold,
            params.tp_sell_percentage,
            params.sl_sell_percentage,
            params.shares_per_signal,
        )

        # Check cache
        if cache_key in memo_cache:
            eval_results = memo_cache[cache_key]
        else:
            try:
                # Use cached split prices
                eval_results = evaluate_params_optimized(params, split_prices_cache, split)
                memo_cache[cache_key] = eval_results
            except Exception:
                # Skip failed evaluations
                continue

        # Store results
        is_results_list.append({
            "params": params,
            "metrics": eval_results.get("is", {}),
        })
        os_results_list.append({
            "params": params,
            "metrics": eval_results.get("os", {}),
        })

        # Intermediate save (only if save_every is set)
        if save_every is not None and (i + 1) % save_every == 0:
            _save_results(is_results_list, os_results_list, output_dir, suffix=f"_partial_{i+1}")

    # Save final results
    _save_results(is_results_list, os_results_list, output_dir)

    # Build summary DataFrame (optimized: use list comprehension)
    summary_data = [
        {
            "threshold": is_res["params"].threshold * 100,
            "tp_threshold": is_res["params"].tp_threshold * 100 if is_res["params"].tp_threshold else None,
            "sl_threshold": is_res["params"].sl_threshold * 100 if is_res["params"].sl_threshold else None,
            "tp_sell": is_res["params"].tp_sell_percentage * 100,
            "sl_sell": is_res["params"].sl_sell_percentage * 100,
            "IS_CAGR": is_res["metrics"].get("CAGR", 0.0),
            "IS_MDD": is_res["metrics"].get("MDD", 0.0),
            "IS_Sharpe": is_res["metrics"].get("Sharpe", 0.0),
            "IS_Sortino": is_res["metrics"].get("Sortino", 0.0),
            "IS_CumulativeReturn": is_res["metrics"].get("CumulativeReturn", 0.0),
            "IS_Trades": is_res["metrics"].get("Trades", 0.0),
            "IS_HitDays": is_res["metrics"].get("HitDays", 0.0),
            "IS_NumTP": is_res["metrics"].get("NumTP", 0.0),
            "IS_NumSL": is_res["metrics"].get("NumSL", 0.0),
            "OS_CAGR": os_res["metrics"].get("CAGR", 0.0),
            "OS_MDD": os_res["metrics"].get("MDD", 0.0),
            "OS_Sharpe": os_res["metrics"].get("Sharpe", 0.0),
            "OS_Sortino": os_res["metrics"].get("Sortino", 0.0),
            "OS_CumulativeReturn": os_res["metrics"].get("CumulativeReturn", 0.0),
            "OS_Trades": os_res["metrics"].get("Trades", 0.0),
            "OS_HitDays": os_res["metrics"].get("HitDays", 0.0),
            "OS_NumTP": os_res["metrics"].get("NumTP", 0.0),
            "OS_NumSL": os_res["metrics"].get("NumSL", 0.0),
        }
        for is_res, os_res in zip(is_results_list, os_results_list)
    ]

    summary_df = pd.DataFrame(summary_data)

    # Find best params and get constraint statistics
    best_params, constraint_stats = rank_results(is_results_list)

    return summary_df, best_params, constraint_stats


def _save_results(
    is_results: list[dict],
    os_results: list[dict],
    output_dir: Path,
    suffix: str = "",
) -> None:
    """Save results to CSV files."""
    # IS results
    is_data = []
    for res in is_results:
        params = res["params"]
        metrics = res["metrics"]
        is_data.append({
            "threshold": params.threshold * 100,
            "tp_threshold": params.tp_threshold * 100 if params.tp_threshold else None,
            "sl_threshold": params.sl_threshold * 100 if params.sl_threshold else None,
            "tp_sell": params.tp_sell_percentage * 100,
            "sl_sell": params.sl_sell_percentage * 100,
            **metrics,
        })

    is_df = pd.DataFrame(is_data)
    is_df.to_csv(output_dir / f"optimization_results_IS{suffix}.csv", index=False)

    # OS results
    os_data = []
    for res in os_results:
        params = res["params"]
        metrics = res["metrics"]
        os_data.append({
            "threshold": params.threshold * 100,
            "tp_threshold": params.tp_threshold * 100 if params.tp_threshold else None,
            "sl_threshold": params.sl_threshold * 100 if params.sl_threshold else None,
            "tp_sell": params.tp_sell_percentage * 100,
            "sl_sell": params.sl_sell_percentage * 100,
            **metrics,
        })

    os_df = pd.DataFrame(os_data)
    os_df.to_csv(output_dir / f"optimization_results_OS{suffix}.csv", index=False)


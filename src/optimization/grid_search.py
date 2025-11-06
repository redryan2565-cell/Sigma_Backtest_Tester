from __future__ import annotations

import csv
import itertools
import random
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestParams, run_backtest
from ..backtest.metrics import sharpe_ratio, sortino_ratio
from ..strategy.dip_buy import AllocationMode


def generate_param_space(
    mode: str = "grid",
    seed: Optional[int] = None,
    budget_n: int = 100,
    base_params: Optional[BacktestParams] = None,
) -> List[BacktestParams]:
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
            tp_threshold=0.30,
            sl_threshold=-0.20,
            tp_sell_percentage=0.5,
            sl_sell_percentage=1.0,
            reset_baseline_after_tp_sl=True,
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


def evaluate_params(
    params: BacktestParams,
    prices: pd.DataFrame,
    split: Dict[str, Tuple[date, date]],
) -> Dict[str, Dict[str, float]]:
    """Evaluate parameters on IS and OS splits.
    
    Args:
        params: BacktestParams to evaluate
        prices: Full price DataFrame
        split: Dict with "is" and "os" keys, each containing (start_date, end_date)
        
    Returns:
        Dict with "is" and "os" keys, each containing metrics dict
    """
    results = {}
    
    for split_name, (start_date, end_date) in split.items():
        # Filter prices to split range
        split_prices = prices.loc[
            (prices.index.date >= start_date) & (prices.index.date <= end_date)
        ].copy()
        
        if split_prices.empty:
            results[split_name] = {
                "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
                "CumulativeReturn": 0.0, "Trades": 0.0, "HitDays": 0.0,
                "NumTP": 0.0, "NumSL": 0.0,
            }
            continue
        
        try:
            ledger, metrics = run_backtest(split_prices, params)
            
            # Calculate Sharpe/Sortino from NAV series
            nav_series = ledger["NAV"]
            sharpe = sharpe_ratio(nav_series)
            sortino = sortino_ratio(nav_series)
            
            # Get TP/SL counts from ledger if available
            num_tp = float(ledger.get("TP_triggered", pd.Series([0])).sum()) if "TP_triggered" in ledger.columns else 0.0
            num_sl = float(ledger.get("SL_triggered", pd.Series([0])).sum()) if "SL_triggered" in ledger.columns else 0.0
            
            results[split_name] = {
                "CAGR": metrics.get("CAGR", 0.0),
                "MDD": metrics.get("MDD", 0.0),
                "Sharpe": sharpe,
                "Sortino": sortino,
                "CumulativeReturn": metrics.get("CumulativeReturn", 0.0),
                "Trades": metrics.get("Trades", 0.0),
                "HitDays": metrics.get("HitDays", 0.0),
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


def rank_key(row: Dict[str, float]) -> Tuple[float, ...]:
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


def rank_results(is_results: List[Dict[str, any]]) -> Optional[BacktestParams]:
    """Rank IS results and return best parameter.
    
    Args:
        is_results: List of dicts with "params" and "metrics" keys
        
    Returns:
        Best BacktestParams, or None if no valid results
    """
    if not is_results:
        return None
    
    # Rank each result
    ranked = []
    for result in is_results:
        metrics = result.get("metrics", {}).get("is", {})
        rank_val = rank_key(metrics)
        ranked.append((rank_val, result["params"]))
    
    # Sort (highest rank first)
    ranked.sort(key=lambda x: x[0], reverse=True)
    
    # Return best params
    if ranked and ranked[0][0][0] > -1e8:  # Valid result
        return ranked[0][1]
    return None


def run_search(
    param_space: List[BacktestParams],
    prices: pd.DataFrame,
    split: Dict[str, Tuple[date, date]],
    output_dir: Optional[Path] = None,
    save_every: int = 25,
) -> Tuple[pd.DataFrame, Optional[BacktestParams]]:
    """Run parameter search.
    
    Args:
        param_space: List of BacktestParams to test
        prices: Full price DataFrame
        split: Dict with "is" and "os" keys
        output_dir: Directory to save CSV results
        save_every: Save intermediate results every N samples
        
    Returns:
        Tuple of (summary DataFrame, best_params)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    is_results_list = []
    os_results_list = []
    memo_cache: Dict[Tuple, Dict[str, Dict[str, float]]] = {}
    
    # Process each parameter combination
    for i, params in enumerate(param_space):
        # Memoization key
        cache_key = (
            params.threshold,
            params.tp_threshold,
            params.sl_threshold,
            params.tp_sell_percentage,
            params.sl_sell_percentage,
        )
        
        # Check cache
        if cache_key in memo_cache:
            eval_results = memo_cache[cache_key]
        else:
            try:
                eval_results = evaluate_params(params, prices, split)
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
        
        # Intermediate save
        if (i + 1) % save_every == 0:
            _save_results(is_results_list, os_results_list, output_dir, suffix=f"_partial_{i+1}")
    
    # Save final results
    _save_results(is_results_list, os_results_list, output_dir)
    
    # Build summary DataFrame
    summary_data = []
    for is_res, os_res in zip(is_results_list, os_results_list):
        params = is_res["params"]
        is_metrics = is_res["metrics"]
        os_metrics = os_res["metrics"]
        
        summary_data.append({
            "threshold": params.threshold * 100,
            "tp_threshold": params.tp_threshold * 100 if params.tp_threshold else None,
            "sl_threshold": params.sl_threshold * 100 if params.sl_threshold else None,
            "tp_sell": params.tp_sell_percentage * 100,
            "sl_sell": params.sl_sell_percentage * 100,
            "IS_CAGR": is_metrics.get("CAGR", 0.0),
            "IS_MDD": is_metrics.get("MDD", 0.0),
            "IS_Sharpe": is_metrics.get("Sharpe", 0.0),
            "IS_Sortino": is_metrics.get("Sortino", 0.0),
            "IS_CumulativeReturn": is_metrics.get("CumulativeReturn", 0.0),
            "IS_Trades": is_metrics.get("Trades", 0.0),
            "IS_HitDays": is_metrics.get("HitDays", 0.0),
            "IS_NumTP": is_metrics.get("NumTP", 0.0),
            "IS_NumSL": is_metrics.get("NumSL", 0.0),
            "OS_CAGR": os_metrics.get("CAGR", 0.0),
            "OS_MDD": os_metrics.get("MDD", 0.0),
            "OS_Sharpe": os_metrics.get("Sharpe", 0.0),
            "OS_Sortino": os_metrics.get("Sortino", 0.0),
            "OS_CumulativeReturn": os_metrics.get("CumulativeReturn", 0.0),
            "OS_Trades": os_metrics.get("Trades", 0.0),
            "OS_HitDays": os_metrics.get("HitDays", 0.0),
            "OS_NumTP": os_metrics.get("NumTP", 0.0),
            "OS_NumSL": os_metrics.get("NumSL", 0.0),
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Find best params
    best_params = rank_results(is_results_list)
    
    return summary_df, best_params


def _save_results(
    is_results: List[Dict],
    os_results: List[Dict],
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


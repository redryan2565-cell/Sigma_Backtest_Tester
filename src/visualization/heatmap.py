from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_monthly_returns_heatmap(
    equity_series: pd.Series,
    position_cost_series: pd.Series,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure | None:
    """Create a monthly returns heatmap from Portfolio Return Ratio (Equity / PositionCost).

    This uses Portfolio Return Ratio instead of NAV to exclude the effect of new investments,
    showing only the actual portfolio performance.

    Args:
        equity_series: Series of Equity values with DatetimeIndex.
        position_cost_series: Series of PositionCost values with DatetimeIndex.
        title: Chart title.

    Returns:
        Plotly figure if Plotly is available, None otherwise.
    """
    if not PLOTLY_AVAILABLE:
        return None

    if equity_series.empty or position_cost_series.empty:
        return None

    # Ensure indices are DatetimeIndex and aligned
    equity_series.index = pd.to_datetime(equity_series.index)
    position_cost_series.index = pd.to_datetime(position_cost_series.index)

    # Align series (use intersection of dates)
    common_index = equity_series.index.intersection(position_cost_series.index)
    if len(common_index) < 2:
        return None

    equity_aligned = equity_series.loc[common_index]
    position_cost_aligned = position_cost_series.loc[common_index]

    # Calculate Portfolio Return Ratio: equity / position_cost
    # This ratio represents the value multiplier relative to cost basis
    portfolio_ratio = pd.Series(np.nan, index=common_index)
    valid_mask = (position_cost_aligned > 1e-6) & (equity_aligned > 1e-9)
    portfolio_ratio.loc[valid_mask] = equity_aligned.loc[valid_mask] / position_cost_aligned.loc[valid_mask]

    # Remove NaN values (no position)
    portfolio_ratio_valid = portfolio_ratio.dropna()

    if len(portfolio_ratio_valid) < 2:
        return None

    # Calculate monthly returns using first and last Portfolio Return Ratio of each month
    monthly_returns_list = []
    monthly_dates = []

    # Group by year-month
    portfolio_ratio_grouped = portfolio_ratio_valid.groupby([portfolio_ratio_valid.index.year, portfolio_ratio_valid.index.month])

    for (year, month), group in portfolio_ratio_grouped:
        if len(group) < 1:
            continue

        # Filter valid ratios (finite and > 1e-9)
        valid_mask = (group > 1e-9) & np.isfinite(group)
        valid_group = group[valid_mask]
        
        if len(valid_group) < 1:
            continue

        # Get first and last valid ratio of the month
        # This handles cases where the first day of the month has no position
        first_ratio = valid_group.iloc[0]
        last_ratio = valid_group.iloc[-1]

        # Calculate monthly return: (last_ratio / first_ratio) - 1
        # This shows the change in portfolio value multiplier during the month
        monthly_return = (last_ratio / first_ratio) - 1.0

        # Filter out infinity and NaN values
        if np.isfinite(monthly_return):
            monthly_returns_list.append(monthly_return)
            # Use the last valid day of the month as the index (safely handle day)
            last_valid_day = valid_group.index[-1]
            last_day = min(last_valid_day.day, 28)  # Cap at 28 to avoid month-end issues
            try:
                monthly_dates.append(pd.Timestamp(year=year, month=month, day=last_day))
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                # Fallback: use first valid day of next month minus 1 day
                try:
                    next_month = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=1)
                    monthly_dates.append(next_month - pd.Timedelta(days=1))
                except Exception:
                    # Final fallback: use last valid day of group
                    monthly_dates.append(last_valid_day)

    if not monthly_returns_list:
        return None

    # Create monthly returns series
    monthly_returns = pd.Series(monthly_returns_list, index=pd.DatetimeIndex(monthly_dates))

    # Extract year and month
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values,
    })

    # Create pivot table
    pivot_table = monthly_returns_df.pivot_table(
        index='Year',
        columns='Month',
        values='Return',
        aggfunc='sum',
    )

    # Fill missing months with NaN
    pivot_table = pivot_table.reindex(
        columns=range(1, 13),
        fill_value=np.nan
    )

    # Replace infinity and NaN with NaN for proper display
    z_values = pivot_table.values * 100  # Convert to percentage
    z_values = np.where(np.isfinite(z_values), z_values, np.nan)

    # Create text for display (handle NaN and infinity)
    text_values = np.round(z_values, 2)
    text_values = np.where(np.isfinite(text_values), text_values, np.nan)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot_table.index.astype(str),
        colorscale='RdYlGn',
        colorbar=dict(title="Return (%)"),
        text=text_values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Month",
        yaxis_title="Year",
        height=500,
        template='plotly_white',
    )

    return fig


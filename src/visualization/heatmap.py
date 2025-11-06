from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_monthly_returns_heatmap(
    nav_series: pd.Series,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure | None:
    """Create a monthly returns heatmap from NAV series.
    
    Args:
        nav_series: Series of NAV values with DatetimeIndex.
        title: Chart title.
        
    Returns:
        Plotly figure if Plotly is available, None otherwise.
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if nav_series.empty or len(nav_series) < 2:
        return None
    
    # Calculate daily returns
    daily_returns = nav_series.pct_change().dropna()
    
    # Convert to monthly returns
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Extract year and month
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
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
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values * 100,  # Convert to percentage
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot_table.index.astype(str),
        colorscale='RdYlGn',
        colorbar=dict(title="Return (%)"),
        text=np.round(pivot_table.values * 100, 2),
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


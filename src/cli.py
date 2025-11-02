from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import typer

from .backtest.engine import BacktestParams, run_backtest
from .data.yfin import YFinanceFeed


app = typer.Typer(help="normal-dip-bt CLI")


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


@app.command("run")
def run(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g., TQQQ"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    threshold: float = typer.Option(-0.041, help="Dip threshold (daily return)"),
    weekly_budget: float = typer.Option(500.0, help="Weekly budget in currency"),
    mode: str = typer.Option("split", help="Allocation mode: split | first_hit"),
    carryover: bool = typer.Option(True, help="Carry unused weekly budget"),
    fee_rate: float = typer.Option(0.0005, help="Fee rate per trade"),
    slippage_rate: float = typer.Option(0.0005, help="Slippage rate applied to price"),
    plot: bool = typer.Option(False, help="Show NAV plot"),
    export_csv: Path | None = typer.Option(None, help="Export daily results to CSV"),
    export_chart: Path | None = typer.Option(None, help="Save NAV chart to file (e.g., nav.png)"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    _setup_logging(debug)
    # parse dates
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    if end_d < start_d:
        raise typer.BadParameter("end must be >= start")

    feed = YFinanceFeed()
    try:
        prices = feed.get_daily(ticker, start_d, end_d)
    except Exception as exc:
        typer.secho(f"Failed to fetch data: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    params = BacktestParams(
        threshold=float(threshold),
        weekly_budget=float(weekly_budget),
        mode="split" if mode == "split" else "first_hit",
        carryover=bool(carryover),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
    )

    daily, metrics = run_backtest(prices, params)

    typer.echo(json.dumps(metrics, indent=2, default=float))

    if plot:
        ax = daily["NAV"].plot(title=f"NAV - {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        plt.tight_layout()
        if export_chart:
            plt.savefig(export_chart, dpi=150)
        plt.show()

    if export_csv:
        try:
            daily.to_csv(export_csv, encoding="utf-8-sig")
            typer.secho(f"Saved daily results to {export_csv}", fg=typer.colors.GREEN)
        except Exception as exc:  # pragma: no cover - filesystem issues
            typer.secho(f"Failed to save CSV: {exc}", fg=typer.colors.RED)


if __name__ == "__main__":  # pragma: no cover
    app()



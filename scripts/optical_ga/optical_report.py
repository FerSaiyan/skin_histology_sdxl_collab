#!/usr/bin/env python
"""
Lightweight reporting for optical GA optimisation runs.

Reads ga_history.csv and produces a summary JSON with convergence
analysis, best genome / Lab, and parameter statistics.

Usage:
    python scripts/optical_ga/optical_report.py \\
        --history outputs/optical_ga/ga_history.csv \\
        --output-dir outputs/optical_ga
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _compute_convergence_slope(
    df: pd.DataFrame,
    col: str = "best_fitness",
) -> Dict[str, Any]:
    """Simple linear regression on the fitness trace."""
    if len(df) < 2:
        return {
            "slope": 0.0,
            "intercept": float(df[col].iloc[0]) if len(df) == 1 else 0.0,
            "r_squared": 0.0,
        }

    x = df["generation"].values.astype(float)
    y = df[col].values.astype(float)

    n = len(x)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return {"slope": 0.0, "intercept": float(y.mean()), "r_squared": 0.0}

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_sq)}


def _summarise_column(col: pd.Series) -> Dict[str, float]:
    """Return basic stats for a numeric column."""
    vals = col.dropna().values.astype(float)
    if len(vals) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
    }


def generate_report(
    history_csv: Path,
    config_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read GA history CSV and produce a summary report dict."""
    if not history_csv.exists():
        return {"error": f"History file not found: {history_csv}"}

    df = pd.read_csv(history_csv)

    if len(df) == 0:
        return {"error": "History CSV is empty."}

    # Identify numeric columns
    num_cols = [
        "best_fitness", "mean_fitness", "worst_fitness",
        "median_fitness", "std_fitness", "diversity",
    ]
    lab_cols = ["best_L", "best_a", "best_b"]

    summary: Dict[str, Any] = {
        "num_generations": len(df),
        "first_generation": int(df["generation"].iloc[0]) if "generation" in df.columns else 0,
        "last_generation": int(df["generation"].iloc[-1]) if "generation" in df.columns else 0,
        "best_fitness_overall": float(df["best_fitness"].max()),
        "best_fitness_final": float(df["best_fitness"].iloc[-1]),
        "worst_fitness_overall": float(df["worst_fitness"].min()),
        "mean_fitness_final": float(df["mean_fitness"].iloc[-1]),
        "best_lab": {},
    }

    # Extract best Lab from last generation
    if "best_L" in df.columns:
        last = df.iloc[-1]
        summary["best_lab"] = {
            "L": float(last.get("best_L", 0)),
            "a": float(last.get("best_a", 0)),
            "b": float(last.get("best_b", 0)),
        }

    # Column summaries
    for col in num_cols + lab_cols:
        if col in df.columns:
            summary[f"{col}_stats"] = _summarise_column(df[col])

    # Convergence slope for best_fitness
    slope_info = _compute_convergence_slope(df, "best_fitness")
    summary["convergence"] = {
        "best_fitness_slope": slope_info["slope"],
        "best_fitness_r_squared": slope_info["r_squared"],
        "interpretation": (
            "improving" if slope_info["slope"] > 0.01
            else "converged" if abs(slope_info["slope"]) <= 0.01
            else "degrading"
        ),
    }

    # Diversity trend
    if "diversity" in df.columns:
        div_slope = _compute_convergence_slope(df, "diversity")
        summary["diversity_trend"] = {
            "slope": div_slope["slope"],
            "initial": float(df["diversity"].iloc[0]),
            "final": float(df["diversity"].iloc[-1]),
        }

    # Load config if available
    if config_json and config_json.exists():
        try:
            summary["config"] = json.loads(config_json.read_text(encoding="utf-8"))
        except Exception as exc:
            summary["config_warning"] = str(exc)

    return summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Optical GA report generator."
    )
    ap.add_argument("--history", required=True, help="Path to ga_history.csv")
    ap.add_argument("--output-dir", required=True, help="Output directory for the report")
    ap.add_argument("--config", default=None, help="Optional path to ga_config.json")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    history_path = Path(args.history)

    if not history_path.exists():
        print(f"ERROR: history file not found: {history_path}", file=sys.stderr)
        sys.exit(1)

    config_path: Optional[Path] = None
    if args.config:
        config_path = Path(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = generate_report(history_csv=history_path, config_json=config_path)

    report_path = output_dir / "optical_ga_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {report_path}")

    # Print summary
    print(f"\nOptical GA Report Summary:")
    print(f"  Generations: {report.get('num_generations', '?')}")
    print(f"  Best fitness (overall): {report.get('best_fitness_overall', '?'):.4f}")
    print(f"  Best fitness (final):   {report.get('best_fitness_final', '?'):.4f}")
    print(f"  Best Lab: {report.get('best_lab', {})}")


if __name__ == "__main__":
    main()

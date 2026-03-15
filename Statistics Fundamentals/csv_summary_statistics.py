"""
Statistics Fundamentals: CSV Summary Statistics
================================================
Reads a CSV file and outputs descriptive summary statistics for every column.

Usage
-----
    python csv_summary_statistics.py [path/to/file.csv]

If no path is supplied the bundled `sample_data.csv` is used.

For each **numeric** column the script reports:
    count, missing, mean, median, std dev, min, 25th / 50th / 75th percentile, max

For each **categorical** column the script reports:
    count, missing, unique values, most-common value (mode)
"""

import csv
import math
import os
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Helper statistics functions (no third-party libraries required)
# ---------------------------------------------------------------------------

def _mean(values):
    return sum(values) / len(values)


def _variance(values):
    """Sample variance (Bessel's correction, divides by n-1)."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def _std(values):
    return math.sqrt(_variance(values))


def _sorted_values(values):
    return sorted(values)


def _median(values):
    s = _sorted_values(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2


def _percentile(values, p):
    """Return the p-th percentile (0–100) using linear interpolation."""
    s = _sorted_values(values)
    n = len(s)
    if n == 1:
        return s[0]
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return s[-1]
    frac = index - lower
    return s[lower] + frac * (s[upper] - s[lower])


def _mode(values):
    """Return the most frequent value (first one encountered on a tie)."""
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def read_csv(filepath):
    """Return (headers, rows) where rows is a list of dicts keyed by header."""
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


# ---------------------------------------------------------------------------
# Column classification and value extraction
# ---------------------------------------------------------------------------

def _try_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_columns(headers, rows):
    """Split columns into numeric and categorical, collecting their values."""
    numeric = {}
    categorical = {}

    for header in headers:
        raw_values = [row[header] for row in rows]
        numeric_values = [_try_float(v) for v in raw_values]
        valid_numeric = [v for v in numeric_values if v is not None]
        missing_count = sum(1 for v in raw_values if v is None or v.strip() == "")

        # Treat as numeric if every non-empty value parses as a float
        non_empty = [v for v in raw_values if v is not None and v.strip() != ""]
        if non_empty and len(valid_numeric) == len(non_empty):
            numeric[header] = {
                "values": valid_numeric,
                "missing": missing_count,
                "total": len(raw_values),
            }
        else:
            categorical[header] = {
                "values": non_empty,
                "missing": missing_count,
                "total": len(raw_values),
            }

    return numeric, categorical


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def numeric_summary(name, info):
    vals = info["values"]
    n = len(vals)
    if n == 0:
        return {
            "column": name,
            "count": 0,
            "missing": info["missing"],
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "max": None,
        }
    return {
        "column": name,
        "count": n,
        "missing": info["missing"],
        "mean": round(_mean(vals), 4),
        "median": round(_median(vals), 4),
        "std": round(_std(vals), 4),
        "min": round(min(vals), 4),
        "25%": round(_percentile(vals, 25), 4),
        "50%": round(_percentile(vals, 50), 4),
        "75%": round(_percentile(vals, 75), 4),
        "max": round(max(vals), 4),
    }


def categorical_summary(name, info):
    vals = info["values"]
    n = len(vals)
    return {
        "column": name,
        "count": n,
        "missing": info["missing"],
        "unique": len(set(vals)),
        "mode": _mode(vals) if n > 0 else None,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_SEP_WIDTH = 60


def _print_header(title):
    print("\n" + "=" * _SEP_WIDTH)
    print(f"  {title}")
    print("=" * _SEP_WIDTH)


def _print_numeric_table(summaries):
    if not summaries:
        return
    _print_header("Numeric Columns — Descriptive Statistics")
    fields = ["column", "count", "missing", "mean", "median", "std",
              "min", "25%", "50%", "75%", "max"]
    col_widths = {f: max(len(f), max(len(str(s[f])) for s in summaries))
                  for f in fields}
    header_row = "  ".join(f.ljust(col_widths[f]) for f in fields)
    print(header_row)
    print("-" * len(header_row))
    for s in summaries:
        print("  ".join(str(s[f]).ljust(col_widths[f]) for f in fields))


def _print_categorical_table(summaries):
    if not summaries:
        return
    _print_header("Categorical Columns — Descriptive Statistics")
    fields = ["column", "count", "missing", "unique", "mode"]
    col_widths = {f: max(len(f), max(len(str(s[f])) for s in summaries))
                  for f in fields}
    header_row = "  ".join(f.ljust(col_widths[f]) for f in fields)
    print(header_row)
    print("-" * len(header_row))
    for s in summaries:
        print("  ".join(str(s[f]).ljust(col_widths[f]) for f in fields))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def summarize(filepath):
    """Read *filepath* and print summary statistics to stdout."""
    if not os.path.isfile(filepath):
        print(f"Error: file not found — {filepath}", file=sys.stderr)
        sys.exit(1)

    headers, rows = read_csv(filepath)

    if not rows:
        print("The CSV file contains no data rows.")
        return

    print(f"\nFile  : {filepath}")
    print(f"Rows  : {len(rows)}")
    print(f"Columns ({len(headers)}): {', '.join(headers)}")

    numeric, categorical = extract_columns(headers, rows)

    numeric_summaries = [numeric_summary(name, info) for name, info in numeric.items()]
    categorical_summaries = [categorical_summary(name, info) for name, info in categorical.items()]

    _print_numeric_table(numeric_summaries)
    _print_categorical_table(categorical_summaries)
    print()


if __name__ == "__main__":
    default_csv = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    summarize(csv_path)

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


# ============================================================
# CONFIGURATION
# ============================================================

# SKAB data directory.
DATA_DIR = Path("SKAB-master/data")

# Aggregation factors (1 = raw data, 2 = mean of every 2 rows, ...).
AGGREGATION_FACTORS = [1, 2, 5, 10, 20, 50, 100]

# "any":      group is anomalous if at least one raw row is anomalous.
# "majority": group is anomalous if >= 50% of its raw rows are anomalous.
ANOMALY_MODE = "any"

# Output folder includes the mode, so runs with different modes
# never overwrite each other.
OUTPUT_DIR = Path(f"SKAB_aggregated_output_{ANOMALY_MODE}")

# Keep the changepoint column (aggregated with max()) if it exists.
# Not used for labels, metrics or plots.
KEEP_CHANGEPOINT_COLUMN = True

# SKAB's native CSV delimiter (used for reading AND writing).
OUTPUT_SEPARATOR = ";"

# Files without an `anomaly` column (e.g. anomaly-free.csv) are skipped.
SKIP_FILES_WITHOUT_ANOMALY = True

MAKE_DATASET_PLOTS = True
MAKE_SUMMARY_PLOTS = True
FIGURE_DPI = 160

# Row-based aggregation; timestamp irregularities are only reported.
EXPECTED_SAMPLE_INTERVAL_SECONDS = 1.0
SAMPLING_TOLERANCE_SECONDS = 0.1


# ============================================================
# CONSTANTS
# ============================================================

DATETIME_COLUMN = "datetime"
ANOMALY_COLUMN = "anomaly"
CHANGEPOINT_COLUMN = "changepoint"

VALID_ANOMALY_MODES = {"any", "majority"}

NON_SENSOR_COLUMNS = {
    DATETIME_COLUMN,
    ANOMALY_COLUMN,
    CHANGEPOINT_COLUMN,
}


# ============================================================
# DATA LOADING
# ============================================================

def read_skab_csv(path: Path) -> pd.DataFrame:
    """Read one SKAB CSV (semicolon delimited) and validate it."""

    df = pd.read_csv(path, sep=OUTPUT_SEPARATOR)
    df.columns = [str(col).strip() for col in df.columns]

    required = {DATETIME_COLUMN, ANOMALY_COLUMN}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        errors="coerce",
    )

    if df[DATETIME_COLUMN].isna().any():
        bad = int(df[DATETIME_COLUMN].isna().sum())
        raise ValueError(f"{path}: {bad} invalid datetime values found.")

    df[ANOMALY_COLUMN] = pd.to_numeric(
        df[ANOMALY_COLUMN],
        errors="raise",
    )

    if not df[ANOMALY_COLUMN].isin([0, 1]).all():
        raise ValueError(
            f"{path}: anomaly column contains values other than 0/1."
        )

    # Changepoint is only preserved, never analysed. Be tolerant:
    # non-numeric / missing entries are treated as "no changepoint".
    if CHANGEPOINT_COLUMN in df.columns:
        df[CHANGEPOINT_COLUMN] = (
            pd.to_numeric(df[CHANGEPOINT_COLUMN], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric sensor columns (metadata/labels excluded)."""

    candidates = [
        col for col in df.columns
        if col not in NON_SENSOR_COLUMNS
    ]

    numeric_columns: list[str] = []

    for col in candidates:
        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().all():
            df[col] = converted
            numeric_columns.append(col)

    if not numeric_columns:
        raise ValueError("No numeric sensor columns were found.")

    return numeric_columns


def sampling_statistics(df: pd.DataFrame) -> dict[str, float | int]:
    """Timestamp-spacing diagnostics. Does not alter the data."""

    deltas = (
        df[DATETIME_COLUMN]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    if deltas.empty:
        return {
            "median_sample_interval_s": math.nan,
            "mean_sample_interval_s": math.nan,
            "max_sample_interval_s": math.nan,
            "irregular_gap_count": 0,
            "irregular_gap_fraction": 0.0,
        }

    irregular = (
        (deltas - EXPECTED_SAMPLE_INTERVAL_SECONDS).abs()
        > SAMPLING_TOLERANCE_SECONDS
    )

    return {
        "median_sample_interval_s": float(deltas.median()),
        "mean_sample_interval_s": float(deltas.mean()),
        "max_sample_interval_s": float(deltas.max()),
        "irregular_gap_count": int(irregular.sum()),
        "irregular_gap_fraction": float(irregular.mean()),
    }


def make_dataset_id(path: Path, data_dir: Path) -> str:
    """Collision-free dataset ID from the relative path (valve1/1.csv -> valve1_1)."""

    parts = list(path.relative_to(data_dir).parts)
    parts[-1] = Path(parts[-1]).stem

    return "_".join(parts).replace(" ", "_")


def collect_dataset_files(data_dir: Path) -> list[Path]:
    """Find all CSV files below DATA_DIR."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"SKAB data directory not found: {data_dir.resolve()}"
        )

    files = sorted(data_dir.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found below: {data_dir.resolve()}"
        )

    return files


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_dataframe(
    df: pd.DataFrame,
    factor: int,
    anomaly_mode: str,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """
    Aggregate into non-overlapping groups of `factor` rows.

    Sensors:     arithmetic mean
    Datetime:    first timestamp of the group
    anomaly:     "any" or "majority" rule over raw rows
    changepoint: max() within the group (if preserved)

    The final incomplete group is discarded.
    """

    if factor < 1:
        raise ValueError("Aggregation factor must be >= 1.")

    if anomaly_mode not in VALID_ANOMALY_MODES:
        raise ValueError(
            f"Unknown ANOMALY_MODE={anomaly_mode!r}. "
            f"Use one of {sorted(VALID_ANOMALY_MODES)}."
        )

    sensor_columns = get_sensor_columns(df)

    original_rows = len(df)
    usable_rows = (original_rows // factor) * factor
    discarded_rows = original_rows - usable_rows

    has_changepoint = (
        KEEP_CHANGEPOINT_COLUMN
        and CHANGEPOINT_COLUMN in df.columns
    )

    total_raw_anomalies = int(df[ANOMALY_COLUMN].sum())

    # --------------------------------------------------------
    # FILE SHORTER THAN FACTOR
    # --------------------------------------------------------

    if usable_rows == 0:

        empty = pd.DataFrame(
            columns=[DATETIME_COLUMN, *sensor_columns, ANOMALY_COLUMN]
        )

        if has_changepoint:
            empty[CHANGEPOINT_COLUMN] = pd.Series(dtype="int64")

        meta = {
            "original_rows": original_rows,
            "aggregated_rows": 0,
            "discarded_rows": discarded_rows,
            "raw_anomaly_rows_total": total_raw_anomalies,
            "raw_anomaly_rows_in_complete_groups": 0,
            "raw_anomaly_rows_in_discarded_tail": total_raw_anomalies,
            "raw_rows_covered_by_anomalous_blocks": 0,
            "normal_rows_covered_by_anomalous_blocks": 0,
            "anomaly_recall_after_label_aggregation": math.nan,
            "anomalous_block_purity": math.nan,
            "rows_covered_per_true_anomaly": (
                0.0 if total_raw_anomalies > 0 else math.nan
            ),
            "anomaly_disappears_completely": total_raw_anomalies > 0,
            "anomaly_disappears_from_complete_groups": False,
            "tail_contains_anomaly": total_raw_anomalies > 0,
            "positive_anomaly_block_count": 0,
            "aggregated_anomaly_rate": 0.0,
        }

        return empty, meta

    # --------------------------------------------------------
    # GROUPING
    # --------------------------------------------------------

    work = df.iloc[:usable_rows].copy()

    work["_group"] = np.arange(usable_rows, dtype=np.int64) // factor

    grouped = work.groupby("_group", sort=True, observed=True)

    aggregated = grouped[sensor_columns].mean().reset_index(drop=True)

    timestamps = (
        grouped[DATETIME_COLUMN]
        .first()
        .reset_index(drop=True)
    )

    anomaly_fraction = (
        grouped[ANOMALY_COLUMN]
        .mean()
        .reset_index(drop=True)
    )

    if anomaly_mode == "any":
        aggregated_anomaly = (anomaly_fraction > 0).astype("int64")
    else:
        aggregated_anomaly = (anomaly_fraction >= 0.5).astype("int64")

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = pd.concat(
        [
            pd.DataFrame({DATETIME_COLUMN: timestamps}),
            aggregated,
        ],
        axis=1,
    )

    result[ANOMALY_COLUMN] = aggregated_anomaly.to_numpy()

    if has_changepoint:
        result[CHANGEPOINT_COLUMN] = (
            grouped[CHANGEPOINT_COLUMN]
            .max()
            .astype("int64")
            .to_numpy()
        )

    # --------------------------------------------------------
    # LABEL-DISTORTION METRICS
    # --------------------------------------------------------

    complete_group_anomalies = int(work[ANOMALY_COLUMN].sum())

    tail_anomalies = int(
        df.iloc[usable_rows:][ANOMALY_COLUMN].sum()
    )

    positive_flags = aggregated_anomaly.to_numpy().astype(bool)
    positive_group_ids = np.flatnonzero(positive_flags)

    if positive_group_ids.size > 0:

        positive_group_mask = work["_group"].isin(positive_group_ids)

        raw_anomalies_in_positive_groups = int(
            work.loc[positive_group_mask, ANOMALY_COLUMN].sum()
        )

        raw_rows_covered = int(positive_group_mask.sum())

    else:

        raw_anomalies_in_positive_groups = 0
        raw_rows_covered = 0

    normal_rows_covered = (
        raw_rows_covered - raw_anomalies_in_positive_groups
    )

    anomaly_recall = (
        raw_anomalies_in_positive_groups / complete_group_anomalies
        if complete_group_anomalies > 0
        else math.nan
    )

    block_purity = (
        raw_anomalies_in_positive_groups / raw_rows_covered
        if raw_rows_covered > 0
        else math.nan
    )

    rows_covered_per_true_anomaly = (
        raw_rows_covered / total_raw_anomalies
        if total_raw_anomalies > 0
        else math.nan
    )

    disappears_from_complete_groups = (
        complete_group_anomalies > 0
        and raw_anomalies_in_positive_groups == 0
    )

    disappears_completely = (
        total_raw_anomalies > 0
        and int(aggregated_anomaly.sum()) == 0
    )

    meta = {
        "original_rows": original_rows,
        "aggregated_rows": len(result),
        "discarded_rows": discarded_rows,
        "raw_anomaly_rows_total": total_raw_anomalies,
        "raw_anomaly_rows_in_complete_groups": complete_group_anomalies,
        "raw_anomaly_rows_in_discarded_tail": tail_anomalies,
        "raw_rows_covered_by_anomalous_blocks": raw_rows_covered,
        "normal_rows_covered_by_anomalous_blocks": normal_rows_covered,
        "anomaly_recall_after_label_aggregation": anomaly_recall,
        "anomalous_block_purity": block_purity,
        "rows_covered_per_true_anomaly": rows_covered_per_true_anomaly,
        "anomaly_disappears_completely": disappears_completely,
        "anomaly_disappears_from_complete_groups": disappears_from_complete_groups,
        "tail_contains_anomaly": tail_anomalies > 0,
        "positive_anomaly_block_count": int(aggregated_anomaly.sum()),
        "aggregated_anomaly_rate": float(aggregated_anomaly.mean()),
    }

    return result, meta


def check_factor_one_is_identity(
    df: pd.DataFrame,
    aggregated: pd.DataFrame,
    meta: dict,
) -> None:
    """Sanity check: factor 1 must reproduce the raw labels exactly."""

    if len(aggregated) != len(df):
        raise AssertionError(
            f"factor 1 changed the row count: {len(df)} -> {len(aggregated)}"
        )

    if not np.array_equal(
        aggregated[ANOMALY_COLUMN].to_numpy(),
        df[ANOMALY_COLUMN].to_numpy(),
    ):
        raise AssertionError("factor 1 changed the anomaly labels.")

    covered = meta["rows_covered_per_true_anomaly"]

    if not (math.isnan(covered) or math.isclose(covered, 1.0)):
        raise AssertionError(
            f"factor 1 rows_covered_per_true_anomaly = {covered}, expected 1.0"
        )


# ============================================================
# PLOTTING
# ============================================================

def plot_dataset_comparison(
    dataset_name: str,
    results: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    """Aggregated anomaly locations for one dataset, one row per factor."""

    factors = list(results.keys())

    fig, ax = plt.subplots(
        figsize=(14, max(3.5, 1.0 + 0.65 * len(factors)))
    )

    for row, factor in enumerate(factors):

        data = results[factor]

        anomalies = data.loc[
            data[ANOMALY_COLUMN] == 1,
            DATETIME_COLUMN,
        ]

        if not anomalies.empty:
            ax.scatter(
                anomalies,
                [row] * len(anomalies),
                s=18 if factor == 1 else 30,
                marker="|",
                linewidths=1.5,
            )

    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([f"×{factor}" for factor in factors])
    ax.set_xlabel("Time")
    ax.set_ylabel("Aggregation")
    ax.set_title(
        f"SKAB point anomalies: {dataset_name}\n"
        f"Label mode: {ANOMALY_MODE} | changepoint ignored"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def heatmap_from_summary(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    """Dataset × aggregation-factor heatmap."""

    pivot = summary.pivot(
        index="dataset",
        columns="factor",
        values=value_column,
    )

    values = pivot.to_numpy(dtype=float)

    fig_height = max(6, 0.32 * len(pivot) + 2.5)

    fig, ax = plt.subplots(figsize=(11, fig_height))

    positive = values[np.isfinite(values) & (values > 0)]

    if log_scale and positive.size > 0:

        image = ax.imshow(
            np.ma.masked_less_equal(
                np.ma.masked_invalid(values),
                0,
            ),
            aspect="auto",
            norm=LogNorm(
                vmin=float(positive.min()),
                vmax=float(positive.max()),
            ),
        )

    else:

        image = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
        )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"×{int(x)}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Aggregation factor")
    ax.set_ylabel("Dataset")
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def make_summary_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    """Heatmaps for anomaly rate, label coverage and recall."""

    heatmap_from_summary(
        summary,
        value_column="aggregated_anomaly_rate",
        title=f"Aggregated anomaly-block rate\nMode: {ANOMALY_MODE}",
        colorbar_label="Fraction of aggregated rows labelled anomalous",
        output_path=output_dir / "summary_anomaly_rate.png",
    )

    heatmap_from_summary(
        summary,
        value_column="rows_covered_per_true_anomaly",
        title=(
            "Label coverage / distortion\n"
            "Raw rows covered by anomalous blocks ÷ true anomalous rows"
        ),
        colorbar_label="Covered rows per true anomaly (log scale)",
        output_path=output_dir / "summary_label_coverage.png",
        log_scale=True,
    )

    heatmap_from_summary(
        summary,
        value_column="anomaly_recall_after_label_aggregation",
        title=f"Raw-anomaly recall after label aggregation\nMode: {ANOMALY_MODE}",
        colorbar_label="Recall on anomalies in complete groups",
        output_path=output_dir / "summary_anomaly_recall.png",
    )


# ============================================================
# MAIN
# ============================================================

SUMMARY_META_KEYS = [
    "original_rows",
    "aggregated_rows",
    "discarded_rows",
    "raw_anomaly_rows_total",
    "raw_anomaly_rows_in_complete_groups",
    "raw_anomaly_rows_in_discarded_tail",
    "raw_rows_covered_by_anomalous_blocks",
    "normal_rows_covered_by_anomalous_blocks",
    "rows_covered_per_true_anomaly",
    "anomaly_recall_after_label_aggregation",
    "anomalous_block_purity",
    "positive_anomaly_block_count",
    "aggregated_anomaly_rate",
    "anomaly_disappears_completely",
    "anomaly_disappears_from_complete_groups",
    "tail_contains_anomaly",
]


def main() -> None:

    if ANOMALY_MODE not in VALID_ANOMALY_MODES:
        raise ValueError(
            f"Invalid ANOMALY_MODE={ANOMALY_MODE!r}; "
            f"choose from {sorted(VALID_ANOMALY_MODES)}"
        )

    if not AGGREGATION_FACTORS:
        raise ValueError("AGGREGATION_FACTORS cannot be empty.")

    if any(factor < 1 for factor in AGGREGATION_FACTORS):
        raise ValueError("All aggregation factors must be >= 1.")

    if len(set(AGGREGATION_FACTORS)) != len(AGGREGATION_FACTORS):
        raise ValueError("AGGREGATION_FACTORS contains duplicates.")

    dataset_files = collect_dataset_files(DATA_DIR)

    csv_root = OUTPUT_DIR / "csv"
    plot_root = OUTPUT_DIR / "plots"

    csv_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    dataset_ids: dict[str, Path] = {}

    print(f"Found {len(dataset_files)} CSV file(s).")
    print(f"Anomaly mode: {ANOMALY_MODE}")
    print(f"Aggregation factors: {AGGREGATION_FACTORS}")
    print(f"Delimiter: '{OUTPUT_SEPARATOR}'")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    for dataset_path in dataset_files:

        dataset_id = make_dataset_id(dataset_path, DATA_DIR)

        if dataset_id in dataset_ids:
            raise RuntimeError(
                "Dataset ID collision detected:\n"
                f"  {dataset_ids[dataset_id]}\n"
                f"  {dataset_path}\n"
                f"Both map to '{dataset_id}'."
            )

        dataset_ids[dataset_id] = dataset_path

        relative_path = str(dataset_path.relative_to(DATA_DIR))

        print(f"Processing: {relative_path}")

        try:

            # Peek at the header to decide whether to skip.
            header = pd.read_csv(
                dataset_path,
                sep=OUTPUT_SEPARATOR,
                nrows=0,
            )

            header_columns = [str(c).strip() for c in header.columns]

            if ANOMALY_COLUMN not in header_columns:

                if SKIP_FILES_WITHOUT_ANOMALY:

                    skipped.append(
                        {
                            "dataset": relative_path,
                            "reason": f"missing '{ANOMALY_COLUMN}' column",
                            "columns_found": ",".join(header_columns),
                        }
                    )

                    print("  -> skipped: no 'anomaly' column")
                    continue

                raise ValueError(
                    f"{dataset_path}: missing '{ANOMALY_COLUMN}' column."
                )

            df = read_skab_csv(dataset_path)

            sample_stats = sampling_statistics(df)

            results_for_plot: dict[int, pd.DataFrame] = {}

            for factor in AGGREGATION_FACTORS:

                aggregated, meta = aggregate_dataframe(
                    df.copy(),
                    factor=factor,
                    anomaly_mode=ANOMALY_MODE,
                )

                if factor == 1:
                    check_factor_one_is_identity(df, aggregated, meta)

                results_for_plot[factor] = aggregated

                factor_dir = csv_root / f"factor_{factor}"
                factor_dir.mkdir(parents=True, exist_ok=True)

                aggregated.to_csv(
                    factor_dir / f"{dataset_id}.csv",
                    sep=OUTPUT_SEPARATOR,
                    index=False,
                )

                row: dict[str, object] = {
                    "dataset": dataset_id,
                    "dataset_path": relative_path,
                    "factor": factor,
                }

                for key in SUMMARY_META_KEYS:
                    row[key] = meta[key]

                row.update(sample_stats)

                summary_rows.append(row)

            if MAKE_DATASET_PLOTS:

                plot_dataset_comparison(
                    dataset_id,
                    results_for_plot,
                    plot_root / f"{dataset_id}_anomalies.png",
                )

            print(
                f"  -> done ({len(df)} raw rows, "
                f"{int(df[ANOMALY_COLUMN].sum())} anomaly rows)"
            )

        except Exception as exc:

            failures.append(
                {
                    "dataset": relative_path,
                    "error": str(exc),
                }
            )

            print(f"  -> ERROR: {exc}")

    # ========================================================
    # REPORT FILES
    # ========================================================

    summary = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "summary.csv"
    summary.to_csv(summary_path, sep=OUTPUT_SEPARATOR, index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(
            OUTPUT_DIR / "skipped_files.csv",
            sep=OUTPUT_SEPARATOR,
            index=False,
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "errors.csv",
            sep=OUTPUT_SEPARATOR,
            index=False,
        )

    if MAKE_SUMMARY_PLOTS and not summary.empty:
        make_summary_plots(summary, OUTPUT_DIR)

    # ========================================================
    # FINAL REPORT
    # ========================================================

    processed_datasets = (
        int(summary["dataset"].nunique())
        if not summary.empty
        else 0
    )

    print("\nFinished.")
    print(f"Processed anomaly datasets: {processed_datasets}")
    print(f"Successful factor runs: {len(summary)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Failed files: {len(failures)}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"CSV output: {csv_root.resolve()}")

    if MAKE_DATASET_PLOTS:
        print(f"Dataset plots: {plot_root.resolve()}")

    if MAKE_SUMMARY_PLOTS and not summary.empty:
        print(f"Summary plots: {OUTPUT_DIR.resolve()}")

    if skipped:
        print(
            f"Skipped-file report: "
            f"{(OUTPUT_DIR / 'skipped_files.csv').resolve()}"
        )

    if failures:
        print(
            f"Error report: "
            f"{(OUTPUT_DIR / 'errors.csv').resolve()}"
        )


if __name__ == "__main__":
    main()
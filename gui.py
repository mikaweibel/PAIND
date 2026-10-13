"""
SKAB aggregation explorer (customtkinter GUI)
=============================================

Visualises
  * the output of skab_aggregate.py        (AGG_DIR)
  * the output of skab_ml_experiment.py    (ML_DIR)

Tabs
  1. Data explorer     - raw vs. aggregated sensor signal + label strips
  2. Label distortion  - what aggregation does to the labels themselves
  3. Model performance - metric vs. aggregation factor, per model
  4. Per dataset       - heatmap / box plot: which files break first
  5. Detection view    - score, threshold and alarms of one model on one file
  6. Explainability    - SHAP / reconstruction-error importances per factor

Install:  pip install customtkinter matplotlib pandas numpy
"""

from __future__ import annotations

import math
from pathlib import Path
from tkinter import filedialog, ttk

import matplotlib

matplotlib.use("TkAgg")

import customtkinter as ctk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure


# ============================================================
# CONFIGURATION
# ============================================================

ANOMALY_MODE = "any"   # default folders; can be changed in the GUI

DEFAULT_AGG_DIR = Path(f"SKAB_aggregated_output_{ANOMALY_MODE}")
DEFAULT_ML_DIR = Path(f"SKAB_ml_results_{ANOMALY_MODE}")

CSV_SEP = ";"

DATETIME_COLUMN = "datetime"
ANOMALY_COLUMN = "anomaly"
NON_FEATURE_COLUMNS = {"datetime", "anomaly", "changepoint"}

MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "pca_spe": "PCA (SPE)",
    "autoencoder": "Autoencoder",
    "lof": "LOF",
    "xgboost_shap": "XGBoost (supervised)",
}

# metric -> (label, higher_is_better)
PERFORMANCE_METRICS = {
    "pr_auc": ("PR-AUC (average precision)", True),
    "roc_auc": ("ROC-AUC", True),
    "best_f1": ("Best F1 (any threshold)", True),
    "f1": ("F1 @ operating threshold", True),
    "precision": ("Precision @ threshold", True),
    "recall": ("Recall @ threshold", True),
    "far": ("FAR - false alarm rate", False),
    "mar": ("MAR - missed alarm rate", False),
    "detection_delay_rows": ("Detection delay (raw rows)", False),
    "train_rows": ("Training rows", True),
    "fit_time_s": ("Fit time (s)", False),
}

LEVELS = {
    "Raw resolution (fair across factors)": "raw",
    "Native (aggregated labels)": "native",
}

# ---- plot theme -------------------------------------------------------

BG = "#242424"
PANEL = "#2b2b2b"
FG = "#e0e0e0"
GRID = "#4a4a4a"
RED = "#e5534b"
ORANGE = "#e0a030"

MODEL_COLORS = {
    "isolation_forest": "#4c9be8",
    "pca_spe": "#e8a33c",
    "autoencoder": "#57c785",
    "lof": "#c678dd",
    "xgboost_shap": "#e5534b",
}
FALLBACK_COLORS = ["#4c9be8", "#e8a33c", "#57c785", "#c678dd", "#e5534b",
                   "#56b6c2", "#d19a66", "#98c379"]


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def model_color(name: str, index: int = 0) -> str:
    return MODEL_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def style_axes(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.6)


def new_figure_state(fig: Figure) -> None:
    fig.clear()
    fig.set_facecolor(BG)


def message_plot(fig: Figure, text: str) -> None:
    """Draw a centred message (missing files, no selection, ...)."""
    new_figure_state(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=FG,
            fontsize=11, wrap=True, transform=ax.transAxes)


def style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(FG)


def factor_label(factor) -> str:
    return f"×{int(factor)}"


# ============================================================
# DATA LAYER (no GUI code - testable headlessly)
# ============================================================

class DataStore:
    """Lazy, cached access to everything the two scripts wrote."""

    def __init__(self, agg_dir: Path, ml_dir: Path):
        self.agg_dir = Path(agg_dir)
        self.ml_dir = Path(ml_dir)
        self._cache: dict = {}

        self.label_summary = self._read(self.agg_dir / "summary.csv")
        self.per_file = self._read(self.ml_dir / "results_per_file.csv")
        self.summary = self._read(self.ml_dir / "results_summary.csv")
        self.explanations = self._read(self.ml_dir / "explanations.csv")

        for frame in (self.label_summary,):
            if frame is not None:
                for col in frame.columns:
                    if frame[col].dtype == bool:
                        frame[col] = frame[col].astype(float)

    # ------------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, sep=CSV_SEP)
        except Exception:
            return None

    # ------------------------------------------------------------------

    @property
    def has_aggregated(self) -> bool:
        return (self.agg_dir / "csv").exists()

    @property
    def has_ml(self) -> bool:
        return self.summary is not None and len(self.summary) > 0

    def factors(self) -> list[int]:
        root = self.agg_dir / "csv"
        found = []
        if root.exists():
            for path in root.glob("factor_*"):
                try:
                    found.append(int(path.name.split("_", 1)[1]))
                except ValueError:
                    pass
        return sorted(found)

    def datasets(self) -> list[str]:
        folder = self.agg_dir / "csv" / "factor_1"
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.glob("*.csv"))

    def models(self) -> list[str]:
        if self.summary is None:
            return []
        present = list(self.summary["model"].unique())
        order = [m for m in MODEL_LABELS if m in present]
        return order + [m for m in present if m not in order]

    def ml_factors(self) -> list[int]:
        if self.summary is None:
            return []
        return sorted(int(f) for f in self.summary["factor"].unique())

    def explained_models(self) -> list[str]:
        if self.explanations is None:
            return []
        present = list(self.explanations["model"].unique())
        return [m for m in MODEL_LABELS if m in present] + [
            m for m in present if m not in MODEL_LABELS
        ]

    # ------------------------------------------------------------------

    def aggregated(self, dataset: str, factor: int) -> pd.DataFrame | None:
        key = ("agg", dataset, factor)
        if key not in self._cache:
            path = self.agg_dir / "csv" / f"factor_{factor}" / f"{dataset}.csv"
            frame = None
            if path.exists():
                try:
                    frame = pd.read_csv(
                        path, sep=CSV_SEP, parse_dates=[DATETIME_COLUMN]
                    )
                except Exception:
                    frame = None
            self._cache[key] = frame
        return self._cache[key]

    def predictions(self, dataset: str, factor: int) -> pd.DataFrame | None:
        key = ("pred", dataset, factor)
        if key not in self._cache:
            path = (self.ml_dir / "predictions" / f"factor_{factor}"
                    / f"{dataset}.csv")
            frame = None
            if path.exists():
                try:
                    frame = pd.read_csv(
                        path, sep=CSV_SEP, parse_dates=[DATETIME_COLUMN]
                    )
                except Exception:
                    frame = None
            self._cache[key] = frame
        return self._cache[key]

    def sensors(self, dataset: str) -> list[str]:
        frame = self.aggregated(dataset, 1)
        if frame is None:
            return []
        return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]

    # ------------------------------------------------------------------

    def curve(self, level: str, model: str, metric: str) -> pd.DataFrame:
        """factor, mean, std, n for one model/metric (macro over files)."""
        if self.summary is None:
            return pd.DataFrame(columns=["factor", "mean", "std", "n"])
        sub = self.summary[
            (self.summary["level"] == level) & (self.summary["model"] == model)
        ].sort_values("factor")
        return pd.DataFrame({
            "factor": sub["factor"].to_numpy(),
            "mean": sub[f"{metric}_mean"].to_numpy(),
            "std": sub[f"{metric}_std"].to_numpy(),
            "n": sub[f"{metric}_n"].to_numpy(),
        })

    def file_matrix(self, level: str, model: str, metric: str) -> pd.DataFrame:
        """dataset x factor matrix of one metric."""
        if self.per_file is None:
            return pd.DataFrame()
        sub = self.per_file[
            (self.per_file["level"] == level) & (self.per_file["model"] == model)
        ]
        return sub.pivot_table(index="dataset", columns="factor",
                               values=metric, aggfunc="mean")

    def file_metrics(self, level: str, model: str, factor: int,
                     dataset: str) -> pd.Series | None:
        if self.per_file is None:
            return None
        sub = self.per_file[
            (self.per_file["level"] == level)
            & (self.per_file["model"] == model)
            & (self.per_file["factor"] == factor)
            & (self.per_file["dataset"] == dataset)
        ]
        return sub.iloc[0] if len(sub) else None

    def importance(self, model: str, column: str) -> pd.DataFrame:
        """feature x factor matrix of mean importance share (over folds)."""
        if self.explanations is None:
            return pd.DataFrame()
        sub = self.explanations[self.explanations["model"] == model]
        return sub.pivot_table(index="feature", columns="factor",
                               values=column, aggfunc="mean")

    def explanation_method(self, model: str) -> str:
        if self.explanations is None:
            return ""
        sub = self.explanations[self.explanations["model"] == model]
        return str(sub["method"].iloc[0]) if len(sub) else ""

    def explanation_drift(self, model: str, column: str) -> pd.Series:
        """Spearman rank correlation of the importance vector with factor 1."""
        matrix = self.importance(model, column)
        if matrix.empty or 1 not in matrix.columns:
            return pd.Series(dtype=float)
        ref = matrix[1].rank()
        return pd.Series(
            {int(f): float(matrix[f].rank().corr(ref)) for f in matrix.columns}
        ).sort_index()


# ============================================================
# PLOT FUNCTIONS (Figure in, drawing out - testable headlessly)
# ============================================================

def plot_data_explorer(fig, store, dataset, sensor, factors) -> None:
    if not dataset or not sensor or not factors:
        message_plot(fig, "Select a dataset, a sensor and at least one factor.")
        return

    frames = {f: store.aggregated(dataset, f) for f in factors}
    frames = {f: d for f, d in frames.items() if d is not None and len(d)}

    if not frames:
        message_plot(fig, "No aggregated data found for this selection.")
        return

    new_figure_state(fig)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax = fig.add_subplot(grid[0])
    ax_lab = fig.add_subplot(grid[1], sharex=ax)

    style_axes(ax)
    style_axes(ax_lab)

    cmap = matplotlib.colormaps["viridis"]
    ordered = sorted(frames)

    raw = store.aggregated(dataset, 1)

    if raw is not None and ANOMALY_COLUMN in raw:
        ax.fill_between(
            raw[DATETIME_COLUMN], 0, 1,
            where=raw[ANOMALY_COLUMN].to_numpy() == 1,
            transform=ax.get_xaxis_transform(),
            color=RED, alpha=0.13, step="post",
            label="true anomaly (raw)",
        )

    for i, f in enumerate(ordered):
        d = frames[f]
        color = "#9a9a9a" if f == 1 else cmap(i / max(1, len(ordered) - 1))
        ax.plot(d[DATETIME_COLUMN], d[sensor], color=color,
                lw=0.8 if f == 1 else 1.4, alpha=0.8 if f == 1 else 1.0,
                label=factor_label(f), drawstyle="steps-post" if f > 1 else "default")

    ax.set_ylabel(sensor)
    ax.set_title(f"{dataset} - {sensor}: raw vs. block means", fontsize=10)
    style_legend(ax.legend(loc="upper right", fontsize=8, ncol=4))
    ax.tick_params(labelbottom=False)

    for row, f in enumerate(ordered):
        d = frames[f]
        ax_lab.fill_between(
            d[DATETIME_COLUMN], row, row + 0.8,
            where=d[ANOMALY_COLUMN].to_numpy() == 1,
            color=RED if f == 1 else ORANGE, alpha=0.85, step="post",
        )

    ax_lab.set_yticks([r + 0.4 for r in range(len(ordered))])
    ax_lab.set_yticklabels([factor_label(f) for f in ordered])
    ax_lab.set_ylim(0, len(ordered))
    ax_lab.set_ylabel("label")
    ax_lab.grid(False)
    fig.autofmt_xdate()


def plot_label_distortion(fig, store, metric, view) -> None:
    frame = store.label_summary

    if frame is None or metric not in frame.columns:
        message_plot(fig, "summary.csv from skab_aggregate.py not found.")
        return

    new_figure_state(fig)
    ax = fig.add_subplot(111)
    style_axes(ax)

    factors = sorted(frame["factor"].unique())

    if view == "Mean per factor":
        grouped = frame.groupby("factor")[metric]
        mean = grouped.mean().reindex(factors)
        q25 = grouped.quantile(0.25).reindex(factors)
        q75 = grouped.quantile(0.75).reindex(factors)
        x = np.arange(len(factors))
        ax.plot(x, mean.to_numpy(), marker="o", color="#4c9be8", lw=2,
                label="mean over datasets")
        ax.fill_between(x, q25.to_numpy(), q75.to_numpy(), color="#4c9be8",
                        alpha=0.2, label="25-75 % range")
        ax.set_xticks(x)
        ax.set_xticklabels([factor_label(f) for f in factors])
        ax.set_xlabel("aggregation factor")
        ax.set_ylabel(metric)
        style_legend(ax.legend(fontsize=8))
    else:
        matrix = frame.pivot_table(index="dataset", columns="factor",
                                   values=metric, aggfunc="mean")
        values = matrix.to_numpy(dtype=float)
        ax.grid(False)
        positive = values[np.isfinite(values) & (values > 0)]
        use_log = (metric in ("rows_covered_per_true_anomaly",
                              "normal_rows_covered_by_anomalous_blocks")
                   and positive.size > 0)
        norm = (LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
                if use_log else None)
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto",
                          cmap="magma", norm=norm)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([factor_label(f) for f in matrix.columns])
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=6)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.ax.tick_params(colors=FG, labelsize=8)
        colorbar.set_label(metric + (" (log)" if use_log else ""), color=FG)

    ax.set_title(f"Label distortion: {metric}", fontsize=10)
    fig.tight_layout()


def _direction_arrow(higher_is_better: bool) -> str:
    return "↑ better" if higher_is_better else "↓ better"


def plot_performance(fig, store, level, metric, models, relative, show_band):
    if not store.has_ml:
        message_plot(fig, "results_summary.csv not found.\n"
                          "Run skab_ml_experiment.py first.")
        return

    if not models:
        message_plot(fig, "Select at least one model.")
        return

    new_figure_state(fig)
    ax = fig.add_subplot(111)
    style_axes(ax)

    label, higher = PERFORMANCE_METRICS[metric]
    factors = store.ml_factors()
    x_of = {f: i for i, f in enumerate(factors)}

    drawn = False

    for idx, model in enumerate(store.models()):
        if model not in models:
            continue

        curve = store.curve(level, model, metric)
        curve = curve[np.isfinite(curve["mean"])]

        if curve.empty:
            continue

        mean = curve["mean"].to_numpy(dtype=float)
        std = curve["std"].to_numpy(dtype=float)

        if relative:
            base = curve.loc[curve["factor"] == 1, "mean"]
            if base.empty or not np.isfinite(base.iloc[0]) or base.iloc[0] == 0:
                continue
            mean = mean / base.iloc[0]
            std = std / base.iloc[0]

        x = np.array([x_of[f] for f in curve["factor"]])
        color = model_color(model, idx)

        ax.plot(x, mean, marker="o", lw=2, color=color,
                label=model_label(model),
                linestyle="--" if model == "xgboost_shap" else "-")

        if show_band:
            lo = mean - np.nan_to_num(std)
            hi = mean + np.nan_to_num(std)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12)

        drawn = True

    if not drawn:
        message_plot(fig, "No data for this metric / selection.")
        return

    if relative:
        ax.axhline(1.0, color=GRID, lw=1)

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([factor_label(f) for f in factors])
    ax.set_xlabel("aggregation factor (mean of N consecutive rows)")
    ax.set_ylabel(("relative to ×1 - " if relative else "") + label)
    level_name = "raw-resolution" if level == "raw" else "native"
    ax.set_title(
        f"{label} vs. aggregation  [{level_name}, {_direction_arrow(higher)}]",
        fontsize=10,
    )
    style_legend(ax.legend(fontsize=8, loc="best"))
    fig.tight_layout()


def plot_per_dataset(fig, store, level, model, metric) -> None:
    if store.per_file is None or not model:
        message_plot(fig, "results_per_file.csv not found.")
        return

    matrix = store.file_matrix(level, model, metric)

    if matrix.empty:
        message_plot(fig, "No data for this selection.")
        return

    new_figure_state(fig)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.28)
    ax_h = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    style_axes(ax_h)
    style_axes(ax_b)
    ax_h.grid(False)

    label, higher = PERFORMANCE_METRICS[metric]
    values = matrix.to_numpy(dtype=float)

    image = ax_h.imshow(
        np.ma.masked_invalid(values), aspect="auto",
        cmap="viridis" if higher else "viridis_r",
    )
    ax_h.set_xticks(range(len(matrix.columns)))
    ax_h.set_xticklabels([factor_label(f) for f in matrix.columns])
    ax_h.set_yticks(range(len(matrix.index)))
    ax_h.set_yticklabels(matrix.index, fontsize=6)

    if values.shape[0] <= 40:
        finite = values[np.isfinite(values)]
        mid = (np.nanmin(finite) + np.nanmax(finite)) / 2 if finite.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    dark = (v > mid) == higher
                    ax_h.text(j, i, f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                              ha="center", va="center", fontsize=5,
                              color="black" if dark else "white")

    colorbar = fig.colorbar(image, ax=ax_h, fraction=0.04, pad=0.02)
    colorbar.ax.tick_params(colors=FG, labelsize=7)
    ax_h.set_title(f"{model_label(model)}: {label}", fontsize=10)

    data = [matrix[f].dropna().to_numpy() for f in matrix.columns]
    boxes = ax_b.boxplot(
        data, patch_artist=True, showfliers=True,
        medianprops={"color": "white"},
        flierprops={"markeredgecolor": FG, "markersize": 3},
        whiskerprops={"color": FG}, capprops={"color": FG},
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor(model_color(model))
        patch.set_alpha(0.6)
    ax_b.set_xticks(range(1, len(matrix.columns) + 1))
    ax_b.set_xticklabels([factor_label(f) for f in matrix.columns])
    ax_b.set_title("distribution over datasets", fontsize=10)
    ax_b.set_ylabel(label)

    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.08)


def plot_detection(fig, store, dataset, factor, model, sensor, level) -> None:
    if not dataset or not model:
        message_plot(fig, "Select dataset, factor and model.")
        return

    pred = store.predictions(dataset, factor)
    agg = store.aggregated(dataset, factor)

    if pred is None or agg is None:
        message_plot(fig, "No prediction file for this selection\n"
                          "(dataset too short for this factor, or the run "
                          "was skipped).")
        return

    score_col = f"{model}__score"

    if score_col not in pred.columns:
        message_plot(fig, f"No scores of {model_label(model)} in this file.")
        return

    new_figure_state(fig)
    grid = fig.add_gridspec(3, 1, height_ratios=[2, 2.4, 0.9], hspace=0.14)
    ax_s = fig.add_subplot(grid[0])
    ax_p = fig.add_subplot(grid[1], sharex=ax_s)
    ax_l = fig.add_subplot(grid[2], sharex=ax_s)
    for axis in (ax_s, ax_p, ax_l):
        style_axes(axis)

    t = pred[DATETIME_COLUMN]
    label = pred[ANOMALY_COLUMN].to_numpy()
    flag = pred[f"{model}__pred"].to_numpy()
    score = pred[score_col].to_numpy()
    threshold = float(pred[f"{model}__threshold"].iloc[0])

    if sensor and sensor in agg.columns:
        ax_s.plot(agg[DATETIME_COLUMN], agg[sensor], color="#9ab", lw=1.2)
        ax_s.set_ylabel(sensor)
    ax_s.fill_between(t, 0, 1, where=label == 1,
                      transform=ax_s.get_xaxis_transform(),
                      color=RED, alpha=0.18, step="post")
    ax_s.tick_params(labelbottom=False)

    ax_p.plot(t, score, color=model_color(model), lw=1.1)
    ax_p.axhline(threshold, color=ORANGE, ls="--", lw=1.2, label="threshold")
    ax_p.fill_between(t, 0, 1, where=label == 1,
                      transform=ax_p.get_xaxis_transform(),
                      color=RED, alpha=0.12, step="post")
    hit = flag == 1
    ax_p.scatter(t[hit], score[hit], s=10, color=ORANGE, zorder=3,
                 label="alarm")
    ax_p.set_ylabel("anomaly score" if model != "xgboost_shap"
                    else "P(anomaly)")
    positive = score[np.isfinite(score) & (score > 0)]
    if model != "xgboost_shap" and positive.size and positive.max() / max(
            positive.min(), 1e-12) > 200:
        ax_p.set_yscale("log")
    style_legend(ax_p.legend(fontsize=8, loc="upper left"))
    ax_p.tick_params(labelbottom=False)

    ax_l.fill_between(t, 0.55, 1.0, where=label == 1, color=RED, alpha=0.9,
                      step="post")
    ax_l.fill_between(t, 0.0, 0.45, where=flag == 1, color=ORANGE, alpha=0.9,
                      step="post")
    ax_l.set_yticks([0.22, 0.77])
    ax_l.set_yticklabels(["alarm", "label"])
    ax_l.set_ylim(0, 1)
    ax_l.grid(False)

    stats = store.file_metrics(level, model, factor, dataset)
    stats_text = ""
    if stats is not None:
        def fmt(key):
            value = stats.get(key, math.nan)
            return "n/a" if not np.isfinite(value) else f"{value:.2f}"
        stats_text = (f"   PR-AUC {fmt('pr_auc')} | F1 {fmt('f1')} | "
                      f"FAR {fmt('far')} | MAR {fmt('mar')}")
        if level == "raw":
            stats_text += f" | delay {fmt('detection_delay_rows')} rows"

    ax_s.set_title(
        f"{dataset} @ {factor_label(factor)} - {model_label(model)}"
        f"{stats_text}", fontsize=9)
    fig.autofmt_xdate()


def plot_explainability(fig, store, view, model, factor, target) -> None:
    if store.explanations is None:
        message_plot(fig, "explanations.csv not found.\n"
                          "Run skab_ml_experiment.py with "
                          "COMPUTE_EXPLANATIONS = True.")
        return

    column = ("importance_anomalous" if target == "anomalous rows"
              else "importance_normal")

    new_figure_state(fig)

    if view == "Feature × factor heatmap":
        matrix = store.importance(model, column)
        if matrix.empty:
            message_plot(fig, "No explanation data for this model.")
            return
        ax = fig.add_subplot(111)
        style_axes(ax)
        ax.grid(False)
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto",
                          cmap="magma")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([factor_label(f) for f in matrix.columns])
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center",
                            va="center", fontsize=7,
                            color="white" if values[i, j] < np.nanmax(values) * 0.6
                            else "black")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.ax.tick_params(colors=FG, labelsize=8)
        colorbar.set_label("share of total importance", color=FG)
        method = store.explanation_method(model)
        ax.set_title(f"{model_label(model)} - which sensors drive the "
                     f"score ({target}; {method})", fontsize=10)
        ax.set_xlabel("aggregation factor")

    elif view == "Models compared at one factor":
        ax = fig.add_subplot(111)
        style_axes(ax)
        models = store.explained_models()
        features = None
        series = []
        for idx, m in enumerate(models):
            matrix = store.importance(m, column)
            if matrix.empty or factor not in matrix.columns:
                continue
            if features is None:
                features = list(matrix.index)
            series.append((m, idx, matrix[factor].reindex(features)))
        if not series:
            message_plot(fig, "No explanation data for this factor.")
            return
        width = 0.8 / len(series)
        x = np.arange(len(features))
        for k, (m, idx, values) in enumerate(series):
            ax.bar(x + k * width - 0.4 + width / 2, values.to_numpy(),
                   width=width, color=model_color(m, idx),
                   label=model_label(m))
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha="right")
        ax.set_ylabel("share of total importance")
        ax.set_title(f"Do the models agree on the important sensors? "
                     f"({factor_label(factor)}, {target})", fontsize=10)
        style_legend(ax.legend(fontsize=8))

    else:  # drift
        ax = fig.add_subplot(111)
        style_axes(ax)
        drawn = False
        all_factors = store.ml_factors()
        for idx, m in enumerate(store.explained_models()):
            drift = store.explanation_drift(m, column)
            if drift.empty:
                continue
            x_pos = [all_factors.index(f) for f in drift.index
                     if f in all_factors]
            values = [drift[f] for f in drift.index if f in all_factors]
            ax.plot(x_pos, values, marker="o", lw=2,
                    color=model_color(m, idx), label=model_label(m))
            drawn = True
        if not drawn:
            message_plot(fig, "Need explanations for factor 1 to measure "
                              "drift.")
            return
        ax.set_xticks(range(len(all_factors)))
        ax.set_xticklabels([factor_label(f) for f in all_factors])
        ax.axhline(1.0, color=GRID, lw=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Spearman ρ of sensor ranking vs. ×1")
        ax.set_xlabel("aggregation factor")
        ax.set_title(f"Explanation drift: does aggregation change *why* an "
                     f"anomaly is flagged? ({target})", fontsize=10)
        style_legend(ax.legend(fontsize=8))

    fig.tight_layout()


# ============================================================
# GUI
# ============================================================

class PlotPanel(ctk.CTkFrame):
    """A matplotlib Figure with toolbar inside a customtkinter frame."""

    def __init__(self, master):
        super().__init__(master, fg_color=PANEL)
        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        toolbar_frame = ctk.CTkFrame(self, fg_color=PANEL)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw(self) -> None:
        self.canvas.draw_idle()


def labelled(parent, text: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=text, anchor="w")


class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("SKAB - aggregation vs. anomaly detection")
        self.geometry("1400x900")
        self.minsize(1100, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.store: DataStore | None = None

        self._build_topbar()

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        for name in ("Data explorer", "Label distortion",
                     "Model performance", "Per dataset",
                     "Detection view", "Explainability"):
            self.tabs.add(name)

        self._build_explorer_tab(self.tabs.tab("Data explorer"))
        self._build_distortion_tab(self.tabs.tab("Label distortion"))
        self._build_performance_tab(self.tabs.tab("Model performance"))
        self._build_per_dataset_tab(self.tabs.tab("Per dataset"))
        self._build_detection_tab(self.tabs.tab("Detection view"))
        self._build_explain_tab(self.tabs.tab("Explainability"))

        self.reload()

    # ------------------------------------------------------------------
    # top bar
    # ------------------------------------------------------------------

    def _build_topbar(self) -> None:
        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=10, pady=10)

        self.agg_var = ctk.StringVar(value=str(DEFAULT_AGG_DIR))
        self.ml_var = ctk.StringVar(value=str(DEFAULT_ML_DIR))

        labelled(bar, "Aggregated data:").grid(row=0, column=0, padx=(10, 4),
                                               pady=4, sticky="w")
        ctk.CTkEntry(bar, textvariable=self.agg_var, width=380).grid(
            row=0, column=1, padx=4, pady=4)
        ctk.CTkButton(bar, text="Browse", width=70,
                      command=lambda: self._browse(self.agg_var)).grid(
            row=0, column=2, padx=4)

        labelled(bar, "ML results:").grid(row=0, column=3, padx=(16, 4),
                                          sticky="w")
        ctk.CTkEntry(bar, textvariable=self.ml_var, width=380).grid(
            row=0, column=4, padx=4)
        ctk.CTkButton(bar, text="Browse", width=70,
                      command=lambda: self._browse(self.ml_var)).grid(
            row=0, column=5, padx=4)

        ctk.CTkButton(bar, text="Load / reload", width=110,
                      command=self.reload).grid(row=0, column=6, padx=(16, 10))

        self.status = ctk.CTkLabel(bar, text="", anchor="w",
                                   text_color="#9aa")
        self.status.grid(row=1, column=0, columnspan=7, padx=10, pady=(0, 4),
                         sticky="w")

    def _browse(self, var) -> None:
        folder = filedialog.askdirectory(initialdir=var.get() or ".")
        if folder:
            var.set(folder)

    # ------------------------------------------------------------------
    # tab builders
    # ------------------------------------------------------------------

    def _controls_and_plot(self, tab, width: int = 250):
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        controls = ctk.CTkScrollableFrame(tab, width=width)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 8), pady=4)
        panel = PlotPanel(tab)
        panel.grid(row=0, column=1, sticky="nsew", pady=4)
        return controls, panel

    # ---- 1. data explorer ----

    def _build_explorer_tab(self, tab) -> None:
        controls, self.explorer_plot = self._controls_and_plot(tab)

        labelled(controls, "Dataset").pack(anchor="w")
        self.ex_dataset = ctk.CTkOptionMenu(controls, values=["-"],
                                            command=self._explorer_dataset_changed)
        self.ex_dataset.pack(fill="x", pady=(0, 8))

        labelled(controls, "Sensor").pack(anchor="w")
        self.ex_sensor = ctk.CTkOptionMenu(controls, values=["-"],
                                           command=lambda _: self.refresh_explorer())
        self.ex_sensor.pack(fill="x", pady=(0, 8))

        labelled(controls, "Factors to overlay").pack(anchor="w")
        self.ex_factor_frame = ctk.CTkFrame(controls, fg_color="transparent")
        self.ex_factor_frame.pack(fill="x")
        self.ex_factor_vars: dict[int, ctk.BooleanVar] = {}

    def _explorer_dataset_changed(self, _=None) -> None:
        self._fill_sensor_menu()
        self.refresh_explorer()

    def _fill_sensor_menu(self) -> None:
        if not self.store:
            return
        sensors = self.store.sensors(self.ex_dataset.get()) or ["-"]
        self.ex_sensor.configure(values=sensors)
        if self.ex_sensor.get() not in sensors:
            self.ex_sensor.set(sensors[0])

    # ---- 2. label distortion ----

    def _build_distortion_tab(self, tab) -> None:
        controls, self.distortion_plot = self._controls_and_plot(tab)

        labelled(controls, "Quantity").pack(anchor="w")
        self.dist_metric = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_distortion())
        self.dist_metric.pack(fill="x", pady=(0, 8))

        labelled(controls, "View").pack(anchor="w")
        self.dist_view = ctk.CTkSegmentedButton(
            controls, values=["Mean per factor", "Heatmap"],
            command=lambda _: self.refresh_distortion())
        self.dist_view.set("Mean per factor")
        self.dist_view.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            controls, justify="left", wraplength=230, text_color="#9aa",
            text=("Computed by skab_aggregate.py.\n\n"
                  "rows_covered_per_true_anomaly > 1 means normal rows are "
                  "relabelled anomalous ('any' mode).\n\n"
                  "recall < 1 or anomaly_disappears_* means labels are lost "
                  "('majority' mode).")).pack(anchor="w", pady=10)

    # ---- 3. performance ----

    def _build_performance_tab(self, tab) -> None:
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkScrollableFrame(tab, width=250)
        controls.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 8),
                      pady=4)

        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=1)

        self.perf_plot = PlotPanel(right)
        self.perf_plot.grid(row=0, column=0, sticky="nsew", pady=(4, 4))

        self.perf_table_frame = ctk.CTkFrame(right)
        self.perf_table_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 4))

        labelled(controls, "Evaluation level").pack(anchor="w")
        self.perf_level = ctk.CTkOptionMenu(
            controls, values=list(LEVELS),
            command=lambda _: self.refresh_performance())
        self.perf_level.pack(fill="x", pady=(0, 8))

        labelled(controls, "Metric").pack(anchor="w")
        self.perf_metric = ctk.CTkOptionMenu(
            controls, values=[v[0] for v in PERFORMANCE_METRICS.values()],
            command=lambda _: self.refresh_performance())
        self.perf_metric.pack(fill="x", pady=(0, 8))

        self.perf_relative = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(controls, text="relative to ×1",
                        variable=self.perf_relative,
                        command=self.refresh_performance).pack(anchor="w",
                                                               pady=2)
        self.perf_band = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(controls, text="show ±1 std over datasets",
                        variable=self.perf_band,
                        command=self.refresh_performance).pack(anchor="w",
                                                               pady=2)

        labelled(controls, "Models").pack(anchor="w", pady=(10, 0))
        self.perf_model_frame = ctk.CTkFrame(controls, fg_color="transparent")
        self.perf_model_frame.pack(fill="x")
        self.perf_model_vars: dict[str, ctk.BooleanVar] = {}

        ctk.CTkLabel(
            controls, justify="left", wraplength=230, text_color="#9aa",
            text=("Raw resolution: every block prediction is stretched back "
                  "over the raw rows it covers and judged against the "
                  "ORIGINAL labels - the fair way to compare factors.\n\n"
                  "PR-AUC / ROC-AUC / best F1 do not depend on the alarm "
                  "threshold; F1, FAR, MAR do.\n\n"
                  "XGBoost is supervised (dashed) - a reference, not a "
                  "competitor.")).pack(anchor="w", pady=10)

    # ---- 4. per dataset ----

    def _build_per_dataset_tab(self, tab) -> None:
        controls, self.perdata_plot = self._controls_and_plot(tab)

        labelled(controls, "Evaluation level").pack(anchor="w")
        self.pd_level = ctk.CTkOptionMenu(
            controls, values=list(LEVELS),
            command=lambda _: self.refresh_per_dataset())
        self.pd_level.pack(fill="x", pady=(0, 8))

        labelled(controls, "Model").pack(anchor="w")
        self.pd_model = ctk.CTkOptionMenu(
            controls, values=["-"],
            command=lambda _: self.refresh_per_dataset())
        self.pd_model.pack(fill="x", pady=(0, 8))

        labelled(controls, "Metric").pack(anchor="w")
        self.pd_metric = ctk.CTkOptionMenu(
            controls, values=[v[0] for v in PERFORMANCE_METRICS.values()],
            command=lambda _: self.refresh_per_dataset())
        self.pd_metric.pack(fill="x", pady=(0, 8))

    # ---- 5. detection view ----

    def _build_detection_tab(self, tab) -> None:
        controls, self.detect_plot = self._controls_and_plot(tab)

        labelled(controls, "Dataset").pack(anchor="w")
        self.dt_dataset = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_detection())
        self.dt_dataset.pack(fill="x", pady=(0, 8))

        labelled(controls, "Aggregation factor").pack(anchor="w")
        self.dt_factor = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_detection())
        self.dt_factor.pack(fill="x", pady=(0, 8))

        labelled(controls, "Model").pack(anchor="w")
        self.dt_model = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_detection())
        self.dt_model.pack(fill="x", pady=(0, 8))

        labelled(controls, "Background sensor").pack(anchor="w")
        self.dt_sensor = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_detection())
        self.dt_sensor.pack(fill="x", pady=(0, 8))

        labelled(controls, "Metrics shown for level").pack(anchor="w")
        self.dt_level = ctk.CTkOptionMenu(
            controls, values=list(LEVELS),
            command=lambda _: self.refresh_detection())
        self.dt_level.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            controls, justify="left", wraplength=230, text_color="#9aa",
            text=("Red = true anomaly (aggregated label). Orange = alarm of "
                  "the model. The dashed line is its alarm threshold.\n\n"
                  "Step through the factors to watch the score curve flatten "
                  "and the alarms arrive later.")).pack(anchor="w", pady=10)

    # ---- 6. explainability ----

    def _build_explain_tab(self, tab) -> None:
        controls, self.explain_plot = self._controls_and_plot(tab)

        labelled(controls, "View").pack(anchor="w")
        self.ex_view = ctk.CTkOptionMenu(
            controls,
            values=["Feature × factor heatmap",
                    "Models compared at one factor",
                    "Explanation drift vs. ×1"],
            command=lambda _: self.refresh_explain())
        self.ex_view.pack(fill="x", pady=(0, 8))

        labelled(controls, "Model (heatmap view)").pack(anchor="w")
        self.ex_model = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_explain())
        self.ex_model.pack(fill="x", pady=(0, 8))

        labelled(controls, "Factor (comparison view)").pack(anchor="w")
        self.ex_factor = ctk.CTkOptionMenu(
            controls, values=["-"], command=lambda _: self.refresh_explain())
        self.ex_factor.pack(fill="x", pady=(0, 8))

        labelled(controls, "Explain").pack(anchor="w")
        self.ex_target = ctk.CTkSegmentedButton(
            controls, values=["anomalous rows", "normal rows"],
            command=lambda _: self.refresh_explain())
        self.ex_target.set("anomalous rows")
        self.ex_target.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            controls, justify="left", wraplength=230, text_color="#9aa",
            text=("XGBoost & Isolation Forest: mean |SHAP| (TreeExplainer).\n"
                  "PCA & Autoencoder: share of the squared reconstruction "
                  "error per sensor.\n\n"
                  "All values are normalised to sum to 1 per model, so they "
                  "can be compared across methods.\n\n"
                  "Drift = Spearman correlation of the sensor ranking with "
                  "the ranking at ×1. A drop means aggregation changes the "
                  "reason why a detector fires.")).pack(anchor="w", pady=10)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def reload(self) -> None:
        self.store = DataStore(Path(self.agg_var.get()),
                               Path(self.ml_var.get()))
        s = self.store

        parts = []
        parts.append(f"{len(s.datasets())} datasets, factors "
                     f"{[factor_label(f) for f in s.factors()]}"
                     if s.has_aggregated else "aggregated data: NOT FOUND")
        parts.append(f"models: {', '.join(model_label(m) for m in s.models())}"
                     if s.has_ml else "ML results: NOT FOUND "
                                      "(run skab_ml_experiment.py)")
        self.status.configure(text="   |   ".join(parts))

        datasets = s.datasets() or ["-"]
        factors = [str(f) for f in s.ml_factors()] or [
            str(f) for f in s.factors()] or ["-"]
        models = s.models() or ["-"]
        model_names = [model_label(m) for m in models]
        self._model_by_label = {model_label(m): m for m in models}

        # explorer
        self.ex_dataset.configure(values=datasets)
        self.ex_dataset.set(datasets[0])
        self._fill_sensor_menu()

        for child in self.ex_factor_frame.winfo_children():
            child.destroy()
        self.ex_factor_vars = {}
        for f in s.factors():
            var = ctk.BooleanVar(value=f in (1, 10, 50, 100)
                                 or f == s.factors()[-1])
            self.ex_factor_vars[f] = var
            ctk.CTkCheckBox(self.ex_factor_frame, text=factor_label(f),
                            variable=var,
                            command=self.refresh_explorer).pack(anchor="w",
                                                                pady=1)

        # distortion
        metric_options = []
        if s.label_summary is not None:
            skip = {"dataset", "dataset_path", "factor"}
            metric_options = [
                c for c in s.label_summary.columns
                if c not in skip
                and pd.api.types.is_numeric_dtype(s.label_summary[c])
            ]
        preferred = ["rows_covered_per_true_anomaly",
                     "anomaly_recall_after_label_aggregation",
                     "anomalous_block_purity", "aggregated_anomaly_rate",
                     "normal_rows_covered_by_anomalous_blocks",
                     "anomaly_disappears_completely"]
        metric_options = ([m for m in preferred if m in metric_options]
                          + [m for m in metric_options if m not in preferred]
                          ) or ["-"]
        self.dist_metric.configure(values=metric_options)
        self.dist_metric.set(metric_options[0])

        # performance
        for child in self.perf_model_frame.winfo_children():
            child.destroy()
        self.perf_model_vars = {}
        for m in models:
            if m == "-":
                continue
            var = ctk.BooleanVar(value=True)
            self.perf_model_vars[m] = var
            ctk.CTkCheckBox(self.perf_model_frame, text=model_label(m),
                            variable=var,
                            command=self.refresh_performance).pack(anchor="w",
                                                                   pady=1)

        # per dataset
        self.pd_model.configure(values=model_names)
        self.pd_model.set(model_names[0])

        # detection
        self.dt_dataset.configure(values=datasets)
        self.dt_dataset.set(datasets[0])
        self.dt_factor.configure(values=factors)
        self.dt_factor.set(factors[min(2, len(factors) - 1)])
        self.dt_model.configure(values=model_names)
        self.dt_model.set(model_names[0])
        sensors = s.sensors(datasets[0]) or ["-"]
        self.dt_sensor.configure(values=sensors)
        self.dt_sensor.set(sensors[0])

        # explainability
        explained = [model_label(m) for m in s.explained_models()] or ["-"]
        self.ex_model.configure(values=explained)
        self.ex_model.set(explained[-1] if "XGBoost (supervised)" in explained
                          else explained[0])
        if "XGBoost (supervised)" in explained:
            self.ex_model.set("XGBoost (supervised)")
        self.ex_factor.configure(values=factors)
        self.ex_factor.set(factors[min(2, len(factors) - 1)])

        self.refresh_all()

    def refresh_all(self) -> None:
        self.refresh_explorer()
        self.refresh_distortion()
        self.refresh_performance()
        self.refresh_per_dataset()
        self.refresh_detection()
        self.refresh_explain()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _model_from_label(self, text: str) -> str:
        return getattr(self, "_model_by_label", {}).get(text, text)

    @staticmethod
    def _metric_from_label(text: str) -> str:
        for key, (label, _) in PERFORMANCE_METRICS.items():
            if label == text:
                return key
        return "pr_auc"

    @staticmethod
    def _int_or_none(text: str) -> int | None:
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # refreshers
    # ------------------------------------------------------------------

    def refresh_explorer(self) -> None:
        if not self.store:
            return
        factors = sorted(f for f, v in self.ex_factor_vars.items() if v.get())
        plot_data_explorer(self.explorer_plot.figure, self.store,
                           self.ex_dataset.get(), self.ex_sensor.get(),
                           factors)
        self.explorer_plot.draw()

    def refresh_distortion(self) -> None:
        if not self.store:
            return
        plot_label_distortion(self.distortion_plot.figure, self.store,
                              self.dist_metric.get(), self.dist_view.get())
        self.distortion_plot.draw()

    def refresh_performance(self) -> None:
        if not self.store:
            return
        level = LEVELS[self.perf_level.get()]
        metric = self._metric_from_label(self.perf_metric.get())
        models = [m for m, v in self.perf_model_vars.items() if v.get()]

        plot_performance(self.perf_plot.figure, self.store, level, metric,
                         models, self.perf_relative.get(),
                         self.perf_band.get())
        self.perf_plot.draw()
        self._fill_performance_table(level, metric, models)

    def _fill_performance_table(self, level, metric, models) -> None:
        for child in self.perf_table_frame.winfo_children():
            child.destroy()

        if not self.store or not self.store.has_ml:
            return

        factors = self.store.ml_factors()

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=PANEL, foreground=FG,
                        fieldbackground=PANEL, rowheight=22, borderwidth=0)
        style.configure("Treeview.Heading", background="#3a3a3a",
                        foreground=FG, relief="flat")
        style.map("Treeview", background=[("selected", "#1f538d")])

        columns = ["model"] + [factor_label(f) for f in factors]
        tree = ttk.Treeview(self.perf_table_frame, columns=columns,
                            show="headings", height=max(3, len(models) + 1))

        for col in columns:
            tree.heading(col, text="" if col == "model" else col)
            tree.column(col, width=170 if col == "model" else 80,
                        anchor="w" if col == "model" else "e")

        for model in self.store.models():
            if model not in models:
                continue
            curve = self.store.curve(level, model, metric).set_index("factor")
            row = [model_label(model)]
            for f in factors:
                value = curve["mean"].get(f, math.nan)
                std = curve["std"].get(f, math.nan)
                if not np.isfinite(value):
                    row.append("-")
                elif np.isfinite(std):
                    row.append(f"{value:.3f} ± {std:.2f}")
                else:
                    row.append(f"{value:.3f}")
            tree.insert("", "end", values=row)

        tree.pack(fill="both", expand=True)

    def refresh_per_dataset(self) -> None:
        if not self.store:
            return
        plot_per_dataset(
            self.perdata_plot.figure, self.store,
            LEVELS[self.pd_level.get()],
            self._model_from_label(self.pd_model.get()),
            self._metric_from_label(self.pd_metric.get()))
        self.perdata_plot.draw()

    def refresh_detection(self) -> None:
        if not self.store:
            return
        factor = self._int_or_none(self.dt_factor.get())
        if factor is None:
            message_plot(self.detect_plot.figure, "No ML results loaded.")
            self.detect_plot.draw()
            return

        sensors = self.store.sensors(self.dt_dataset.get()) or ["-"]
        self.dt_sensor.configure(values=sensors)
        if self.dt_sensor.get() not in sensors:
            self.dt_sensor.set(sensors[0])

        plot_detection(
            self.detect_plot.figure, self.store, self.dt_dataset.get(),
            factor, self._model_from_label(self.dt_model.get()),
            self.dt_sensor.get(), LEVELS[self.dt_level.get()])
        self.detect_plot.draw()

    def refresh_explain(self) -> None:
        if not self.store:
            return
        factor = self._int_or_none(self.ex_factor.get()) or 1
        plot_explainability(
            self.explain_plot.figure, self.store, self.ex_view.get(),
            self._model_from_label(self.ex_model.get()), factor,
            self.ex_target.get())
        self.explain_plot.draw()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
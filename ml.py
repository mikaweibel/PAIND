"""
SKAB aggregation vs. anomaly-detection accuracy
================================================

Input : the output folder of skab_aggregate.py   (csv/factor_N/<dataset>.csv)
Output: SKAB_ml_results_<mode>/                  (read by skab_gui.py)

PROTOCOL
--------
* Datasets are split into N_FOLDS groups of whole files (no file is ever in
  training and test at the same time).
* Unsupervised detectors (Isolation Forest, PCA-SPE, Autoencoder, LOF) are
  fitted ONLY on normal rows (label 0) of the training files. Labels are used
  for nothing else but picking those rows.
* The alarm threshold of the unsupervised detectors is label-free: the
  THRESHOLD_QUANTILE of the scores on normal rows of held-out *calibration*
  files (a slice of the training files that the detector never saw).
* XGBoost is the SUPERVISED reference ("how good could it be with labels?").
  It is explained with SHAP (TreeExplainer). Isolation Forest is explained with
  SHAP too; PCA / Autoencoder via per-sensor reconstruction error.

EVALUATION (two levels, both reported)
--------------------------------------
* "native": predictions vs. the aggregated labels of that factor.
* "raw"   : every block prediction is stretched back over the raw rows it
            covers and compared with the ORIGINAL labels (factor 1). This is
            the fair level for comparing factors, because the aggregated
            labels themselves change with the factor.

Install:  pip install numpy pandas scikit-learn xgboost shap
"""

from __future__ import annotations

import json
import math
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

# Must match the ANOMALY_MODE used in skab_aggregate.py.
ANOMALY_MODE = "any"

AGG_DIR = Path(f"SKAB_aggregated_output_{ANOMALY_MODE}")
ML_OUTPUT_DIR = Path(f"SKAB_ml_results_{ANOMALY_MODE}")

CSV_SEP = ";"

# Any subset of:
#   "isolation_forest", "pca_spe", "autoencoder", "lof", "xgboost_shap"
MODELS = [
    "isolation_forest",
    "pca_spe",
    "autoencoder",
    "lof",
    "xgboost_shap",
]

# None = every factor found in AGG_DIR/csv (factor 1 is always required).
FACTORS: list[int] | None = None

N_FOLDS = 5
CALIBRATION_FRACTION = 0.20      # share of training files used to set thresholds
THRESHOLD_QUANTILE = 0.99        # unsupervised alarm threshold
SUPERVISED_THRESHOLD = 0.5       # XGBoost probability threshold

MAX_TRAIN_ROWS = 4000            # random subsample cap per fit
MIN_TRAIN_ROWS = 40              # fewer rows -> that run is skipped
RANDOM_SEED = 42

# Model hyper-parameters
PCA_VARIANCE = 0.95
LOF_NEIGHBORS = 20
IFOREST_TREES = 200
XGB_TREES = 200

# Explainability
COMPUTE_EXPLANATIONS = True
EXPLAIN_MAX_ROWS = 800           # rows per fold used for SHAP / residual shares


# ============================================================
# CONSTANTS
# ============================================================

DATETIME_COLUMN = "datetime"
ANOMALY_COLUMN = "anomaly"
CHANGEPOINT_COLUMN = "changepoint"
NON_FEATURE_COLUMNS = {DATETIME_COLUMN, ANOMALY_COLUMN, CHANGEPOINT_COLUMN}

METRIC_COLUMNS = [
    "roc_auc",
    "pr_auc",
    "best_f1",
    "f1",
    "precision",
    "recall",
    "far",
    "mar",
    "detection_delay_rows",
    "train_rows",
    "fit_time_s",
]

MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "pca_spe": "PCA (SPE)",
    "autoencoder": "Autoencoder",
    "lof": "LOF",
    "xgboost_shap": "XGBoost (supervised)",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# MODELS
#   score(X): higher = more anomalous
#   explain(X): (n, n_features) non-negative contributions, or None
# ============================================================

class BaseModel:
    supervised = False

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, X, y=None):
        raise NotImplementedError

    def score(self, X) -> np.ndarray:
        raise NotImplementedError

    def explain(self, X) -> np.ndarray | None:
        return None


class IsolationForestModel(BaseModel):
    def fit(self, X, y=None):
        self.model = IsolationForest(
            n_estimators=IFOREST_TREES,
            random_state=self.seed,
            n_jobs=-1,
        ).fit(X)
        return self

    def score(self, X):
        # score_samples is "higher = more normal" -> flip the sign.
        return -self.model.score_samples(X)

    def explain(self, X):
        import shap

        values = shap.TreeExplainer(self.model).shap_values(
            X, check_additivity=False
        )
        return np.abs(_first_output(values))


class PCAModel(BaseModel):
    """PCA reconstruction error (squared prediction error, 'Q' statistic)."""

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        d = Z.shape[1]

        full = PCA().fit(Z)
        cumulative = np.cumsum(full.explained_variance_ratio_)
        k = int(np.searchsorted(cumulative, PCA_VARIANCE) + 1)
        k = max(1, min(k, d - 1))           # k == d would give zero error

        self.pca = PCA(n_components=k).fit(Z)
        self.n_components = k
        return self

    def _residual_sq(self, X):
        Z = self.scaler.transform(X)
        recon = self.pca.inverse_transform(self.pca.transform(Z))
        return (Z - recon) ** 2

    def score(self, X):
        return self._residual_sq(X).sum(axis=1)

    def explain(self, X):
        return self._residual_sq(X)


class AutoencoderModel(BaseModel):
    """Small bottleneck autoencoder (scikit-learn MLP, no torch needed)."""

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)

        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 4, 16),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.003,
            batch_size=min(128, len(Z)),
            max_iter=300,
            tol=1e-5,
            n_iter_no_change=20,
            random_state=self.seed,
        )
        self.model.fit(Z, Z)
        return self

    def _residual_sq(self, X):
        Z = self.scaler.transform(X)
        return (Z - self.model.predict(Z)) ** 2

    def score(self, X):
        return self._residual_sq(X).sum(axis=1)

    def explain(self, X):
        return self._residual_sq(X)


class LOFModel(BaseModel):
    """Local Outlier Factor in novelty mode (fit on normal data only)."""

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        k = max(2, min(LOF_NEIGHBORS, len(Z) - 1))

        self.model = LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,
            n_jobs=-1,
        ).fit(Z)
        return self

    def score(self, X):
        return -self.model.score_samples(self.scaler.transform(X))


class XGBoostModel(BaseModel):
    """Supervised reference model, explained with SHAP."""

    supervised = True

    def fit(self, X, y=None):
        import xgboost as xgb

        y = np.asarray(y).astype(int)
        positives = int(y.sum())
        negatives = len(y) - positives

        self.model = xgb.XGBClassifier(
            n_estimators=XGB_TREES,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=negatives / max(positives, 1),
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def explain(self, X):
        import shap

        values = shap.TreeExplainer(self.model).shap_values(
            X, check_additivity=False
        )
        return np.abs(_first_output(values))


MODEL_CLASSES = {
    "isolation_forest": IsolationForestModel,
    "pca_spe": PCAModel,
    "autoencoder": AutoencoderModel,
    "lof": LOFModel,
    "xgboost_shap": XGBoostModel,
}


def _first_output(values) -> np.ndarray:
    """Normalise the different shapes shap.TreeExplainer can return."""

    if isinstance(values, list):
        values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[..., -1]

    return values


# ============================================================
# DATA
# ============================================================

def discover_factors(agg_dir: Path) -> list[int]:
    root = agg_dir / "csv"

    if not root.exists():
        raise FileNotFoundError(
            f"{root.resolve()} not found. Run skab_aggregate.py first "
            f"(and check ANOMALY_MODE)."
        )

    factors = []

    for path in root.glob("factor_*"):
        if path.is_dir():
            try:
                factors.append(int(path.name.split("_", 1)[1]))
            except ValueError:
                pass

    return sorted(factors)


def load_factor(agg_dir: Path, factor: int) -> dict[str, pd.DataFrame]:
    """All aggregated datasets of one factor, keyed by dataset id."""

    folder = agg_dir / "csv" / f"factor_{factor}"
    data: dict[str, pd.DataFrame] = {}

    for path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(path, sep=CSV_SEP, parse_dates=[DATETIME_COLUMN])

        if len(df) == 0:
            continue

        data[path.stem] = df

    return data


def make_folds(ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """Deterministic random split of dataset ids into folds."""

    ids = sorted(ids)
    n_folds = max(2, min(n_folds, len(ids)))

    permutation = np.random.default_rng(seed).permutation(len(ids))

    return [
        [ids[i] for i in chunk]
        for chunk in np.array_split(permutation, n_folds)
    ]


def split_calibration(
    train_ids: list[str],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Split training files into (fit files, calibration files)."""

    ids = sorted(train_ids)

    if len(ids) < 2:
        return ids, []

    k = int(round(len(ids) * fraction))
    k = max(1, min(k, len(ids) - 1))

    calibration = set(rng.choice(ids, size=k, replace=False).tolist())

    fit = [i for i in ids if i not in calibration]

    return fit, sorted(calibration)


def stack_rows(
    data: dict[str, pd.DataFrame],
    ids: list[str],
    features: list[str],
    normal_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []

    for ds in ids:
        df = data[ds]

        if normal_only:
            df = df[df[ANOMALY_COLUMN] == 0]

        X_parts.append(df[features].to_numpy(float))
        y_parts.append(df[ANOMALY_COLUMN].to_numpy(int))

    if not X_parts:
        return np.empty((0, len(features))), np.empty((0,), dtype=int)

    return np.vstack(X_parts), np.concatenate(y_parts)


def subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, y

    idx = np.sort(rng.choice(len(X), size=max_rows, replace=False))

    return X[idx], y[idx]


def sanitize(scores) -> np.ndarray:
    """Replace NaN/inf scores by the largest finite score."""

    s = np.asarray(scores, dtype=float).copy()
    bad = ~np.isfinite(s)

    if bad.any():
        s[bad] = s[~bad].max() if (~bad).any() else 0.0

    return s


# ============================================================
# METRICS
# ============================================================

def binary_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    """Threshold-free (ROC/PR-AUC, best F1) and thresholded (F1/FAR/MAR)."""

    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    positives = tp + fn
    negatives = fp + tn

    out: dict[str, float] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_eval_rows": len(y_true),
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else math.nan
        ),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else math.nan,
        "recall": tp / positives if positives > 0 else math.nan,
        # SKAB convention: FAR = false-positive rate, MAR = miss rate.
        "far": fp / negatives if negatives > 0 else math.nan,
        "mar": fn / positives if positives > 0 else math.nan,
        "roc_auc": math.nan,
        "pr_auc": math.nan,
        "best_f1": math.nan,
    }

    if positives > 0 and negatives > 0:

        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))

        precision, recall, _ = precision_recall_curve(y_true, score)
        denom = precision + recall

        f_values = np.divide(
            2 * precision * recall,
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )

        out["best_f1"] = float(f_values.max())

    return out


def detection_delay_rows(
    y_raw: np.ndarray,
    block_pred: np.ndarray,
    factor: int,
) -> float:
    """
    Raw rows between the first true anomalous row and the first alarm that is
    *available* at or after it. A block decision is only available once the
    block is complete, so the delay can never be below ~factor/2 on average.
    NaN if there is no anomaly or it is never detected.
    """

    anomalous = np.flatnonzero(y_raw == 1)

    if anomalous.size == 0:
        return math.nan

    onset = int(anomalous[0])

    decision_row = (np.arange(len(block_pred)) + 1) * factor - 1

    candidates = np.flatnonzero(
        (block_pred == 1) & (decision_row >= onset)
    )

    if candidates.size == 0:
        return math.nan

    return float(decision_row[candidates[0]] - onset)


def importance_share(values: np.ndarray) -> np.ndarray | None:
    """Mean contribution per feature, normalised to sum to 1."""

    if values is None or len(values) == 0:
        return None

    mean = values.mean(axis=0)
    total = mean.sum()

    if not np.isfinite(total) or total <= 0:
        return None

    return mean / total


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    started = time.perf_counter()

    unknown = [m for m in MODELS if m not in MODEL_CLASSES]

    if unknown:
        raise ValueError(f"Unknown model(s) in MODELS: {unknown}")

    if "xgboost_shap" in MODELS or COMPUTE_EXPLANATIONS:
        try:
            import shap  # noqa: F401
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"{exc}\nInstall with:  pip install xgboost shap"
            ) from exc

    available = discover_factors(AGG_DIR)

    factors = (
        [f for f in available if f in FACTORS]
        if FACTORS is not None
        else available
    )

    if 1 not in factors:
        raise ValueError(
            "Factor 1 is required (it provides the raw labels for the "
            "raw-resolution evaluation)."
        )

    raw = load_factor(AGG_DIR, 1)

    if len(raw) < 3:
        raise ValueError(f"Too few datasets found: {len(raw)}")

    ids = sorted(raw)

    first = raw[ids[0]]

    features = [
        c for c in first.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    for ds in ids:
        missing = [c for c in features if c not in raw[ds].columns]
        if missing:
            raise ValueError(f"{ds}: missing feature columns {missing}")

    raw_labels = {
        ds: raw[ds][ANOMALY_COLUMN].to_numpy(int)
        for ds in ids
    }

    folds = make_folds(ids, N_FOLDS, RANDOM_SEED)

    print(f"Datasets : {len(ids)}")
    print(f"Features : {features}")
    print(f"Factors  : {factors}")
    print(f"Models   : {MODELS}")
    print(f"Folds    : {[len(f) for f in folds]} files each")
    print(f"Output   : {ML_OUTPUT_DIR.resolve()}\n")

    pred_root = ML_OUTPUT_DIR / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    per_file_rows: list[dict] = []
    explanation_rows: list[dict] = []
    skipped_runs: list[dict] = []

    for factor in factors:

        data = raw if factor == 1 else load_factor(AGG_DIR, factor)

        print(f"=== factor ×{factor}: {len(data)} datasets ===")

        pred_frames: dict[str, pd.DataFrame] = {}

        for k, fold_ids in enumerate(folds):

            fold_set = set(fold_ids)

            test_ids = [i for i in fold_ids if i in data]
            train_ids = [
                i for i in ids
                if i in data and i not in fold_set
            ]

            if not test_ids or len(train_ids) < 2:
                skipped_runs.append(
                    {
                        "factor": factor,
                        "fold": k,
                        "model": "*",
                        "reason": "not enough datasets at this factor",
                    }
                )
                continue

            rng = np.random.default_rng(
                RANDOM_SEED + 1000 * factor + k
            )

            fit_ids, calib_ids = split_calibration(
                train_ids,
                CALIBRATION_FRACTION,
                rng,
            )

            fit_normal, fit_normal_y = stack_rows(
                data, fit_ids, features, normal_only=True
            )
            fit_normal, _ = subsample(
                fit_normal, fit_normal_y, MAX_TRAIN_ROWS, rng
            )

            calib_normal, _ = stack_rows(
                data, calib_ids, features, normal_only=True
            )

            fit_X, fit_y = stack_rows(
                data, fit_ids, features, normal_only=False
            )
            fit_X, fit_y = subsample(fit_X, fit_y, MAX_TRAIN_ROWS, rng)

            test_X_all, test_y_all = stack_rows(
                data, test_ids, features, normal_only=False
            )

            for name in MODELS:

                model = MODEL_CLASSES[name](RANDOM_SEED + k)

                # ---------------- training ----------------

                if model.supervised:

                    if (
                        len(fit_y) < MIN_TRAIN_ROWS
                        or len(np.unique(fit_y)) < 2
                    ):
                        skipped_runs.append(
                            {
                                "factor": factor,
                                "fold": k,
                                "model": name,
                                "reason": (
                                    f"supervised training set unusable "
                                    f"({len(fit_y)} rows, "
                                    f"{len(np.unique(fit_y))} class(es))"
                                ),
                            }
                        )
                        continue

                    train_rows = len(fit_y)

                else:

                    if len(fit_normal) < MIN_TRAIN_ROWS:
                        skipped_runs.append(
                            {
                                "factor": factor,
                                "fold": k,
                                "model": name,
                                "reason": (
                                    f"only {len(fit_normal)} normal "
                                    f"training rows"
                                ),
                            }
                        )
                        continue

                    train_rows = len(fit_normal)

                t0 = time.perf_counter()

                try:
                    if model.supervised:
                        model.fit(fit_X, fit_y)
                    else:
                        model.fit(fit_normal)

                except Exception as exc:
                    skipped_runs.append(
                        {
                            "factor": factor,
                            "fold": k,
                            "model": name,
                            "reason": f"fit failed: {exc}",
                        }
                    )
                    print(f"  [fold {k}] {name}: FIT FAILED: {exc}")
                    continue

                fit_time = time.perf_counter() - t0

                # ---------------- threshold ----------------

                if model.supervised:

                    threshold = SUPERVISED_THRESHOLD

                else:

                    reference = (
                        calib_normal
                        if len(calib_normal) >= 5
                        else fit_normal
                    )

                    threshold = float(
                        np.quantile(
                            sanitize(model.score(reference)),
                            THRESHOLD_QUANTILE,
                        )
                    )

                # ---------------- test files ----------------

                for ds in test_ids:

                    df = data[ds]
                    n = len(df)

                    y_raw = raw_labels[ds][: n * factor]

                    if len(y_raw) != n * factor:
                        skipped_runs.append(
                            {
                                "factor": factor,
                                "fold": k,
                                "model": name,
                                "reason": (
                                    f"{ds}: raw label length does not "
                                    f"match {n} blocks × {factor}"
                                ),
                            }
                        )
                        continue

                    X = df[features].to_numpy(float)
                    y_native = df[ANOMALY_COLUMN].to_numpy(int)

                    score = sanitize(model.score(X))

                    if model.supervised:
                        pred = (score >= threshold).astype(int)
                    else:
                        pred = (score > threshold).astype(int)

                    common = {
                        "model": name,
                        "factor": factor,
                        "dataset": ds,
                        "fold": k,
                        "threshold": threshold,
                        "train_rows": train_rows,
                        "fit_time_s": fit_time,
                    }

                    # native level
                    native = binary_metrics(y_native, score, pred)
                    native["detection_delay_rows"] = math.nan

                    per_file_rows.append(
                        {"level": "native", **common, **native}
                    )

                    # raw level
                    raw_metrics = binary_metrics(
                        y_raw,
                        np.repeat(score, factor),
                        np.repeat(pred, factor),
                    )
                    raw_metrics["detection_delay_rows"] = (
                        detection_delay_rows(y_raw, pred, factor)
                    )

                    per_file_rows.append(
                        {"level": "raw", **common, **raw_metrics}
                    )

                    # stored for the GUI
                    frame = pred_frames.get(ds)

                    if frame is None:
                        frame = pd.DataFrame(
                            {
                                DATETIME_COLUMN: df[DATETIME_COLUMN],
                                ANOMALY_COLUMN: y_native,
                                "raw_anomaly_fraction": (
                                    y_raw.reshape(n, factor).mean(axis=1)
                                ),
                            }
                        )
                        pred_frames[ds] = frame

                    frame[f"{name}__score"] = score
                    frame[f"{name}__pred"] = pred
                    frame[f"{name}__threshold"] = threshold

                # ---------------- explanations ----------------

                if COMPUTE_EXPLANATIONS:

                    try:
                        anomalous_idx = np.flatnonzero(test_y_all == 1)
                        normal_idx = np.flatnonzero(test_y_all == 0)

                        half = EXPLAIN_MAX_ROWS // 2

                        if len(anomalous_idx) > half:
                            anomalous_idx = rng.choice(
                                anomalous_idx, half, replace=False
                            )

                        if len(normal_idx) > half:
                            normal_idx = rng.choice(
                                normal_idx, half, replace=False
                            )

                        if len(anomalous_idx) and len(normal_idx):

                            chosen = np.concatenate(
                                [anomalous_idx, normal_idx]
                            )

                            contributions = model.explain(
                                test_X_all[chosen]
                            )

                            if contributions is not None:

                                n_anom = len(anomalous_idx)

                                share_anom = importance_share(
                                    contributions[:n_anom]
                                )
                                share_norm = importance_share(
                                    contributions[n_anom:]
                                )

                                if (
                                    share_anom is not None
                                    and share_norm is not None
                                ):

                                    method = (
                                        "shap_tree"
                                        if name
                                        in (
                                            "xgboost_shap",
                                            "isolation_forest",
                                        )
                                        else "reconstruction_error"
                                    )

                                    for j, feat in enumerate(features):
                                        explanation_rows.append(
                                            {
                                                "model": name,
                                                "factor": factor,
                                                "fold": k,
                                                "feature": feat,
                                                "method": method,
                                                "importance_anomalous":
                                                    float(share_anom[j]),
                                                "importance_normal":
                                                    float(share_norm[j]),
                                            }
                                        )

                    except Exception as exc:
                        print(
                            f"  [fold {k}] {name}: explanation failed: "
                            f"{exc}"
                        )

                print(
                    f"  [fold {k}] {MODEL_LABELS[name]:<22} "
                    f"train={train_rows:>5}  thr={threshold:.4g}  "
                    f"fit={fit_time:.1f}s"
                )

        # ---------------- write predictions of this factor ----------------

        factor_dir = pred_root / f"factor_{factor}"
        factor_dir.mkdir(parents=True, exist_ok=True)

        for ds, frame in pred_frames.items():
            frame.to_csv(
                factor_dir / f"{ds}.csv",
                sep=CSV_SEP,
                index=False,
            )

    # ========================================================
    # RESULT FILES
    # ========================================================

    if not per_file_rows:
        raise RuntimeError(
            "No results were produced; see skipped_runs.csv."
        )

    per_file = pd.DataFrame(per_file_rows)

    per_file.to_csv(
        ML_OUTPUT_DIR / "results_per_file.csv",
        sep=CSV_SEP,
        index=False,
    )

    summary_rows = []

    for (level, model_name, factor), g in per_file.groupby(
        ["level", "model", "factor"],
        sort=True,
    ):

        row: dict[str, object] = {
            "level": level,
            "model": model_name,
            "factor": int(factor),
            "n_files": int(g["dataset"].nunique()),
        }

        for metric in METRIC_COLUMNS:
            values = g[metric].dropna()

            row[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else math.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
            row[f"{metric}_n"] = int(len(values))

        tp, fp, tn, fn = (
            int(g[c].sum()) for c in ("tp", "fp", "tn", "fn")
        )

        row["micro_f1"] = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else math.nan
        )
        row["micro_far"] = fp / (fp + tn) if (fp + tn) > 0 else math.nan
        row["micro_mar"] = fn / (fn + tp) if (fn + tp) > 0 else math.nan

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        ML_OUTPUT_DIR / "results_summary.csv",
        sep=CSV_SEP,
        index=False,
    )

    if explanation_rows:
        pd.DataFrame(explanation_rows).to_csv(
            ML_OUTPUT_DIR / "explanations.csv",
            sep=CSV_SEP,
            index=False,
        )

    pd.DataFrame(
        skipped_runs,
        columns=["factor", "fold", "model", "reason"],
    ).to_csv(
        ML_OUTPUT_DIR / "skipped_runs.csv",
        sep=CSV_SEP,
        index=False,
    )

    def _version(module_name: str) -> str:
        try:
            module = __import__(module_name)
            return getattr(module, "__version__", "?")
        except ImportError:
            return "not installed"

    run_info = {
        "anomaly_mode": ANOMALY_MODE,
        "agg_dir": str(AGG_DIR),
        "factors": factors,
        "models": MODELS,
        "features": features,
        "n_datasets": len(ids),
        "n_folds": len(folds),
        "folds": folds,
        "calibration_fraction": CALIBRATION_FRACTION,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "supervised_threshold": SUPERVISED_THRESHOLD,
        "max_train_rows": MAX_TRAIN_ROWS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "random_seed": RANDOM_SEED,
        "runtime_s": round(time.perf_counter() - started, 1),
        "python": platform.python_version(),
        "versions": {
            name: _version(name)
            for name in (
                "numpy",
                "pandas",
                "sklearn",
                "xgboost",
                "shap",
            )
        },
    }

    with open(
        ML_OUTPUT_DIR / "run_info.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(run_info, handle, indent=2)

    print("\nFinished.")
    print(f"Result rows     : {len(per_file)}")
    print(f"Skipped runs    : {len(skipped_runs)}")
    print(f"Runtime         : {run_info['runtime_s']} s")
    print(f"Results         : {ML_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
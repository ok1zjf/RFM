"""Multiclass classification metrics + ROC/PR/confusion-matrix plots."""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)

# Noteevery dataset's CI comes from one
# implementation at one replicate count/seed, whether or not it actually has
# a patient-clustering correction applied (see `resolve_groups`).
N_BOOT = 4000
BOOT_SEED = 20260921
MESSIDOR_PAIRS_CSV = "datasets/messidor2_pairs.csv"  # relative to the repo root


def _auroc_score(y_true, y_prob, num_classes, labels):
    """sklearn's roc_auc_score(..., multi_class="ovr") only accepts a 2D score
    array for 3+ classes; for true binary (num_classes == 2) it needs the plain
    1D positive-class score instead """
    if num_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    return roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro")


def _auprc_score(y_true, y_prob, num_classes):
    if num_classes == 2:
        return average_precision_score(y_true, y_prob[:, 1])
    return average_precision_score(np.eye(num_classes)[y_true], y_prob, average="macro")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict:
    """y_true: [N] int labels. y_prob: [N, num_classes] softmax probabilities."""
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(num_classes))
    return {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "auroc": _auroc_score(y_true, y_prob, num_classes, labels),
        "auprc": _auprc_score(y_true, y_prob, num_classes),
    }


def _papila_patient_groups(csv_path: str, ids) -> np.ndarray | None:
    """PAPILA filenames encode patient and laterality: RET002OS.jpg -> RET002."""
    lut = pd.read_csv(csv_path).set_index("id")["filename"]
    fn = pd.Series(ids).map(lut)
    pat = fn.str.extract(r"(RET\d+)", expand=False)
    return None if pat.isna().any() else pat.to_numpy()


def _messidor_pairs(repo_root: str) -> dict:
    """messidor2_pairs.csv lists one 'left;right' pair per patient. Returns
    {image basename -> patient key}. Ported from cluster_ci.py:_messidor_pairs."""
    raw = pd.read_csv(os.path.join(repo_root, MESSIDOR_PAIRS_CSV))
    col = raw.columns[0]
    out = {}
    for i, cell in enumerate(raw[col].astype(str)):
        for name in cell.split(";"):
            name = name.strip()
            if name:
                out[os.path.basename(name)] = f"pat{i:05d}"
    return out


def _messidor_patient_groups(csv_path: str, repo_root: str, ids) -> np.ndarray | None:
    """Ported from RFM_pub/results/cluster_ci.py:_groups_messidor."""
    lut = pd.read_csv(csv_path).set_index("id")["filename"]
    fn = pd.Series(ids).map(lut)
    if fn.isna().any():
        return None
    base = fn.map(os.path.basename)
    pairs = _messidor_pairs(repo_root)
    pat = base.map(pairs)
    # An image absent from the pairing table is its own patient rather than a
    # silent merge into one bucket.
    missing = int(pat.isna().sum())
    if missing:
        pat = pat.fillna(pd.Series([f"solo{i}" for i in range(len(pat))], index=pat.index))
    return pat.to_numpy()


def resolve_groups(csv_path: str, repo_root: str, ids) -> np.ndarray | None:
    """Patient-level group key per test row, for datasets where one patient
    contributes more than one image (PAPILA, Messidor-2: 2 images/patient).
    Returns None for every other dataset (IDRiD, GFID, APTOS publish no
    subject identifiers, so the image is the only available unit) -- callers
    then resample rows individually. """
    name = os.path.basename(csv_path).lower()
    if "papila" in name:
        return _papila_patient_groups(csv_path, ids)
    if "messidor2" in name:
        return _messidor_patient_groups(csv_path, repo_root, ids)
    return None


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_prob: np.ndarray, groups=None,
                  n_bootstrap: int = N_BOOT, ci: float = 0.95, seed: int = BOOT_SEED):
    """Percentile bootstrap CI for a scalar metric.

    `groups` (one key per row) resamples unique group keys with replacement,
    carrying all rows of a drawn key together, instead of resampling rows
    independently -- required whenever a group (e.g. a patient) contributes
    more than one correlated row, otherwise the interval is falsely narrow.
    `groups=None` (default) puts every row in its own group, which reduces to
    a plain row-level bootstrap. Also cover the ungrouped case in one implementation.

    Returns (lower, upper, std), or (nan, nan, nan) if too few resamples succeed
    (e.g. a class missing from a resample breaks a metric like AUROC).

    A resample that's missing a class (common for rare classes in a small test
    set, e.g. IDRiD) can make sklearn return NaN instead of raising — a NaN
    silently mixed into `scores` would poison every percentile/std computed
    from the list, so those draws are explicitly dropped, not just exceptions.
    """
    n = len(y_true)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    n_g = len(uniq)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=n_g)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        drawn = rng.integers(0, n_g, size=n_g)
        c = counts[drawn]
        total = int(c.sum())
        if total == 0:
            continue
        base = np.repeat(starts[drawn], c)
        within = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        idx = order[base + within]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = float(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue
        if not np.isnan(value):
            scores.append(value)
    if len(scores) < 10:
        return float("nan"), float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(scores, 100 * alpha)),
        float(np.percentile(scores, 100 * (1 - alpha))),
        float(np.std(scores)),
    )


def compute_metrics_with_ci(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int,
                             groups=None, n_bootstrap: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Full test-time metrics report matching the original test.csv columns:
    acc/auprc/auroc/balanced_acc/f1 each with a bootstrap CI (ci_lower/ci_upper/std),
    plus precision/recall (no CI) and a constant `mae=0.0` placeholder — the original
    codebase always reports mae=0 for categorical phenotypes (MAE is only ever
    populated for continuous/regression phenotypes).
    """
    labels = list(range(num_classes))

    def _acc(yt, yp):
        return accuracy_score(yt, yp.argmax(axis=1))

    def _balanced_acc(yt, yp):
        return balanced_accuracy_score(yt, yp.argmax(axis=1))

    def _f1(yt, yp):
        return f1_score(yt, yp.argmax(axis=1), labels=labels, average="macro", zero_division=0)

    def _auroc(yt, yp):
        return _auroc_score(yt, yp, num_classes, labels)

    def _auprc(yt, yp):
        return _auprc_score(yt, yp, num_classes)

    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "acc": _acc(y_true, y_prob),
        "auprc": _auprc(y_true, y_prob),
        "auroc": _auroc(y_true, y_prob),
        "balanced_acc": _balanced_acc(y_true, y_prob),
        "f1": _f1(y_true, y_prob),
        "mae": 0.0,
        "precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    for name, fn in [("acc", _acc), ("auprc", _auprc), ("auroc", _auroc),
                      ("balanced_acc", _balanced_acc), ("f1", _f1)]:
        lo, hi, std = bootstrap_ci(fn, y_true, y_prob, groups=groups, n_bootstrap=n_bootstrap, seed=seed)
        metrics[f"{name}_ci_lower"] = lo
        metrics[f"{name}_ci_upper"] = hi
        metrics[f"{name}_std"] = std
    return metrics


def _bootstrap_curve_band(curve_fn, y_true_binary, y_score, grid, groups=None,
                           n_bootstrap=N_BOOT, seed=BOOT_SEED):
    """Bootstrap a 95% CI band for a per-class (one-vs-rest) curve, interpolated
    onto a fixed x-axis `grid` so bootstrap draws can be averaged pointwise.

    `groups` (one key per row, see `bootstrap_ci`) resamples patients rather
    than rows when a patient contributes more than one correlated image
    (PAPILA, Messidor-2); `groups=None` resamples rows, unified on the same
    n_bootstrap/seed as everything else (RFM_pub/results/cluster_ci.py).

    Generalizes the original codebase's binary-only `ci_band` to each class of
    a multiclass one-vs-rest curve — the original never plots a multiclass band
    at all (`ci_band` there only takes effect when `num_classes == 2`), but
    every phenotype in this codebase (dr_grade, glaucoma_grade) is multiclass.

    curve_fn(y_true_b, y_score_b) -> (x, y) arrays, e.g. sklearn's roc_curve
    (x=fpr, y=tpr) or precision_recall_curve (x=recall, y=precision, decreasing).

    Returns (lower, upper) arrays aligned with `grid`, or None if fewer than 10
    resamples succeed (e.g. the class is too rare to survive most resamples).
    """
    n = len(y_true_binary)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    n_g = len(uniq)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=n_g)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    rng = np.random.default_rng(seed)
    boot_curves = []
    for _ in range(n_bootstrap):
        drawn = rng.integers(0, n_g, size=n_g)
        c = counts[drawn]
        total = int(c.sum())
        if total == 0:
            continue
        base = np.repeat(starts[drawn], c)
        within = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        idx = order[base + within]
        if len(np.unique(y_true_binary[idx])) < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y = curve_fn(y_true_binary[idx], y_score[idx])
        except Exception:
            continue
        if x[0] > x[-1]:  # precision_recall_curve returns recall in decreasing order
            x, y = x[::-1], y[::-1]
        boot_curves.append(np.interp(grid, x, y))
    if len(boot_curves) < 10:
        return None
    boot_curves = np.array(boot_curves)
    return np.percentile(boot_curves, 2.5, axis=0), np.percentile(boot_curves, 97.5, axis=0)


def plot_roc_curve(y_true, y_prob, num_classes, class_names, out_path, ci_band=False, groups=None):
    y_onehot = np.eye(num_classes)[y_true]
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr_grid = np.linspace(0, 1, 300)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c in range(num_classes):
        color = colors[c % len(colors)]
        fpr, tpr, _ = roc_curve(y_onehot[:, c], y_prob[:, c])
        auc = roc_auc_score(y_onehot[:, c], y_prob[:, c])
        if ci_band:
            def _roc_curve(yt, ys):
                x, y, _ = roc_curve(yt, ys)
                return x, y
            band = _bootstrap_curve_band(_roc_curve, y_onehot[:, c], y_prob[:, c], fpr_grid, groups=groups)
            if band is not None:
                ax.fill_between(fpr_grid, band[0], band[1], alpha=0.2, color=color, linewidth=0)
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve" + (" (shaded: 95% bootstrap CI)" if ci_band else ""))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, num_classes, class_names, out_path, ci_band=False, groups=None):
    y_onehot = np.eye(num_classes)[y_true]
    fig, ax = plt.subplots(figsize=(6, 6))
    recall_grid = np.linspace(0, 1, 300)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c in range(num_classes):
        color = colors[c % len(colors)]
        precision, recall, _ = precision_recall_curve(y_onehot[:, c], y_prob[:, c])
        ap = average_precision_score(y_onehot[:, c], y_prob[:, c])
        if ci_band:
            def _pr_curve(yt, ys):
                p, r, _ = precision_recall_curve(yt, ys)
                return r, p
            band = _bootstrap_curve_band(_pr_curve, y_onehot[:, c], y_prob[:, c], recall_grid, groups=groups)
            if band is not None:
                ax.fill_between(recall_grid, band[0], band[1], alpha=0.2, color=color, linewidth=0)
        ax.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve" + (" (shaded: 95% bootstrap CI)" if ci_band else ""))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_prob, num_classes, class_names, out_path):
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

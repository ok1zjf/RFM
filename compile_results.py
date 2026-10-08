"""Compile every experiment listed in outputs/results_manifest.txt into a single
outputs/results.pdf: a summary metrics table plus each experiment's
ROC/PR/confusion-matrix plots.

Usage:
    python compile_results.py [--manifest outputs/results_manifest.txt] [--out outputs/results.pdf]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf

from metrics import resolve_groups

METRIC_COLUMNS = ["auroc", "auprc", "balanced_acc", "f1", "acc"]

SPLIT_LABELS = {"t": "Train", "v": "Val", "e": "Test"}
DATASET_DISPLAY_NAMES = {
    "papila": "PAPILA", "messidor2": "Messidor-2", "aptos-2019": "APTOS 2019",
    "idrid": "IDRiD", "gfid": "GFID",
}


def _format_with_ci(test_row: pd.Series, prefix: str, metric: str) -> str:
    value = test_row.get(f"{prefix}_{metric}")
    lo = test_row.get(f"{prefix}_{metric}_ci_lower")
    hi = test_row.get(f"{prefix}_{metric}_ci_upper")
    if value is None:
        return ""
    if lo is not None and hi is not None and not (pd.isna(lo) or pd.isna(hi)):
        return f"{value:.4f} [{lo:.3f}, {hi:.3f}]"
    return f"{value:.4f}"


def load_run(run_dir: str) -> dict:
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    test_row = pd.read_csv(os.path.join(run_dir, "test.csv")).iloc[0]
    phenotype = cfg.data.phenotype
    row = {"experiment": cfg.experiment_name, "phenotype": phenotype, "run_dir": run_dir}
    for metric in METRIC_COLUMNS:
        row[metric] = _format_with_ci(test_row, phenotype, metric)
    return row


def add_table_page(pdf: PdfPages, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(18, 0.5 + 0.4 * len(df)))
    ax.axis("off")
    display_df = df[["experiment"] + METRIC_COLUMNS]
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("RFM / DINOv3 head-training results — summary", fontsize=13, pad=20)
    pdf.savefig(fig)
    plt.close(fig)


def dataset_key(csv_path: str) -> str:
    """'datasets/papila_splits.csv' -> 'papila'; 'datasets/messidor2_splits.csv'
    -> 'messidor2'; etc - the <name> in every config's <name>_splits.csv."""
    base = os.path.basename(csv_path)
    for suffix in ("_splits.csv", "_split.csv"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def collect_datasets(run_dirs: list, repo_root: str) -> dict:
    """One entry per distinct dataset (keyed by csv_path), taken from the
    first run that uses it - every run training on the same dataset points at
    the identical CSV, so there is nothing to reconcile across runs."""
    datasets = {}
    for run_dir in run_dirs:
        cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        csv_path = cfg.data.csv_path if os.path.isabs(cfg.data.csv_path) else os.path.join(repo_root, cfg.data.csv_path)
        key = dataset_key(csv_path)
        if key not in datasets:
            splits = [s for s in (cfg.data.get("train_split"), cfg.data.get("val_split"),
                                   cfg.data.get("test_split")) if s]
            datasets[key] = {"csv_path": csv_path, "splits": splits}
    return datasets


def _count_table(df: pd.DataFrame, splits: list, group_col: str = None):
    """[[label, n, pct], ...] + a Total row, counting either rows (group_col=None)
    or unique values of group_col, per split value."""
    if group_col is None:
        total = len(df)
        counts = {s: int((df["split"].astype(str) == str(s)).sum()) for s in splits}
    else:
        total = df[group_col].nunique()
        counts = {s: df.loc[df["split"].astype(str) == str(s), group_col].nunique() for s in splits}
    rows = [[SPLIT_LABELS.get(s, s), counts[s], f"{100 * counts[s] / total:.1f}%"] for s in splits]
    rows.append(["Total", total, "100.0%"])
    return rows


def add_dataset_split_page(pdf: PdfPages, name: str, info: dict, repo_root: str):
    """One page per dataset: image counts per split, and - for datasets where
    patient identity is recoverable (PAPILA: encoded in the filename;
    Messidor-2: via datasets/messidor2_pairs.csv; see metrics.resolve_groups)
    - patient counts per split plus a cross-split patient-leakage check.
    Datasets with no recoverable patient identifier (APTOS 2019, IDRiD, GFID)
    get the image-count table only - there is nothing else to verify."""
    df = pd.read_csv(info["csv_path"])
    splits = info["splits"]
    display_name = DATASET_DISPLAY_NAMES.get(name, name)

    groups = resolve_groups(info["csv_path"], repo_root, df["id"].tolist())
    has_patients = groups is not None
    leak_note = None
    if has_patients:
        df = df.assign(_patient=groups)
        image_rows = _count_table(df, splits)
        patient_rows = _count_table(df, splits, group_col="_patient")
        span = df.groupby("_patient")["split"].nunique()
        leaking = int((span > 1).sum())
        leak_note = (f"Patient-leakage check: {leaking} of {df['_patient'].nunique()} patients "
                     f"appear in more than one split — {'VERIFIED, none do' if leaking == 0 else 'LEAKAGE FOUND'}.")
    else:
        image_rows = _count_table(df, splits)
        patient_rows = None

    n_tables = 2 if has_patients else 1
    fig, axes = plt.subplots(1, n_tables, figsize=(6 * n_tables, 1.6 + 0.5 * (len(splits) + 1)))
    axes = [axes] if n_tables == 1 else list(axes)

    def render(ax, rows, title):
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["Split", "N", "%"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        ax.set_title(title, fontsize=11, pad=14)

    render(axes[0], image_rows, "Images")
    if has_patients:
        render(axes[1], patient_rows, "Patients")

    suptitle = f"{display_name} — dataset split summary"
    if leak_note:
        fig.suptitle(f"{suptitle}\n{leak_note}", fontsize=12)
    else:
        fig.suptitle(f"{suptitle}\nNo patient identifier available for this dataset "
                      f"(image-level split only)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    pdf.savefig(fig)
    plt.close(fig)


def add_experiment_page(pdf: PdfPages, row: dict):
    phenotype = row["phenotype"]
    plot_names = ["roc_curve", "pr_curve", "confusion_matrix"]
    fig, axes = plt.subplots(1, len(plot_names), figsize=(15, 5))
    fig.suptitle(row["experiment"], fontsize=13)
    for ax, name in zip(axes, plot_names):
        path = os.path.join(row["run_dir"], f"{phenotype}_{name}.png")
        ax.axis("off")
        if os.path.exists(path):
            ax.imshow(plt.imread(path))
        else:
            ax.text(0.5, 0.5, f"missing: {name}", ha="center", va="center")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/results_manifest.txt")
    parser.add_argument("--out", default="outputs/results.pdf")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    with open(args.manifest) as f:
        run_dirs = [line.strip() for line in f if line.strip()]

    rows = [load_run(run_dir) for run_dir in run_dirs]
    df = pd.DataFrame(rows)
    datasets = collect_datasets(run_dirs, repo_root)

    with PdfPages(args.out) as pdf:
        add_table_page(pdf, df)
        for name in sorted(datasets):
            add_dataset_split_page(pdf, name, datasets[name], repo_root)
        for row in rows:
            add_experiment_page(pdf, row)

    print(f"Wrote {args.out} with {len(rows)} experiments and {len(datasets)} dataset-split summaries.")


if __name__ == "__main__":
    main()

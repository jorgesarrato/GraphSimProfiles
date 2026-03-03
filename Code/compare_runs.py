"""
compare_runs.py — Compare mass-ratio plots across multiple training runs.

Loads plot_data.pkl files produced by train_graphs.py and overlays the
Testing split for each run on a single figure.  Classical estimators and
Genina+20 are drawn from the first run listed (they are simulation-pair
dependent, so mixing runs with different src/tgt pairs is not meaningful).

Usage
-----
    python compare_runs.py /path/to/run1 /path/to/run2 ... [options]

    # Compare all subdirectories under a parent folder automatically:
    python compare_runs.py --glob "/path/to/work/Graph+Flow_Mocks_NH/new/*"

Options
-------
    --split {Training,Validation,Testing}
        Which data split to compare (default: Testing).
    --labels LABEL [LABEL ...]
        Custom legend labels for each run (must match number of runs).
    --out PATH
        Output path for the figure (default: compare_runs.png).
    --no_classical
        Omit classical estimator points.
    --no_genina
        Omit Genina+20 reference band.
    --title TITLE
        Figure title.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Genina+20 reference (copied from train_graphs.py) ─────────────────────────
GENINA_MED = dict(
    x=[-0.3765, -0.3253, -0.2534, -0.2222, -0.1473, -0.0932, -0.0569,
        0.0403,  0.1248,  0.2138,  0.2687,  0.3496,  0.4208,  0.4912,  0.5305],
    y=[ 0.9655,  0.9655,  0.9655,  0.9545,  0.9509,  0.9473,  0.9509,
        0.9473,  0.9364,  0.9255,  0.9327,  0.9400,  0.9473,  0.9436,  0.9364],
)
GENINA_UP = dict(
    x=[-0.3773, -0.3543, -0.2237, -0.1904, -0.0925, -0.0606,  0.0403,
        0.0707,  0.2049,  0.3681,  0.4334,  0.5001,  0.5290],
    y=[ 1.5073,  1.4927,  1.3364,  1.3073,  1.2055,  1.2055,  1.1436,
        1.1327,  1.1327,  1.1764,  1.2127,  1.2309,  1.2236],
)
GENINA_DOWN = dict(
    x=[-0.3787, -0.2564, -0.2245, -0.1911, -0.0621,  0.0536,  0.1693,
        0.2739,  0.3911,  0.4868,  0.5290],
    y=[ 0.6273,  0.6891,  0.6855,  0.7036,  0.7545,  0.7873,  0.7909,
        0.7691,  0.7400,  0.7109,  0.6927],
)

MARKER_STYLES   = {"Walker": "o", "Wolf": "d", "Amorisco": "*", "Campbell": "<", "Errani": "^"}
ESTIMATOR_RADII = {"Walker": 1.0, "Wolf": 4/3, "Amorisco": 1.7, "Campbell": 1.8, "Errani": 1.8}


def auto_label(pd: dict) -> str:
    """Generate a compact legend label from a plot_data dict."""
    parts = []
    if pd.get("dann"):
        parts.append(f"DANN src={pd['dann_source']}")
        lam = pd.get("dann_lambda")
        if lam is not None:
            parts.append(f"λ={lam}")
    else:
        parts.append(f"{pd['train_set']}→{pd['test_set']}")
    parts.append(f"N={pd['Nstars']}")
    return "  ".join(parts)


def load_plot_data(folder: str) -> dict:
    path = os.path.join(folder, "plot_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No plot_data.pkl found in {folder}")
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mass-ratio plots across training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folders", nargs="*",
        help="Model folders containing plot_data.pkl",
    )
    parser.add_argument(
        "--glob", default=None,
        help="Shell glob pattern to collect folders automatically",
    )
    parser.add_argument(
        "--split", default="Testing",
        choices=["Training", "Validation", "Testing"],
        help="Data split to compare (default: Testing)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Custom legend labels (one per run)",
    )
    parser.add_argument(
        "--out", default="compare_runs.png",
        help="Output figure path (default: compare_runs.png)",
    )
    parser.add_argument(
        "--no_classical", action="store_true",
        help="Omit classical estimator points",
    )
    parser.add_argument(
        "--no_genina", action="store_true",
        help="Omit Genina+20 reference band",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Collect folders ────────────────────────────────────────────────────────
    folders = list(args.folders)
    if args.glob:
        folders += sorted(glob_module.glob(args.glob))
    folders = [f for f in folders if os.path.isdir(f)]

    if not folders:
        print("No valid folders found. Pass model folder paths or use --glob.", file=sys.stderr)
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    datasets = []
    for folder in folders:
        try:
            datasets.append(load_plot_data(folder))
        except FileNotFoundError as e:
            print(f"Warning: {e}", file=sys.stderr)

    if not datasets:
        print("No plot_data.pkl files could be loaded.", file=sys.stderr)
        sys.exit(1)

    # ── Labels ────────────────────────────────────────────────────────────────
    if args.labels:
        if len(args.labels) != len(datasets):
            print(
                f"--labels has {len(args.labels)} entries but {len(datasets)} runs were loaded.",
                file=sys.stderr,
            )
            sys.exit(1)
        labels = args.labels
    else:
        labels = [auto_label(pd) for pd in datasets]

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = list(mcolors.TABLEAU_COLORS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1, ls="--", color="k", lw=0.8, zorder=0)

    for i, (pd, label) in enumerate(zip(datasets, labels)):
        radii = np.array(pd["label_mass_radii_frac"])
        split_data = pd["ratio_stats"].get(args.split)
        if split_data is None:
            print(f"Warning: split '{args.split}' not found in run '{label}', skipping.", file=sys.stderr)
            continue

        med = np.array(split_data["med"])
        p16 = np.array(split_data["p16"])
        p84 = np.array(split_data["p84"])
        c   = colors[i % len(colors)]

        ax.plot(radii, med, color=c, label=label, lw=1.8)
        ax.fill_between(radii, p16, p84, color=c, alpha=0.15)

    # ── Genina+20 (from first run, as reference) ──────────────────────────────
    if not args.no_genina:
        ax.plot(10 ** np.array(GENINA_MED["x"]),  GENINA_MED["y"],
                ls=":", color="darkgray", lw=1.2, label="Genina+20")
        ax.plot(10 ** np.array(GENINA_UP["x"]),   GENINA_UP["y"],
                ls=":", color="darkgray", lw=1.0)
        ax.plot(10 ** np.array(GENINA_DOWN["x"]), GENINA_DOWN["y"],
                ls=":", color="darkgray", lw=1.0)

    # ── Classical estimators (from first run) ─────────────────────────────────
    if not args.no_classical:
        first = datasets[0]
        classical = first.get("classical_masses", {})
        estim     = np.array(first.get("estim_masses", []))
        src_size  = first.get("N_files_train", None)

        # classical_masses keys are estimator names → arrays of mass estimates
        # estim_masses shape: (N_projections, 5) matching Walker/Wolf/Amorisco/Errani/Campbell
        estim_col = {"Walker": 0, "Wolf": 1, "Amorisco": 2, "Errani": 3, "Campbell": 4}

        for name, marker in MARKER_STYLES.items():
            if name not in classical:
                continue
            class_vals = np.array(classical[name])
            if estim.ndim == 2 and name in estim_col:
                ref_vals = estim[:len(class_vals), estim_col[name]]
            else:
                continue
            rel    = class_vals / ref_vals
            med_r  = np.nanmedian(rel)
            err    = np.array([
                [med_r - np.nanpercentile(rel, 16)],
                [np.nanpercentile(rel, 84) - med_r],
            ])
            ax.errorbar(
                ESTIMATOR_RADII[name], med_r, err,
                fmt=marker, capsize=4, markersize=4,
                color="k", label=name, zorder=5,
            )

    ax.set_xlabel(r"r / R$_{\rm h}$")
    ax.set_ylabel(r"M$_{\rm pred}$(<r) / M$_{\rm true}$(<r)")
    ax.set_ylim(bottom=0)

    title = args.title or f"Comparison — {args.split} split"
    ax.set_title(title)

    # Deduplicate legend entries (classical estimators appear once)
    handles, legend_labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, legend_labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

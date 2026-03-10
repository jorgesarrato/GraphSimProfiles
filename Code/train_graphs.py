"""
train_graphs.py — Train a GNN + Masked Autoregressive Flow posterior for
stellar mass estimation from graph-structured phase-space data.

Two training modes
------------------
Standard (default)
    train_set  — labelled source of training/validation graphs
    test_set   — labelled source of test graphs

DANN  (activated by --dann_source KEY)
    --dann_source  — labelled source domain  (task loss + domain loss)
    test_set       — target domain: unlabelled during training, then
                     evaluated as the test set after training
    train_set      — ignored in DANN mode (keep for positional-arg
                     compatibility; pass any valid population key)
"""

# TODO: check why mask mode BIF produces NaNs
# TODO: Turn off scheduler during DA lambda ramp up?
# TODO: Correctly print log_prob and DA loss contributions separately during DA training

from __future__ import annotations

import argparse
import os
import pickle
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from scipy.interpolate import interp1d
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

import mlflow

from utils import (
    DANNFlowPosterior,
    MMDFlowPosterior,
    DomainClassifier,
    FlowPosterior,
    GraphCreator,
    GraphNN,
    build_maf_flow,
    ganin_lambda_schedule,
    sample_with_timeout,
)

POPULATION_KEYS = [
    "NIHAO_lo", "NIHAO_hi", "NIHAO_all", "NIHAO_shared",
    "AURIGA_lo", "AURIGA_hi", "AURIGA_all", "AURIGA_shared",
    "ALL", "ALL_shared",
]

GRAPH_LAYER_CONFIGS: dict[str, dict] = {
    "Cheb": {
        "graph_layer_name": "ChebConv",
        "graph_layer_params": {"K": 4, "normalization": "sym", "bias": True},
    },
    "GCN": {
        "graph_layer_name": "GCNConv",
        "graph_layer_params": {"normalize": True, "bias": True},
    },
    "GAT": {
        "graph_layer_name": "GATConv",
        "graph_layer_params": {"heads": 1, "bias": True},
    },
}

KM_TO_M    = 1e3
KPC_TO_M   = 3.086e19
KG_TO_MSUN = 1.0 / (2e30)
G_SI       = 6.6743e-11   # m^3 kg^-1 s^-2

ESTIMATOR_COEFFICIENTS = {
    "Walker":   2.5,
    "Wolf":     4.0,
    "Amorisco": 5.8,
    "Campbell": 6.0,
    "Errani":   3.5 * 1.8,
}

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNN + normalizing flow posterior.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Population keys
---------------
  NIHAO_lo      low-resolution  NIHAO  (eps_dm >  1.0 kpc)
  NIHAO_hi      high-resolution NIHAO  (eps_dm <= 1.0 kpc)
  NIHAO_all     all NIHAO galaxies
  NIHAO_shared  NIHAO galaxies in the eps_dm range shared with AURIGA
  AURIGA_lo     low-resolution  AURIGA (eps_dm >  0.25 kpc)
  AURIGA_hi     high-resolution AURIGA (eps_dm <= 0.25 kpc)
  AURIGA_all    all AURIGA galaxies
  AURIGA_shared AURIGA galaxies in the eps_dm range shared with NIHAO
  ALL           every galaxy from both simulations
  ALL_shared    every galaxy from both simulations, shared range only

  Shared range: [max(NIHAO_min, AURIGA_min), min(NIHAO_max, AURIGA_max)]
  computed directly from eps_dm values in the data.

Standard examples
-----------------
  # NIHAO low-res -> NIHAO high-res
  train_graphs.py train NIHAO_lo NIHAO_hi 1 100 Cheb 8 0

  # all NIHAO -> all AURIGA
  train_graphs.py train NIHAO_all AURIGA_all 1 100 Cheb 8 0

  # shared-resolution NIHAO -> shared-resolution AURIGA
  train_graphs.py train NIHAO_shared AURIGA_shared 1 100 Cheb 8 0

DANN examples
-------------
  # source=NIHAO_all, target/test=AURIGA_all
  train_graphs.py train NIHAO_all AURIGA_all 1 100 Cheb 8 0 --dann_source NIHAO_all

  # source=NIHAO_shared, target/test=AURIGA_shared
  train_graphs.py train NIHAO_shared AURIGA_shared 1 100 Cheb 8 0 --dann_source NIHAO_shared
""",
    )
    parser.add_argument(
        "mode", choices=["train", "sample", "sampletest"],
        help="train, sample (train+val+test), or sampletest (test only)",
    )
    parser.add_argument(
        "train_set", choices=POPULATION_KEYS,
        help="Population to train on in standard mode (ignored with --dann_source)",
    )
    parser.add_argument(
        "test_set", choices=POPULATION_KEYS,
        help="Population to test on; also the DANN target domain when --dann_source is set",
    )
    parser.add_argument(
        "test_long", type=int, choices=[0, 1],
        help="Full test set (1) or first 100 galaxies (0)",
    )
    parser.add_argument("N_stars",         type=int,                 help="Expected number of stars (Poisson mean)")
    parser.add_argument("GraphNN_type",    choices=["Cheb", "GCN", "GAT"])
    parser.add_argument("N_proj_per_gal",  type=int,                 help="Projections per galaxy")
    parser.add_argument("PCAfilter",       type=int, choices=[0, 1], help="PCA-filtered data (0|1)")
    parser.add_argument(
        "--hlr_std", type=int, choices=[0, 1], default=1,
        help="Append hlr/std scalars to embedding (default: 1)",
    )
    # DANN arguments
    parser.add_argument(
        "--dann_source", choices=POPULATION_KEYS, default=None, metavar="KEY",
        help=(
            "Enable DANN. KEY = labelled source domain. "
            "test_set becomes the unlabelled target domain during training "
            "and the evaluated test set after training."
        ),
    )
    parser.add_argument(
        "--dann_lambda", type=float, default=1.0,
        help="Maximum GRL reversal strength lambda (default: 1.0)",
    )
    parser.add_argument(
        "--dann_gamma", type=float, default=10.0,
        help="Steepness of the Ganin lambda schedule (default: 10.0)",
    )
    parser.add_argument(
        "--dann_domain_hidden", type=int, default=64,
        help="Hidden width of the domain-classifier MLP (default: 64)",
    )
    
    parser.add_argument(
        "--mask_missing", choices=["mask", "BIF", "drop"], default=None,
        help=(
            "Strategy for handling unresolved radii: "
            "'mask' (context-aware masking), "
            "'BIF' (Bayesian imputation), "
            "'drop' (discard any sample with at least one unresolved bin)."
        ),
    )
    
    parser.add_argument(
        "--da_method", choices=["dann", "mmd"], default="dann",
        help="Domain adaptation method when --dann_source is set (default: dann)",
    )
    parser.add_argument(
        "--mmd_lambda", type=float, default=1.0,
        help="MMD loss weight (only used with --da_method mmd, default: 1.0)",
    )

    parser.add_argument(
        "--mmd_gamma", type=float, default=10.0,
        help="Steepness of the Ganin lambda schedule for MMD (default: 10.0)",
    )

    parser.add_argument(
        "--hidden_dim", type=int, default=128,
        help="Hidden width for graph layers AND FC layers (default: 128)",
    )
    parser.add_argument(
        "--num_graph_layers", type=int, default=3,
        help="Number of graph convolution layers (default: 3)",
    )
    parser.add_argument(
        "--num_fc_layers", type=int, default=2,
        help="Number of FC layers after pooling (default: 2)",
    )
    parser.add_argument(
        "--use_residuals", type=int, choices=[0, 1], default=1,
        help="Skip connections between graph layers (default: 1)",
    )
    parser.add_argument(
        "--use_batch_norm", type=int, choices=[0, 1], default=1,
        help="BatchNorm1d after each graph/FC layer (default: 1)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability after each graph/FC layer (default: 0.1)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Training batch size (default: 128)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader worker processes (default: 2)",
    )
    parser.add_argument(
        "--mlflow_uri", type=str, default=None,
        help=(
            "MLflow tracking URI. If omitted, MLflow tracking is disabled. "
            "Examples: 'file:///path/to/mlruns'  'http://host:5000'"
        ),
    )
    parser.add_argument(
        "--mlflow_experiment", type=str, default="graph_flow_posterior",
        help="MLflow experiment name (default: 'graph_flow_posterior')",
    )
    return parser.parse_args()



# ─────────────────────────────────────────────────────────────────────────────
# MLflow helpers
# ─────────────────────────────────────────────────────────────────────────────

class _NullRun:
    """Drop-in replacement for an mlflow ActiveRun when tracking is disabled.

    Every attribute access and method call returns self or a no-op so that
    all ``mlflow.*`` call sites can be written unconditionally.
    """
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def __getattr__(self, _): return lambda *a, **kw: None


def setup_mlflow(args, model_str: str):
    """Configure MLflow and return an active run context manager.

    If ``--mlflow_uri`` is not provided, returns a ``_NullRun`` that silently
    ignores all logging calls so the rest of the code is unchanged.

    Parameters
    ----------
    args : argparse.Namespace
    model_str : str
        Used as the MLflow run name for easy identification in the UI.

    Returns
    -------
    run : mlflow.ActiveRun | _NullRun
        Use as ``with setup_mlflow(...) as run:``.
    """
    if args.mlflow_uri is None:
        return _NullRun()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    return mlflow.start_run(run_name=model_str)


def mlflow_log_hparams(args, data: dict) -> None:
    """Log all hyperparameters as MLflow params (strings/scalars only)."""
    use_dann = args.dann_source is not None
    params = dict(
        # Data / sampling
        train_set      = data["src_key"],
        test_set       = data["tgt_key"],
        N_stars        = args.N_stars,
        N_proj_per_gal = args.N_proj_per_gal,
        N_src_galaxies = len(data["src_indices"]),
        N_tgt_galaxies = len(data["tgt_indices"]),
        mask_missing   = str(args.mask_missing),
        PCAfilter      = args.PCAfilter,
        hlr_std        = args.hlr_std,
        # Architecture
        GraphNN_type      = args.GraphNN_type,
        hidden_dim        = args.hidden_dim,
        num_graph_layers  = args.num_graph_layers,
        num_fc_layers     = args.num_fc_layers,
        use_residuals     = args.use_residuals,
        use_batch_norm    = args.use_batch_norm,
        dropout           = args.dropout,
        # Optimisation
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        # Domain adaptation
        da_enabled        = use_dann,
        da_method         = args.da_method if use_dann else "none",
        dann_source       = str(args.dann_source),
        dann_lambda       = args.dann_lambda,
        dann_gamma        = args.dann_gamma,
        mmd_lambda        = args.mmd_lambda,
        mmd_gamma         = args.mmd_gamma,
    )
    mlflow.log_params(params)


def mlflow_log_epoch(
    epoch: int,
    *,
    train_lp: float,
    val_lp: float,
    lr: float,
    domain_loss: float | None = None,
    domain_acc:  float | None = None,
    mmd_loss:    float | None = None,
    lam:         float | None = None,
) -> None:
    """Log per-epoch scalars to the active MLflow run."""
    metrics = {
        "train_log_prob": train_lp,
        "val_log_prob":   val_lp,
        "learning_rate":  lr,
    }
    if domain_loss is not None: metrics["domain_loss"] = domain_loss
    if domain_acc  is not None: metrics["domain_acc"]  = domain_acc
    if mmd_loss    is not None: metrics["mmd_loss"]    = mmd_loss
    if lam         is not None: metrics["lambda"]      = lam
    mlflow.log_metrics(metrics, step=epoch)


def mlflow_log_results(ratio_stats: dict, ratio_stats_resolved: dict) -> None:
    """Log final mass-ratio summary statistics as MLflow metrics."""
    for tag, rs in [("all", ratio_stats), ("resolved", ratio_stats_resolved)]:
        for split, (med, p16, p84) in rs.items():
            prefix = f"{tag}_{split.lower()}"
            # Log per-bin median and scatter as flat scalar arrays
            for i, (m, lo, hi) in enumerate(zip(med, p16, p84)):
                mlflow.log_metrics({
                    f"{prefix}_median_bin{i}": float(m),
                    f"{prefix}_scatter_bin{i}": float(hi - lo),
                }, step=i)
            # Summary scalars: mean over bins
            mlflow.log_metrics({
                f"{prefix}_median_mean":  float(np.mean(np.abs(med - 1))),
                f"{prefix}_scatter_mean": float(np.mean(p84 - p16)),
            })


def mlflow_log_artefacts(model_folder: str, args) -> None:
    """Log files from model_folder as MLflow artefacts."""
    for fname in ("posterior.pkl", "plot_data.pkl"):
        path = os.path.join(model_folder, fname)
        if os.path.exists(path):
            mlflow.log_artifact(path)


def load_label_file(data_folder: str, pca_filter: bool) -> pd.DataFrame:
    """Read the label CSV and annotate each row with its simulation origin."""
    suffix   = "PCAfilt" if pca_filter else "PCAnofilt"
    csv_path = (
        f"/net/debut/project/jsarrato/Paper-GraphSimProfiles/work/"
        f"proj_data_NIHAO_and_AURIGA_{suffix}_samplearr.csv"
    )
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    df["sim"] = df["name"].apply(lambda x: 1 if x.startswith("halo") else 0)
    return df

def shared_eps_dm_range(label_file: pd.DataFrame) -> tuple[float, float]:
    """Return the overlapping eps_dm range between NIHAO and AURIGA.

    Computes [max(nh_min, au_min), min(nh_max, au_max)] from the actual data.
    Raises ValueError if there is no overlap.
    """
    nh_eps = label_file.loc[label_file["sim"] == 0, "eps_dm"]
    au_eps = label_file.loc[label_file["sim"] == 1, "eps_dm"]
    lo = max(nh_eps.min(), au_eps.min())
    hi = min(nh_eps.max(), au_eps.max())
    if lo >= hi:
        raise ValueError(
            f"No overlapping eps_dm range: NIHAO [{nh_eps.min():.3f}, {nh_eps.max():.3f}], "
            f"AURIGA [{au_eps.min():.3f}, {au_eps.max():.3f}]."
        )
    return lo, hi

def population_indices(label_file: pd.DataFrame, key: str) -> np.ndarray:
    """Return the i_index array for a named galaxy population.

    Population keys
    ---------------
    NIHAO_lo      eps_dm >  1.0 kpc, NIHAO
    NIHAO_hi      eps_dm <= 1.0 kpc, NIHAO
    NIHAO_all     all NIHAO
    NIHAO_shared  NIHAO within the shared eps_dm range
    AURIGA_lo     eps_dm >  0.25 kpc, AURIGA
    AURIGA_hi     eps_dm <= 0.25 kpc, AURIGA
    AURIGA_all    all AURIGA
    AURIGA_shared AURIGA within the shared eps_dm range
    ALL           everything
    ALL_shared    everything within the shared eps_dm range
    """
    is_nihao  = label_file["sim"] == 0
    is_auriga = label_file["sim"] == 1

    masks: dict[str, pd.Series] = {
        "NIHAO_lo":   is_nihao  & (label_file["eps_dm"] >  1.0),
        "NIHAO_hi":   is_nihao  & (label_file["eps_dm"] <= 1.0),
        "NIHAO_all":  is_nihao,
        "AURIGA_lo":  is_auriga & (label_file["eps_dm"] >  0.25),
        "AURIGA_hi":  is_auriga & (label_file["eps_dm"] <= 0.25),
        "AURIGA_all": is_auriga,
        "ALL":        pd.Series(True, index=label_file.index),
    }


    if "_shared" in key:
        lo, hi    = shared_eps_dm_range(label_file)
        in_shared = (label_file["eps_dm"] >= lo) & (label_file["eps_dm"] <= hi)
        masks["NIHAO_shared"]  = is_nihao  & in_shared
        masks["AURIGA_shared"] = is_auriga & in_shared
        masks["ALL_shared"]    = in_shared

    if key not in masks:
        raise ValueError(f"Unknown population key '{key}'. Valid: {POPULATION_KEYS}")

    return np.array(label_file[masks[key]]["i_index"], dtype=int)


def resolve_indices(
    label_file: pd.DataFrame,
    key: str,
    limit: int | None = None,
) -> np.ndarray:

    if "_shared" in key:
        lo, hi = shared_eps_dm_range(label_file)
        print(f"  Shared eps_dm range for '{key}': [{lo:.4f}, {hi:.4f}] kpc")
    idx = population_indices(label_file, key)
    if limit is not None:
        idx = idx[:limit]
    return idx
    
def get_split_indices(label_file, key, test_size=0.1, limit=None, seed=None):
    """Split key into source (train/val) and target (test) indices.

    Parameters
    ----------
    label_file : pd.DataFrame
    key : str
        Population key (same set accepted by ``resolve_indices``).
    test_size : float
        Fraction of galaxies to hold out as the target/test set.
    limit : int | None
        If given, cap the *target* indices at this number.
        The source indices are never capped here.
    seed : int | None
        Seed for the local RNG used to shuffle. Does not affect the global
        NumPy or PyTorch RNG state.

    Returns
    -------
    src_indices, tgt_indices : np.ndarray
    """
    full_idx = resolve_indices(label_file, key)
    
    rng = np.random.default_rng(seed)
    rng.shuffle(full_idx)
    
    split_point = int(len(full_idx) * (1 - test_size))
    
    src_indices = full_idx[:split_point]
    tgt_indices = full_idx[split_point:]
    
    if limit is not None:
        tgt_indices = tgt_indices[:limit]
        
    return src_indices, tgt_indices

def load_phase_space(
    data_folder: str,
    file_indices_src: np.ndarray,
    file_indices_tgt: np.ndarray,
    n_proj_per_gal: int,
    nstars_arr: np.ndarray,
    label_mass_radii_frac: np.ndarray,
    mass_estim_radii_frac: np.ndarray,
    label_file: pd.DataFrame,
) -> dict:
    """Read all galaxy files for source and target sets.

    Returns dict with keys:
        positions, velocities, labels, file_indices,
        hlrs, stds, estimator_masses, train_and_val_size,
        masks
    """
    masks = []
    all_indices = np.concatenate([file_indices_src, file_indices_tgt])
    len_src     = len(file_indices_src)

    positions, velocities, labels = [], [], []
    file_indices_out              = []
    hlrs, stds, estimator_masses  = [], [], []
    train_and_val_size            = None

    for i, idx in enumerate(tqdm(all_indices, desc="Reading data")):
        masses_name = data_folder + f"mass_interp{idx}.npz"
        posvel_name = data_folder + f"posvel_{idx}.pkl"

        mass_data         = np.load(masses_name)
        mass_interpolator = interp1d(mass_data["x"], mass_data["y"])
        posvel            = torch.load(posvel_name, weights_only=False)

        proj_indices      = np.random.choice(len(posvel), n_proj_per_gal)

        eps_dm            = label_file.loc[label_file["i_index"] == idx, "eps_dm"].values[0]

        for j, proj_idx in enumerate(proj_indices):
            nstars = nstars_arr[i * n_proj_per_gal + j]
            data   = posvel[proj_idx]

            rxy  = np.linalg.norm(data[:nstars, :2], axis=1)
            hlr  = np.median(rxy)
            vstd = np.std(data[:nstars, -1])

            try:
                masses_idx = np.log10(mass_interpolator(label_mass_radii_frac * hlr))
                estim      = mass_interpolator(mass_estim_radii_frac * hlr)
            except Exception:
                continue

            if np.any(~np.isfinite(data)) or np.any(~np.isfinite(masses_idx)):
                continue

            mask = (label_mass_radii_frac * hlr) >= (3.0 * eps_dm)

            positions.append(data[:nstars, :2])
            velocities.append(data[:nstars, -1])
            labels.append(masses_idx)
            file_indices_out.append(i)
            hlrs.append(hlr)
            stds.append(vstd)
            estimator_masses.append(estim)
            masks.append(mask)

        if i == len_src - 1:
            train_and_val_size = len(labels)

    return dict(
        positions          = positions,
        velocities         = velocities,
        labels             = labels,
        file_indices       = np.array(file_indices_out),
        hlrs               = np.array(hlrs),
        stds               = np.array(stds),
        estimator_masses   = np.array(estimator_masses),
        train_and_val_size = train_and_val_size,
        masks              = np.array(masks),
    )

def compute_standardization_stats(
    graphs: list,
    mask_missing: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute input and output standardization statistics from training graphs.

    Parameters
    ----------
    graphs : list of Data
        Source **training** graphs only (val/test excluded to avoid leakage).
    mask_missing : str | None
        When ``"mask"`` or ``"BIF"``, label statistics for each dimension are
        computed only over samples where that dimension is resolved
        (``mask[:, d] == 1``).  Unmasked dimensions use the full set.
        When ``None``, all samples contribute to every dimension's statistics.

    Returns
    -------
    x_mean, x_std : Tensor, shape ``(n_node_features,)``
        Per-feature mean and std computed across all nodes in all graphs.
    y_mean, y_std : Tensor, shape ``(n_labels,)``
        Per-label mean and std; masked dimensions exclude unresolved samples.
    """

    all_x = torch.cat([g.x for g in graphs], dim=0)
    x_mean = all_x.mean(dim=0)
    x_std  = all_x.std(dim=0).clamp(min=1e-6)

    all_hlr = torch.cat([g.hlr for g in graphs], dim=0)  # (N, 1)
    all_std = torch.cat([g.std for g in graphs], dim=0)  # (N, 1)
    hlr_mean = all_hlr.mean()
    hlr_std  = all_hlr.std().clamp(min=1e-6)
    s_mean   = all_std.mean()
    s_std    = all_std.std().clamp(min=1e-6)

    # graph.y  shape: (1, n_labels)  — the raw (unstandardized) label
    # graph.mask shape: (1, n_labels) — float 1 = resolved, 0 = unresolved
    all_y    = torch.cat([g.y    for g in graphs], dim=0)   # (N, n_labels)
    n_labels = all_y.shape[1]

    if mask_missing in ("mask", "BIF"):
        # Per-dimension statistics using only resolved samples for each dim.
        all_m = torch.cat([g.mask for g in graphs], dim=0)  # (N, n_labels)
        y_mean = torch.zeros(n_labels)
        y_std  = torch.ones(n_labels)
        for d in range(n_labels):
            resolved = all_m[:, d].bool()
            vals = all_y[resolved, d]
            if vals.numel() > 1:
                y_mean[d] = vals.mean()
                y_std[d]  = vals.std().clamp(min=1e-6)
            else:
                # Fallback if a dimension is fully masked: use global stats
                y_mean[d] = all_y[:, d].mean()
                y_std[d]  = all_y[:, d].std().clamp(min=1e-6)
    else:
        # "drop" mode: every sample is fully resolved, so global stats are correct.
        # None mode: no masking at all.
        y_mean = all_y.mean(dim=0)
        y_std  = all_y.std(dim=0).clamp(min=1e-6)

    return x_mean, x_std, y_mean, y_std, hlr_mean, hlr_std, s_mean, s_std

def build_graph_list(
    data: dict,
    graph_creator: GraphCreator,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Apply graph_creator to every sample; return (graphs, hlrs, stds)."""
    graphs, hlrs, stds = [], [], []

    for i in tqdm(range(len(data["labels"])), desc="Building graphs"):
        graph = graph_creator(
            positions  = data["positions"][i],
            velocities = data["velocities"][i],
            labels     = data["labels"][i],
        )
        graph.mask = torch.tensor(data["masks"][i], dtype=torch.float32).reshape(1, -1)
        graph.hlr = torch.tensor(
            torch.quantile(10 ** graph.x[:, 0], 0.5), dtype=torch.float32
        ).reshape(1, 1)
        graph.std = torch.tensor(
            torch.std(graph.x[:, 1]), dtype=torch.float32
        ).reshape(1, 1)
        hlrs.append(float(graph.hlr))
        stds.append(float(graph.std))
        graphs.append(graph)

    return graphs, np.array(hlrs), np.array(stds)

def filter_fully_resolved(
    graphs: list,
    hlrs: np.ndarray,
    stds: np.ndarray,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Discard graphs that have any unresolved label dimension.

    A graph is kept only when every element of its ``mask`` is 1, i.e.
    ``mask.min() == 1``.  This corresponds to ``--mask_missing drop``.

    Parameters
    ----------
    graphs : list of Data
        Graph list as returned by :func:`build_graph_list`.
    hlrs : np.ndarray, shape ``(N,)``
        Half-light radii aligned with ``graphs``.
    stds : np.ndarray, shape ``(N,)``
        Velocity dispersions aligned with ``graphs``.

    Returns
    -------
    graphs_out, hlrs_out, stds_out
        Filtered copies — all three arrays are aligned to the same kept indices.
    """
    keep = [g.mask.min().item() == 1.0 for g in graphs]
    mask_arr    = np.array(keep)
    graphs_out  = [g for g, k in zip(graphs, keep) if k]
    hlrs_out    = hlrs[mask_arr]
    stds_out    = stds[mask_arr]
    n_before    = len(graphs)
    n_after     = len(graphs_out)
    print(
        f"filter_fully_resolved: kept {n_after} / {n_before} graphs "
        f"({100 * n_after / max(n_before, 1):.1f}%)"
    )
    return graphs_out, hlrs_out, stds_out


def compute_classical_estimators(
    stds: np.ndarray,
    hlrs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Walker/Wolf/etc. mass estimates in solar masses."""
    unit_factor = (G_SI ** -1) * (KM_TO_M ** 2) * KPC_TO_M * KG_TO_MSUN
    return {
        name: coeff * unit_factor * stds ** 2 * hlrs
        for name, coeff in ESTIMATOR_COEFFICIENTS.items()
    }

def train_val_split(
    graphs: list,
    file_indices: np.ndarray,
    val_fraction: float = 0.2,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Split graphs into train/val by galaxy file (no leakage across projections).

    Returns (train_graphs, val_graphs, train_mask, val_mask).
    """
    unique_files = np.unique(file_indices)
    n_val        = max(1, int(val_fraction * len(unique_files)))
    val_files    = np.random.choice(unique_files, n_val, replace=False)
    val_mask     = np.isin(file_indices, val_files)
    train_mask   = ~val_mask
    train_graphs = [g for g, m in zip(graphs, train_mask) if m]
    val_graphs   = [g for g, m in zip(graphs, val_mask)   if m]
    return train_graphs, val_graphs, train_mask, val_mask

def _make_loader(
    graphs: list,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    augment: bool = False,
) -> torch.utils.data.DataLoader:
    collater = Collater(graphs)

    def collate_fn(batch):
        batch = collater(batch)
        if augment:
            batch = batch.clone()
            flip = torch.randint(0, 2, (batch.num_graphs,), device=batch.x.device) * 2 - 1
            flip_per_node = flip[batch.batch] 
            batch.x[:, 1] = batch.x[:, 1] * flip_per_node
        return batch, batch.y

    return torch.utils.data.DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, prefetch_factor=2,
        num_workers=num_workers, collate_fn=collate_fn,
    )

def make_data_loaders(
    train_graphs: list,
    val_graphs: list,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return (
        _make_loader(train_graphs, batch_size, shuffle=True,  num_workers=num_workers, augment=True),
        _make_loader(val_graphs,   batch_size, shuffle=False, num_workers=num_workers, augment=False),
    )

def make_dann_loaders(
    src_train: list,
    src_val: list,
    tgt_all: list,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Return (src_train_loader, src_val_loader, tgt_loader).

    The target loader uses the same collate_fn so that graphs
    can be evaluated after training; labels are never used in the DANN step.
    """
    return (
        _make_loader(src_train, batch_size, shuffle=True,  num_workers=num_workers, augment=True),
        _make_loader(src_val,   batch_size, shuffle=False, num_workers=num_workers, augment=False),
        _make_loader(tgt_all,   batch_size, shuffle=True,  num_workers=num_workers, augment=True),
    )

def train_posterior(
    posterior: FlowPosterior,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    learning_rate: float = 4e-4,
    max_epochs: int = 500,
    stop_after_epochs: int = 20,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    mlflow_run=None,
) -> dict:
    """Train FlowPosterior with early stopping on validation log-prob.

    Parameters
    ----------
    lr_factor : float
        Multiplicative factor by which the LR is reduced when validation
        log-prob stops improving.  Passed to ``ReduceLROnPlateau``.
    lr_patience : int
        Number of epochs with no improvement before the LR is reduced.
        Passed to ``ReduceLROnPlateau``.

    Returns dict with training_log_probs and validation_log_probs.
    """
    posterior.to(device)
    optimizer = torch.optim.Adam(posterior.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience,
    )
    best_val_lp    = float("-inf")
    epochs_no_impr = 0
    train_lps, val_lps = [], []

    for epoch in range(max_epochs):
        posterior.train()
        epoch_train = []
        for batch, labels in train_loader:
            batch  = batch.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            lp = posterior.log_prob(labels, batch).mean()
            (-lp).backward()
            optimizer.step()
            epoch_train.append(lp.item())

        posterior.eval()
        epoch_val = []
        with torch.no_grad():
            for batch, labels in val_loader:
                batch  = batch.to(device)
                labels = labels.to(device).float()
                epoch_val.append(posterior.log_prob(labels, batch).mean().item())

        mean_train = float(np.mean(epoch_train))
        mean_val   = float(np.mean(epoch_val))
        train_lps.append(mean_train)
        val_lps.append(mean_val)
        scheduler.step(mean_val)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:4d} | "
            f"train log-prob: {mean_train:.4f} | "
            f"val log-prob:   {mean_val:.4f} | "
            f"lr: {current_lr:.2e}"
        )
        mlflow_log_epoch(epoch, train_lp=mean_train, val_lp=mean_val, lr=current_lr)

        if mean_val > best_val_lp:
            best_val_lp    = mean_val
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= stop_after_epochs:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    return {"training_log_probs": train_lps, "validation_log_probs": val_lps}

def train_dann_posterior(
    posterior: DANNFlowPosterior,
    src_train_loader: torch.utils.data.DataLoader,
    src_val_loader: torch.utils.data.DataLoader,
    tgt_loader: torch.utils.data.DataLoader,
    device: torch.device,
    learning_rate: float = 4e-4,
    max_epochs: int = 500,
    stop_after_epochs: int = 20,
    lambda_max: float = 1.0,
    gamma: float = 10.0,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    mlflow_run=None,
) -> dict:
    """DANN training with early stopping on source-domain validation log-prob.

    The LR scheduler monitors the **task** validation log-prob on the source
    domain only.  Domain loss is intentionally excluded: as domain confusion
    improves (loss → log 2 ≈ 0.693), the combined loss would mislead the
    scheduler into reducing LR prematurely.

    Parameters
    ----------
    lr_factor : float
        Multiplicative LR reduction factor for ``ReduceLROnPlateau``.
    lr_patience : int
        Epochs without task-val improvement before LR is reduced.

    Returns dict with:
        training_log_probs      – per-epoch source task log-probs
        validation_log_probs    – per-epoch source val log-probs
        training_domain_losses  – per-epoch domain BCE losses
        training_domain_accs    – per-epoch domain classifier accuracy
    """
    posterior.to(device)
    optimizer = torch.optim.Adam(posterior.parameters(), lr=learning_rate, weight_decay=5e-5)
    # Monitor task val log-prob only — domain loss is excluded deliberately.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience,
    )
    best_val_lp    = float("-inf")
    epochs_no_impr = 0
    train_lps, val_lps         = [], []
    domain_losses, domain_accs = [], []

    domain_acc_ema = 1.0          
    ema_alpha      = 0.1          
    lam            = 0.0          

    for epoch in range(max_epochs):
        if domain_acc_ema > 0.52:
            lam = ganin_lambda_schedule(epoch, max_epochs, gamma) * lambda_max

        posterior.domain_classifier.set_lambda(lam)

        posterior.train()
        tgt_iter = iter(tgt_loader)
        epoch_task, epoch_dom, epoch_acc = [], [], []

        for src_batch, src_labels in src_train_loader:
            try:
                tgt_batch, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter     = iter(tgt_loader)
                tgt_batch, _ = next(tgt_iter)

            src_batch  = src_batch.to(device)
            src_labels = src_labels.to(device).float()
            tgt_batch  = tgt_batch.to(device)

            optimizer.zero_grad()
            task_loss, domain_loss, domain_acc = posterior.dann_forward(
                src_batch, src_labels, tgt_batch
            )
            (task_loss + domain_loss).backward()
            optimizer.step()

            epoch_task.append(task_loss.item())
            epoch_dom.append(domain_loss.item())
            epoch_acc.append(domain_acc.item())

        posterior.eval()
        epoch_val = []
        with torch.no_grad():
            for batch, labels in src_val_loader:
                batch  = batch.to(device)
                labels = labels.to(device).float()
                epoch_val.append(posterior.log_prob(labels, batch).mean().item())

        mean_task   = float(np.mean(epoch_task))
        mean_domain = float(np.mean(epoch_dom))
        mean_acc    = float(np.mean(epoch_acc))
        mean_val    = float(np.mean(epoch_val))
        
        median_val  = float(np.median(epoch_val)) # Ignore outliers causing loss jumps

        domain_acc_ema = (1 - ema_alpha) * domain_acc_ema + ema_alpha * mean_acc

        # Store -task_loss so sign convention matches standard mode (higher = better)
        train_lps.append(-mean_task)
        val_lps.append(mean_val)
        domain_losses.append(mean_domain)
        domain_accs.append(mean_acc)
        scheduler.step(median_val)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:4d} | lambda={lam:.3f} | "
            f"task: {mean_task:.4f} | domain: {mean_domain:.4f} | "
            f"dom-acc: {mean_acc:.3f} | val: {mean_val:.4f} | "
            f"lr: {current_lr:.2e}"
        )
        mlflow_log_epoch(
            epoch, train_lp=-mean_task, val_lp=mean_val, lr=current_lr,
            domain_loss=mean_domain, domain_acc=mean_acc, lam=lam,
        )

        if median_val > best_val_lp:
            best_val_lp    = median_val
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= stop_after_epochs:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    return {
        "training_log_probs":     train_lps,
        "validation_log_probs":   val_lps,
        "training_domain_losses": domain_losses,
        "training_domain_accs":   domain_accs,
    }

def plot_training_curves(summary: dict, save_path: str) -> None:
    has_dann = "training_domain_losses" in summary
    has_mmd  = "mmd_losses" in summary
    ncols    = 2 if (has_dann or has_mmd) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    ax1 = axes[0] if ncols == 2 else axes

    colors = list(mcolors.TABLEAU_COLORS)
    ax1.plot(summary["training_log_probs"],   ls="-",  label="train", c=colors[0])
    ax1.plot(summary["validation_log_probs"], ls="--", label="val",   c=colors[0])
    ax1.set_xlim(0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log probability")
    ax1.set_title("Task (flow log-prob)")
    ax1.legend()

    if has_dann:
        ax2  = axes[1]
        ax2r = ax2.twinx()
        ax2.plot( summary["training_domain_losses"], label="domain loss", c=colors[1])
        ax2r.plot(summary["training_domain_accs"],   label="domain acc",  c=colors[2], ls="--")
        ax2.set_xlim(0)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Domain BCE loss")
        ax2r.set_ylabel("Domain accuracy")
        ax2.set_title("Domain adversary (DANN)")
        lines = ax2.get_lines() + ax2r.get_lines()
        ax2.legend(lines, [l.get_label() for l in lines])

    if has_mmd:
        ax2 = axes[1]
        ax2.plot(summary["mmd_losses"], label="MMD loss", c=colors[1])
        ax2.set_xlim(0)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MMD² × λ")
        ax2.set_title("Domain alignment (MMD)")
        ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def train_mmd_posterior(
    posterior: MMDFlowPosterior,
    src_train_loader: torch.utils.data.DataLoader,
    src_val_loader: torch.utils.data.DataLoader,
    tgt_loader: torch.utils.data.DataLoader,
    device: torch.device,
    learning_rate: float = 4e-4,
    max_epochs: int = 500,
    stop_after_epochs: int = 20,
    lambda_max: float = 1.0,
    gamma: float = 10.0,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    mlflow_run=None,
) -> dict:
    """MMD training with early stopping on source validation log-prob.

    No lambda schedule — MMD weight is fixed at ``posterior.mmd_lambda``.
    Returns dict with training_log_probs, validation_log_probs, mmd_losses.
    """
    posterior.to(device)
    optimizer = torch.optim.Adam(posterior.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience,
    )
    best_val_lp    = float("-inf")
    epochs_no_impr = 0
    train_lps, val_lps, mmd_losses = [], [], []

    lam       = 0.0

    for epoch in range(max_epochs):
        lam = ganin_lambda_schedule(epoch, max_epochs, gamma) * lambda_max
        posterior.mmd_lambda = lam 
        posterior.train()
        tgt_iter = iter(tgt_loader)
        epoch_task, epoch_mmd = [], []

        for src_batch, src_labels in src_train_loader:
            try:
                tgt_batch, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter      = iter(tgt_loader)
                tgt_batch, _  = next(tgt_iter)

            src_batch  = src_batch.to(device)
            src_labels = src_labels.to(device).float()
            tgt_batch  = tgt_batch.to(device)

            optimizer.zero_grad()
            task_loss, mmd_loss = posterior.mmd_forward(src_batch, src_labels, tgt_batch)
            (task_loss + mmd_loss).backward()
            optimizer.step()

            epoch_task.append(task_loss.item())
            epoch_mmd.append(mmd_loss.item())

        posterior.eval()
        epoch_val = []
        with torch.no_grad():
            for batch, labels in src_val_loader:
                batch  = batch.to(device)
                labels = labels.to(device).float()
                epoch_val.append(posterior.log_prob(labels, batch).mean().item())

        mean_task = float(np.mean(epoch_task))
        mean_mmd  = float(np.mean(epoch_mmd))
        mean_val  = float(np.mean(epoch_val))
        median_val = float(np.median(epoch_val))

        train_lps.append(-mean_task)
        val_lps.append(mean_val)
        mmd_losses.append(mean_mmd)
        scheduler.step(median_val)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:4d} | lambda={lam:.3f} | "
            f"task: {mean_task:.4f} | mmd: {mean_mmd:.6f} | "
            f"val: {mean_val:.4f} | lr: {current_lr:.2e}"
        )
        mlflow_log_epoch(
            epoch, train_lp=-mean_task, val_lp=mean_val, lr=current_lr,
            mmd_loss=mean_mmd, lam=lam,
        )

        if mean_val > best_val_lp:
            best_val_lp    = mean_val
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= stop_after_epochs:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    return {
        "training_log_probs":   train_lps,
        "validation_log_probs": val_lps,
        "mmd_losses":           mmd_losses,
    }

def apply_resolution_mask(
    samples: np.ndarray,
    truths: np.ndarray,
    resolution_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of samples and truths with unresolved dims set to NaN.

    Parameters
    ----------
    samples : ndarray, shape ``(n_samples, N, n_labels)``
    truths  : ndarray, shape ``(N, n_labels)``
    resolution_mask : ndarray, shape ``(N, n_labels)``, bool
        True = resolved (keep), False = unresolved (NaN out).
    """
    samples = samples.copy()
    truths  = truths.copy().astype(float)
    # samples[:, i, d] = NaN for all unresolved (i, d)
    unresolved = ~resolution_mask                        # (N, n_labels)
    samples[:, unresolved] = np.nan
    truths[unresolved]     = np.nan
    return samples, truths
    
def compute_ratio_stats(
    samples: np.ndarray,
    truths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, p16, p84) of M_pred/M_true across galaxies."""
    medians = 10 ** np.nanmedian(samples, axis=0) / (10 ** truths)
    return (
        np.nanmedian(medians,      axis=0),
        np.nanpercentile(medians, 16, axis=0),
        np.nanpercentile(medians, 84, axis=0),
    )


def plot_mass_ratios(
    label_mass_radii_frac: np.ndarray,
    ratio_stats: dict[str, tuple],
    classical_rel: dict[str, np.ndarray],
    sim: str,
    nstars: int,
    save_path: str,
    capsize: int = 5,
    markersize: int = 3,
) -> None:
    marker_styles   = {"Walker": "o", "Wolf": "d", "Amorisco": "*", "Campbell": "<", "Errani": "^"}
    estimator_radii = {"Walker": 1.0, "Wolf": 4/3, "Amorisco": 1.7, "Campbell": 1.8, "Errani": 1.8}

    fig, ax = plt.subplots()
    ax.axhline(1, ls="--", color="k")

    for split, (med, p16, p84) in ratio_stats.items():
        ax.plot(label_mass_radii_frac, med, label=split)
        ax.fill_between(label_mass_radii_frac, p16, p84, alpha=0.5)

    ax.plot(10 ** np.array(GENINA_MED["x"]),  GENINA_MED["y"],  ls=":", color="darkgray", label="Genina+20")
    ax.plot(10 ** np.array(GENINA_UP["x"]),   GENINA_UP["y"],   ls=":", color="darkgray")
    ax.plot(10 ** np.array(GENINA_DOWN["x"]), GENINA_DOWN["y"], ls=":", color="darkgray")

    for name, rel_values in classical_rel.items():
        r     = estimator_radii[name]
        med_r = np.nanmedian(rel_values)
        err   = np.array([
            [med_r - np.nanpercentile(rel_values, 16)],
            [np.nanpercentile(rel_values, 84) - med_r],
        ])
        ax.errorbar(r, med_r, err, capsize=capsize,
                    fmt=marker_styles[name], markersize=markersize, label=name)

    ax.set_xlabel(r"r/R$_{\rm h}$")
    ax.set_ylabel(r"M(<r)$_{\rm pred}$/M(<r)$_{\rm true}$")
    ax.set_title(f"{sim} {nstars} stars")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_label_distributions(
    labels_src: np.ndarray,
    labels_tgt: np.ndarray,
    labels_tgt_filtered: np.ndarray,
    save_dir: str,
) -> None:
    for i in range(labels_src.shape[1]):
        fig, ax = plt.subplots()
        ax.hist(labels_src[:, i],          bins=30, density=True, alpha=0.5, label="Source (train/val)")
        ax.hist(labels_tgt_filtered[:, i], bins=30, density=True, alpha=0.5, label="Target/test (filtered)")
        ax.hist(labels_tgt[:, i],          bins=30, density=True, alpha=0.5, label="Target/test (all)")
        ax.set_xlabel(f"Feature {i}")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Distribution of feature {i}")
        fig.savefig(os.path.join(save_dir, f"feature_{i}_distribution.png"), bbox_inches="tight")
        plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# build_data  — load raw phase-space, build graphs, split, standardize
# ─────────────────────────────────────────────────────────────────────────────

def build_data(args, label_file, data_folder, seed):
    """Load raw data, build graphs, apply mask_missing='drop' if requested,
    split into train/val, compute standardization statistics.

    Returns a dict with all data artefacts needed by build_model() and
    run_evaluation().
    """
    use_dann  = args.dann_source is not None
    test_long = bool(args.test_long)

    test_limit = None if test_long else 100
    src_key    = args.dann_source if use_dann else args.train_set
    tgt_key    = args.test_set

    if src_key == tgt_key:
        src_indices, tgt_indices = get_split_indices(
            label_file, src_key, limit=test_limit, seed=seed
        )
    else:
        src_indices = resolve_indices(label_file, src_key)
        tgt_indices = resolve_indices(label_file, tgt_key, limit=test_limit)

    mode_tag = "DANN" if use_dann else "Standard"
    print(
        f"{mode_tag}  |  source: {src_key} ({len(src_indices)} galaxies)  |  "
        f"{'target' if use_dann else 'test'}: {tgt_key} ({len(tgt_indices)} galaxies)"
    )

    n_total    = (len(src_indices) + len(tgt_indices)) * args.N_proj_per_gal
    nstars_arr = np.random.poisson(args.N_stars, size=n_total)

    label_mass_radii_frac = np.arange(0.6, 2.6, 0.2)
    mass_estim_radii_frac = np.array([1.0, 4/3, 1.67, 1.77, 1.8])

    raw      = load_phase_space(
        data_folder, src_indices, tgt_indices,
        args.N_proj_per_gal, nstars_arr,
        label_mass_radii_frac, mass_estim_radii_frac,
        label_file,
    )
    src_size = raw["train_and_val_size"]

    labels_all = np.array(raw["labels"])
    labels_src = labels_all[:src_size]
    labels_tgt = labels_all[src_size:]
    hlrs_src   = raw["hlrs"][:src_size]
    stds_src   = raw["stds"][:src_size]
    hlrs_tgt   = raw["hlrs"][src_size:]
    stds_tgt   = raw["stds"][src_size:]

    mask_test = np.ones(len(labels_tgt), dtype=bool)
    for dim in range(labels_tgt.shape[1]):
        lo, hi = labels_src[:, dim].min(), labels_src[:, dim].max()
        mask_test &= (labels_tgt[:, dim] > lo) & (labels_tgt[:, dim] < hi)
    mask_test &= (hlrs_tgt > hlrs_src.min()) & (hlrs_tgt < hlrs_src.max())
    mask_test &= (stds_tgt > stds_src.min()) & (stds_tgt < stds_src.max())
    print(f"Test projections after filtering: {mask_test.sum()} / {len(mask_test)}")

    labels_tgt_filtered = labels_tgt[mask_test]

    k_neighbors   = min(args.N_stars, 20)
    graph_creator = GraphCreator(
        graph_type   = "KNNGraph",
        graph_config = {"k": k_neighbors, "force_undirected": True, "loop": True},
        use_log_radius = True,
    )
    graphs_all, hlrs_from_graphs, stds_from_graphs = build_graph_list(raw, graph_creator)

    graphs_src     = graphs_all[:src_size]
    graphs_tgt_all = graphs_all[src_size:]
    graphs_test    = [g for g, m in zip(graphs_tgt_all, mask_test) if m]

    if args.mask_missing == "drop":
        graphs_src, hlrs_from_graphs_src, stds_from_graphs_src = filter_fully_resolved(
            graphs_src,
            hlrs_from_graphs[:src_size],
            stds_from_graphs[:src_size],
        )
        # Rebuild aligned file_indices after filtering
        keep_mask       = np.array([g.mask.min().item() == 1.0
                                    for g in graphs_all[:src_size]])
        file_indices_src = raw["file_indices"][:src_size][keep_mask]
        labels_src       = labels_src[keep_mask]
        masks_src_arr    = raw["masks"][:src_size][keep_mask]
    else:
        hlrs_from_graphs_src = hlrs_from_graphs[:src_size]
        stds_from_graphs_src = stds_from_graphs[:src_size]
        file_indices_src     = raw["file_indices"][:src_size]
        masks_src_arr        = raw["masks"][:src_size]

    src_train_graphs, src_val_graphs, train_mask, val_mask = train_val_split(
        graphs_src, file_indices_src
    )

    x_mean, x_std, y_mean, y_std, hlr_mean, hlr_std, s_mean, s_std = \
        compute_standardization_stats(src_train_graphs, args.mask_missing)
    print(
        f"Input  standardization — mean: {x_mean.numpy().round(4)}  "
        f"std: {x_std.numpy().round(4)}"
    )
    print(
        f"Output standardization — mean: {y_mean.numpy().round(4)}  "
        f"std: {y_std.numpy().round(4)}"
    )

    # ── Classical estimators ──────────────────────────────────────────────────
    classical_masses_src = compute_classical_estimators(
        stds_from_graphs_src, hlrs_from_graphs_src
    )
    estim_masses = raw["estimator_masses"]

    # classical_rel uses only src projections (pre-drop filter aligns indices)
    src_proj_count = len(graphs_src)
    classical_rel = {
        "Walker":   classical_masses_src["Walker"]   / estim_masses[:src_size, 0][:src_proj_count],
        "Wolf":     classical_masses_src["Wolf"]     / estim_masses[:src_size, 1][:src_proj_count],
        "Amorisco": classical_masses_src["Amorisco"] / estim_masses[:src_size, 2][:src_proj_count],
        "Errani":   classical_masses_src["Errani"]   / estim_masses[:src_size, 3][:src_proj_count],
        "Campbell": classical_masses_src["Campbell"] / estim_masses[:src_size, 4][:src_proj_count],
    }

    return dict(
        # keys / indices
        src_key              = src_key,
        tgt_key              = tgt_key,
        src_indices          = src_indices,
        tgt_indices          = tgt_indices,
        nstars_arr           = nstars_arr,
        label_mass_radii_frac= label_mass_radii_frac,
        # graph lists
        src_train_graphs     = src_train_graphs,
        src_val_graphs       = src_val_graphs,
        graphs_tgt_all       = graphs_tgt_all,
        graphs_test          = graphs_test,
        # labels / masks
        labels_src           = labels_src,
        labels_tgt_filtered  = labels_tgt_filtered,
        masks_src_arr        = masks_src_arr,
        train_mask           = train_mask,
        val_mask             = val_mask,
        mask_test            = mask_test,
        # standardization tensors
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
        hlr_mean=hlr_mean, hlr_std=hlr_std,
        s_mean=s_mean, s_std=s_std,
        # diagnostics
        hlrs_from_graphs     = hlrs_from_graphs,
        stds_from_graphs     = stds_from_graphs,
        classical_rel        = classical_rel,
        estim_masses         = estim_masses,
        raw                  = raw,
        n_labels             = len(label_mass_radii_frac),
    )


# ─────────────────────────────────────────────────────────────────────────────
# build_model  — construct posterior and register standardization buffers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args, data):
    """Construct embedding, flow, and posterior from CLI args and data stats.

    Returns the posterior (compiled) ready for training or loading.
    """
    use_dann     = args.dann_source is not None
    use_mmd      = use_dann and args.da_method == "mmd"
    use_dann_adv = use_dann and args.da_method == "dann"
    use_hlr_std  = bool(args.hlr_std)
    n_labels     = data["n_labels"]

    layer_cfg = GRAPH_LAYER_CONFIGS[args.GraphNN_type]
    embedding = GraphNN(
        in_channels           = 2,
        out_channels          = args.hidden_dim,
        hidden_graph_channels = args.hidden_dim,
        num_graph_layers      = args.num_graph_layers,
        hidden_fc_channels    = args.hidden_dim,
        num_fc_layers         = args.num_fc_layers,
        dropout               = args.dropout,
        hlr_std               = use_hlr_std,
        use_residuals         = bool(args.use_residuals),
        use_batch_norm        = bool(args.use_batch_norm),
        **layer_cfg,
    )

    # "mask" mode appends the resolution mask vector to the context
    context_dim = args.hidden_dim
    if args.mask_missing == "mask":
        context_dim += n_labels

    flow = build_maf_flow(
        features         = n_labels,
        context_features = context_dim,
        hidden_features  = args.hidden_dim,
        num_transforms   = 4,
    )

    if use_dann_adv:
        domain_cls = DomainClassifier(
            in_features     = args.hidden_dim,
            hidden_features = args.dann_domain_hidden,
            lambda_         = 0.0,
        )
        posterior: DANNFlowPosterior | MMDFlowPosterior | FlowPosterior = DANNFlowPosterior(
            embedding_net      = embedding,
            flow               = flow,
            domain_classifier  = domain_cls,
            mask_missing       = args.mask_missing,
        )
    elif use_mmd:
        posterior = MMDFlowPosterior(
            embedding_net = embedding,
            flow          = flow,
            mmd_lambda    = args.mmd_lambda,
            mask_missing  = args.mask_missing,
        )
    else:
        posterior = FlowPosterior(
            embedding_net = embedding,
            flow          = flow,
            mask_missing  = args.mask_missing,
        )

    posterior = torch.compile(posterior, dynamic=True)
    posterior.set_standardization(
        data["x_mean"], data["x_std"],
        data["y_mean"], data["y_std"],
        data["hlr_mean"], data["hlr_std"],
        data["s_mean"], data["s_std"],
    )
    return posterior


# ─────────────────────────────────────────────────────────────────────────────
# run_training  — select training loop, train, save
# ─────────────────────────────────────────────────────────────────────────────

def run_training(args, posterior, data, device, model_folder, mlflow_run=None):
    """Run the appropriate training loop and save the model.

    Returns the training summary dict.
    """
    use_dann     = args.dann_source is not None
    use_mmd      = use_dann and args.da_method == "mmd"
    use_dann_adv = use_dann and args.da_method == "dann"

    src_train_graphs = data["src_train_graphs"]
    src_val_graphs   = data["src_val_graphs"]
    graphs_tgt_all   = data["graphs_tgt_all"]

    if use_dann_adv:
        src_train_loader, src_val_loader, tgt_loader = make_dann_loaders(
            src_train_graphs, src_val_graphs, graphs_tgt_all,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        summary = train_dann_posterior(
            posterior, src_train_loader, src_val_loader, tgt_loader, device,
            learning_rate=4e-4, max_epochs=500, stop_after_epochs=20,
            lambda_max=args.dann_lambda, gamma=args.dann_gamma,
            lr_factor=0.5, lr_patience=15,
            mlflow_run=mlflow_run,
        )
    elif use_mmd:
        src_train_loader, src_val_loader, tgt_loader = make_dann_loaders(
            src_train_graphs, src_val_graphs, graphs_tgt_all,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        summary = train_mmd_posterior(
            posterior, src_train_loader, src_val_loader, tgt_loader, device,
            learning_rate=4e-4, max_epochs=500, stop_after_epochs=20,
            lambda_max=args.mmd_lambda, gamma=args.mmd_gamma,
            lr_factor=0.5, lr_patience=15,
            mlflow_run=mlflow_run,
        )
    else:
        train_loader, val_loader = make_data_loaders(
            src_train_graphs, src_val_graphs,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        summary = train_posterior(
            posterior, train_loader, val_loader, device,
            learning_rate=4e-4, max_epochs=500, stop_after_epochs=10,
            lr_factor=0.5, lr_patience=15,
            mlflow_run=mlflow_run,
        )

    plot_training_curves(summary, os.path.join(model_folder, "loss.png"))
    with open(os.path.join(model_folder, "posterior.pkl"), "wb") as f:
        pickle.dump(posterior, f)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# run_evaluation  — sample, compute ratio stats, plot, save
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args, posterior, data, device, model_folder, seed, nstars_arr, mlflow_run=None):
    """Sample the posterior on train/val/test sets, compute mass-ratio
    statistics, produce plots, and save plot_data.pkl.
    """
    use_dann  = args.dann_source is not None
    n_labels  = data["n_labels"]
    n_samples   = 1000
    timeout_sec = 5 * 60

    src_train_graphs    = data["src_train_graphs"]
    src_val_graphs      = data["src_val_graphs"]
    graphs_test         = data["graphs_test"]
    labels_src          = data["labels_src"]
    labels_tgt_filtered = data["labels_tgt_filtered"]
    train_mask          = data["train_mask"]
    val_mask            = data["val_mask"]
    mask_test           = data["mask_test"]
    masks_src_arr       = data["masks_src_arr"]
    src_key             = data["src_key"]
    tgt_key             = data["tgt_key"]
    classical_rel       = data["classical_rel"]
    label_mass_radii_frac = data["label_mass_radii_frac"]

    samples_train_path = os.path.join(model_folder, "samples_train.npy")
    samples_val_path   = os.path.join(model_folder, "samples_val.npy")
    samples_test_path  = os.path.join(model_folder, "samples_test.npy")

    if args.mode in ("train", "sample"):
        print("Sampling source training set ...")
        samples_train = sample_with_timeout(
            posterior, src_train_graphs, n_samples, device, timeout_sec, n_labels
        )
        np.save(samples_train_path, samples_train)

        print("Sampling source validation set ...")
        samples_val = sample_with_timeout(
            posterior, src_val_graphs, n_samples, device, timeout_sec, n_labels
        )
        np.save(samples_val_path, samples_val)
    else:
        samples_train = np.load(samples_train_path)
        samples_val   = np.load(samples_val_path)

    print("Sampling test (target) set ...")
    samples_test = sample_with_timeout(
        posterior, graphs_test, n_samples, device, timeout_sec, n_labels
    )
    np.save(samples_test_path, samples_test)

    truths_train = labels_src[train_mask]
    truths_val   = labels_src[val_mask]
    truths_test  = labels_tgt_filtered

    masks_train = masks_src_arr[train_mask]
    masks_val   = masks_src_arr[val_mask]
    masks_tgt_all = data["raw"]["masks"][len(labels_src):]
    masks_test    = masks_tgt_all[mask_test]

    ratio_stats = {
        "Training":   compute_ratio_stats(samples_train, truths_train),
        "Validation": compute_ratio_stats(samples_val,   truths_val),
        "Testing":    compute_ratio_stats(samples_test,  truths_test),
    }

    if args.mask_missing in ("mask", "BIF", "drop"):
        s_train_m, t_train_m = apply_resolution_mask(samples_train, truths_train, masks_train)
        s_val_m,   t_val_m   = apply_resolution_mask(samples_val,   truths_val,   masks_val)
        s_test_m,  t_test_m  = apply_resolution_mask(samples_test,  truths_test,  masks_test)
        ratio_stats_resolved = {
            "Training":   compute_ratio_stats(s_train_m, t_train_m),
            "Validation": compute_ratio_stats(s_val_m,   t_val_m),
            "Testing":    compute_ratio_stats(s_test_m,  t_test_m),
        }
    else:
        ratio_stats_resolved = ratio_stats

    for label, rs in [("ALL DIMS", ratio_stats), ("RESOLVED ONLY", ratio_stats_resolved)]:
        print(f"\n── {label} ──")
        for split, (med, p16, p84) in rs.items():
            print(f"\n{split.upper()}")
            print("  median:",   med)
            print("  p84-p16:", p84 - p16)

    plot_title = (
        f"DANN  src={src_key}  tgt={tgt_key}" if use_dann
        else f"{src_key} -> {tgt_key}"
    )
    plot_mass_ratios(
        label_mass_radii_frac, ratio_stats, classical_rel,
        sim       = plot_title,
        nstars    = args.N_stars,
        save_path = os.path.join(model_folder, "TrainingVsValidationVsTest.png"),
    )
    if args.mask_missing:
        plot_mass_ratios(
            label_mass_radii_frac, ratio_stats_resolved, classical_rel,
            sim       = plot_title + "  [resolved only]",
            nstars    = args.N_stars,
            save_path = os.path.join(model_folder, "TrainingVsValidationVsTest_resolved.png"),
        )

    def _rs_to_dict(rs):
        return {k: {"med": v[0], "p16": v[1], "p84": v[2]} for k, v in rs.items()}

    plot_data = dict(
        label_mass_radii_frac = label_mass_radii_frac,
        ratio_stats           = _rs_to_dict(ratio_stats),
        ratio_stats_resolved  = _rs_to_dict(ratio_stats_resolved),
        train_set        = src_key,
        test_set         = tgt_key,
        dann             = use_dann,
        dann_source      = args.dann_source,
        dann_lambda      = args.dann_lambda if use_dann else None,
        dann_gamma       = args.dann_gamma  if use_dann else None,
        Nstars           = args.N_stars,
        actual_n         = nstars_arr,
        seed             = seed,
        N_files_train    = len(data["src_indices"]),
        N_files_test     = len(data["tgt_indices"]),
        N_proj_per_gal   = args.N_proj_per_gal,
        hlrs             = data["hlrs_from_graphs"].tolist(),
        stds             = data["stds_from_graphs"].tolist(),
        estim_masses     = data["estim_masses"].tolist(),
        classical_masses = classical_rel,
        test_long        = bool(args.test_long),
        model_folder     = model_folder,
    )
    plot_data_path = os.path.join(model_folder, "plot_data.pkl")
    with open(plot_data_path, "wb") as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to {plot_data_path}")

    mlflow_log_results(ratio_stats, ratio_stats_resolved)
    mlflow_log_artefacts(model_folder, args)


# ─────────────────────────────────────────────────────────────────────────────
# main  — orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    use_dann   = args.dann_source is not None
    pca_filter = bool(args.PCAfilter)

    accelerator = Accelerator()
    device      = accelerator.device

    seed = 22133
    np.random.seed(seed)

    pca_suffix    = "PCAfilt" if pca_filter else "PCAnofilt"
    data_folder   = (
        f"/net/debut/project/jsarrato/Paper-GraphSimProfiles/work/"
        f"arrs_NIHAO_and_AURIGA_{pca_suffix}_1000/"
    )
    main_work_dir = "/net/deimos/scratch/jsarrato/Wolf_for_FIRE/work/"

    label_file = load_label_file(data_folder, pca_filter)

    data = build_data(args, label_file, data_folder, seed)

    src_key  = data["src_key"]
    tgt_key  = data["tgt_key"]
    dann_tag = f"_DANN-src{src_key}" if use_dann else ""
    mask_tag = f"_mask-{args.mask_missing}" if args.mask_missing else ""
    if use_dann:
        if args.da_method == "mmd":
            da_tag = f"_mmd-lam{args.mmd_lambda}-gam{args.mmd_gamma}"
        else:
            da_tag = f"_dann-lam{args.dann_lambda}-gam{args.dann_gamma}"
    else:
        da_tag = ""
    model_str = (
        f"{args.GraphNN_type}"
        f"_src{src_key}_tgt{tgt_key}{dann_tag}{da_tag}{mask_tag}"
        f"_h{args.hidden_dim}_gl{args.num_graph_layers}_fc{args.num_fc_layers}"
        f"_res{args.use_residuals}_bn{args.use_batch_norm}_do{args.dropout}"
        f"_poisson{args.N_stars}_Nfiles{len(data['src_indices'])}"
        f"_Nproj{args.N_proj_per_gal}_hlrstd{args.hlr_std}"
    )
    model_folder = os.path.join(main_work_dir, "Graph+Flow_Mocks_NH/new/", model_str)
    os.makedirs(model_folder, exist_ok=True)

    plot_label_distributions(
        np.array(data["raw"]["labels"])[:data["raw"]["train_and_val_size"]],
        np.array(data["raw"]["labels"])[data["raw"]["train_and_val_size"]:],
        data["labels_tgt_filtered"],
        model_folder,
    )

    import json as _json
    with open(os.path.join(model_folder, "graphs_train.pkl"), "wb") as _f:
        pickle.dump(data["src_train_graphs"], _f)
    with open(os.path.join(model_folder, "graphs_val.pkl"), "wb") as _f:
        pickle.dump(data["src_val_graphs"], _f)
    _std = dict(
        x_mean   = data["x_mean"].tolist(),   x_std    = data["x_std"].tolist(),
        y_mean   = data["y_mean"].tolist(),   y_std    = data["y_std"].tolist(),
        hlr_mean = float(data["hlr_mean"]),   hlr_std  = float(data["hlr_std"]),
        s_mean   = float(data["s_mean"]),     s_std    = float(data["s_std"]),
    )
    with open(os.path.join(model_folder, "std_stats.json"), "w") as _f:
        _json.dump(_std, _f)

    posterior      = build_model(args, data)
    posterior_path = os.path.join(model_folder, "posterior.pkl")

    with setup_mlflow(args, model_str) as mlflow_run:
        mlflow_log_hparams(args, data)
        mlflow.log_param("model_folder", model_folder)

        if args.mode == "train":
            run_training(args, posterior, data, device, model_folder,
                         mlflow_run=mlflow_run)
        else:
            with open(posterior_path, "rb") as f:
                posterior = pickle.load(f)

        run_evaluation(
            args, posterior, data, device, model_folder,
            seed=seed, nstars_arr=data["nstars_arr"],
            mlflow_run=mlflow_run,
        )


if __name__ == "__main__":
    main()

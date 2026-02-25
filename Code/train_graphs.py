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

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from scipy.interpolate import interp1d
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from utils import (
    DANNFlowPosterior,
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
        "--mask_missing", choices=["mask", "BIF"], default=None,
        help="Strategy for masking unresolved radii: 'mask' (context aware) or 'BIF' (Bayesian Imputation)",
    )
    return parser.parse_args()


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
        hlrs, stds, estimator_masses, train_and_val_size
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
    num_workers: int = 1,
) -> torch.utils.data.DataLoader:
    collater = Collater(graphs)

    def collate_fn(batch):
        batch = collater(batch)
        return batch, batch.y

    return torch.utils.data.DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn,
    )

def make_data_loaders(
    train_graphs: list,
    val_graphs: list,
    batch_size: int = 64,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return (
        _make_loader(train_graphs, batch_size, shuffle=True),
        _make_loader(val_graphs,   batch_size, shuffle=False),
    )

def make_dann_loaders(
    src_train: list,
    src_val: list,
    tgt_all: list,
    batch_size: int = 64,
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
        _make_loader(src_train, batch_size, shuffle=True),
        _make_loader(src_val,   batch_size, shuffle=False),
        _make_loader(tgt_all,   batch_size, shuffle=True),
    )

def train_posterior(
    posterior: FlowPosterior,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    learning_rate: float = 4e-4,
    max_epochs: int = 500,
    stop_after_epochs: int = 10,
) -> dict:
    """Train FlowPosterior with early stopping on validation log-prob.

    Returns dict with training_log_probs and validation_log_probs.
    """
    posterior.to(device)
    optimizer = torch.optim.Adam(posterior.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.5, patience=5
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

        print(
            f"Epoch {epoch + 1:4d} | "
            f"train log-prob: {mean_train:.4f} | "
            f"val log-prob:   {mean_val:.4f}"
        )

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
    stop_after_epochs: int = 10,
    lambda_max: float = 1.0,
    gamma: float = 10.0,
) -> dict:
    """DANN training with early stopping on source-domain validation log-prob.

    1. Ramp lambda via the Ganin sigmoid schedule (0 -> lambda_max).
    2. For each source batch, draw one target batch,
    compute task + domain losses, back-propagate
    3. Validate task log-prob on the source validation split only.

    Returns dict with:
        training_log_probs      – per-epoch source task log-probs
        validation_log_probs    – per-epoch source val log-probs
        training_domain_losses  – per-epoch domain BCE losses
        training_domain_accs    – per-epoch domain classifier accuracy
    """
    posterior.to(device)
    optimizer = torch.optim.Adam(posterior.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.5, patience=5
    )
    best_val_lp    = float("-inf")
    epochs_no_impr = 0
    train_lps, val_lps         = [], []
    domain_losses, domain_accs = [], []

    for epoch in range(max_epochs):
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

        # Store -task_loss so sign convention matches standard mode (higher = better)
        train_lps.append(-mean_task)
        val_lps.append(mean_val)
        domain_losses.append(mean_domain)
        domain_accs.append(mean_acc)
        scheduler.step(mean_val)

        print(
            f"Epoch {epoch + 1:4d} | lambda={lam:.3f} | "
            f"task: {mean_task:.4f} | domain: {mean_domain:.4f} | "
            f"dom-acc: {mean_acc:.3f} | val: {mean_val:.4f}"
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
        "training_log_probs":     train_lps,
        "validation_log_probs":   val_lps,
        "training_domain_losses": domain_losses,
        "training_domain_accs":   domain_accs,
    }

def plot_training_curves(summary: dict, save_path: str) -> None:
    """Task log-prob panel; adds domain loss/acc panel for DANN runs."""
    has_dann = "training_domain_losses" in summary
    ncols    = 2 if has_dann else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    ax1 = axes[0] if has_dann else axes

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
        ax2.set_title("Domain adversary")
        lines  = ax2.get_lines() + ax2r.get_lines()
        ax2.legend(lines, [l.get_label() for l in lines])

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


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

def main() -> None:
    args        = parse_args()
    use_dann    = args.dann_source is not None
    test_long   = bool(args.test_long)
    pca_filter  = bool(args.PCAfilter)
    use_hlr_std = bool(args.hlr_std)

    accelerator = Accelerator()
    device      = accelerator.device

    seed = 22133
    np.random.seed(seed)

    pca_suffix  = "PCAfilt" if pca_filter else "PCAnofilt"
    data_folder = (
        f"/net/debut/project/jsarrato/Paper-GraphSimProfiles/work/"
        f"arrs_NIHAO_and_AURIGA_{pca_suffix}_1000/"
    )
    main_work_dir = "/net/deimos/scratch/jsarrato/Wolf_for_FIRE/work/"

    label_file            = load_label_file(data_folder, pca_filter)
    label_mass_radii_frac = np.arange(0.6, 2.6, 0.2)
    mass_estim_radii_frac = np.array([1.0, 4/3, 1.67, 1.77, 1.8])
    n_labels              = len(label_mass_radii_frac)

    test_limit = None if test_long else 100

    src_key = args.dann_source if use_dann else args.train_set
    tgt_key = args.test_set

    if src_key == tgt_key:
        src_indices, tgt_indices = get_split_indices(label_file, src_key, limit=test_limit, seed=seed)
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

    raw        = load_phase_space(
        data_folder, src_indices, tgt_indices,
        args.N_proj_per_gal, nstars_arr,
        label_mass_radii_frac, mass_estim_radii_frac,
        label_file
    )
    src_size = raw["train_and_val_size"]   # projection-level boundary

    dann_tag  = f"_DANN-src{src_key}" if use_dann else ""
    mask_tag = f"_mask-{args.mask_missing}" if args.mask_missing else ""
    model_str = (
        f"{args.GraphNN_type}"
        f"_src{src_key}_tgt{tgt_key}{dann_tag}{mask_tag}"
        f"_poisson{args.N_stars}_Nfiles{len(src_indices)}"
        f"_Nproj{args.N_proj_per_gal}_hlrstd{args.hlr_std}"
    )
    model_folder = os.path.join(main_work_dir, "Graph+Flow_Mocks_NH/new/", model_str)
    os.makedirs(model_folder, exist_ok=True)

    labels_all = np.array(raw["labels"])
    labels_src = labels_all[:src_size]
    labels_tgt = labels_all[src_size:]

    hlrs_src = raw["hlrs"][:src_size]
    stds_src = raw["stds"][:src_size]
    hlrs_tgt = raw["hlrs"][src_size:]
    stds_tgt = raw["stds"][src_size:]

    # Filter target projections to the source distribution
    mask_test = np.ones(len(labels_tgt), dtype=bool)
    for dim in range(labels_tgt.shape[1]):
        lo, hi = labels_src[:, dim].min(), labels_src[:, dim].max()
        mask_test &= (labels_tgt[:, dim] > lo) & (labels_tgt[:, dim] < hi)
    mask_test &= (hlrs_tgt > hlrs_src.min()) & (hlrs_tgt < hlrs_src.max())
    mask_test &= (stds_tgt > stds_src.min()) & (stds_tgt < stds_src.max())
    print(f"Test projections after filtering: {mask_test.sum()} / {len(mask_test)}")

    labels_tgt_filtered = labels_tgt[mask_test]
    plot_label_distributions(labels_src, labels_tgt, labels_tgt_filtered, model_folder)


    k_neighbors   = min(args.N_stars, 20)
    graph_creator = GraphCreator(
        graph_type    = "KNNGraph",
        graph_config  = {"k": k_neighbors, "force_undirected": True, "loop": True},
        use_log_radius= True,
    )

    graphs_all, hlrs_from_graphs, stds_from_graphs = build_graph_list(raw, graph_creator)

    graphs_src     = graphs_all[:src_size]
    graphs_tgt_all = graphs_all[src_size:]
    graphs_test    = [g for g, m in zip(graphs_tgt_all, mask_test) if m]

    estim_masses         = raw["estimator_masses"]
    classical_masses_src = compute_classical_estimators(
        stds_from_graphs[:src_size], hlrs_from_graphs[:src_size]
    )
    classical_rel = {
        "Walker":   classical_masses_src["Walker"]   / estim_masses[:src_size, 0],
        "Wolf":     classical_masses_src["Wolf"]     / estim_masses[:src_size, 1],
        "Amorisco": classical_masses_src["Amorisco"] / estim_masses[:src_size, 2],
        "Errani":   classical_masses_src["Errani"]   / estim_masses[:src_size, 3],
        "Campbell": classical_masses_src["Campbell"] / estim_masses[:src_size, 4],
    }

    file_indices_src                                     = raw["file_indices"][:src_size]
    src_train_graphs, src_val_graphs, train_mask, val_mask = train_val_split(
        graphs_src, file_indices_src
    )

    layer_cfg = GRAPH_LAYER_CONFIGS[args.GraphNN_type]
    embedding = GraphNN(
        in_channels           = 2,
        out_channels          = 128,
        hidden_graph_channels = 128,
        num_graph_layers      = 3,
        hidden_fc_channels    = 128,
        num_fc_layers         = 2,
        hlr_std               = use_hlr_std,
        **layer_cfg,
    )

    context_dim = 128
    if args.mask_missing == "mask":
        context_dim += n_labels

    flow = build_maf_flow(
        features         = n_labels,
        context_features = context_dim,
        hidden_features  = 128,
        num_transforms   = 4,
    )

    if use_dann:
        # The domain classifier always receives the raw 128-dim embedding, never the
        # mask-augmented context.
        domain_cls = DomainClassifier(
            in_features     = 128,
            hidden_features = args.dann_domain_hidden,
            lambda_         = 0.0,   # ramped up from zero during training
        )
        posterior: DANNFlowPosterior | FlowPosterior = DANNFlowPosterior(
            embedding_net     = embedding,
            flow              = flow,
            domain_classifier = domain_cls,
            mask_missing      = args.mask_missing,
        )
    else:
        posterior = FlowPosterior(embedding_net=embedding, flow=flow, mask_missing= args.mask_missing)

    posterior_path = os.path.join(model_folder, "posterior.pkl")

    if args.mode == "train":
        if use_dann:
            src_train_loader, src_val_loader, tgt_loader = make_dann_loaders(
                src_train_graphs, src_val_graphs, graphs_tgt_all, batch_size=64
            )
            summary = train_dann_posterior(
                posterior,
                src_train_loader, src_val_loader, tgt_loader,
                device,
                learning_rate     = 4e-4,
                max_epochs        = 500,
                stop_after_epochs = 10,
                lambda_max        = args.dann_lambda,
                gamma             = args.dann_gamma,
            )
        else:
            train_loader, val_loader = make_data_loaders(
                src_train_graphs, src_val_graphs, batch_size=64
            )
            summary = train_posterior(
                posterior, train_loader, val_loader, device,
                learning_rate     = 4e-4,
                max_epochs        = 500,
                stop_after_epochs = 10,
            )

        plot_training_curves(summary, os.path.join(model_folder, "loss.png"))
        with open(posterior_path, "wb") as f:
            pickle.dump(posterior, f)
    else:
        with open(posterior_path, "rb") as f:
            posterior = pickle.load(f)

    n_samples   = 1000
    timeout_sec = 5 * 60

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

    ratio_stats = {
        "Training":   compute_ratio_stats(samples_train, truths_train),
        "Validation": compute_ratio_stats(samples_val,   truths_val),
        "Testing":    compute_ratio_stats(samples_test,  truths_test),
    }

    for split, (med, p16, p84) in ratio_stats.items():
        print(f"\n{split.upper()}")
        print("  median:",   med)
        print("  p84-p16:", p84 - p16)

    plot_title = (
        f"DANN  src={src_key}  tgt={tgt_key}"
        if use_dann else
        f"{src_key} -> {tgt_key}"
    )
    plot_mass_ratios(
        label_mass_radii_frac,
        ratio_stats,
        classical_rel,
        sim       = plot_title,
        nstars    = args.N_stars,
        save_path = os.path.join(model_folder, "TrainingVsValidationVsTest.png"),
    )

    plot_data = dict(
        label_mass_radii_frac = label_mass_radii_frac,
        ratio_stats = {
            k: {"med": v[0], "p16": v[1], "p84": v[2]}
            for k, v in ratio_stats.items()
        },
        train_set        = src_key,
        test_set         = tgt_key,
        dann             = use_dann,
        dann_source      = args.dann_source,
        dann_lambda      = args.dann_lambda if use_dann else None,
        dann_gamma       = args.dann_gamma  if use_dann else None,
        Nstars           = args.N_stars,
        actual_n         = nstars_arr,
        seed             = seed,
        N_files_train    = len(src_indices),
        N_files_test     = len(tgt_indices),
        N_proj_per_gal   = args.N_proj_per_gal,
        hlrs             = hlrs_from_graphs.tolist(),
        stds             = stds_from_graphs.tolist(),
        estim_masses     = raw["estimator_masses"].tolist(),
        classical_masses = classical_masses_src,
        test_long        = test_long,
        model_folder     = model_folder,
    )
    plot_data_path = os.path.join(model_folder, "plot_data.pkl")
    with open(plot_data_path, "wb") as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to {plot_data_path}")


if __name__ == "__main__":
    main()

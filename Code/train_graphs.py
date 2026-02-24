"""
train_graphs.py — Train a GNN + Masked Autoregressive Flow posterior for
stellar mass estimation from graph-structured phase-space data.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from scipy.interpolate import interp1d
from torch_geometric.data import Batch
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from utils import (
    FlowPosterior,
    GraphCreator,
    GraphNN,
    build_maf_flow,
    sample_with_timeout,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNN + normalizing flow posterior."
    )
    parser.add_argument(
        "mode",
        choices=["train", "sample", "sampletest"],
        help="Run mode: train, sample (train+val+test), or sampletest (test only)",
    )
    parser.add_argument("sim", choices=["AURIGA", "NIHAO"], help="Simulation type")
    parser.add_argument(
        "test_long", type=int, choices=[0, 1], help="Use full test set (1) or first 100 (0)"
    )
    parser.add_argument(
        "hlr_std", type=int, choices=[0, 1], help="Append hlr/std scalars to embedding (0|1)"
    )
    parser.add_argument("N_stars", type=int, help="Expected number of stars (Poisson mean)")
    parser.add_argument("GraphNN_type", choices=["Cheb", "GCN", "GAT"])
    parser.add_argument("N_proj_per_gal", type=int, help="Projections per galaxy")
    parser.add_argument("PCAfilter", type=int, choices=[0, 1], help="PCA-filtered data (0|1)")
    parser.add_argument(
        "SAME", type=int, choices=[0, 1], help="Train and test on same simulation (0|1)"
    )
    parser.add_argument(
        "train_on_highres",
        type=int,
        choices=[0, 1],
        help="Train on high-res, test on low-res within same sim (0|1)",
    )
    return parser.parse_args()

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

KM_TO_M = 1e3
KPC_TO_M = 3.086e19
KG_TO_MSUN = 1.0 / (2e30)
G_SI = 6.6743e-11  # m³ kg⁻¹ s⁻²

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


def load_label_file(data_folder: str, pca_filter: bool) -> pd.DataFrame:
    """Read the label CSV and annotate each row with its simulation origin."""
    suffix = "PCAfilt" if pca_filter else "PCAnofilt"
    csv_path = (
        f"/net/debut/project/jsarrato/Paper-GraphSimProfiles/work/"
        f"proj_data_NIHAO_and_AURIGA_{suffix}_samplearr.csv"
    )
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    df["sim"] = df["name"].apply(lambda x: 1 if x.startswith("halo") else 0)
    return df


def resolve_file_indices(
    label_file: pd.DataFrame,
    sim: str,
    same: bool,
    train_on_highres: bool,
    test_long: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) into label_file."""
    au_idx = np.array(label_file[label_file["sim"] == 1]["i_index"], dtype=int)
    nh_idx = np.array(label_file[label_file["sim"] == 0]["i_index"], dtype=int)

    if same:
        res_limits = {"AURIGA": 0.25, "NIHAO": 1.0}
        sim_flag = 1 if sim == "AURIGA" else 0
        res_lim = res_limits[sim]
        mask = label_file["sim"] == sim_flag
        if train_on_highres:
            train_mask = mask & (label_file["eps_dm"] <= res_lim)
            test_mask  = mask & (label_file["eps_dm"] >  res_lim)
        else:
            train_mask = mask & (label_file["eps_dm"] >  res_lim)
            test_mask  = mask & (label_file["eps_dm"] <= res_lim)
        train_idx = np.array(label_file[train_mask]["i_index"], dtype=int)
        test_idx  = np.array(label_file[test_mask]["i_index"],  dtype=int)
    else:
        if sim == "AURIGA":
            train_idx, test_idx = au_idx, nh_idx
        else:
            train_idx, test_idx = nh_idx, au_idx

    if not test_long:
        test_idx = test_idx[:100]

    return train_idx, test_idx


def load_phase_space(
    data_folder: str,
    file_indices_train: np.ndarray,
    file_indices_test: np.ndarray,
    n_proj_per_gal: int,
    nstars_arr: np.ndarray,
    label_mass_radii_frac: np.ndarray,
    mass_estim_radii_frac: np.ndarray,
) -> dict:
    """Read all galaxy files and extract stellar features and mass labels.

    Returns a dict with keys:
        positions, velocities, labels, file_indices,
        hlrs, stds, estimator_masses, train_and_val_size
    """
    all_indices = np.concatenate([file_indices_train, file_indices_test])
    len_train = len(file_indices_train)

    positions, velocities, labels = [], [], []
    file_indices_out = []
    hlrs, stds, estimator_masses = [], [], []
    train_and_val_size = None

    for i, idx in enumerate(tqdm(all_indices, desc="Reading data")):
        masses_name = data_folder + f"mass_interp{idx}.npz"
        posvel_name = data_folder + f"posvel_{idx}.pkl"

        mass_data = np.load(masses_name)
        mass_interpolator = interp1d(mass_data["x"], mass_data["y"])
        posvel = torch.load(posvel_name, weights_only=False)

        proj_indices = np.random.choice(len(posvel), n_proj_per_gal)

        for j, proj_idx in enumerate(proj_indices):
            nstars = nstars_arr[i * n_proj_per_gal + j]
            data = posvel[proj_idx]

            rxy = np.linalg.norm(data[:nstars, :2], axis=1)
            hlr = np.median(rxy)
            vstd = np.std(data[:nstars, -1])

            try:
                masses_idx = np.log10(mass_interpolator(label_mass_radii_frac * hlr))
                estim = mass_interpolator(mass_estim_radii_frac * hlr)
            except Exception:
                continue

            if np.any(~np.isfinite(data)) or np.any(~np.isfinite(masses_idx)):
                continue

            positions.append(data[:nstars, :2])
            velocities.append(data[:nstars, -1])
            labels.append(masses_idx)
            file_indices_out.append(i)
            hlrs.append(hlr)
            stds.append(vstd)
            estimator_masses.append(estim)

        if i == len_train - 1:
            train_and_val_size = len(labels)

    return dict(
        positions=positions,
        velocities=velocities,
        labels=labels,
        file_indices=np.array(file_indices_out),
        hlrs=np.array(hlrs),
        stds=np.array(stds),
        estimator_masses=np.array(estimator_masses),
        train_and_val_size=train_and_val_size,
    )

def build_graph_list(
    data: dict,
    graph_creator: GraphCreator,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Apply *graph_creator* to every sample; return graphs, hlrs, stds."""
    graphs, hlrs, stds = [], [], []

    for i in tqdm(range(len(data["labels"])), desc="Building graphs"):
        graph = graph_creator(
            positions=data["positions"][i],
            velocities=data["velocities"][i],
            labels=data["labels"][i],
        )
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
    """Split *graphs* into train and val sets by galaxy file (no data leakage).

    Returns (train_graphs, val_graphs, train_mask, val_mask).
    """
    unique_files = np.unique(file_indices)
    n_val = max(1, int(val_fraction * len(unique_files)))
    val_files = np.random.choice(unique_files, n_val, replace=False)

    val_mask = np.isin(file_indices, val_files)
    train_mask = ~val_mask

    train_graphs = [g for g, m in zip(graphs, train_mask) if m]
    val_graphs   = [g for g, m in zip(graphs, val_mask)   if m]
    return train_graphs, val_graphs, train_mask, val_mask

def make_data_loaders(
    train_graphs: list,
    val_graphs: list,
    batch_size: int = 64,
    num_workers: int = 1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create PyTorch DataLoaders that return (batch, labels) tuples."""
    collater = Collater(train_graphs)

    def collate_fn(batch):
        batch = collater(batch)
        return batch, batch.y

    train_loader = torch.utils.data.DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader

def train_posterior(
    posterior: FlowPosterior,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    learning_rate: float = 4e-4,
    max_epochs: int = 500,
    stop_after_epochs: int = 10,
) -> dict:
    """Train the FlowPosterior with early stopping.

    Returns a dict with ``"training_log_probs"`` and ``"validation_log_probs"``
    lists (one entry per epoch), mirroring the ili summary format.
    """
    posterior.to(device)
    optimizer = torch.optim.Adam(posterior.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=5)

    best_val_lp = float("-inf")
    epochs_no_improve = 0
    train_log_probs, val_log_probs = [], []

    for epoch in range(max_epochs):
        posterior.train()
        epoch_train_lp = []
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            lp = posterior.log_prob(labels, batch).mean()
            (-lp).backward()
            optimizer.step()
            epoch_train_lp.append(lp.item())

        posterior.eval()
        epoch_val_lp = []
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = batch.to(device)
                labels = labels.to(device).float()
                lp = posterior.log_prob(labels, batch).mean()
                epoch_val_lp.append(lp.item())

        mean_train = float(np.mean(epoch_train_lp))
        mean_val   = float(np.mean(epoch_val_lp))
        train_log_probs.append(mean_train)
        val_log_probs.append(mean_val)
        scheduler.step(mean_val)

        print(
            f"Epoch {epoch + 1:4d} | "
            f"train log-prob: {mean_train:.4f} | "
            f"val log-prob:   {mean_val:.4f}"
        )

        if mean_val > best_val_lp:
            best_val_lp = mean_val
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= stop_after_epochs:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    return {"training_log_probs": train_log_probs, "validation_log_probs": val_log_probs}

def plot_training_curves(summary: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = list(mcolors.TABLEAU_COLORS)
    ax.plot(summary["training_log_probs"],   ls="-",  label="train", c=colors[0])
    ax.plot(summary["validation_log_probs"], ls="--", label="val",   c=colors[0])
    ax.set_xlim(0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log probability")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def compute_ratio_stats(
    samples: np.ndarray,
    truths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, p16, p84) of predicted/true mass ratio across the sample."""
    medians = 10 ** np.nanmedian(samples, axis=0) / (10 ** truths)
    return (
        np.nanmedian(medians, axis=0),
        np.nanpercentile(medians, 16, axis=0),
        np.nanpercentile(medians, 84, axis=0),
    )


def plot_mass_ratios(
    label_mass_radii_frac: np.ndarray,
    ratio_stats: dict[str, tuple],
    classical_rel: dict[str, tuple],
    sim: str,
    nstars: int,
    save_path: str,
    capsize: int = 5,
    markersize: int = 3,
) -> None:
    """Plot predicted/true mass ratio vs. r/R_h for all splits + estimators."""
    marker_styles = {"Walker": "o", "Wolf": "d", "Amorisco": "*", "Campbell": "<", "Errani": "^"}
    estimator_radii = {"Walker": 1.0, "Wolf": 4/3, "Amorisco": 1.7, "Campbell": 1.8, "Errani": 1.8}

    fig, ax = plt.subplots()
    ax.axhline(1, ls="--", color="k")

    for split, (med, p16, p84) in ratio_stats.items():
        ax.plot(label_mass_radii_frac, med, label=split)
        ax.fill_between(label_mass_radii_frac, p16, p84, alpha=0.5)

    # Genina+20 reference
    ax.plot(10 ** np.array(GENINA_MED["x"]), GENINA_MED["y"], ls=":", color="darkgray", label="Genina+20")
    ax.plot(10 ** np.array(GENINA_UP["x"]),   GENINA_UP["y"],   ls=":", color="darkgray")
    ax.plot(10 ** np.array(GENINA_DOWN["x"]), GENINA_DOWN["y"], ls=":", color="darkgray")

    for name, rel_values in classical_rel.items():
        r = estimator_radii[name]
        med_r = np.nanmedian(rel_values)
        err = np.array([
            [med_r - np.nanpercentile(rel_values, 16)],
            [np.nanpercentile(rel_values, 84) - med_r],
        ])
        ax.errorbar(r, med_r, err, capsize=capsize, fmt=marker_styles[name],
                    markersize=markersize, label=name)

    ax.set_xlabel(r"r/R$_{\rm h}$")
    ax.set_ylabel(r"M(<r)$_{\rm pred}$/M(<r)$_{\rm true}$")
    ax.set_title(f"{sim} {nstars} stars")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_label_distributions(
    labels_train_val: np.ndarray,
    labels_test: np.ndarray,
    labels_test_filtered: np.ndarray,
    save_dir: str,
) -> None:
    """Save per-feature distribution plots to *save_dir*."""
    for i in range(labels_train_val.shape[1]):
        fig, ax = plt.subplots()
        ax.hist(labels_train_val[:, i],      bins=30, density=True, alpha=0.5, label="Train/Val")
        ax.hist(labels_test_filtered[:, i],  bins=30, density=True, alpha=0.5, label="Test (masked)")
        ax.hist(labels_test[:, i],           bins=30, density=True, alpha=0.5, label="Test (full)")
        ax.set_xlabel(f"Feature {i}")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Distribution of feature {i}")
        fig.savefig(os.path.join(save_dir, f"feature_{i}_distribution.png"), bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    args = parse_args()

    test_long        = bool(args.test_long)
    pca_filter       = bool(args.PCAfilter)
    same             = bool(args.SAME)
    train_on_highres = bool(args.train_on_highres)
    use_hlr_std      = bool(args.hlr_std)
    if train_on_highres:
        assert same, "train_on_highres=1 requires SAME=1"

    accelerator = Accelerator()
    device = accelerator.device

    seed = 22133
    np.random.seed(seed)

    pca_suffix  = "PCAfilt" if pca_filter else "PCAnofilt"
    data_folder = (
        f"/net/debut/project/jsarrato/Paper-GraphSimProfiles/work/"
        f"arrs_NIHAO_and_AURIGA_{pca_suffix}_1000/"
    )
    main_work_dir = "/net/deimos/scratch/jsarrato/Wolf_for_FIRE/work/"

    model_str = (
        f"{args.GraphNN_type}_{args.sim}"
        f"_testonsame{same}_trainonhigh{train_on_highres}"
        f"_poisson{args.N_stars}_Nfiles{{len_train}}"
        f"_Nproj{args.N_proj_per_gal}_hlrstd{args.hlr_std}"
    )

    label_file = load_label_file(data_folder, pca_filter)
    train_indices, test_indices = resolve_file_indices(
        label_file, args.sim, same, train_on_highres, test_long
    )
    print(f"Train galaxies: {len(train_indices)} | Test galaxies: {len(test_indices)}")

    label_mass_radii_frac = np.arange(0.6, 2.6, 0.2)
    mass_estim_radii_frac = np.array([1.0, 4/3, 1.67, 1.77, 1.8])
    n_labels = len(label_mass_radii_frac)

    n_total = (len(train_indices) + len(test_indices)) * args.N_proj_per_gal
    nstars_arr = np.random.poisson(args.N_stars, size=n_total)

    raw = load_phase_space(
        data_folder, train_indices, test_indices,
        args.N_proj_per_gal, nstars_arr,
        label_mass_radii_frac, mass_estim_radii_frac,
    )

    train_and_val_size = raw["train_and_val_size"]
    len_train = len(train_indices)

    model_folder = os.path.join(
        main_work_dir, "Graph+Flow_Mocks_NH/new/",
        model_str.format(len_train=len_train),
    )
    os.makedirs(model_folder, exist_ok=True)

    labels_all      = np.array(raw["labels"])
    labels_train_val = labels_all[:train_and_val_size]
    labels_test      = labels_all[train_and_val_size:]

    hlrs_train_val  = raw["hlrs"][:train_and_val_size]
    stds_train_val  = raw["stds"][:train_and_val_size]
    hlrs_test       = raw["hlrs"][train_and_val_size:]
    stds_test       = raw["stds"][train_and_val_size:]

    mask_test = np.ones(len(labels_test), dtype=bool)
    for dim in range(labels_test.shape[1]):
        lo, hi = labels_train_val[:, dim].min(), labels_train_val[:, dim].max()
        mask_test &= (labels_test[:, dim] > lo) & (labels_test[:, dim] < hi)
    mask_test &= (hlrs_test > hlrs_train_val.min()) & (hlrs_test < hlrs_train_val.max())
    mask_test &= (stds_test > stds_train_val.min()) & (stds_test < stds_train_val.max())
    print(f"Test points after filtering: {mask_test.sum()} / {len(mask_test)}")

    labels_test_filtered = labels_test[mask_test]

    plot_label_distributions(labels_train_val, labels_test, labels_test_filtered, model_folder)

    k_neighbors = min(args.N_stars, 20)
    graph_creator = GraphCreator(
        graph_type="KNNGraph",
        graph_config={"k": k_neighbors, "force_undirected": True, "loop": True},
        use_log_radius=True,
    )

    graphs_all, hlrs_from_graphs, stds_from_graphs = build_graph_list(raw, graph_creator)

    estim_masses   = raw["estimator_masses"]
    stds_combined  = stds_from_graphs  # shape (N_total,)
    hlrs_combined  = hlrs_from_graphs

    classical_masses_train = compute_classical_estimators(
        stds_combined[:train_and_val_size], hlrs_combined[:train_and_val_size]
    )
    classical_rel = {
        "Walker":   classical_masses_train["Walker"]   / estim_masses[:train_and_val_size, 0],
        "Wolf":     classical_masses_train["Wolf"]     / estim_masses[:train_and_val_size, 1],
        "Amorisco": classical_masses_train["Amorisco"] / estim_masses[:train_and_val_size, 2],
        "Errani":   classical_masses_train["Errani"]   / estim_masses[:train_and_val_size, 3],
        "Campbell": classical_masses_train["Campbell"] / estim_masses[:train_and_val_size, 4],
    }

    graphs_train_val = graphs_all[:train_and_val_size]
    graphs_test_all  = graphs_all[train_and_val_size:]
    graphs_test      = [g for g, m in zip(graphs_test_all, mask_test) if m]

    file_indices_tv  = raw["file_indices"][:train_and_val_size]
    train_graphs, val_graphs, train_mask, val_mask = train_val_split(
        graphs_train_val, file_indices_tv
    )

    train_loader, val_loader = make_data_loaders(
        train_graphs, val_graphs, batch_size=64
    )

    layer_cfg = GRAPH_LAYER_CONFIGS[args.GraphNN_type]
    embedding = GraphNN(
        in_channels=2,
        out_channels=128,
        hidden_graph_channels=128,
        num_graph_layers=3,
        hidden_fc_channels=128,
        num_fc_layers=2,
        hlr_std=use_hlr_std,
        **layer_cfg,
    )

    flow = build_maf_flow(
        features=n_labels,
        context_features=128,
        hidden_features=128,
        num_transforms=4,
    )

    posterior = FlowPosterior(embedding_net=embedding, flow=flow)

    posterior_path = os.path.join(model_folder, "posterior.pkl")

    if args.mode == "train":
        summary = train_posterior(
            posterior, train_loader, val_loader, device,
            learning_rate=4e-4,
            max_epochs=500,
            stop_after_epochs=10,
        )
        plot_training_curves(summary, os.path.join(model_folder, "loss.png"))
        with open(posterior_path, "wb") as f:
            pickle.dump(posterior, f)
    else:
        with open(posterior_path, "rb") as f:
            posterior = pickle.load(f)

    n_samples = 1000
    timeout_sec = 5 * 60  # 5 minutes per galaxy

    samples_train_path = os.path.join(model_folder, "samples_train.npy")
    samples_val_path   = os.path.join(model_folder, "samples_val.npy")
    samples_test_path  = os.path.join(model_folder, "samples_test.npy")

    if args.mode in ("train", "sample"):
        print("Sampling training set …")
        samples_train = sample_with_timeout(
            posterior, train_graphs, n_samples, device, timeout_sec, n_labels
        )
        np.save(samples_train_path, samples_train)

        print("Sampling validation set …")
        samples_val = sample_with_timeout(
            posterior, val_graphs, n_samples, device, timeout_sec, n_labels
        )
        np.save(samples_val_path, samples_val)
    else:
        samples_train = np.load(samples_train_path)
        samples_val   = np.load(samples_val_path)

    print("Sampling test set …")
    samples_test = sample_with_timeout(
        posterior, graphs_test, n_samples, device, timeout_sec, n_labels
    )
    np.save(samples_test_path, samples_test)

    truths_train = labels_train_val[train_mask]
    truths_val   = labels_train_val[val_mask]
    truths_test  = labels_test_filtered

    ratio_stats = {
        "Training":   compute_ratio_stats(samples_train, truths_train),
        "Validation": compute_ratio_stats(samples_val,   truths_val),
        "Testing":    compute_ratio_stats(samples_test,  truths_test),
    }

    for split, (med, p16, p84) in ratio_stats.items():
        print(f"\n{split.upper()}")
        print("  median:", med)
        print("  p84-p16:", p84 - p16)

    plot_mass_ratios(
        label_mass_radii_frac,
        ratio_stats,
        classical_rel,
        sim=args.sim,
        nstars=args.N_stars,
        save_path=os.path.join(model_folder, "TrainingVsValidationVsTest.png"),
    )

    plot_data = dict(
        label_mass_radii_frac=label_mass_radii_frac,
        ratio_stats={k: {"med": v[0], "p16": v[1], "p84": v[2]} for k, v in ratio_stats.items()},
        sim=args.sim,
        Nstars=args.N_stars,
        actual_n=nstars_arr,
        seed=seed,
        N_files_train=len_train,
        N_files_test=len(test_indices),
        N_proj_per_gal=args.N_proj_per_gal,
        hlrs=hlrs_from_graphs.tolist(),
        stds=stds_from_graphs.tolist(),
        estim_masses=raw["estimator_masses"].tolist(),
        classical_masses=classical_masses_train,
        test_long=test_long,
        model_folder=model_folder,
    )
    plot_data_path = os.path.join(model_folder, "plot_data.pkl")
    with open(plot_data_path, "wb") as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to {plot_data_path}")


if __name__ == "__main__":
    main()

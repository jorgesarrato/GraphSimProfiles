# Graph Flow Posterior — Dark Matter Mass Estimation from Phase-Space Data

A GNN + normalizing flow pipeline for inferring enclosed dark matter mass profiles of dispersion-supported galaxies from stellar phase-space observations. The model is trained on cosmological simulations (NIHAO, AURIGA) and supports cross-simulation transfer via domain adaptation.

---

## Overview

Each galaxy is represented as a k-NN graph over observed stars, with node features `(log₁₀ radius, line-of-sight velocity)`. A GNN encoder maps this graph to a fixed-size context vector, which conditions a Masked Autoregressive Flow (MAF) that models the posterior over 10 enclosed mass labels at radii `0.6–2.4 × r_half`.

**Labels:** `M(<f × r_half)` at `f ∈ {0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4}` (10 bins).

Three training modes are supported:

| Mode | Description |
|---|---|
| Standard | Train and test on the same simulation family |
| DANN | Adversarial domain adaptation (Ganin et al. 2016) |
| MMD | Maximum Mean Discrepancy alignment |

---

## Repository Structure

```
train_graphs.py   — Main training script (CLI entry point)
utils.py          — GNN, MAF, posterior classes, domain adaptation modules
```

---

## Installation

```bash
pip install torch torch-geometric nflows accelerate mlflow tqdm scipy pandas matplotlib
```

A CUDA-capable GPU is strongly recommended. The code uses `accelerate` for device management and `torch.compile` for the posterior.

---

## Usage

### Standard training

```bash
python train_graphs.py train NIHAO_hi NIHAO_lo 1 100 Cheb 8 0
#                      mode  src       tgt      test_long N_stars arch N_proj PCA
```

### Cross-simulation with MMD domain adaptation

```bash
python train_graphs.py train NIHAO_all AURIGA_all 1 100 Cheb 8 0 \
    --dann_source NIHAO_all \
    --da_method mmd \
    --mmd_lambda 300 \
    --mmd_gamma 5.0
```

### Sample from a trained model

```bash
python train_graphs.py sample NIHAO_hi NIHAO_lo 1 100 Cheb 8 0
```

---

## Population Keys

| Key | Description |
|---|---|
| `NIHAO_lo` | Low-resolution NIHAO (`eps_dm > 1.0 kpc`) |
| `NIHAO_hi` | High-resolution NIHAO (`eps_dm ≤ 1.0 kpc`) |
| `NIHAO_all` | All NIHAO galaxies |
| `NIHAO_shared` | NIHAO galaxies in the resolution range shared with AURIGA |
| `AURIGA_lo` | Low-resolution AURIGA (`eps_dm > 0.25 kpc`) |
| `AURIGA_hi` | High-resolution AURIGA (`eps_dm ≤ 0.25 kpc`) |
| `AURIGA_all` | All AURIGA galaxies |
| `AURIGA_shared` | AURIGA galaxies in the shared resolution range |
| `ALL` | All galaxies from both simulations |
| `ALL_shared` | All galaxies within the shared resolution range |

---

## CLI Reference

### Positional arguments

| Argument | Description |
|---|---|
| `mode` | `train`, `sample` (train+val+test), or `sampletest` (test only) |
| `train_set` | Source population key (ignored when `--dann_source` is set) |
| `test_set` | Target population key |
| `test_long` | `1` = full test set, `0` = first 100 galaxies |
| `N_stars` | Poisson mean for number of observed stars per projection |
| `GraphNN_type` | `Cheb`, `GCN`, or `GAT` |
| `N_proj_per_gal` | Number of line-of-sight projections per galaxy |
| `PCAfilter` | `1` = use PCA-filtered data, `0` = raw |

### Architecture

| Flag | Default | Description |
|---|---|---|
| `--hidden_dim` | 128 | Hidden width for graph and FC layers |
| `--num_graph_layers` | 3 | Number of graph convolution layers |
| `--num_fc_layers` | 2 | Number of FC layers after pooling |
| `--use_residuals` | 1 | Skip connections between graph layers |
| `--use_batch_norm` | 1 | BatchNorm1d after each layer |
| `--dropout` | 0.1 | Dropout probability |
| `--hlr_std` | 1 | Append `(hlr, σ_v)` scalars to the embedding |

### Unresolved label handling

| Flag | Description |
|---|---|
| `--mask_missing mask` | Context-aware: append resolution mask to flow context, zero unresolved labels |
| `--mask_missing BIF` | Bayesian Imputation: impute unresolved labels from the flow's own prior |
| `--mask_missing drop` | Drop any source projection with at least one unresolved bin (`mask_min < 3 eps_dm`) |

### Domain adaptation

| Flag | Default | Description |
|---|---|---|
| `--dann_source KEY` | — | Enable DA; KEY = labelled source domain |
| `--da_method` | `dann` | `dann` (adversarial) or `mmd` (kernel alignment) |
| `--dann_lambda` | 1.0 | Max GRL reversal strength |
| `--dann_gamma` | 10.0 | Steepness of Ganin λ schedule |
| `--dann_domain_hidden` | 64 | Hidden width of domain classifier MLP |
| `--mmd_lambda` | 1.0 | MMD loss weight |
| `--mmd_gamma` | 10.0 | Steepness of MMD λ schedule |

### Training

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | 128 | Training batch size |
| `--num_workers` | 2 | DataLoader worker processes (recommend `num_workers + 1` CPU cores) |

### MLflow tracking

| Flag | Default | Description |
|---|---|---|
| `--mlflow_uri` | — | Tracking URI; omit to disable. E.g. `file:///path/to/mlruns` or `http://host:5000` |
| `--mlflow_experiment` | `graph_flow_posterior` | Experiment name |

When enabled, each run logs: all hyperparameters, per-epoch `train_log_prob` / `val_log_prob` / `learning_rate` / domain loss (DANN) or MMD loss, final mass-ratio statistics (median and scatter per bin, for all dims and resolved-only), and `posterior.pkl` + `plot_data.pkl` as artefacts.

---

## Model Architecture

```
Stars (N × 2: log₁₀r, v_los)
        │
        ▼
  k-NN Graph (k = min(N_stars, 20), undirected, self-loops)
        │
        ▼
  GNN Encoder  [num_graph_layers × (Conv → BN → ReLU → Dropout)]
               Conv types: ChebConv (K=4), GCNConv, GATConv
               Optional residuals between layers i > 0
        │
        ▼
  Global Mean + Max Pooling  →  concat  →  (2 × hidden_dim,)
        │
  Optional: append (hlr, σ_v)
        │
        ▼
  FC Head  [num_fc_layers × (Linear → BN → ReLU → Dropout)]
        │
        ▼
  Context vector  (hidden_dim,)        [+ mask vector if --mask_missing mask]
        │
        ▼
  MAF  [4 × (RandomPermutation → MaskedAffineAutoregressive)]
        │
        ▼
  log p(M_0, …, M_9 | context)
```

### Domain adaptation

**DANN:** A domain classifier MLP with a Gradient Reversal Layer (GRL) is attached to the encoder. The GRL reversal strength follows the Ganin schedule `λ(t) = 2/(1 + exp(−γ·t/T)) − 1`, and is additionally gated by a domain-accuracy EMA so adversarial pressure is only applied once the classifier is reliably distinguishing domains.

**MMD:** The encoder is trained to minimise the maximum mean discrepancy between source and target embeddings in addition to the task loss. The MMD weight follows the same Ganin schedule. BatchNorm running statistics are updated only on source batches (target batches use `eval`-mode BN) to prevent target-domain contamination.

---

## Output Files

All outputs are written to:
```
{main_work_dir}/Graph+Flow_Mocks_NH/new/{model_str}/
```

where `model_str` encodes the full configuration, e.g.:
```
Cheb_srcNIHAO_hi_tgtNIHAO_lo_mmd-lam300.0-gam5.0_mask-drop_h128_gl3_fc2_res1_bn1_do0.1_poisson100_Nfiles1163_Nproj8_hlrstd1
```

| File | Description |
|---|---|
| `posterior.pkl` | Trained model (pickle) |
| `std_stats.json` | Input/output standardization statistics |
| `graphs_train.pkl` | Training graph list (for Optuna / dataloader benchmarks) |
| `graphs_val.pkl` | Validation graph list |
| `samples_train.npy` | Posterior samples on training set `(N, 1000, 10)` |
| `samples_val.npy` | Posterior samples on validation set |
| `samples_test.npy` | Posterior samples on test set |
| `loss.png` | Training curves |
| `TrainingVsValidationVsTest.png` | Mass ratio plots (all bins) |
| `TrainingVsValidationVsTest_resolved.png` | Mass ratio plots (resolved bins only) |
| `plot_data.pkl` | All plot data + ratio statistics |

---

## Classical Estimator Baselines

The pipeline computes five classical mass estimators for comparison:

| Estimator | Coefficient |
|---|---|
| Walker et al. | 2.5 |
| Wolf et al. | 4.0 |
| Amorisco et al. | 5.8 |
| Campbell et al. | 6.0 |
| Errani et al. | 6.3 |

All use the form `M = C · σ_v² · r_half / G`.

---

## HPC / SLURM Notes

DataLoader workers are spawned as separate processes. Allocate `--num_workers + 1` CPU cores to avoid contention between the main process and workers.

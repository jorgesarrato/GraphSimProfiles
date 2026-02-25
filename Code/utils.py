"""
utils.py — Graph construction, GNN embedding, normalizing flow posterior, and helpers.
"""

from __future__ import annotations

import signal
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.transforms as T
from torch import Tensor
from torch.autograd import Function
from torch_geometric.data import Batch, Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn import ChebConv, GATConv, GCNConv
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    RandomPermutation,
)


@functional_transform("knn_graph_grouped")
class KNNGraphGrouped(BaseTransform):
    """k-NN graph that restricts edges to within user-defined node groups.

    Parameters
    ----------
    k : int
        Number of nearest neighbours per node.
    loop : bool
        Include self-loops.
    force_undirected : bool
        Make edges bidirectional.
    flow : str
        Message-passing flow direction.
    cosine : bool
        Use cosine distance instead of Euclidean.
    num_workers : int
        CPU workers for k-NN search (ignored on GPU).
    group_condition : Callable, optional
        Maps a ``Data`` object to a 1-D integer tensor of group IDs per node.
        Nodes share edges only within the same group.
    """

    def __init__(
        self,
        k: int = 6,
        loop: bool = False,
        force_undirected: bool = False,
        flow: str = "source_to_target",
        cosine: bool = False,
        num_workers: int = 1,
        group_condition: Optional[Callable] = None,
    ) -> None:
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow
        self.cosine = cosine
        self.num_workers = num_workers
        self.group_condition = group_condition

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        if self.group_condition is not None:
            group_indices = self.group_condition(data)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=data.pos.device)

            for group in group_indices.unique():
                mask = group_indices == group
                group_edge_index = torch_geometric.nn.knn_graph(
                    data.pos[mask],
                    self.k,
                    data.batch[mask] if data.batch is not None else None,
                    loop=self.loop,
                    flow=self.flow,
                    cosine=self.cosine,
                    num_workers=self.num_workers,
                )
                global_edge_index = mask.nonzero(as_tuple=True)[0][group_edge_index]
                edge_index = torch.cat([edge_index, global_edge_index], dim=1)
        else:
            edge_index = torch_geometric.nn.knn_graph(
                data.pos,
                self.k,
                data.batch,
                loop=self.loop,
                flow=self.flow,
                cosine=self.cosine,
                num_workers=self.num_workers,
            )

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index
        data.edge_attr = None
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(k={self.k}, group_condition={self.group_condition})"
        )

class GraphNN(nn.Module):
    """GNN encoder that maps a graph batch to a fixed-size context vector.

    Architecture: ``num_graph_layers`` message-passing layers → global mean
    pooling → optional (hlr, std) concatenation → ``num_fc_layers`` linear
    layers.
    
    This module acts as the shared feature extractor in both the standard
    training and the DANN setup.

    Parameters
    ----------
    in_channels : int
        Node feature dimensionality.
    out_channels : int
        Output (context) dimensionality.
    hidden_graph_channels : int
        Hidden width of graph layers.
    num_graph_layers : int
        Number of graph convolution layers.
    hidden_fc_channels : int
        Hidden width of FC layers.
    num_fc_layers : int
        Number of FC layers.
    hlr_std : bool
        Append pre-computed (hlr, std) scalars after pooling.
    graph_layer_name : str
        One of ``"ChebConv"``, ``"GATConv"``, ``"GCNConv"``.
    graph_layer_params : dict, optional
        Extra keyword arguments forwarded to the graph layer constructor.
    activation : str | Callable | torch.nn.Module
        Activation function or its name in ``torch.nn.functional``.
    activation_params : dict, optional
        Extra kwargs forwarded to the activation at each call.
    """

    GRAPH_LAYERS: dict[str, type] = {
        "ChebConv": ChebConv,
        "GATConv": GATConv,
        "GCNConv": GCNConv,
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_graph_channels: int = 128,
        num_graph_layers: int = 3,
        hidden_fc_channels: int = 128,
        num_fc_layers: int = 2,
        hlr_std: bool = True,
        graph_layer_name: str = "ChebConv",
        graph_layer_params: Optional[dict] = None,
        activation: Union[str, nn.Module, Callable] = "relu",
        activation_params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.hlr_std = hlr_std
        graph_layer_params = graph_layer_params or {}
        activation_params = activation_params or {}

        self.graph_layers = nn.ModuleList()
        for i in range(num_graph_layers):
            n_in = in_channels if i == 0 else hidden_graph_channels
            self.graph_layers.append(
                self._build_graph_layer(
                    n_in, hidden_graph_channels, graph_layer_name, graph_layer_params
                )
            )

        self.fc_layers = nn.ModuleList()
        for i in range(num_fc_layers):
            # After pooling, optionally concatenate hlr and std (2 extra dims)
            n_in = (hidden_graph_channels + 2) if (i == 0 and hlr_std) else hidden_fc_channels
            n_out = out_channels if i == num_fc_layers - 1 else hidden_fc_channels
            self.fc_layers.append(nn.Linear(n_in, n_out))

        if isinstance(activation, str):
            self.activation = getattr(nn.functional, activation)
        elif isinstance(activation, (nn.Module, Callable)):
            self.activation = activation
        else:
            raise ValueError(f"Unsupported activation type: {type(activation)}")
        self.activation_params = activation_params

    def forward(self, data: Data) -> Tensor:
        """Return context vectors of shape ``(batch_size, out_channels)``."""
        x = data.x

        for layer in self.graph_layers:
            x = layer(x, data.edge_index)
            x = self.activation(x, **self.activation_params)

        x = torch_geometric.nn.global_mean_pool(x, data.batch)

        if self.hlr_std:
            x = torch.cat([x, data.hlr, data.std], dim=1)

        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.activation(x, **self.activation_params)
        x = self.fc_layers[-1](x)

        return x

    def _build_graph_layer(
        self,
        in_dim: int,
        out_dim: int,
        layer_name: str,
        layer_params: dict,
    ) -> nn.Module:
        if layer_name not in self.GRAPH_LAYERS:
            raise ValueError(
                f"Unknown graph layer '{layer_name}'. "
                f"Available: {list(self.GRAPH_LAYERS.keys())}"
            )
        return self.GRAPH_LAYERS[layer_name](in_dim, out_dim, **layer_params)

def build_maf_flow(
    features: int,
    context_features: int,
    hidden_features: int = 128,
    num_transforms: int = 4,
) -> Flow:
    """Build a Masked Autoregressive Flow conditioned on a context vector.

    Parameters
    ----------
    features : int
        Dimensionality of the quantity to be modelled (number of labels).
    context_features : int
        Dimensionality of the embedding / context vector.
    hidden_features : int
        Hidden width of each MADE block.
    num_transforms : int
        Number of autoregressive transforms.

    Returns
    -------
    nflows.flows.Flow
        An ``nflows`` ``Flow`` object with ``.log_prob(x, context)`` and
        ``.sample(n, context)`` methods.
    """
    transforms = []
    for _ in range(num_transforms):
        transforms.append(RandomPermutation(features=features))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                context_features=context_features,
            )
        )

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([features]),
    )


class FlowPosterior(nn.Module):
    """Wraps a GNN embedding network and a conditional normalizing flow.

    Parameters
    ----------
    embedding_net : GraphNN
        Maps graph batches to context vectors.
    flow : nflows.flows.Flow
        Conditional normalizing flow.
    """

    def __init__(
        self, 
        embedding_net: GraphNN, 
        flow: Flow, 
        mask_missing: Optional[str] = None
    ) -> None:
        super().__init__()
        self.embedding_net = embedding_net
        self.flow = flow
        self.mask_missing = mask_missing

    def log_prob(self, labels: Tensor, data: Data) -> Tensor:
        """Return log p(labels | data) for a batch."""
        context = self.embedding_net(data)
        
        if self.mask_missing == "mask":
            labels = labels * data.mask
            context = torch.cat([context, data.mask], dim=1)
            
        elif self.mask_missing == "BIF": 
            # TO DO: Probably delay BIF activation until x epochs...
            # Initially sample imputations from a fixed prior?

            # Bayesian Imputation: Use the flow to guess the missing values
            with torch.no_grad():
                imputed = self.flow.sample(1, context=context).squeeze(1)
            # Combine known labels with imputed labels
            labels = (labels * data.mask) + (imputed * (1.0 - data.mask))
            
        return self.flow.log_prob(labels, context=context)

    @torch.no_grad()
    def sample(self, num_samples: int, data: Data) -> Tensor:
        """Draw ``num_samples`` samples from p(labels | data).
    
        ``data`` must contain exactly one graph. Returns shape ``(num_samples, n_labels)``.
        """
        context = self.embedding_net(data)
        if self.mask_missing == "mask":
            context = torch.cat([context, data.mask], dim=1)
        return self.flow.sample(num_samples, context=context)

class GradientReversalFunction(Function):
    """Identity in the forward pass; negates and scales gradients in backward.

    Implements the gradient reversal layer of Ganin et al. (2016) without
    any learnable parameters.
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Thin ``nn.Module`` wrapper around `GradientReversalFunction`.

    Parameters
    ----------
    lambda_ : float
        Reversal strength.  Increase from 0 → 1 over training following the
        Ganin schedule (see :func:`ganin_lambda_schedule`).
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lambda_={self.lambda_:.4f})"


class DomainClassifier(nn.Module):
    """Binary MLP that predicts domain membership (source=0, target=1).

    The gradient reversal layer ensures the shared encoder learns features
    that are indistinguishable across domains while the classifier tries
    to discriminate them.

    Parameters
    ----------
    in_features : int
        Dimensionality of the context vector produced by :class:`GraphNN`.
    hidden_features : int
        Width of the hidden layers.
    lambda_ : float
        Initial reversal strength.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        lambda_: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),  # logit for P(target domain)
        )

    def forward(self, context: Tensor) -> Tensor:
        """Return domain logits; shape ``(batch_size, 1)``."""
        return self.classifier(self.grl(context))

    def set_lambda(self, lambda_: float) -> None:
        """Update reversal strength (call once per epoch during training)."""
        self.grl.lambda_ = lambda_


class DANNFlowPosterior(nn.Module):
    """Full DANN model: shared GNN encoder + flow task head + domain classifier.

    Parameters
    ----------
    embedding_net : GraphNN
        Shared feature extractor.
    flow : Flow
        Conditional normalizing flow (task head).
    domain_classifier : DomainClassifier
        Adversarial domain head (contains the GRL).
    """

    def __init__(
        self,
        embedding_net: GraphNN,
        flow: Flow,
        domain_classifier: DomainClassifier,
        mask_missing: Optional[str] = None
    ) -> None:
        super().__init__()
        self.embedding_net     = embedding_net
        self.flow              = flow
        self.domain_classifier = domain_classifier
        self.mask_missing      = mask_missing

    def log_prob(self, labels: Tensor, data: Data) -> Tensor:
        """Per-sample log p(labels | data). Used during validation."""
        context = self.embedding_net(data)
        if self.mask_missing == "mask":
            labels = labels * data.mask
            context = torch.cat([context, data.mask], dim=1)
        elif self.mask_missing == "BIF":
            with torch.no_grad():
                imputed = self.flow.sample(1, context=context).squeeze(1)
            labels = (labels * data.mask) + (imputed * (1.0 - data.mask))
            
        return self.flow.log_prob(labels, context=context)

    @torch.no_grad()
    def sample(self, num_samples: int, data: Data) -> Tensor:
        """Shape: ``(num_samples, n_labels)``."""
        context = self.embedding_net(data)
        if self.mask_missing == "mask":
            context = torch.cat([context, data.mask], dim=1)
        return self.flow.sample(num_samples, context=context)

    def dann_forward(
        self,
        src_batch: Data,
        src_labels: Tensor,
        tgt_batch: Data,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single DANN forward pass.

        Parameters
        ----------
        src_batch : Data
            Batched graphs from the **source** (labelled) domain.
        src_labels : Tensor
            Mass labels for ``src_batch``; shape ``(n_src, n_labels)``.
        tgt_batch : Data
            Batched graphs from the **target** (unlabelled) domain.

        Returns
        -------
        task_loss : Tensor
            ``-mean log p(y | x)`` on source samples.
        domain_loss : Tensor
            Binary cross-entropy of the domain classifier on all samples.
        domain_acc : Tensor
            Fraction of correctly classified domain labels (diagnostic).
        """
        n_src = src_batch.num_graphs
        n_tgt = tgt_batch.num_graphs

        src_context = self.embedding_net(src_batch)
        tgt_context = self.embedding_net(tgt_batch)

        if self.mask_missing == "mask":
            src_labels_flow = src_labels * src_batch.mask
            src_context_flow = torch.cat([src_context, src_batch.mask], dim=1)
            task_loss = -self.flow.log_prob(src_labels_flow, context=src_context_flow).mean()
            
        elif self.mask_missing == "BIF":
            with torch.no_grad():
                imputed = self.flow.sample(1, context=src_context).squeeze(1)
            src_labels_flow = (src_labels * src_batch.mask) + (imputed * (1.0 - src_batch.mask))
            task_loss = -self.flow.log_prob(src_labels_flow, context=src_context).mean()
            
        else:
            task_loss = -self.flow.log_prob(src_labels, context=src_context).mean()

        domain_labels = torch.cat([
            torch.zeros(n_src, 1, device=src_context.device),
            torch.ones( n_tgt, 1, device=tgt_context.device),
        ])
        all_context   = torch.cat([src_context, tgt_context], dim=0)
        domain_logits = self.domain_classifier(all_context)   # GRL reverses grad
        domain_loss   = nn.functional.binary_cross_entropy_with_logits(
            domain_logits, domain_labels
        )

        with torch.no_grad():
            preds      = (domain_logits.sigmoid() > 0.5).float()
            domain_acc = (preds == domain_labels).float().mean()

        return task_loss, domain_loss, domain_acc

def ganin_lambda_schedule(epoch: int, max_epochs: int, gamma: float = 10.0) -> float:
    """Smooth 0→1 schedule for the gradient reversal strength. (Ganin et al. 2016)

    Parameters
    ----------
    epoch : int
        Current epoch (0-indexed).
    max_epochs : int
        Total number of training epochs.
    gamma : float
        Controls the steepness of the sigmoid ramp.
    """
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    
class GraphCreator:
    """Convert raw phase-space arrays into ``torch_geometric.data.Data`` graphs.

    Parameters
    ----------
    graph_type : str
        One of ``"KNNGraph"``, ``"RadiusGraph"``, ``"KNNGraph_grouped"``.
    graph_config : dict, optional
        Keyword arguments forwarded to the graph transform constructor.
    use_radius : bool
        Replace 2-D positions with projected radius as node feature.
    use_log_radius : bool
        Apply log₁₀ to the radius (implies ``use_radius=True``).
    tensor_dtype : torch.dtype
        Dtype for all created tensors.
    """

    GRAPH_TYPES: dict[str, type] = {
        "KNNGraph": T.KNNGraph,
        "RadiusGraph": T.RadiusGraph,
        "KNNGraph_grouped": KNNGraphGrouped,
    }

    def __init__(
        self,
        graph_type: str = "KNNGraph",
        graph_config: Optional[dict] = None,
        use_radius: bool = True,
        use_log_radius: bool = True,
        tensor_dtype: Union[str, torch.dtype] = torch.float32,
    ) -> None:
        self.use_radius = use_radius
        self.use_log_radius = use_log_radius
        self.tensor_dtype = tensor_dtype
        self.graph_fn = self._init_graph(graph_type, graph_config or {})

    def _init_graph(self, graph_type: str, graph_config: dict):
        if graph_type not in self.GRAPH_TYPES:
            raise KeyError(
                f"Unknown graph type '{graph_type}'. "
                f"Available: {list(self.GRAPH_TYPES.keys())}"
            )
        return self.GRAPH_TYPES[graph_type](**graph_config)

    def __call__(
        self,
        positions: Union[np.ndarray, Tensor],
        velocities: Union[np.ndarray, Tensor],
        additional_features: Optional[Union[np.ndarray, Tensor]] = None,
        labels: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> Data:
        """Build a graph from positions, velocities and optional labels."""
        positions = self._to_tensor(positions)
        velocities = self._to_tensor(velocities)
        if additional_features is not None:
            additional_features = self._to_tensor(additional_features)
        if labels is not None:
            labels = self._to_tensor(labels)

        node_features = self._build_node_features(positions, velocities, additional_features)
        graph = Data(pos=positions, x=node_features)
        graph = self.graph_fn(graph)

        if labels is not None:
            graph.y = labels.reshape(1, -1)

        return graph

    def _to_tensor(self, data: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=self.tensor_dtype)
        return data

    def _build_node_features(
        self,
        positions: Tensor,
        velocities: Tensor,
        additional_features: Optional[Tensor],
    ) -> Tensor:
        if velocities.ndim == 1:
            velocities = velocities.unsqueeze(-1)
        if additional_features is not None and additional_features.ndim == 1:
            additional_features = additional_features.unsqueeze(-1)

        if self.use_radius or self.use_log_radius:
            radius = torch.linalg.norm(positions, ord=2, dim=1, keepdim=True)
            if self.use_log_radius:
                radius = torch.log10(radius)
            features = torch.hstack([radius, velocities])
        else:
            features = torch.hstack([positions, velocities])

        if additional_features is not None:
            features = torch.hstack([features, additional_features])

        return features

class TimeoutException(Exception):
    pass


def sample_with_timeout(
    posterior: Union[FlowPosterior, DANNFlowPosterior],
    data_list: list,
    n_samples: int,
    device: torch.device,
    timeout_sec: int,
    ndims: int,
) -> np.ndarray:
    """Sample from the posterior for every graph in *data_list*.

    A per-sample SIGALRM timeout is applied so that hanging GPU calls do not
    block indefinitely.  Timed-out entries are filled with NaN.

    Parameters
    ----------
    posterior : FlowPosterior
        Trained posterior model.
    data_list : list of Data
        Graph data to condition on.
    n_samples : int
        Number of posterior samples per data point.
    device : torch.device
        Device to run inference on.
    timeout_sec : int
        Maximum seconds to wait per data point.
    ndims : int
        Dimensionality of the posterior (number of labels).

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, len(data_list), ndims)``.
    """
    samples = np.full((n_samples, len(data_list), ndims), np.nan)

    def _timeout_handler(signum, frame):
        raise TimeoutException

    for idx, graph in enumerate(data_list):
        print(f"Sampling {idx + 1}/{len(data_list)}")
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)
        try:
            graph = graph.to(device)
            # FlowPosterior.sample expects a batched Data object
            # Wrap single graph in a batch
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([graph])
            draw = posterior.sample(n_samples, batch)  # (n_samples, ndims)
            samples[:, idx, :] = draw.detach().cpu().numpy()
            del graph, batch, draw
        except TimeoutException:
            print(
                f"  Warning: data point {idx} exceeded timeout "
                f"({timeout_sec // 60} min). Filling with NaN."
            )
        finally:
            signal.alarm(0)

    return samples

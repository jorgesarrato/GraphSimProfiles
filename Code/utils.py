"""
utils.py — Graph construction, GNN embedding, normalizing flow posterior, and helpers.
"""

from __future__ import annotations

import signal
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data
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

class GraphNN(torch.nn.Module):
    """GNN encoder that maps a graph batch to a fixed-size context vector.

    Architecture: ``num_graph_layers`` message-passing layers → global mean
    pooling → optional (hlr, std) concatenation → ``num_fc_layers`` linear
    layers.

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
        activation: Union[str, torch.nn.Module, Callable] = "relu",
        activation_params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.hlr_std = hlr_std
        graph_layer_params = graph_layer_params or {}
        activation_params = activation_params or {}

        self.graph_layers = torch.nn.ModuleList()
        for i in range(num_graph_layers):
            n_in = in_channels if i == 0 else hidden_graph_channels
            self.graph_layers.append(
                self._build_graph_layer(
                    n_in, hidden_graph_channels, graph_layer_name, graph_layer_params
                )
            )

        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            # After pooling, optionally concatenate hlr and std (2 extra dims)
            n_in = (hidden_graph_channels + 2) if (i == 0 and hlr_std) else hidden_fc_channels
            n_out = out_channels if i == num_fc_layers - 1 else hidden_fc_channels
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        if isinstance(activation, str):
            self.activation = getattr(torch.nn.functional, activation)
        elif isinstance(activation, (torch.nn.Module, Callable)):
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
    ) -> torch.nn.Module:
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


class FlowPosterior(torch.nn.Module):
    """Wraps a GNN embedding network and a conditional normalizing flow.

    Parameters
    ----------
    embedding_net : GraphNN
        Maps graph batches to context vectors.
    flow : nflows.flows.Flow
        Conditional normalizing flow.
    """

    def __init__(self, embedding_net: GraphNN, flow: Flow) -> None:
        super().__init__()
        self.embedding_net = embedding_net
        self.flow = flow

    def log_prob(self, labels: Tensor, data: Data) -> Tensor:
        """Return log p(labels | data) for a batch."""
        context = self.embedding_net(data)
        return self.flow.log_prob(labels, context=context)

    @torch.no_grad()
    def sample(self, num_samples: int, data: Data) -> Tensor:
        """Draw ``num_samples`` samples from p(labels | data).

        Returns tensor of shape ``(num_samples, n_labels)``.
        """
        context = self.embedding_net(data)
        # nflows expects context shape (batch, context_dim); repeat for n samples
        context_expanded = context.repeat_interleave(num_samples, dim=0)
        samples = self.flow.sample(num_samples, context=context)
        return samples

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
    posterior: FlowPosterior,
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

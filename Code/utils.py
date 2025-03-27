
from typing import Callable, Optional, Union

import torch
import torch_geometric
from torch_geometric.nn import ChebConv, GATConv, GCNConv

import signal

import numpy as np

import torch_geometric.transforms as T

from torch import Tensor

from torch_geometric.data import Data


from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected



@functional_transform('knn_graph_grouped')
class KNNGraph_Grouped(BaseTransform):
    r"""Creates a k-NN graph based on node positions :obj:`data.pos`
    and divides nodes into groups based on a user-defined condition.
    Only nodes within each group can be connected (functional name: :obj:`knn_graph`).

    Args:
        k (int, optional): The number of neighbors. (default: :obj:`6`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        force_undirected (bool, optional): If set to :obj:`True`, new edges
            will be undirected. (default: :obj:`False`)
        flow (str, optional): The flow direction when used in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`).
            If set to :obj:`"source_to_target"`, every target node will have
            exactly :math:`k` source nodes pointing to it.
            (default: :obj:`"source_to_target"`)
        cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        group_condition (Callable, optional): A function that takes in the
            node features or positions and returns a tensor of group indices
            for each node. Nodes with the same group index can be connected.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        k: int = 6,
        loop: bool = False,
        force_undirected: bool = False,
        flow: str = 'source_to_target',
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
            # Compute group indices for each node
            group_indices = self.group_condition(data)

            # Create an empty edge index
            edge_index = torch.empty((2, 0), dtype=torch.long, device=data.pos.device)

            # Process each group separately
            unique_groups = group_indices.unique()
            for group in unique_groups:
                mask = group_indices == group
                group_pos = data.pos[mask]
                group_batch = data.batch[mask] if data.batch is not None else None

                # Compute k-NN graph for the current group
                group_edge_index = torch_geometric.nn.knn_graph(
                    group_pos,
                    self.k,
                    group_batch,
                    loop=self.loop,
                    flow=self.flow,
                    cosine=self.cosine,
                    num_workers=self.num_workers,
                )

                # Map group-specific edge indices back to the original node indices
                global_edge_index = mask.nonzero(as_tuple=True)[0][group_edge_index]
                edge_index = torch.cat([edge_index, global_edge_index], dim=1)
        else:
            # Compute k-NN graph for all nodes
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
        return f'{self.__class__.__name__}(k={self.k}, group_condition={self.group_condition})'


class GraphNN(torch.nn.Module):
    """ Graph Regressor model

    Attributes
    ----------
    graph_layers: torch.nn.ModuleList
        List of graph layers
    fc_layers: torch.nn.ModuleList
        List of fully connected layers
    activation: torch.nn.Module or Callable or str
        Activation function
    activation_params: dict
        Parameters of the activation function.
    flows: torch.nn.ModuleList
        List of normalizing flow layers

    Methods
    -------
    forward(x, edge_index, batch)
        Forward pass of the model
    log_prob(batch, return_context=False)
        Calculate log-likelihood from batch
    sample(batch, num_samples, return_context=False)
        Sample from batch
    log_prob_from_context(x, context)
        Calculate log-likelihood P(x | context)
    sample_from_context(num_samples, context)
        Sample from context

    """
    # Static attributes
    # all implemented graph layers
    GRAPH_LAYERS = {
        "ChebConv": ChebConv,
        "GATConv": GATConv,
        "GCNConv": GCNConv,    }


    def __init__(
            self, in_channels: int, out_channels: int,
            hidden_graph_channels: int = 1,
            num_graph_layers: int = 1,
            hidden_fc_channels: int = 1,
            num_fc_layers: int = 1,
            hlr_std: bool = True,
            graph_layer_name: str = "ChebConv",
            graph_layer_params: Optional[dict] = None,
            activation: Union[str, torch.nn.Module, Callable] = "relu",
            activation_params: Optional[dict] = None            ):
        """
        Parameters
        ----------
        in_channels: int
            Input dimension of the graph layers
        out_channels: int
            Output dimension of the normalizing flow
        hidden_graph_channels: int
            Hidden dimension
        num_graph_layers: int
            Number of graph layers
        hidden_fc_channels: int
            Hidden dimension of the fully connected layers
        num_fc_layers: int
            Number of fully connected layers
        graph_layer_name: str
            Name of the graph layer
        graph_layer_params: dict
            Parameters of the graph layer
        activation: str or torch.nn.Module or Callable
            Activation function
        activation_params: dict
            Parameters of the activation function. Ignored if activation is
            torch.nn.Module
        flow_params: dict
            Parameters of the normalizing flow
        """
        super().__init__()

        self.hlr_std = hlr_std

        if graph_layer_params is None:
            graph_layer_params = {}
        if activation_params is None:
            activation_params = {}

        self.graph_layer_name = graph_layer_name

        # Create the graph layers
        self.graph_layers = torch.nn.ModuleList()
        for i in range(num_graph_layers):
            n_in = in_channels if i == 0 else hidden_graph_channels
            n_out = hidden_graph_channels
            self.graph_layers.append(
                self._get_graph_layer(
                    n_in, n_out, graph_layer_name, graph_layer_params))

        # Create FC layers
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            n_in = hidden_fc_channels+2 if (i == 0 and self.hlr_std) else hidden_fc_channels
            n_out = hidden_fc_channels
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        # Create activation function
        if isinstance(activation, str):
            self.activation = getattr(torch.nn.functional, activation)
            self.activation_params = activation_params
        elif isinstance(activation, torch.nn.Module):
            self.activation = activation
            self.activation_params = {}
        elif isinstance(activation, Callable):
            self.activation = activation
            self.activation_params = activation_params
        else:
            raise ValueError("Invalid activation function")


    def forward(self, dat ):

        x = dat.x
        """ Forward pass of the model

        Parameters
        ----------
        x: torch.Tensor
            Input features
        edge_index: torch.Tensor
            Edge indices


        Returns
        -------
        x: torch.Tensor
            Output features as the flows context
        """
        # Apply graph and FC layers to extract features as the flows context
        # apply graph layers
        for layer in self.graph_layers:
            x = layer(x, dat.edge_index)
            x = self.activation(x, **self.activation_params)
        # pool the features
        x = torch_geometric.nn.global_mean_pool(x, dat.batch)
        
        if self.hlr_std:
            x = torch.cat((x, dat.hlr, dat.std), dim = 1)

        # apply FC layers
        # do not apply activation function to the last layer
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.activation(x, **self.activation_params)
        x = self.fc_layers[-1](x)

        return x

    def _get_graph_layer(
            self, in_dim, out_dim, graph_layer_name, graph_layer_params):
        """ Return a graph layer

        Parameters
        ----------
        in_dim: int
            Input dimension
        out_dim: int
            Output dimension
        graph_layer_name: str
            Name of the graph layer
        graph_layer_params: dict
            Parameters of the graph layer

        Returns
        -------
        graph_layer: torch.nn.Module
        """
        if graph_layer_name not in self.GRAPH_LAYERS:
            raise ValueError(f"Graph layer {graph_layer_name} not implemented")
        return self.GRAPH_LAYERS[graph_layer_name](
            in_dim, out_dim, **graph_layer_params)
    
class TimeoutException(Exception):
    pass

def sample_with_timeout(posterior_ensemble, data_list, n_samples, device, timeout_sec, ndims):
    samples = np.zeros((n_samples, len(data_list), ndims))

    def timeout_handler(signum, frame):
        raise TimeoutException

    for ii in range(len(data_list)):
        print(ii, len(data_list))
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        try:
            data = data_list[ii].to(device)  # Move data to GPU
            sample = posterior_ensemble.sample((n_samples,), data)  # Perform computation on GPU
            samples[:, ii, :] = sample.detach().cpu().numpy()  # Move result back to CPU
            del data, sample  # Explicitly delete tensors to free GPU memory
            #torch.cuda.empty_cache()
        except TimeoutException:
            print(f"Warning: Sampling for data point {ii} took longer than {timeout_sec // 60} minutes. Setting result to NaN.")
            samples[:, ii, :] = np.nan
        finally:
            signal.alarm(0)  # Disable the alarm

    return samples
    

class GraphCreator():
    """
    Process phase space data into graph data.

    Attributes
    ----------
    GRAPH_TYPES: dict
        Dictionary of graph types.
    """

    GRAPH_TYPES = {
        "KNNGraph": T.KNNGraph,
        "RadiusGraph": T.RadiusGraph,
        "KNNGraph_grouped": KNNGraph_Grouped,
    }

    def __init__(
        self,
        graph_type: str = "KNNGraph",
        graph_config: Optional[dict] = None,
        use_radius: bool = True,
        use_log_radius: bool = True,
        tensor_dtype: Union[str, torch.dtype] = torch.float32,
    ):
        """
        Parameters
        ----------
        graph_type: str
            Name of the graph type to use.
        graph_config: dict
            Configuration parameters for the graph.
        use_radius: bool
            Whether to use radius as a feature.
        use_log_radius: bool
            Whether to apply logarithm to the radius.
        tensor_dtype: Union[str, torch.dtype]
            Data type for tensors.
        """
        self.graph_type = graph_type
        self.graph_config = graph_config
        self.use_radius = use_radius
        self.use_log_radius = use_log_radius
        self.tensor_dtype = tensor_dtype

        self.graph_function = self._initialize_graph(graph_type, self.graph_config)

    def _initialize_graph(self, graph_type: str, graph_config: dict):
        """Initialize the graph function based on the provided type and configuration."""
        if graph_type not in self.GRAPH_TYPES:
            raise KeyError(
                f"Unknown graph type \"{graph_type}\". "
                f"Available graph types are: {list(self.GRAPH_TYPES.keys())}"
            )
        return self.GRAPH_TYPES[graph_type](**graph_config)

    def __call__(
        self,
        positions: Tensor,
        velocities: Tensor,
        additional_features: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Data:
        """
        Process input data into a graph.

        Parameters
        ----------
        positions: torch.Tensor or numpy.ndarray
            Position tensor. Shape (N, 2).
        velocities: torch.Tensor or numpy.ndarray
            Velocity tensor. Shape (N, 1).
        additional_features: Optional[torch.Tensor]
            Additional features. Shape (N, *).
        labels: Optional[torch.Tensor]
            Label tensor. If None, return data without labels.

        Returns
        -------
        torch_geometric.data.Data
            Processed graph data.
        """
        # Convert numpy arrays to torch tensors
        positions = self._ensure_tensor(positions)
        velocities = self._ensure_tensor(velocities)
        labels = self._ensure_tensor(labels) if labels is not None else None
        additional_features = self._ensure_tensor(additional_features) if additional_features is not None else None

        # Preprocess features
        node_features = self._preprocess_features(positions, velocities, additional_features)

        # Create graph
        graph_data = Data(pos=positions, x=node_features)
        graph_data = self.graph_function(graph_data)

        # Add labels if provided
        if labels is not None:
            graph_data.y = labels.reshape((1,-1))
        return graph_data

    def _ensure_tensor(self, data: Union[np.ndarray, Tensor]) -> Tensor:
        """Convert numpy arrays to torch tensors if necessary."""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=self.tensor_dtype)
        return data

    def _preprocess_features(
        self,
        positions: Tensor,
        velocities: Tensor,
        additional_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Transform 2D positions and velocities into a 1D feature vector.

        Parameters
        ----------
        positions: torch.Tensor
            Position tensor.
        velocities: torch.Tensor
            Velocity tensor.
        additional_features: Optional[torch.Tensor]
            Additional features.

        Returns
        -------
        torch.Tensor
            Processed feature tensor.
        """
        # Reshape tensors if necessary
        if velocities.ndim == 1:
            velocities = velocities.reshape(-1, 1)
        if additional_features is not None and additional_features.ndim == 1:
            additional_features = additional_features.reshape(-1, 1)

        # Compute radius if required
        if self.use_radius or self.use_log_radius:
            radius = torch.linalg.norm(positions, ord=2, dim=1, keepdims=True)
            if self.use_log_radius:
                radius = torch.log10(radius)
            features = torch.hstack([radius, velocities])
        else:
            features = torch.hstack([positions, velocities])

        # Add additional features if provided
        if additional_features is not None:
            return torch.hstack([features, additional_features])
        return features
    



"""

        import matplotlib.pyplot as plt

        # Generate toy data
        num_nodes = 100
        positions = torch.rand((num_nodes, 2)) * 2 - 1  # Random positions in [-1, 1] x [-1, 1]
        velocities = torch.rand((num_nodes, 1))  # Random velocities
        labels = torch.randint(0, 2, (num_nodes,))  # Random labels

        # Create a GraphCreator instance
        graph_creator = GraphCreator(graph_type="KNNGraph", graph_config={"k": 5})

        # Create graph without condition
        graph_data_no_condition = graph_creator(positions, velocities)

        # Create graph with group condition based on radius
        def group_condition(data):
            radius = torch.linalg.norm(data.pos, dim=1)
            return (radius > 0.5).long()  # Group 0: radius <= 0.5, Group 1: radius > 0.5

        knn_grouped = KNNGraph_Grouped(k=5, group_condition=group_condition)
        graph_data_with_condition = knn_grouped(graph_data_no_condition)

        # Convert to NetworkX for visualization
        G_no_condition = to_networkx(graph_data_no_condition, to_undirected=True)
        G_with_condition = to_networkx(graph_data_with_condition, to_undirected=True)

        # Plot graphs
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot without condition
        ax = axes[0]
        nx.draw(
            G_no_condition,
            pos=graph_data_no_condition.pos.numpy(),
            node_size=50,
            ax=ax,
            with_labels=False
        )
        ax.set_title("Graph without condition")

        # Plot with condition
        ax = axes[1]
        nx.draw(
            G_with_condition,
            pos=graph_data_with_condition.pos.numpy(),
            node_size=50,
            ax=ax,
            with_labels=False
        )
        ax.set_title("Graph with group condition")

        plt.show()

"""

from typing import Callable, Optional, Union

import torch
import torch_geometric
from torch_geometric.nn import ChebConv, GATConv, GCNConv

import signal

import numpy as np

import torch_geometric.transforms as T

from torch import Tensor

from torch_geometric.data import Data

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
            samples[:, ii, :] = posterior_ensemble.sample((n_samples,), data_list[ii].to(device)).detach().cpu().numpy()
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
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import torch
from torch import nn

from modulus.models.layers.activations import get_activation
from modulus.models.layers.segment_csr import segment_csr
from modulus.models.mlp import FullyConnected


class GNO(nn.Module):
    """Implement the Graph Neural Operator (GNO)

    This class computes the GNO transform as described
    in `Neural Operator: Graph Kernel Network for Partial
    Differential Equations` by Li et al.

    By defining as F(x) a concatenation of the PDE parameters
    such as (a,f,g) within the PDE

    $$
        -\nabla\cdot(a(x)\nabla u(x)) = f(x)           for x \in \Omega
        y(x) = g(x)                            for x \in \partial\Omega
    $$

    the GNO  computes the following iterative steps to find the
    solution u(x) to the PDE:

    $$
        v_0(x) = NN_1(x, F(x))
        v_{t+1}(x) = \sigma(Wv_t(x) + NeighbourhoodIntegral(
                v_t, x, y, list_neighbouring_nodes, F())
            )                                      for t = 0, 1, ..., T-1
        u(x) = NN_2(v_T(x))
    $$

    where NN_1 and NN_2 are MLPs, W is a learnable weight matrix,
    and NeighbourhoodIntegral is a function that computes the
    integral of a kernel over the neighbourhood of each point x.

    The computed integral is one of the following:
        (a) \int_{A(x)} k(v_t(x), v_t(y)) dy
        (b) \int_{A(x)} k(v_t(x), v_t(y)) * v_t(y) dy
        (c) \int_{A(x)} k(v_t(x), v_t(y), F(y)) dy
        (d) \int_{A(x)} k(v_t(x), v_t(y), F(y)) * v_t(y) dy

    x : Points for which the output is defined
    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on
           each x) over which to integrate that are defined by
           the list of neightbouring nodes
    k : A kernel parametrized as a MLP
    v_t : Input function to integrate against given
        on the points y

    If F is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    mlp_kernel : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    mlp_kernel_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the MLPLinear class
    mlp_kernel_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        MLPLinear class. Only used if mlp_layers is
        given and mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    """

    def __init__(
        self,
        input_dimension: int = 3,  # dimension of the entry x and y
        embedding_dimension: int = 32,  # dimension of the entry v_t
        F_dimension=None,  # dimension of the entry F
        output_dimension: int = 1,
        mlp_kernel=None,
        mlp_kernel_non_linearity: str = "gelu",
        mlp_kernel_n_layers: int = 2,
        mlp_kernel_layer_size: int = 512,
        # mlp_kernel_input_size: int=None,
        embed_input: bool = True,
        mlp_embedding_layers: int = 2,
        mlp_embedding_non_linearity: str = "gelu",
        transform_type: str = "linear",
        skip_connection: bool = True,
        non_linearity: str = "gelu",
        num_iterations: int = 1,
    ):
        super().__init__()

        # assert mlp_kernel is not None or (
        #     mlp_kernel_input_size is not None
        # )

        self.transform_type = transform_type

        if (
            self.transform_type != "linear_kernelonly"
            and self.transform_type != "linear"
            and self.transform_type != "nonlinear_kernelonly"
            and self.transform_type != "nonlinear"
        ):
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if transform_type.startswith("linear"):
            dimension_kernel_mlp = input_dimension + input_dimension
        else:
            dimension_kernel_mlp = input_dimension + input_dimension + F_dimension

        # Set up the MLPs that embed and decode the input space to v_t
        if embed_input:

            self.mlp_embed = FullyConnected(
                num_layers=mlp_embedding_layers,
                activation_fn=mlp_embedding_non_linearity,
                layer_size=mlp_kernel_layer_size,
                in_features=F_dimension,
                out_features=embedding_dimension,
            )
        else:
            self.mlp_embed = None
            if F_dimension != embedding_dimension:
                raise ValueError(
                    "No Embedding has been selected and for this, the F input needs "
                    " to have the same dimension as the embedding dimension."
                )

        if mlp_kernel is None:
            self.mlp_kernel = FullyConnected(
                num_layers=mlp_kernel_n_layers,
                activation_fn=mlp_kernel_non_linearity,
                layer_size=mlp_kernel_layer_size,
                in_features=dimension_kernel_mlp,
                out_features=embedding_dimension,  # TODO if embed_input else output_dimension  CHECK THIS
            )
        else:
            self.mlp_kernel = mlp_kernel

        self.mlp_decoder = FullyConnected(
            num_layers=mlp_embedding_layers,
            activation_fn=mlp_embedding_non_linearity,
            layer_size=mlp_kernel_layer_size,
            in_features=embedding_dimension,
            out_features=output_dimension,
        )

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.W = torch.nn.Linear(embedding_dimension, embedding_dimension)

        self.non_linearity = non_linearity
        self.num_iterations = num_iterations

    """"
    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def kernel_transform(self, y, neighbors, x=None, F_y=None, weights=None):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        F_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y

        rep_features = y[neighbors["neighbors_index"]]

        # batching only matters if f_y (latent embedding) values are provided
        batched = False

        # f_y has a batch dim IFF batched=True
        if F_y is not None:
            if F_y.ndim == 3:
                batched = True
                batch_size = F_y.shape[0]
                in_features = F_y[:, neighbors["neighbors_index"], :]
            elif F_y.ndim == 2:
                batched = False
                in_features = F_y[neighbors["neighbors_index"]]

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )

        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if F_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        kernel_features = self.mlp_kernel(agg_features)

        # if F_y is not None and self.transform_type != "nonlinear_kernelonly": # TODO HERE THEY ARE MULTIPLYING ALSO IN THE CASE OF LINEAR_kernelonly
        if (F_y is not None) and ("kernelonly" not in self.transform_type):
            kernel_features = kernel_features * in_features

        if weights is not None:
            if weights.ndim != 1:
                raise ValueError("Weights must be of dimension 1 in all cases")
            nbr_weights = weights[neighbors["neighbors_index"]]
            # repeat weights along batch dim if batched
            if batched:
                nbr_weights = nbr_weights.repeat([batch_size] + [1] * nbr_weights.ndim)
            kernel_features = nbr_weights * rep_features
            reduction = "sum"
        else:
            reduction = "mean"

        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(kernel_features, splits, reduce=reduction)

        return out_features

    def compose_operator(
        self,
        v_t: torch.Tensor,
        x: Optional[torch.Tensor],
        y: torch.Tensor,
        neighbors: Dict[str, List],
        F_y: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Compose the iterative operation of the GNO.

        This function computes the iterative module of the GNO iterative process.

            v_{t+1}(x) = \sigma(Wv_t(x) + NeighbourhoodIntegral(
                v_t, x, y, list_neighbouring_nodes, F())
            )
        """

        #  y, neighbors, x=None, F_y=None, weights=None

        v_t1 = self.kernel_transform(y, neighbors, F_y=F_y)

        if self.skip_connection:

            v_t1 = get_activation(self.non_linearity)(v_t1 + self.W(v_t))

        return v_t1

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        neighbors: Dict[str, List],
        F_y: Optional[torch.Tensor],  # =None
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            neighbors (Dict[str, List]): _description_
            F (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """

        if self.mlp_embed is not None:
            v_0 = self.mlp_embed(F_y)
        else:
            v_0 = F_y

        for _ in range(self.num_iterations):
            v_0 = self.compose_operator(v_0, x, y, neighbors, F_y)

        return self.mlp_decoder(v_0)

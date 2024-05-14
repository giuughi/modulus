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
"""
Python implementation of neighbor-search algorithm for use on CPU to avoid
breaking torch_cluster's CPU version.
"""

import torch


def simple_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """

    Parameters
    ----------
    Density-Based Spatial Clustering of Applications with Noise
    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """

    dists = torch.cdist(queries, data).to(
        queries.device
    )  # shaped num query points x num data points
    in_nbr = torch.where(dists <= radius, 1.0, 0.0)  # i,j is one if j is i's neighbor
    nbr_indices = in_nbr.nonzero()[:, 1:].reshape(
        -1,
    )  # only keep the column indices
    nbrhd_sizes = torch.cumsum(
        torch.sum(in_nbr, dim=1), dim=0
    )  # num points in each neighborhood, summed cumulatively
    splits = torch.cat((torch.tensor([0.0]).to(queries.device), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict["neighbors_index"] = nbr_indices.long().to(queries.device)
    nbr_dict["neighbors_row_splits"] = splits.long()
    return nbr_dict

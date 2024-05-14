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

import os

import h5py
import numpy as np
import requests
import scipy
import torch
from sklearn.metrics.pairwise import pairwise_distances
from torch import nn
from torch_geometric.data import Data, DataLoader

from ..datapipe import Datapipe


class Poisson2D(Datapipe):
    """2D Poisson benchmark problem datapipe.

    This datapipe uploads a set of Poisson problems for training and testing a model. The
    training and testing data are uploaded from the files `piececonst_r241_N1024_smooth1.mat`
    and `piececonst_r241_N1024_smooth2.mat`, respectively. The data is then preprocessed
    and normalised before being returned as a dataloader.

    Parameters
    ----------
    resolution : int, optional
        Resolution to run simulation at, by default 256
    batch_size : int, optional
        Batch size of simulations, by default 64
    nr_permeability_freq : int, optional
        Number of frequencies to use for generating random permeability. Higher values
        will give higher freq permeability fields., by default 5
    max_permeability : float, optional
        Max permeability, by default 2.0
    min_permeability : float, optional
        Min permeability, by default 0.5
    max_iterations : int, optional
        Maximum iterations to use for each multi-grid, by default 30000
    convergence_threshold : float, optional
        Solver L-Infinity convergence threshold, by default 1e-6
    iterations_per_convergence_check : int, optional
        Number of Jacobi iterations to run before checking convergence, by default 1000
    nr_multigrids : int, optional
        Number of multi-grid levels, by default 4
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys `permeability` and `darcy`. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"

    Raises
    ------
    ValueError
        Incompatable multi-grid and resolution settings
    """

    def __init__(
        self,
        training_resolution,
        testing_resolution,
        training_radius=0.10,
        testing_radius=0.10,
    ):

        # Check if the data is already downloaded in "./data" otherwise download it
        if not os.path.exists("data/piececonst_r241_N1024_smooth1.mat"):
            url_train = "https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/piececonst_r421_N1024_smooth1.mat"
            response = requests.get(url_train, timeout=30)
            if response.status_code == 200:
                with open("data/piececonst_r421_N1024_smooth1.mat", "wb") as file:
                    file.write(response.content)
                    print("File downloaded successfully!")
            else:
                print("Failed to download the training file.")
            url_test = "https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/piececonst_r421_N1024_smooth2.mat"
            response = requests.get(url_test, timeout=30)
            if response.status_code == 200:
                with open("data/piececonst_r421_N1024_smooth2.mat", "wb") as file:
                    file.write(response.content)
                    print("File downloaded successfully!")
            else:
                print("Failed to download the testing file.")

        (TRAIN_PATH, TEST_PATH) = (
            "data/piececonst_r241_N1024_smooth1.mat",
            "data/piececonst_r241_N1024_smooth2.mat",
        )

        # Upload the data
        reader = MatReader(TRAIN_PATH)

        # Define the resolution of the grids
        self.training_grid_size = reader.read_field("coeff").shape[1]
        self.training_resolution = training_resolution
        self.training_spacing = int(
            ((self.training_grid_size - 1) / self.training_resolution) + 1
        )
        self.training_radius = training_radius

        self.testing_grid_size = reader.read_field("coeff").shape[1]
        self.testing_resolution = testing_resolution
        self.testing_spacing = int(
            ((self.testing_grid_size - 1) / self.training_resolution) + 1
        )
        self.testing_radius = testing_radius

        # Define the subsampled data according to the resolutions
        ntrain = reader.read_field("coeff").shape[0]
        self.train_a = reader.read_field("coeff")[
            :ntrain, :: self.training_resolution, :: self.training_resolution
        ].reshape(ntrain, -1)
        self.train_a_smooth = reader.read_field("Kcoeff")[
            :ntrain, :: self.training_resolution, :: self.training_resolution
        ].reshape(ntrain, -1)
        self.train_a_gradx = reader.read_field("Kcoeff_x")[
            :ntrain, :: self.training_resolution, :: self.training_resolution
        ].reshape(ntrain, -1)
        self.train_a_grady = reader.read_field("Kcoeff_y")[
            :ntrain, :: self.training_resolution, :: self.training_resolution
        ].reshape(ntrain, -1)
        self.train_u = reader.read_field("sol")[
            :ntrain, :: self.training_resolution, :: self.training_resolution
        ].reshape(ntrain, -1)

        reader.load_file(TEST_PATH)
        ntest = reader.read_field("coeff").shape[0]
        self.test_a = reader.read_field("coeff")[
            :ntest, :: self.testing_resolution, :: self.testing_resolution
        ].reshape(ntest, -1)
        self.test_a_smooth = reader.read_field("Kcoeff")[
            :ntest, :: self.testing_resolution, :: self.testing_resolution
        ].reshape(ntest, -1)
        self.test_a_gradx = reader.read_field("Kcoeff_x")[
            :ntest, :: self.testing_resolution, :: self.testing_resolution
        ].reshape(ntest, -1)
        self.test_a_grady = reader.read_field("Kcoeff_y")[
            :ntest, :: self.testing_resolution, :: self.testing_resolution
        ].reshape(ntest, -1)
        self.test_u = reader.read_field("sol")[
            :ntest, :: self.testing_resolution, :: self.testing_resolution
        ].reshape(ntest, -1)

        # Normalise the data
        a_normalizer = GaussianNormalizer(self.train_a)
        self.train_a = a_normalizer.encode(self.train_a)
        self.test_a = a_normalizer.encode(self.test_a)
        as_normalizer = GaussianNormalizer(self.train_a_smooth)
        self.train_a_smooth = as_normalizer.encode(self.train_a_smooth)
        self.test_a_smooth = as_normalizer.encode(self.test_a_smooth)
        agx_normalizer = GaussianNormalizer(self.train_a_gradx)
        self.train_a_gradx = agx_normalizer.encode(self.train_a_gradx)
        self.test_a_gradx = agx_normalizer.encode(self.test_a_gradx)
        agy_normalizer = GaussianNormalizer(self.train_a_grady)
        self.train_a_grady = agy_normalizer.encode(self.train_a_grady)
        self.test_a_grady = agy_normalizer.encode(self.test_a_grady)

        u_normalizer = UnitGaussianNormalizer(self.train_u)
        self.train_u = u_normalizer.encode(self.train_u)
        self.test_u = u_normalizer.encode(self.test_u)

        # Define the underlying grid

        meshgenerator = SquareMeshGenerator(
            [[0, 1], [0, 1]], [self.training_spacing, self.training_spacing]
        )
        self.grid = meshgenerator.get_grid()

        # Define the neighbours
        nb_search_out = NeighborSearch()
        self.neighbours = nb_search_out(self.grid, self.grid, 0.1)

        # Define the data loaders
        data_train = self.setup_data_structure(
            self.train_a,
            self.train_a_smooth,
            self.train_a_gradx,
            self.train_a_grady,
            self.train_u,
        )
        data_test = self.setup_data_structure(
            self.test_a,
            self.test_a_smooth,
            self.test_a_gradx,
            self.test_a_grady,
            self.test_u,
        )

        self.train_loader = DataLoader(data_train, batch_size=1, shuffle=True)
        self.test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    def setup_data_structure(self, a, a_smooth, a_gradx, a_grady, u):
        """Create a data structure returning as feature on the forcing term the values of the
        permeability, its smoothness, and its gradients."""
        data_train = [
            Data(
                x=self.grid,
                y=self.grid,
                F_y=torch.cat(
                    [
                        a[j, :].reshape(-1, 1),
                        a_smooth[j, :].reshape(-1, 1),
                        a_smooth[j, :].reshape(-1, 1),
                        a_gradx[j, :].reshape(-1, 1),
                        a_grady[j, :].reshape(-1, 1),
                    ],
                    dim=1,
                ),
                neighbors=self.neighbours,
                true_solution=u[j, :],
            )
            for j in range(a.shape[0])
        ]
        return data_train


class MatReader(object):
    """Read the mesh files included in mat or h5py files."""

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except ValueError:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class UnitGaussianNormalizer(object):
    """Normalize the nada according to a normal distrrbution."""

    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer(object):
    """Noramlize the data according to a gaussian distribution."""

    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class SquareMeshGenerator(object):
    """Generate a Square Mesh with the option of adding different attributes."""

    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        if len(mesh_size) != self.d:
            raise ValueError(
                "The mesh size does not coincide with the dimension of the real space"
            )

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape(
                (self.n, 1)
            )
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(
                    np.linspace(real_space[j][0], real_space[j][1], mesh_size[j])
                )
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def ball_connectivity(self, r):
        """Identify the neighbouring points"""
        pwd = pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        """Identify the neighbouring points according to a gaussian distribution."""
        pwd = pairwise_distances(self.grid)
        rbf = np.exp(-(pwd**2) / sigma**2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        """Return the grid as a tensor."""
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0 : 2 * self.d] = self.grid[self.edge_index.T].reshape(
                    (self.n_edges, -1)
                )
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0 : self.d], xy[:, self.d :])
            else:
                edge_attr = f(
                    xy[:, 0 : self.d],
                    xy[:, self.d :],
                    theta[self.edge_index[0]],
                    theta[self.edge_index[1]],
                )

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        """Return the bouundaries of the domain."""
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1)
                )
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0 : 2 * self.d] = self.grid[
                    self.edge_index_boundary.T
                ].reshape((self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[
                    self.edge_index_boundary[1]
                ]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape(
                (self.n_edges_boundary, -1)
            )
            if theta is None:
                edge_attr_boundary = f(xy[:, 0 : self.d], xy[:, self.d :])
            else:
                edge_attr_boundary = f(
                    xy[:, 0 : self.d],
                    xy[:, self.d :],
                    theta[self.edge_index_boundary[0]],
                    theta[self.edge_index_boundary[1]],
                )

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class NeighborSearch(nn.Module):
    """Neighbor search within a ball of a given radius

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or torch_cluster
        NOTE: open3d implementation requires 3d data
    """

    def __init__(self, use_open3d=False):
        super().__init__()
        from modulus.models.layers.simple_neighbor_search import simple_neighbor_search

        self.search_fn = simple_neighbor_search
        self.use_open3d = False

    def forward(self, data, queries, radius):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Point for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict["neighbors_index"] = search_return.neighbors_index.long()
            return_dict[
                "neighbors_row_splits"
            ] = search_return.neighbors_row_splits.long()

        else:
            return_dict = self.search_fn(data, queries, radius)

        return return_dict

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

import hydra
from omegaconf import DictConfig
from math import ceil

import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F

from modulus.models.gno import GNO
from modulus.datapipes.benchmarks.poisson import Poisson2D
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def poisson_trainer(cfg: DictConfig) -> None:
    """Training for the 2D poisson benchamrk problem.

    This training script demonstrates how to set up a data-driven model for a 2D Poisson
    problem using Graph Neural Operators (GNO) and acts as a benchmark for this type of operator.

    The training and the testing data are uploaded from the PINO article data.
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="poisson_gno")
    log.file_logging()
    initialize_mlflow(
        experiment_name="Poisson_GNO",
        experiment_desc="training an GNO model for the Poisson problem",
        run_name="Poisson_GNO training",
        run_desc="training GNO for Poisson",
        user_name="G. U.",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=cfg.logger.use_mlflow)  # Modulus launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = GNO(**cfg.gno).to(dist.device)

    model.train()

    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler.scheduler_step,
        gamma=cfg.scheduler.scheduler_gamma,
    )

    # Upload the data loaders
    poisson_data = Poisson2D(**cfg.poisson_data)

    training_data_loader = poisson_data.train_loader
    testing_data_loader = poisson_data.test_loader

    # Iterate to train the model
    for ep in range(cfg.training.epochs):
        train_mse = 0.0
        train_loss = 0.0
        train_l2 = 0.0
        for i, batch in enumerate(training_data_loader):
            if i > 20:
                continue
            batch = batch.to(dist.device)
            # print(batch)
            optimizer.zero_grad()
            out = model(
                batch["x"],
                batch["y"],
                batch["neighbors"],
                batch["F_y"],
            )
            mse = F.mse_loss(out.view(-1, 1), batch.true_solution.view(-1, 1))
            mse.backward(retain_graph=True)
            loss = torch.norm(out.view(-1) - batch.true_solution.view(-1), 1)

            optimizer.step()
            train_mse += mse.item()
            train_loss += loss.item()

        scheduler.step()

        model.eval()

        if ep % 10 == 0:
            print(
                ep,
                " train_mse:",
                train_mse / len(training_data_loader),
                " train_loss:",
                train_loss / len(training_data_loader),
            )

    # save_checkpoint(
    #     path="./checkpoints",
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     models=model,
    #     epoch=cfg.training.max_pseudo_epochs
    # )
    log.success("Training completed *yay*")


if __name__ == "__main__":
    poisson_trainer()

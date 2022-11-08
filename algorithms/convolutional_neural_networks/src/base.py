"""Base Classes for Layer and Optimizer.
Code modified from https://github.com/SkalskiP/ILearnDeepLearning.py.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch


class Layer(ABC):
    @property
    def weights(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns weights tensor if layer is trainable.
        Returns None for non-trainable layers.
        """
        return None

    @property
    def gradients(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns bias tensor if layer is trainable.
        Returns None for non-trainable layers.
        """
        return None

    @abstractmethod
    def forward_pass(self, a_prev: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Perform layer forward propagation logic.
        """

    @abstractmethod
    def backward_pass(self, da_curr: torch.Tensor) -> torch.Tensor:
        """Perform layer backward propagation logic."""

    @abstractmethod
    def set_weights(self, w: torch.Tensor, b: torch.Tensor) -> None:
        """
        Perform layer backward propagation logic.
        """


class Optimizer(ABC):
    @abstractmethod
    def update(self, layers: List[Layer]) -> None:
        """
        Updates value of weights and bias tensors in trainable layers.
        """

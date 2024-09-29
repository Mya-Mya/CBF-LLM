from typing import List
from abc import ABC, abstractmethod
# PyTorch
import torch
Tensor = torch.Tensor
zeros_like = torch.zeros_like
float32 = torch.float32


class Normalizer(ABC):
    """
    The Normalizer R
    """
    @abstractmethod
    def __call__(self, P: Tensor, allowed_tokens: List[int]) -> Tensor:
        pass


class ElementwiseMultiplyNormalizer(Normalizer):
    def __call__(self, P: Tensor, allowed_tokens: List[int]) -> Tensor:
        Q = zeros_like(P, device=P.device, dtype=float32)
        Q[allowed_tokens] = P[allowed_tokens]
        r = P[allowed_tokens].sum()
        Q /= r
        return Q


class ElementwiseAddNormalizer(Normalizer):
    def __call__(self, P: Tensor, allowed_tokens: List[int]) -> Tensor:
        Q = zeros_like(P, device=P.device, dtype=float32)
        r = (1.0 - P[allowed_tokens].sum()) / len(allowed_tokens)
        Q[allowed_tokens] = P[allowed_tokens] + r
        return Q

from typing import List
from abc import ABC, abstractmethod
# PyTorch
import torch
Tensor = torch.Tensor
zeros_like = torch.zeros_like
float32 = torch.float32


class Normalizer(ABC):
    """
    正規化器R：フィルタFからの出力を加工し要素の和を1にして，確率分布の値にする
    """
    @abstractmethod
    def __call__(self, P: Tensor, allowed_tokens: List[int]) -> Tensor:
        pass


class MinJSDNormalizer(Normalizer):
    """
    加工前と後とでJensen-Shanon Divergenceが最小となるような正規化器
    """

    def __call__(self, P: Tensor, allowed_tokens: List[int]) -> Tensor:
        Q = zeros_like(P, device=P.device, dtype=float32)
        Q[allowed_tokens] = P[allowed_tokens]
        r = P[allowed_tokens].sum()
        Q /= r
        return Q


class Min2NormNormalizer(Normalizer):
    """
    加工前と後とで2乗誤差が最小となるような正規化器
    """

    def __call__(self, P: Tensor, allowed_tokens: List[int]) -> Tensor:
        Q = zeros_like(P, device=P.device, dtype=float32)
        r = (1.0 - P[allowed_tokens].sum()) / len(allowed_tokens)
        Q[allowed_tokens] = P[allowed_tokens] + r
        return Q

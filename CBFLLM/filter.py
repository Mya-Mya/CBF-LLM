# CBF-LLM Modules
from .language_constraint_functions import LanguageCF
# Other Modules
from typing import List, Dict, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizerBase
# PyTorch
import torch
Tensor = torch.Tensor
tensor = torch.tensor
ones = torch.ones
hstack = torch.hstack
kron = torch.kron


@dataclass
class FilterResult:
    allowed: List[int] = field(default_factory=list)
    disallowed: List[int] = field(default_factory=list)
    lcf_mapping: Dict[int, float] = field(default_factory=dict)


class Filter(ABC):
    """
    Allows/disallows tokens based on specified constraints.
    """
    @abstractmethod
    def scan(self, x: Tensor, P: Tensor) -> FilterResult:
        pass


def get_next_lcf_list(
    current_x: Tensor,
    next_token_list: Union[Tensor, List[int]],
    tokenizer: PreTrainedTokenizerBase,
    lcf: LanguageCF
) -> List[float]:
    """
    Parameters
    ----------
    current_x: Tensor
        Current text input
    next_token_list: List[int]
        List of next possible tokens
    tokenizer: PreTrainedTokenizerBase
        Tokenizer of the language generation model
    lcf: LanguageCF
        The language-constraint function object
    """
    device = current_x.device
    dtype = current_x.dtype
    height = len(next_token_list)
    I = ones(height, 1, dtype=dtype, device=device)
    if not isinstance(next_token_list, Tensor):
        next_token_list = tensor(
            next_token_list,
            dtype=dtype,
            device=device
        )
    next_token_list = next_token_list.reshape(-1, 1)
    next_x_list = hstack((kron(I, current_x), next_token_list))
    next_xstr_list = tokenizer.batch_decode(next_x_list)
    next_lcf_list = lcf.get_for_texts(next_xstr_list)
    return next_lcf_list

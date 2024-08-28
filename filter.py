# CBF-LLM Modules
from language_constraint_functions import LanguageCF
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
    clf_mapping: Dict[int, float] = field(default_factory=dict)


class Filter(ABC):
    """
    制約条件に従ってトークンを許可/禁止する
    """
    @abstractmethod
    def scan(self, x: Tensor, P: Tensor) -> FilterResult:
        pass


def get_next_CLF_list(
    current_x: Tensor,
    next_token_list: Union[Tensor, List[int]],
    tokenizer: PreTrainedTokenizerBase,
    clf: LanguageCF
) -> List[float]:
    """
    Parameters
    ----------
    current_x: Tensor
        現在のテキスト
    next_token_list: List[int]
        次に続くトークン一覧
    tokenizer: PreTrainedTokenizerBase
        生成言語モデルのトークナイザ
    clf: LanguageCF
        使用している制約言語関数
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
    next_CLF_list = clf.get_for_texts(next_xstr_list)
    return next_CLF_list
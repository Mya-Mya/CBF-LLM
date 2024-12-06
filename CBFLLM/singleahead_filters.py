from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from .torch_utils import *
import torch
import transformers

Tensor = torch.Tensor
PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase


@dataclass
class SingleAheadFilterSpecMeasureResult:
    satisfaction_list: List[bool] = field(default_factory=list)
    lcfvalue_list: List[float] = field(default_factory=list)


class SingleAheadFilterSpec(ABC):
    @abstractmethod
    def measure(
        self,
        x: Tensor,
        next_token_list: Tensor,
        xstr: str,
        next_xstr_list: List[str]
    ) -> SingleAheadFilterSpecMeasureResult:
        """
        Note that `xstr` is equivalent to the decoded `x`, 
        and `next_xstr_list[i]` is equivalent to the decoded `hstack((x, next_token_list[i]))`.

        Parameters
        ----------
        x: Tensor
            Current text in token-id-sequence format
        next_token_list: Tensor
            List of next token id
        xstr: str
            Current text in string format
        next_xstr_list: List[str]
            List of next text in string format
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


LCFValueMapping = Dict[int, float]


@dataclass
class SingleAheadFilterResult:
    allowed: List[int] = field(default_factory=list)
    disallowed: List[int] = field(default_factory=list)
    lcfvalue_mappings: Dict[str, LCFValueMapping] = field(default_factory=dict)


class SingleAheadFilterGroup:
    def __init__(
        self,
        specs: List[SingleAheadFilterSpec],
        top_k: int,
        baselinellm_tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 32
    ):
        self.specs = specs
        self.top_k = top_k
        self.tokenizer = baselinellm_tokenizer
        self.chunk_size = chunk_size

    def filter(self, x: Tensor, P: Tensor) -> SingleAheadFilterResult:
        filter_result = SingleAheadFilterResult()
        filter_result.lcfvalue_mappings = {
            spec.get_name(): {} for spec in self.specs
        }

        sorted_next_tokens = P.argsort(descending=True)

        xstr = self.tokenizer.decode(x)

        chunker = torch.ones(self.chunk_size, 1, dtype=P.dtype).to(P.device) # (chunk_size, 1)
        x_chunk = torch.kron(chunker, x) # (chunk_size, len(x))

        vocab_size = len(P)
        chunk_idx = 0
        while \
                len(filter_result.allowed) < self.top_k and \
                chunk_idx * self.chunk_size < vocab_size:
            offset = self.chunk_size * chunk_idx
            next_tokens_chunk = sorted_next_tokens[offset:offset+self.chunk_size] # (chunk_size, )
            next_tokens_chunk_T = next_tokens_chunk[None].T # (chunk_size, 1)
            next_x_chunk = torch.hstack((x_chunk, next_tokens_chunk_T)) # (chunk_size, len(x)+1)
            next_xstr_chunk = self.tokenizer.batch_decode(next_x_chunk) # (chunk_size,)

            # Measure
            measure_result_list = [
                spec.measure(
                    x,
                    next_tokens_chunk,
                    xstr,
                    next_xstr_list=next_xstr_chunk
                )
                for spec in self.specs
            ]

            # Add each token to either allowed/disallowed
            for i, token in enumerate(next_tokens_chunk):
                is_allowed = all([
                    measure_result.satisfaction_list[i]
                    for measure_result in measure_result_list
                ])
                if is_allowed:
                    filter_result.allowed.append(token)
                else:
                    filter_result.disallowed.append(token)
            
            # Record L-CF values of each spec
            next_tokens_chunk_list = tolist(next_tokens_chunk)
            for spec, measure_result in zip(self.specs, measure_result_list):
                name = spec.get_name()
                lcfvalue_list = measure_result.lcfvalue_list
                filter_result.lcfvalue_mappings[name] = \
                    dict(zip(next_tokens_chunk_list, lcfvalue_list))

            chunk_idx += 1

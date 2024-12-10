from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from .torch_utils import *
import torch
import transformers

Tensor = torch.Tensor
PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase


@dataclass
class SSAFilterSpecMeasureResult:
    satisfaction_list: List[bool] = field(default_factory=list)
    lcfvalue_list: List[float] = field(default_factory=list)


class SSAFilterSpec(ABC):
    @abstractmethod
    def measure(
        self,
        x: Tensor,
        next_token_list: Tensor,
        xstr: str,
        next_xstr_list: List[str]
    ) -> SSAFilterSpecMeasureResult:
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
class SSAFilterResult:
    allowed: List[int] = field(default_factory=list)
    disallowed: List[int] = field(default_factory=list)
    disallowed_by_spec:Dict[str, List[int]] = field(default_factory=list)
    lcfvalue_mappings: Dict[str, LCFValueMapping] = field(default_factory=list)


class SSAFilterGroup:
    def __init__(
        self,
        specs: List[SSAFilterSpec],
        top_k: int,
        baselinellm_tokenizer: PreTrainedTokenizerBase,
        chunk_size: int = 32
    ):
        self.specs = specs
        self.top_k = top_k
        self.tokenizer = baselinellm_tokenizer
        self.chunk_size = chunk_size

    def filter(self, x: Tensor, P: Tensor) -> SSAFilterResult:
        result = SSAFilterResult()
        result.lcfvalue_mappings = {
            spec.get_name(): {} for spec in self.specs
        }
        result.disallowed_by_spec = {
            spec.get_name(): [] for spec in self.specs
        }

        sorted_next_tokens = P.argsort(descending=True)

        xstr = self.tokenizer.decode(x)

        # (chunk_size, 1)
        chunker = torch.ones(self.chunk_size, 1, dtype=x.dtype).to(P.device)
        x_chunk = torch.kron(chunker, x)  # (chunk_size, len(x))

        vocab_size = len(P)
        chunk_idx = 0
        while \
                len(result.allowed) < self.top_k and \
                chunk_idx * self.chunk_size < vocab_size:
            offset = self.chunk_size * chunk_idx
            # (chunk_size, )
            next_tokens_chunk = sorted_next_tokens[offset:offset+self.chunk_size]
            next_tokens_chunk_T = next_tokens_chunk[None].T  # (chunk_size, 1)
            # (chunk_size, len(x)+1)
            next_x_chunk = torch.hstack((x_chunk, next_tokens_chunk_T))
            # (chunk_size,)
            next_xstr_chunk = self.tokenizer.batch_decode(next_x_chunk)

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

            for i, token_id in enumerate(tolist(next_tokens_chunk)):
                # Record L-CF values
                for spec, measure_result in zip(self.specs, measure_result_list):
                    name = spec.get_name()
                    result.lcfvalue_mappings[name][token_id] = measure_result.lcfvalue_list[i]

                # Update disallowed-by-spec
                is_allowed = True
                for spec, measure_result in zip(self.specs, measure_result_list):
                    does_spec_allow = measure_result.satisfaction_list[i]
                    if not does_spec_allow:
                        result.disallowed_by_spec[spec.get_name()].append(token_id)
                        is_allowed = False
                
                # Add the token to either allowed/disallowed
                if is_allowed:
                    result.allowed.append(token_id)
                    if len(result.allowed) == self.top_k:
                        return result
                else:
                    result.disallowed.append(token_id)

            chunk_idx += 1
        return result

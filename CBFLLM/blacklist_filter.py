# CBF-LLM Modules
from .torch_utils import *
from .filter import Filter, FilterResult, get_next_lcf_list
from .language_constraint_functions import LanguageCF
# Other Modules
from typing import Any
from transformers import PreTrainedTokenizerBase
# PyTorch
import torch
Tensor = torch.Tensor
zeros = torch.zeros
float32 = torch.float32


class BlacklistFilter(Filter):
    def __init__(
        self,
        top_k: int,
        tokenizer: PreTrainedTokenizerBase,
        clf: LanguageCF,
        chunk_size: int = 32
    ) -> None:
        """
        Parameters
        ----------
        top_k: int
            Generation parameter top-k
        tokenizer: PreTrainedTokenizerBase
            Tokenizer of the generative language model
        lcf: LanguageCF
            The language-constraint function object
        chunk_size: int, optional (default = 32)
            Chunk size for querying the constraint language function
          """
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.clf = clf
        self.chunk_size = chunk_size

        self.vocab_size = len(tokenizer.get_vocab())
        self.all_disallowed_Q = zeros((self.vocab_size,), dtype=float32)

        self.chunk_size = chunk_size

    def scan(self, x: Tensor, P: Tensor) -> Any:
        assert (P.sum() - 1.0).abs() < 1e-3, f"Sum of input distribution P = {
            tofloat(P.sum())}, expected around 1."
        R = FilterResult()

        sorted_next_tokens = P.argsort(descending=True)

        chunk_idx = 0
        searched_cnt = 0
        allowed_cnt = 0
        while allowed_cnt < self.top_k and searched_cnt < self.vocab_size:
            offset = self.chunk_size * chunk_idx
            chunk_next_token_list = sorted_next_tokens[offset:offset+self.chunk_size]
            chunk_next_h_list = get_next_lcf_list(
                x, chunk_next_token_list, self.tokenizer, self.clf)

            for t, h in zip(tolist(chunk_next_token_list), chunk_next_h_list):
                R.lcf_mapping[t] = h
                if h >= 0:
                    R.allowed.append(t)
                    allowed_cnt += 1
                    if allowed_cnt >= self.top_k:
                        break
                else:
                    R.disallowed.append(t)
                searched_cnt += 1
            chunk_idx += 1
        return R

# CBF-LLM Modules
from .torch_utils import *
from .filter import Filter, FilterResult, get_next_lcf_list
from .language_constraint_functions import LanguageCF
# Other Modules
from typing import Optional
from transformers import PreTrainedTokenizerBase
# PyTorch
import torch
Tensor = torch.Tensor
arange = torch.arange


class JustTopkFilter(Filter):
    def __init__(
            self,
            top_k: int,
            output_lcf_mapping: bool = False,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            lcf: Optional[LanguageCF] = None
    ) -> None:
        """
        Performs only top-k sampling, with no control.
        It is useful fo NoControl case.

        Parameters
        ----------
        top_k: int
            Generation parameter top-k
        output_lcf_mapping: bool = False
            Whether to include the 'lcf_mapping' in the filter results. Enabling this may increase computation time.

        tokenizer: Optional[PreTrainedTokenizerBase] = None
            The tokenizer for the generative language model. If 'lcf_mapping' is specified, this cannot be None.

        lcf: Optional[ConstraintLF] = None
            The language-constraint function object. If 'lcf_mapping' is specified, this cannot be None.
        """
        self.top_k = top_k
        self.output_lcf_mapping = output_lcf_mapping
        self.tokenizer = tokenizer
        self.lcf = lcf

    def scan(self, x: Tensor, P: Tensor) -> FilterResult:
        R = FilterResult()
        minimum = P.topk(self.top_k).values[-1]
        allowed_tokens = arange(len(P)).to(P.device)[P >= minimum]
        R.allowed = tolist(allowed_tokens)
        if not self.output_lcf_mapping:
            return R
        next_clf_list = get_next_lcf_list(
            x, allowed_tokens, self.tokenizer, self.lcf)
        R.lcf_mapping = {t: h for t, h in zip(allowed_tokens, next_clf_list)}
        return R

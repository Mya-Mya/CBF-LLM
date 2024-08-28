# CBF-LLM Modules
from torch_utils import *
from filter import Filter, FilterResult, get_next_CLF_list
from language_constraint_functions import LanguageCF
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
            output_clf_mapping: bool = False,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            clf: Optional[LanguageCF] = None
    ) -> None:
        """
        Top-Kサンプリングをするだけのフィルタ．無制御生成をする際に便利．

        Parameters
        ----------
        top_k: int
            生成パラメータトップk
        output_constraint_mapping: bool=False
            スキャン結果に`constraint_mapping`を載せるか．載せる場合，計算時間がかかる．
        tokenizer: Optional[PreTrainedTokenizerBase]=None
            生成言語モデルのトークナイザ．`output_constraint_mapping`を指定した場合，Noneにしてはならない．
        clf: Optional[ConstraintLF]=None
            制約言語関数．`output_constraint_mapping`を指定した場合，Noneにしてはならない．
        """
        self.top_k = top_k
        self.output_clf_mapping = output_clf_mapping
        self.tokenizer = tokenizer
        self.clf = clf

    def scan(self, x: Tensor, P: Tensor) -> FilterResult:
        R = FilterResult()
        minimum = P.topk(self.top_k).values[-1]
        allowed_tokens = arange(len(P)).to(P.device)[P >= minimum]
        R.allowed = tolist(allowed_tokens)
        if not self.output_clf_mapping:
            return R
        # 制約言語関数値の調査
        next_clf_list = get_next_CLF_list(
            x, allowed_tokens, self.tokenizer, self.clf)
        R.clf_mapping = {t: h for t, h in zip(allowed_tokens, next_clf_list)}
        return R

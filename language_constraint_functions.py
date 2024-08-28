from typing import Callable, Optional, List, Dict, Set, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import no_grad

class LanguageCF:
    """
    制約言語関数(Language-Constraint Function, L-CF) h
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        mapper: Callable[[SequenceClassifierOutput], List[float]],
        name: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        model: PreTrainedModel
            制約言語関数に使うモデル
        tokenizer: PreTrainedTokenizerBase
            制約言語関数に使うモデルのトークナイザ
        mapper: Callable[[SequenceClassifierOutput], List[float]],
            モデルの出力から制約言語関数の値に変換する関数
        name: Optional[str] = None
            この制約言語関数の名前
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mapper = mapper
        if name is None:
            name = model.name_or_path
        self.name = name

        self.cache: Dict[str, float] = dict()
        self.cache_keys: Set[str] = set()

    def get_for_text(self, xstr: str) -> float:
        return self.get_for_texts([xstr])[0]

    def get_for_texts(self, xstr_list: List[str]) -> List[float]:
        # キャッシュに無いものを探す
        has_any_nc = False
        nc_xstr_list = []
        for xstr in xstr_list:
            if not xstr in self.cache_keys:
                has_any_nc = True
                nc_xstr_list.append(xstr)

        # キャッシュに無いものがある
        if has_any_nc:
            self.prepare_cache(nc_xstr_list)

        # 全てのxstrに対してキャッシュがあるはず
        return [self.cache[xstr] for xstr in xstr_list]

    @no_grad
    def prepare_cache(self, xstr_list: List[str]) -> List[float]:
        if len(xstr_list) == 0:
            return []
        inputs = self.tokenizer(xstr_list, padding=True,
                                return_tensors="pt").to(self.model.device)
        output = self.model(**inputs)
        res = self.mapper(output)
        # キャッシュに登録する
        for xstr, h in zip(xstr_list, res):
            self.cache[xstr] = h
        self.cache_keys |= set(xstr_list)

    def __call__(self, xstr: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(xstr, List):
            return self.get_for_texts(xstr)
        if isinstance(xstr, str):
            return self.get_for_text(xstr)
        raise TypeError(f"Invalid Type: Got {type(xstr)}")

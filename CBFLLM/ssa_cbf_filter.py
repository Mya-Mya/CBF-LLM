from .language_constraint_functions import LanguageCF
from .ssa_filters import SSAFilterSpec, SSAFilterSpecMeasureResult


class SSACBFFilterSpec(SSAFilterSpec):
    def __init__(self, lcf: LanguageCF, alpha: float):
        self.lcf = lcf
        self.alpha = alpha
        self.one_minus_alpha = 1. - alpha

    def measure(self, x, next_token_list, xstr, next_xstr_list) -> SSAFilterSpecMeasureResult:
        lcfvalue = self.lcf.get_for_text(xstr)
        threshold = self.one_minus_alpha * lcfvalue
        next_lcfvalue_list = self.lcf.get_for_texts(next_xstr_list)

        result = SSAFilterSpecMeasureResult()
        for next_lcfvalue in next_lcfvalue_list:
            satisfaction = next_lcfvalue >= threshold
            result.satisfaction_list.append(satisfaction)
            result.lcfvalue_list.append(next_lcfvalue)
        return result
    
    def get_name(self):
        return f"CBF(lcf={self.lcf.name}, alpha={self.alpha})"
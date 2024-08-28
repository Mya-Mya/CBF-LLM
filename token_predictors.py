import torch
Tensor = torch.Tensor
softmax = torch.softmax


def distributionify(logit: Tensor, temperature: float) -> Tensor:
    """
    生成言語モデルの出力ロジット`logit`を確率分布Pに変換する．
    Parameters
    ----------
    logit: Tensor
        生成言語モデルの出力ロジット
    temperature: float
        温度パラメータ
    Returns
    -------
    P: Tensor
        各トークンに対する確率分布
    """
    return softmax(logit / temperature, dim=0)

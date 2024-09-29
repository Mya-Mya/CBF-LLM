import torch
Tensor = torch.Tensor
softmax = torch.softmax


def distributionify(logit: Tensor, temperature: float) -> Tensor:
    """
    Convert the output logits of a generative language model to a probability distribution.
    In HuggingFace Transformrs, this process is performed in the last (after the token processors are executed),
    but this process is performed first in my paper.

    Parameters
    ----------
    logit: Tensor
        The output logits from the baseline LLM
    temperature: float
        Temperature parameter
    Returns
    -------
    P: Tensor
        The probability distribution over tokens
    """
    return softmax(logit / temperature, dim=0)

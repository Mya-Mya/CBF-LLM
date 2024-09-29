import numpy


def tonumpy(x) -> numpy.ndarray:
    return x.detach().cpu().numpy()


def tofloat(x) -> float:
    return float(tonumpy(x))


def toint(x) -> int:
    return int(tonumpy(x))


def tolist(x) -> list:
    return tonumpy(x).tolist()


oo = float("Inf")

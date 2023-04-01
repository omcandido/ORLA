import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable
from enum import IntEnum
import matplotlib.pyplot as plt

def max_rand_tie(values: List) -> Tuple[int, float, List]:
    """Returns the maximum element of a list breaking ties at random.

    Args:
        values (List): Values to get the max from.

    Returns:
        Tuple[int, float, List]: `idx` is the index of the chosen max element. `val` is the value of the maximum element. `best` contains a list of the indices of all maximum elements. 
    """
    best = np.argwhere(values == np.amax(values)).flatten().tolist()
    idx = np.random.choice(best)
    val = values[idx]
    return idx, val, best


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_alpha_w(n_args, n_iterations = 1e4):
    alpha_w = 0.1 / (n_iterations * np.mean(range(n_args)))
    return alpha_w

def flatten(items):
    """Yield items from any nested iterable; https://stackoverflow.com/a/40857703"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

class Mode (IntEnum):
    STRICT   = 0
    NON_STRICT = 1


def plot_returns(returns: List[float], window: int):
    returns = pd.DataFrame(returns)/10
    returns = returns.rolling(window).mean()
    plt.plot(-returns)
    plt.xlabel("Episode #")
    plt.ylabel("Return (seconds)")
    plt.show()

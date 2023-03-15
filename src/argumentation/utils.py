import numpy as np
from typing import List, Union

Ranking = List[ Union [str, list[str]]]
Argument = str
Arguments = List[Argument]

def ranking_to_matrix(
        ranking: Ranking, 
        args: Arguments,
        as_bool: bool = False
    ) -> np.ndarray:
    
        """Creates a matrix that encodes the (partial) ranking of arguments.

    Args:
        ranking (Ranking): current (partial) ranking
        args (Arguments): full list of arguments. The returned matrix preserves indices with this list.
        as_bool (bool, optional): whether the returned matrix should be a Boolean matrix. Defaults to False.

    Returns:
        _type_: returns a matrix with the encoded (partial) ranking
    """

        mat = np.zeros((len(args), len(args)), int)
        running_indices = []
        for level in ranking:

            # Transform everything to a list (even if it only has 1 element)
            # to have a unique implementation
            level = [level] if not isinstance(level, list) else level

            for arg in level:
                row = np.ones(len(args))
                row[running_indices] = 0
                mat[args.index(arg), :] = row
            
            # Keep indices of the arguments of the parsed levels
            curr_idx = [args.index(arg) for arg in level]
            running_indices.extend(curr_idx)

            # Strict order: remove diagonal 1's
            np.fill_diagonal(mat, 0)

        if as_bool:
            return np.array(mat, dtype=bool)
        return mat

def save_ranking(path: str , args: Arguments, ranking: Ranking):
    f = open(path, "w")
    for i, elem in enumerate(ranking):
        if (isinstance(elem, list)):
            for e in elem:
                arg_idx = args.index(e)
                f.write("{} {}\n".format(arg_idx, len(ranking)-i))
        else:
            arg_idx = args.index(elem)
            f.write("{} {}\n".format(arg_idx, len(ranking)-i))
    f.close()

def save_values(path: str, args, arg_val: dict):
    f = open(path, "w")
    for arg in arg_val:
        arg_idx = args.index(arg)
        f.write("{} {}\n".format(arg_idx, arg_val[arg]))
    f.close()
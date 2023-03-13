import numpy as np
from typing import List

def order_to_matrix(
        order: List[str], 
        args: List[str],
        as_bool: bool = False
    ) -> np.ndarray:
    
        """Creates a matrix that encodes the (partial) ordering of arguments.

    Args:
        order (List[str]): current (partial) ordering
        args (List[str]): full list of arguments. The returned matrix preserves indices with this list.
        as_bool (bool, optional): whether the returned matrix should be a Boolean matrix. Defaults to False.

    Returns:
        _type_: returns a matrix with the encoded (partial) ordering
    """

        mat = np.zeros((len(args), len(args)), 'int')
        indices_ord = np.array([args.index(ord) for ord in order], 'int')
        for i in range(len(indices_ord)):
            row = np.ones(len(args))
            row[indices_ord[:i+1]] = 0
            mat[indices_ord[i],:] = row

        if as_bool:
            return np.array(mat, dtype=bool)
        return mat
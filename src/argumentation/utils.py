import numpy as np
import pandas as pd
from typing import List, MutableSet, Tuple, Dict

Argument = str
Arguments = List[Argument]
Ranking = List[List[Argument]]

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
            for arg in level:
                row = np.ones(len(args))
                row[running_indices] = 0
                mat[args.index(arg), :] = row
            
            # Keep indices of the arguments of the parsed levels
            curr_idx = [args.index(arg) for arg in level]
            running_indices.extend(curr_idx)

            # Strict order: remove diagonal 1's
            np.fill_diagonal(mat, 0)

        mat = np.array((mat == 0), int)

        if as_bool:
            return np.array(mat, dtype=bool)
        return mat

def values_to_ranking(arg_val):
    unique_values = list(set(arg_val.values()))
    unique_values.sort(reverse=True)

    ranking = []
    for val in unique_values:
        level = []
        for arg in arg_val:
            if arg_val[arg] == val:
                level.append(arg)
        ranking.append(level)
    return ranking

def ranking_to_values(ranking):
    arg_val = dict()
    for i, level in enumerate(ranking):
        val = len(ranking) - i
        for arg in level:
            arg_val[arg] = val
    return arg_val

def text_to_values(path:str, arg_id: Dict):
    arg_val = dict()
    df = pd.read_csv(path, header=None, sep=' ', names=(('argID', 'val')))
    for idx, row in df.iterrows():
        arg_name = arg_id[row['argID']]
        arg_val[arg_name] = row['val']
    return arg_val

def filter_vals_by_arg(arg_val: dict, arguments: Arguments):
    return {a:arg_val[a] for a in arg_val if a in arguments}


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

def save_values(path: str, args, arg_val: Dict):
    f = open(path, "w")
    for arg in arg_val:
        arg_idx = args.index(arg)
        f.write("{} {}\n".format(arg_idx, arg_val[arg]))
    f.close()


def construct_all_attacks(arg_actions: Dict) -> MutableSet[Tuple[str, str]]:
    """Given a dictionary of arguments and their promoted action, returns a set with all attacks among them.
    Args:
        arg_actions (dict): dictionary in the format {argument: action}
    Returns:
        MutableSet[Tuple[str, str]]: set of attacks among
    """
    attacks = set()
    for arg1 in arg_actions:
        for arg2 in arg_actions:
            if arg_actions[arg1] != arg_actions[arg2]:
                attacks.add((arg1, arg2))
                attacks.add((arg2, arg1))      
    return attacks
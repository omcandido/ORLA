from enum import IntEnum
import subprocess
from collections import defaultdict
import numpy as np

from argumentation import utils as argm

argument_types = ["TackleBall", "MinAngle", "MinDist", "FarKeeper", "OpenKeeper"]

def get_global_values(ranking: argm.Ranking):
    args = list(all_arguments.values())
    values = defaultdict(lambda: [])

    for arg in ranking:
        arg = arg[0]
        for arg_type in argument_types:
            if str(arg).startswith(arg_type):
                val = len(all_arguments) - ranking.index([arg])
                values[arg_type].append(val)

    get_global_values = dict()
    for val in values:
        data = np.array(values[val])
        mean = data.mean()
        get_global_values[val] = round(mean)

    get_global_values = dict(sorted(get_global_values.items(), key=lambda x:x[1], reverse=True))

    global_arg_val = []
    for arg in args:
        for glob in get_global_values:
            if str(arg).startswith(glob):
                global_arg_val.append((arg, get_global_values[glob]))
    
    global_arg_val = dict(global_arg_val)
    global_arg_val = dict(sorted(global_arg_val.items(), key=lambda x:x[1], reverse=True))

    return get_global_values, global_arg_val


t1_arguments = {
    0 : 'TackleBall1',
    1 : 'OpenKeeper1,2',
    2 : 'FarKeeper1,2',
    3 : 'MinAngle1,2',
    4 : 'MinDist1,2',
    5 : 'OpenKeeper1,3',
    6 : 'FarKeeper1,3',
    7 : 'MinAngle1,3',
    8 : 'MinDist1,3',
    9 : 'OpenKeeper1,4',
    10 : 'FarKeeper1,4',
    11 : 'MinAngle1,4',
    12 : 'MinDist1,4'}

t2_arguments = {
    13 : 'TackleBall2',
    14 : 'OpenKeeper2,2',
    15 : 'FarKeeper2,2',
    16 : 'MinAngle2,2',
    17 : 'MinDist2,2',
    18 : 'OpenKeeper2,3',
    19 : 'FarKeeper2,3',
    20 : 'MinAngle2,3',
    21 : 'MinDist2,3',
    22 : 'OpenKeeper2,4',
    23 : 'FarKeeper2,4',
    24 : 'MinAngle2,4',
    25 : 'MinDist2,4'}

t3_arguments = {
    26 : 'TackleBall3',
    27 : 'OpenKeeper3,2',
    28 : 'FarKeeper3,2',
    29 : 'MinAngle3,2',
    30 : 'MinDist3,2',
    31 : 'OpenKeeper3,3',
    32 : 'FarKeeper3,3',
    33 : 'MinAngle3,3',
    34 : 'MinDist3,3',
    35 : 'OpenKeeper3,4',
    36 : 'FarKeeper3,4',
    37 : 'MinAngle3,4',
    38 : 'MinDist3,4'}

all_arguments = t1_arguments | t2_arguments | t3_arguments

# Takeaway Actions.
class Actions(IntEnum):
    TackleBall   = 0
    MarkKeeper_2 = 1
    MarkKeeper_3 = 2
    MarkKeeper_4 = 3

arg_actions = {
    # Arguments T1
    'TackleBall1': Actions.TackleBall,
    'OpenKeeper1,2': Actions.MarkKeeper_2,
    'FarKeeper1,2': Actions.MarkKeeper_2,
    'MinAngle1,2': Actions.MarkKeeper_2,
    'MinDist1,2': Actions.MarkKeeper_2,
    'OpenKeeper1,3': Actions.MarkKeeper_3,
    'FarKeeper1,3': Actions.MarkKeeper_3,
    'MinAngle1,3': Actions.MarkKeeper_3,
    'MinDist1,3': Actions.MarkKeeper_3,
    'OpenKeeper1,4': Actions.MarkKeeper_4,
    'FarKeeper1,4': Actions.MarkKeeper_4,
    'MinAngle1,4': Actions.MarkKeeper_4,
    'MinDist1,4': Actions.MarkKeeper_4,
    # Arguments T2
    'TackleBall2': Actions.TackleBall,
    'OpenKeeper2,2': Actions.MarkKeeper_2,
    'FarKeeper2,2': Actions.MarkKeeper_2,
    'MinAngle2,2': Actions.MarkKeeper_2,
    'MinDist2,2': Actions.MarkKeeper_2,
    'OpenKeeper2,3': Actions.MarkKeeper_3,
    'FarKeeper2,3': Actions.MarkKeeper_3,
    'MinAngle2,3': Actions.MarkKeeper_3,
    'MinDist2,3': Actions.MarkKeeper_3,
    'OpenKeeper2,4': Actions.MarkKeeper_4,
    'FarKeeper2,4': Actions.MarkKeeper_4,
    'MinAngle2,4': Actions.MarkKeeper_4,
    'MinDist2,4': Actions.MarkKeeper_4,
    # Arguments T3
    'TackleBall3': Actions.TackleBall,
    'OpenKeeper3,2': Actions.MarkKeeper_2,
    'FarKeeper3,2': Actions.MarkKeeper_2,
    'MinAngle3,2': Actions.MarkKeeper_2,
    'MinDist3,2': Actions.MarkKeeper_2,
    'OpenKeeper3,3': Actions.MarkKeeper_3,
    'FarKeeper3,3': Actions.MarkKeeper_3,
    'MinAngle3,3': Actions.MarkKeeper_3,
    'MinDist3,3': Actions.MarkKeeper_3,
    'OpenKeeper3,4': Actions.MarkKeeper_4,
    'FarKeeper3,4': Actions.MarkKeeper_4,
    'MinAngle3,4': Actions.MarkKeeper_4,
    'MinDist3,4': Actions.MarkKeeper_4,
}

def get_host(
        mode: str = 'recv', 
        path = r"\\wsl.localhost\Ubuntu-16.04\mnt\wsl\resolv.conf"
    ) -> str:

    """This is an auxiliary function to connect to RoboCup when the rcssserver is running on WSL.
       This function returns the IP of the selected host necessary for the UPD connections.
       (Disclaimer: this solution is hacky and ugly, there's probably a better way to manage 
       the connections, but I'm not that familiar with socket programming).

    Args:
        mode (str, optional): whether we want the IP of the server in the client (send) or the IP of the client in WLS (recv).
        path (str, optional): the path to the resolv.conf of you WSL installation. 

    Returns:
        str: the IP of the requested host. 
    """
    if mode == 'send':
        batcmd="wsl hostname -I"
        result = subprocess.check_output(batcmd, shell=True)
        return result.decode("utf-8").strip()
    else:
        f = open(path)
        for line in f:
            li=line.strip()
            if not li.startswith("#"):
                if line.rstrip().split(' ')[0] == 'nameserver':
                    return line.rstrip().split(' ')[1]
        # If we get here, the nameserver could not be read.
        raise RuntimeError("No nameserver has been detected in the resolv.conf file")


def show_ranking_by_taker(ranking):
    arg_val = argm.ranking_to_values(ranking)
    o1 = argm.filter_vals_by_arg(arg_val, t1_arguments.values())
    o1 = argm.values_to_ranking(o1)
    print("    T1: ", o1)
    o2 = argm.filter_vals_by_arg(arg_val, t2_arguments.values())
    o2 = argm.values_to_ranking(o2)
    print("    T2: ", o2)
    o3 = argm.filter_vals_by_arg(arg_val, t3_arguments.values())
    o3 = argm.values_to_ranking(o3)
    print("    T3: ", o3)

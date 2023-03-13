from enum import IntEnum
import subprocess

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


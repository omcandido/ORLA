from typing import List
import socket

class Takeaway():
    """Interface that communicates with rcssserver via sockets.
    It stores the ordering that needs to be evaluated, sends a start signal 
    to rcssserver and returns the episode duration (total reward).
    """
    
    def __init__(
        self,
        args: List[str],
        send_host: str,
        send_port: int,
        recv_host: str,
        recv_port: int,
        ordering_path: str
    ):
        """Initialise Takeaway

        Args:
            args (List[str]): list of arguments to order
            send_host (str): IP of the WSL instance
            send_port (int): port of the WSL instance
            recv_host (str): IP of the windows host
            recv_port (int): port of the windows host
            ordering_path (str): path where the ordering will be written (and read by the RoboCup takers).
        """
    
        self._args = args
        self._size = len(args)
        self._order = []
        self._order_idx = []
        self._send_host = send_host
        self._send_port = send_port
        self._recv_host = recv_host
        self._recv_port = recv_port
        self._ordering_path = ordering_path

    def _save_ordering(self, ordering: List[str]) -> None:
        assert len(ordering)==self._size,  "Error: Ordering is not the right size."
        f = open(self._ordering_path, "w")
        for i, elem in enumerate(ordering):
            arg_idx = self._args.index(elem)
            f.write("{} {}\n".format(arg_idx, self._size-i))
        f.close()

    def play(self, ordering: List[str]) -> float:
        """Send a message to RoboCup to start playing and listen until it receives the final reward.

        Returns:
            float: the reward output by the game
        """
        self._save_ordering(ordering)
        self._start_game()
        reward = self._wait_termination()
        return reward        

    def _start_game(self):
        """Send rcssserver a message to start the episode.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto("start".encode(), (self._send_host, self._send_port))

    def _wait_termination(self) -> float:
        """Wait until it receives a reward from rcssserver, indicating the end of the episode.

        Returns:
            float: final return of the episode.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self._recv_host, self._recv_port))
            reward = ""
            while reward == "":
                reward, addr = s.recvfrom(16)
                reward = float(reward.decode("utf-8"))
                reward = - reward
            return reward
from typing import List
import socket
from argumentation import utils as argm
from environments.takeaway.utils import get_global_values
from environments.environment import Environment

class Takeaway(Environment):
    """Interface that communicates with rcssserver via sockets.
    It stores the ranking that needs to be evaluated, sends a start signal 
    to rcssserver and returns the episode duration (total reward).
    """
    
    def __init__(
        self,
        args: argm.Arguments,
        send_host: str,
        send_port: int,
        recv_host: str,
        recv_port: int,
        ranking_path: str
    ):
        """Initialise Takeaway

        Args:
            args (List[str]): list of arguments to order
            send_host (str): IP of the WSL instance
            send_port (int): port of the WSL instance
            recv_host (str): IP of the windows host
            recv_port (int): port of the windows host
            ranking_path (str): path where the ranking will be written (and read by the RoboCup takers).
        """
    
        super().__init__(args, None)
        self._size = len(args)
        self._order = []
        self._order_idx = []
        self._send_host = send_host
        self._send_port = send_port
        self._recv_host = recv_host
        self._recv_port = recv_port
        self._ranking_path = ranking_path

    def get_premises(self, obs):
        pass

    def get_arguments(self, premises):
        pass

    def update_memory(self, obs, act):
        pass
    
    def reset_memory(self, obs, act):
        pass

    def play(self, ranking: argm.Ranking) -> float:
        """Send a message to RoboCup to start playing and listen until it receives the final reward.

        Returns:
            float: the reward output by the game
        """
        argm.save_ranking(self._ranking_path, self._arguments, ranking)
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
            s.settimeout(10)
            s.bind((self._recv_host, self._recv_port))
            reward = ""
            while reward == "":
                try:
                    reward, addr = s.recvfrom(16)
                    reward = float(reward.decode("utf-8"))
                    reward = - reward
                    s.close()
                except socket.timeout:
                    print("warning: missed result")
                    self._start_game()

            return float(reward)
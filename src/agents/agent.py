from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from argumentation import utils as argm
from utils import flatten

class Agent(ABC):
    def __init__(self, args: List[str]):
        super().__init__()
        self.args = args
        self.n = len(args)
    
    @abstractmethod
    def get_action_probs(self, state: np.ndarray, mask:Optional[np.ndarray] = None) -> np.ndarray:
        pass

    @abstractmethod
    def learn(self):
        pass

    def derive_ranking(self, greedy=False) -> Tuple[List[str], List[float]]:
        ranking = []
        probs = []
        for _ in range(self.n):
            state_t = argm.ranking_to_matrix(ranking, self.args, True)
            remaining = self.reimaining_arguments(ranking)
            remaining, mask = self.mask_remaining(remaining)
            probs_t = self.get_action_probs(state_t, mask)
            if greedy:
                idx = np.argmax(probs_t)
                choice_t = remaining[idx]
            else:
                choice_t = np.random.choice(remaining, p=probs_t)
                idx = remaining.index(choice_t)
            ranking.append(choice_t)
            probs.append(probs_t[idx])
        return ranking, probs

    def reimaining_arguments(self, ranking: argm.Ranking) -> argm.Arguments:
        remaining = [arg for arg in self.args if not arg in flatten(ranking)]
        return remaining

    def mask_remaining(self, remaining: argm.Arguments) -> np.ndarray:
        indices = np.where(np.in1d(self.args, remaining))[0]
        mask = np.zeros(self.n)
        mask[indices] = 1
        return mask.astype(bool)

    def save_ranking(self, path: str, ranking: argm.Ranking = None):
        if ranking == None:
            ranking, _ = self.derive_ranking(True)
        argm.save_ranking(path, self.args, ranking)



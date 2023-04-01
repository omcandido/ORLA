from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from argumentation import utils as argm
from utils import flatten, Mode
import torch
from torch.distributions import Categorical
from typing import List, Optional, Tuple

class Agent(ABC):
    def __init__(self, args: List[str], mode: Mode = Mode.STRICT):
        super().__init__()
        self.args = args
        self.n = len(args)
        self.mode = mode
    
    @abstractmethod
    def get_action_probs(self, state: np.ndarray, mask:Optional[np.ndarray] = None) -> np.ndarray:
        pass

    @abstractmethod
    def learn(self):
        pass

    def remaining_arguments(self, ranking: argm.Ranking) -> argm.Arguments:
        remaining = [arg for arg in self.args if not arg in flatten(ranking)]
        return remaining

    def mask_remaining(self, remaining: argm.Arguments) -> np.ndarray:
        if self.mode == Mode.NON_STRICT:
            mask = self.__mask_remaining_strict(remaining)
            return mask
        else:
            mask = self.__mask_remaining_non_strict(remaining)
            return mask
        
    def __mask_remaining_strict(self, remaining: argm.Arguments) -> np.ndarray:
        indices = np.where(np.in1d(self.args, remaining))[0]
        indices = np.append(indices*2, indices*2 +1)
        mask = np.zeros(2*self.n)
        mask[indices] = 1
        return mask.astype(bool)
    
    def __mask_remaining_non_strict(self, remaining: argm.Arguments) -> np.ndarray:
        indices = np.where(np.in1d(self.args, remaining))[0]
        mask = np.zeros(self.n)
        mask[indices] = 1
        return mask.astype(bool)

    def decode_ranking(self, greedy:bool=False) -> Tuple[List[str], List[float]]:
        if self.mode == Mode.STRICT:
            return self.__decode_ranking_strict(greedy)
        if self.mode == Mode.NON_STRICT:
            return self.__decode_ranking_non_strict(greedy)
        
    def __decode_ranking_strict(self, greedy: bool) -> Tuple[argm.Ranking, torch.Tensor]: 
        if self.mode != Mode.STRICT:
            raise RuntimeError("Attempted strict decoding.")
        ranking = []
        probabilities = []
        for _ in range(self.n):
            state = argm.ranking_to_matrix(ranking, self.args, True)
            remaining = self.remaining_arguments(ranking)
            mask = self.mask_remaining(remaining)
            probs = self.get_action_probs(state, mask)
            distribution = Categorical(probs)
            sample_idx = torch.argmax(probs) if greedy else distribution.sample()
            ranking.append([self.args[sample_idx]])
            probabilities.append(probs[sample_idx])
        probabilities = torch.stack(probabilities)
        return ranking, probabilities

    def __decode_ranking_non_strict(self, greedy: bool) -> Tuple[argm.Ranking, torch.Tensor]: 
        if self.mode != Mode.NON_STRICT:
            raise RuntimeError("Attempted non-strict decoding on a strict .")
        ranking = []
        probabilities = []
        for _ in range(self.n):
            state = argm.ranking_to_matrix(ranking, self.args, True)
            remaining = self.remaining_arguments(ranking)
            mask = self.mask_remaining(remaining)
            probs = self.get_action_probs(state, mask)
            m = Categorical(probs)
            idx = torch.argmax(probs) if greedy else m.sample()
            if idx %2 == 0:
                # Append to new level
                idx_arg = int(idx/2)
                new_arg = self.args[idx_arg]
                ranking.append([new_arg])
            else:
                #Append to previous level
                idx_arg = int((idx-1)/2)
                new_arg = self.args[idx_arg]
                if len(ranking) == 0:
                    ranking.append([new_arg])
                else:
                    ranking[-1].extend([new_arg])
            probabilities.append(probs[idx])
        probabilities = torch.stack(probabilities)
        return ranking, probabilities

    def save_ranking(self, path: str, ranking: argm.Ranking = None):
        if ranking == None:
            ranking, _ = self.decode_ranking(True)
        argm.save_ranking(path, self.args, ranking)



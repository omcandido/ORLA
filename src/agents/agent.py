from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from argumentation import utils as argm

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

    def derive_ordering(self, greedy=False) -> Tuple[List[str], List[float]]:
        ordering = []
        probs = []
        for _ in range(self.n):
            state_t = argm.order_to_matrix(ordering, self.args, True)
            remaining = self.reimaining_arguments(ordering)
            mask = self.mask_remaining(remaining)
            probs_t = self.get_action_probs(state_t, mask)
            if greedy:
                idx = np.argmax(probs_t)
                choice_t = remaining[idx]
            else:
                choice_t = np.random.choice(remaining, p=probs_t)
                idx = remaining.index(choice_t)
            ordering.append(choice_t)
            probs.append(probs_t[idx])
        return ordering, probs

    def reimaining_arguments(self, ordering: List[str]) -> List[str]:
        remaining = [arg for arg in self.args if not arg in ordering]
        return remaining

    def mask_remaining(self, remaining: List[str]) -> np.ndarray:
        indices = np.where(np.in1d(self.args, remaining))[0]
        mask = np.zeros(self.n)
        mask[indices] = 1
        return mask.astype(bool)

    def save_ordering(self, path: str, ordering: Optional[List[str]] = None):
        if ordering == None:
            ordering, _ = self.derive_ordering(True)

        assert len(ordering)==self.n,  "Error: Ordering is not the right size."
        f = open(path, "w")
        for i, elem in enumerate(ordering):
            f.write("{} {}\n".format(self.args.index(elem), self.n-i))
        f.close()    



from agents.agent import Agent
from argumentation import utils as argm

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class ORLA(Agent):
    def __init__(self, args: List[str], alpha_th:float,  device: torch.device):
        super().__init__(args)
        self.net = self.Net(self.n).to(device)
        self.device = device
        self.optimizer = optim.Adam(self.net.parameters(), lr=alpha_th)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=alpha_th)

    class Net(nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.n = n

            self.fc1 = nn.Linear(n*n, n*n)
            # self.fc2 = nn.Linear(n*n*2, n*n)
            self.fc3 = nn.Linear(n*n, n)
            # self.dropout = nn.Dropout(p=0.8)

            self.mask = torch.ones(n)

        def forward(self, x):
            x = self.fc1(x)
            # x = self.dropout(x)
            # x = F.relu(x)
            # x = self.fc2(x)
            # x = self.dropout(x)
            x = F.relu(x)
            logits = self.fc3(x)
            logits[~self.mask] = float('-inf')
            return F.softmax(logits, dim=0)

    def get_action_probs(self, state: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        state_flat = torch.from_numpy(state).float().flatten().to(self.device)
        self.net.mask = torch.from_numpy(mask).bool().to(self.device)
        probs = self.net(state_flat)
        return probs

    def derive_ordering(self, greedy=False) -> Tuple[List[str], torch.Tensor]:
        ordering = []
        probs = []

        for _ in range(self.n):
            state_t = argm.order_to_matrix(ordering, self.args, True)
            remaining = self.reimaining_arguments(ordering)
            mask = self.mask_remaining(remaining)
            probs_t = self.get_action_probs(state_t, mask)
            m = Categorical(probs_t)
            idx = torch.argmax(probs_t) if greedy else m.sample()
            ordering.append(self.args[idx])
            probs.append(probs_t[idx])
        probs = torch.stack(probs)
        return ordering, probs

    def learn(self, probs: torch.Tensor, discounted_returns: torch.Tensor):
        loss = torch.log(probs) * discounted_returns
        loss = - loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ORLABaseline(ORLA):
    def __init__(self, args: List[str], alpha_th: float, alpha_w: float, device: torch.device):
        super().__init__(args, alpha_th, device)
        self.w = np.zeros((self.n, self.n))
        self.alpha_w = alpha_w

    def state_value(self, state: np.ndarray):
        return np.sum(self.w[state])

    def learn(self, orderings: List[List[str]],  probs: List[torch.Tensor], final_returns: List[float]):
        deltas = []
        step_w = np.zeros_like(self.w)

        for i in range(len(final_returns)): # for each episode in the batch
            delta = []
            for t in range(self.n): # for each step in the episode
                state_t = argm.order_to_matrix(orderings[i][:t], self.args, True)
                delta_t = final_returns[i] - self.state_value(state_t)
                step_w[state_t] += self.alpha_w * delta_t
                delta.append(delta_t)
            deltas.append(delta)
        self.w += step_w

        probs_batch = torch.stack(probs)
        deltas_batch = torch.FloatTensor(deltas).to(self.device)
        super().learn(probs_batch, deltas_batch)
from agents.agent import Agent
from argumentation import utils as argm
from utils import Mode

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ORLA(Agent):
    def __init__(self, args: List[str], alpha_th:float,  device: torch.device, mode: Mode):
        super().__init__(args, mode)
        self.net = self.Net(self.n, mode).to(device)
        self.device = device
        self.optimizer = optim.Adam(self.net.parameters(), lr=alpha_th)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=alpha_th)

    class Net(nn.Module):
        def __init__(self, n: int, mode: Mode):
            super().__init__()
            self.n = n

            output_size = n if mode == Mode.STRICT else 2*n
            self.fc_in = nn.Linear(n*n, n*n)
            # self.fc_h = nn.Linear(2*n*n, n*n)
            self.fc_out = nn.Linear(n*n, output_size)
            self.mask = torch.ones(output_size) # mask out the appended arguments
            
        def forward(self, x):
            x = self.fc_in(x)
            x = F.relu(x)
            # x = self.fc_h(x)
            # x = F.relu(x)
            logits = self.fc_out(x)
            logits[~self.mask] = float('-inf') # If not remaining, -inf
            return F.softmax(logits, dim=0)

    def get_action_probs(self, state: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        state_flat = torch.from_numpy(state).float().flatten().to(self.device)
        self.net.mask = torch.from_numpy(mask).bool().to(self.device)
        probs = self.net(state_flat)
        return probs

    def learn(self, probs: torch.Tensor, advantages: torch.Tensor):
        loss = torch.mul(torch.log(probs),advantages)
        loss = - loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ORLABaseline(ORLA):
    def __init__(self, args: List[str], alpha_th: float, alpha_w: float, device: torch.device, mode: Mode):
        super().__init__(args, alpha_th, device, mode)
        self.w = np.zeros((self.n, self.n))
        self.alpha_w = alpha_w

    def state_value(self, state: np.ndarray):
        return np.sum(self.w[state])

    def learn(self, rankings: List[List[str]],  probs: List[torch.Tensor], final_returns: List[float]):
        deltas = []
        step_w = np.zeros_like(self.w)

        for i in range(len(final_returns)): # for each episode in the batch
            delta = []
            for t in range(self.n): # for each step in the episode
                state_t = argm.ranking_to_matrix(rankings[i][:t], self.args, True)
                delta_t = final_returns[i] - self.state_value(state_t)
                step_w[state_t] += self.alpha_w * delta_t
                delta.append(delta_t)
            deltas.append(delta)
        self.w += step_w


        probs_batch = torch.stack(probs)
        deltas_batch = torch.FloatTensor(deltas).to(self.device)
        super().learn(probs_batch, deltas_batch)
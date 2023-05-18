import numpy as np
import gymnasium as gym
from environments.foggy_frozen_lake.utils import FLActions

class FrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, multiple_visits = True):
        super().__init__(env)
        self.t = 0
        self.multiple_visits = multiple_visits
        self.previous_actions = np.full([*env.desc.shape, len(FLActions)], 0)
        self.hist = []
    def step(self, action):
        self.t +=1
        self.hist.append(self.s)
        if self.hist[-2:] == self.hist[-4:-2] == self.hist[-6:-4]:
            return self.s, 0, False, True, self.t

        if not self.multiple_visits and self.previous_actions[self.coordinates][action] == 1:
            return self.s, 0, False, True, self.t
            
        self.previous_actions[self.coordinates][action] = 1
        next_state, reward, terminated, truncated, info = super().step(action)
        info['t'] = self.t
        return next_state, reward, terminated, truncated, info
    def reset(self, **kwargs):
        self.t = 0
        self.previous_actions = np.full([*self.env.desc.shape, len(FLActions)], 0)
        self.hist = []
        return super().reset(**kwargs)
    def index_to_coordinate(self, index):
        return np.unravel_index(index, (self.nrow, self.ncol))
    @property
    def coordinates(self):
        """
        Returns the current state in coordinate form.
        """
        return self.index_to_coordinate(self.s)
    def in_hole(self):
        """
        True if the current cell is a hole. False otherwise.
        """
        r, c = self.coordinates
        if bytes.decode(self.desc[r,c]) == "H":
            return True
        return False

    def in_goal(self):
        """
        True if the current cell is the goal. False otherwise.
        """
        r, c = self.coordinates
        if bytes.decode(self.desc[r,c]) == "G":
            return True
        return False

    def _get_info(self):
        return {
            't': self.t
            }

class FrozenLakeObservationWrapper(gym.ObservationWrapper):
    """
    Transforms an index observation into a coordinate observation (e.g., in a 8x8 matrix, 
    the observation with index 8 corresponds to coordinate [1,0]).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    def observation(self, obs):
        return {
            'neighbours': self.coordinates,
            'previous_actions': self.previous_actions
            }
    
class FrozenLakeRewardWrapper(gym.RewardWrapper):
    """
    Subtracts 1 to the reward if the current cell is a hole.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    def reward(self, rew):
        if self.in_hole():
            return -1
        elif self.in_goal():
            return 1
        else:
            return 0

class FrozenLakeMapObservationWrapper(gym.ObservationWrapper):
    """
    Includes the map in the observation
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    def observation(self, obs):
        tiles = (self.desc == b'H').astype(int).flatten()
        pos = np.zeros(self.nrow * self.ncol)
        pos[self.s] = 1
        return np.concatenate([tiles, pos]).astype(bool)

class FrozenLakeNeighboursObservationWrapper(gym.ObservationWrapper):
    """
    Includes the neighbours in the observation
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    def observation(self, obs):
        neighbours = self.get_neighbours(self.desc, self.s)
        holes = neighbours == 'H'
        margin = neighbours == '0'
        safe = (neighbours == 'F') + (neighbours == 'S') + (neighbours == 'G')
        pos = np.zeros(self.nrow * self.ncol)
        pos[self.s] = 1
        # return np.concatenate([pos]).astype(bool)
        return np.concatenate([safe, holes, margin, pos]).astype(bool)
    
    @staticmethod
    def get_neighbours(tiles, index):
        padded = np.pad(tiles, 1)
        size = len(tiles)
        row, col = np.unravel_index(index, (size, size))
        index = (row+1)*(size+2) + col + 1
        row, col = np.unravel_index(index, (size+2, size+2))
        res = np.empty((8,), dtype="str")
        res[0] = padded[row, col-1]
        res[1:4] = padded[row-1, col-1:col+2]
        res[4] = padded[row, col+1]
        res[5:9] = np.flip(padded[row+1, col-1:col+2])
        return res
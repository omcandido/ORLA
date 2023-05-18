import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake  import generate_random_map

import numpy as np
from copy import deepcopy

from environments.foggy_frozen_lake.utils import FLActions
from environments.environment import Environment
from environments.foggy_frozen_lake.utils import Direction

from environments.foggy_frozen_lake.world import FrozenLakeWrapper, FrozenLakeRewardWrapper, FrozenLakeNeighboursObservationWrapper

class FFL(Environment):
    def __init__(self, arg_actions, n=8, p=0.8):
        env = gym.make("FrozenLake-v1",  is_slippery=False, desc=generate_random_map(n, p))
        env = FrozenLakeWrapper(env, multiple_visits=True)
        env = FrozenLakeNeighboursObservationWrapper(env)
        env = FrozenLakeRewardWrapper(env)
        super().__init__(arg_actions, env)
        self.n = n
        self.p = p
        self.reset_memory()

    def get_premises(self, observation):
        # Extract relevant vectors for convenience.
        safe = observation[0:8]
        holes = observation[0:16]
        margin = observation[16:24]

        # Get the tile index.
        tile_idx = np.where(observation[24:])[0][0]
        # Get the map size.
        map_size = int(np.sqrt(len(observation)-24))

        # Pad memory and convert to a 3D array for convenience.
        memory_pad = deepcopy(self.memory)
        memory_pad = memory_pad.reshape(map_size, map_size, 4)
        memory_pad = np.pad(memory_pad, pad_width=((1,1), (1,1), (0,0)))
        
        # Get the coordinates of the current tile in the padded memory.
        row, col = np.unravel_index(tile_idx, (map_size, map_size))
        tile_idx = (row+1)*(map_size+2) + col + 1
        row, col = np.unravel_index(tile_idx, (map_size+2, map_size+2))

        safe_up = safe[Direction.TOP]
        safe_down = safe[Direction.BOTTOM]
        safe_left = safe[Direction.LEFT]
        safe_right = safe[Direction.RIGHT]

        visited_up = any(memory_pad[row-1, col])
        visited_down = any(memory_pad[row+1, col])
        visited_left = any(memory_pad[row, col-1])
        visited_right = any(memory_pad[row, col+1])

        res = {
            'safe_up'    : safe_up,
            'safe_down'  : safe_down,
            'safe_left'  : safe_left,
            'safe_right' : safe_right,
            'visited_up'     : visited_up,
            'visited_down'   : visited_down,
            'visited_left'   : visited_left,
            'visited_right'  : visited_right,
        }
        
        return res
    
    def get_arguments(self, premises):
        args = []
        if premises['safe_up']:
            args.append('U')
        if premises['safe_down']:
            args.append('D')
        if premises['safe_left']:
            args.append('L')
        if premises['safe_right']:
            args.append('R')

        if premises['safe_up'] and not premises['visited_up']:
            args.append('nU')
        if premises['safe_down'] and not premises['visited_down']:
            args.append('nD')
        if premises['safe_left'] and not premises['visited_left']:
            args.append('nL')
        if premises['safe_right'] and not premises['visited_right']:
            args.append('nR')

        return args
    
    def update_memory(self, obs, act):
        # Make memory[tile_index][action] = True, 
        # to remember what actions were aready taken in the current tile.
        self.memory[obs[24:]==True, act] = True
    
    def reset_memory(self):
        # We want an array where for each tile, we can store what actions we took.
        # There are map_size x map_size x #actions bits to store.
        # Table format is chosen because map tiles are identified by tile index.
        self.memory = np.zeros((self.n*self.n, len(FLActions)), dtype=bool)


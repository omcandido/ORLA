from hashlib import new
from typing import Tuple, List
from enum import IntEnum
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake  import generate_random_map
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation
from copy import deepcopy

# Frozen Lake observation. E.g., (4,6)
FLObservation = Tuple[int,int]

fl_safe_cells = ('S', 'F', 'G')
fl_unsafe_cells = ('H', '0')


# Frozen Lake Actions.
class FLActions(IntEnum):
    LEFT  = 0
    DOWN  = 1
    RIGHT = 2
    UP    = 3

# This is a shorthand to identify the index of the neighbouring tile. I.e.:
# [1][2][3]
# [0]🤖[4]
# [7][6][5]
class Direction(IntEnum):
    LEFT         = 0
    TOP          = 2
    RIGHT        = 4
    BOTTOM       = 6

argument_actions = {
    'U': FLActions.UP,
    'L': FLActions.LEFT,
    'R': FLActions.RIGHT,
    'D': FLActions.DOWN,
    'nD': FLActions.DOWN,
    'nL': FLActions.LEFT,
    'nR': FLActions.RIGHT,
    'nU': FLActions.UP
}

def fl_map_to_str(map):
    return map.astype('U13')

def fl_plot_run(map, stat_from, stat_to, time, act, rew):
    map_size = len(map)
    def init_func():
        for r in range(map_size):
            for c in range(map_size):
                if map[r,c] == "H":
                    background_color = (0,0,1)
                elif map[r,c] == "S":
                    background_color = (0,1,1)
                elif map[r,c] == "G":
                    background_color = (0,1,0)
                else:
                    background_color = (1,1,1)
                char_x = 0.05 + ((1-0.05)/map_size)*c
                char_y = 1 - (0.1+(((1-0.05)/map_size)*r))
                ax.text(char_x, char_y, map[r,c],
                    fontsize=int(((1-0.2)/map_size)*200),
                    backgroundcolor=background_color,
                    family='monospace')
        # return [artist_state, artist_info]

    def animate(i, map, stat_from, stat_to, time, act, rew):

        def state_to_indices(state):
            """Transform the one-hot part of the state into coordinates in the map"""
            if type(state) is str:
                return state
            n = len(map)
            position = state[24:]
            index = np.argmax(position, axis=0)
            return np.unravel_index(index, (n, n))

        map_size = len(map)
        r,c = state_to_indices(stat_from[i])
        char_x = 0.05 + ((1-0.05)/map_size)*c
        char_y = 1 - (0.1+(((1-0.05)/map_size)*r))
        artist_state.set_x(char_x)
        artist_state.set_y(char_y)
        artist_state.set_text(map[r,c])
        action_str = 'None' if act[i] == 'None' else str(FLActions(act[i]))
        artist_info.set_text("\nTime step: {} \nAction: {} \nState: {} -> {} \nCurrent total reward: {:.4f}".format(time[i], action_str, state_to_indices(stat_from[i]), state_to_indices(stat_to[i]), rew[i]))
        # return [artist_state, artist_info]

    map = fl_map_to_str(map)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.axis('off')
    artist_state = ax.text(0.0, 0., "",
        fontsize=int(((1-0.2)/map_size)*200),
        backgroundcolor=(1,0,0),
        family='monospace',
        zorder=99999)
    artist_info = fig.text(0, 0.01,"")        
    ani = matplotlib.animation.FuncAnimation(fig, animate,
        fargs=(map,  stat_from, stat_to, time, act, rew), 
        init_func=init_func,
        # blit= True,
        frames=len(stat_from),
        interval=100)
    plt.close()
    # ani.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'],)
    return HTML(ani.to_jshtml())
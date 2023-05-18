from abc import ABC, abstractmethod
from argumentation import utils as argm
from argumentation.classes import ValuebasedArgumentationFramework
from copy import deepcopy
import numpy as np
import random
import time
from typing import Dict, List

class Environment(ABC):
    """Interface that ORLA uses to play a game and obtain the reward.
    This class is meant to be inherited by your own implementation.
    """
	
    def __init__(
            self,
            arg_actions: Dict,
            env,
    ):
        self._arg_actions = arg_actions
        self._env = env

        self._arguments = list(arg_actions.keys())
        self._attacks = argm.construct_all_attacks(arg_actions)

    @abstractmethod
    def get_premises(self, obs):
        pass

    @abstractmethod
    def get_arguments(self, premises):
        pass

    @abstractmethod
    def update_memory(self, obs, act):
        self.mem = NotImplemented

    @abstractmethod
    def reset_memory(self):
        self.mem = NotImplemented

    def play(self, ranking: argm.Ranking, render=False) -> float:
        """Uses the ranking to play the RL game and returns the total reward.

        Args:
            ranking (argm.Ranking): Arguments ORLA can use to play the game

        Returns:
            float: total reward emitted by the environment at the end of the task
        """
        vaf = ValuebasedArgumentationFramework(self._arguments, self._attacks, ranking)
        observation, _ = self._env.reset()
        self.reset_memory()
        total_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if render:
                self.render_frame()
            action = self.select_action(vaf, observation)
            self.update_memory(observation, action)
            observation, reward, terminated, truncated, _ = self._env.step(action)

            total_reward += reward
        
        if render:
            self.render_frame()
        return total_reward

    def render_frame(self):
        # self._env.render()
        # time.sleep(0.25)
        pass

    def select_action(self, vaf, obs) ->int:
        """Select an action according to the VAF it has been initialised with.
        Args:
            obs (_type_): observation of the game.
        Returns:
            int: index of the selected action.
        """
        vsaf = self.get_vsaf(vaf, obs)
        ext = self.get_extension(vsaf)
        action = self.get_extension_action(ext)
        return action
    
    def get_vsaf(self, vaf, obs) -> ValuebasedArgumentationFramework:
        """Get the value-based situation-specific argumentation framework (VSAF) given the current observation of the game.
        Args:
            obs (_type_): current observation of the game.
        Returns:
            ValuebasedArgumentationFramework: the VSAF.
        """
        vsaf = deepcopy(vaf)
        prems = self.get_premises(obs)
        valid_args = self.get_arguments(prems)
        invalid_args = set(vsaf.args) - set(valid_args)
        vsaf.remove_arguments(invalid_args)
        return vsaf
    
    @staticmethod
    def get_extension(vsaf: ValuebasedArgumentationFramework):
        """Returns the grounded extension. In a total strict order (such as as ours), the grounded extension contains always one argument.
        Args:
            vsaf (ValuebasedArgumentationFramework): the VSAF given the current observation of the game.
        Returns:
            _type_: arguments in the grounded extension.
        """
        sum_cols = vsaf.mat.sum(axis=0)
        indices = np.nonzero(sum_cols==0)
        ext = np.take(vsaf.args, indices)[0]
        return ext
    

    def get_extension_action(self, ext: list) -> int:
        """Gets the action promoted by the arguments in the grounded extension. 
        Args:
            ext (list): the grounded extension.
        Returns:
            int: index of the promoted action.
        """
        if len(ext) == 0:
            print("No extension: performing random action...")
            return random.sample(sorted(self._arg_actions.values()), 1)[0]
        return self._arg_actions[ext[0]]
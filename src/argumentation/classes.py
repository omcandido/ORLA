from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
from typing import Tuple, List

from utils import flatten

# Attacks are just tupples (attacker, attacked).
# An Attack type is created for convenience.
Attack = Tuple[str,str]

class ArgumentationFramework:
    """An Argumentation Framework (AF) à la Dung.
    """
    def __init__(self,
        args: List[str] = [],
        atts: List[Attack] = []
    ):
        """Initialise the Argumentation Framework
        Args:
            args (List[str], optional): List of arguments that comprise the AF. Defaults to [].
            atts (List[Attack], optional): List of attacks in the AF. Defaults to [].
        """
        self.args = []
        self.atts = []
        # The entire AF can be represented with a binary matrix NxN, where mat[attacker, attacked] = 1
        self.mat = np.zeros((0, 0))
        self.add_arguments(args)
        self.add_attacks(atts)

    def add_argument(self,
        argument: str
    ):
        assert argument not in self.args, "{} already in arguments".format(argument)
        self.args.append(argument)
        self.expand_mat()

    def add_arguments(self,
        arguments: List[str]
    ):
        for arg in arguments:
            self.add_argument(arg)

    def add_attacks(self,
        attacks: List[Attack]
    ):
        for att in attacks:
            self.add_attack(att)

    def add_attack(self,
        attack: Attack
    ):
        for arg in attack:
            if arg not in self.args:
                self.args.append(arg)
        self.expand_mat()
        attacker = self.args.index(attack[0])
        attacked = self.args.index(attack[1])
        if attack not in self.atts:
            self.atts.append(attack)
            self.mat[attacker, attacked] = 1

    def remove_attack(self,
        attack: Attack
    ):
        i_attacker = self.args.index(attack[0])
        i_attacked = self.args.index(attack[1])
        self.mat[i_attacker, i_attacked] = 0
        self.atts.remove(attack)

    def remove_attacks(self,
        attacks: List[Attack]
    ):
        for att in attacks:
            self.remove_attack(att)

    def remove_argument(self,
        argument: str
    ):
        i_arg = self.args.index(argument)
        self.mat = np.delete(self.mat, i_arg, 0)
        self.mat = np.delete(self.mat, i_arg, 1)
        self.args.remove(argument)

    def remove_arguments(self,
        arguments: List[str]
    ):
        for arg in arguments:
            self.remove_argument(arg)

    def expand_mat(self):
        """The matrix needs to be expanded when a new argument is added."""
        nA = len(self.args)
        nM = len(self.mat)
        diff = nA - nM
        if diff == 0:
            return
        padding = np.zeros((diff, nM))
        self.mat = np.vstack((self.mat, padding))
        padding = np.zeros((nA, diff))
        self.mat = np.hstack((self.mat, padding))


    def draw(self):
        G = nx.from_numpy_array(
            self.mat,
            create_using=nx.DiGraph)
        G = nx.relabel_nodes(G, lambda x: self.args[x])
        nx.draw(G, with_labels = True)

class ValuebasedArgumentationFramework(ArgumentationFramework):
    """The value-based argumentation framework. This is actually a preference-based AF, since the ordering is strict. 
    Args:
        ArgumentationFramework (_type_): the original AF
    """
    def __init__(
        self,
        args: List[str] = [],
        atts: List[Attack] = [],
        order : List[str] = [],
        update_on_init: bool = True
    ):
        """Initialise the VAF
        Args:
            args (List[str], optional): list of arguments that comprise the AF. Defaults to [].
            atts (List[Attack], optional): list of attacks in the AF. Defaults to [].
            order (List[str], optional): order of arguments. Defaults to [].
            update_on_init (bool, optional): whether to update the attacks of the AF on initialisation. Defaults to True.
        """
        super().__init__(args, atts)
        self.order = order
        if update_on_init:
            self.update_vaf()

    def update_vaf(self):
        """Remove the attacks from all arguments with lower preference.
        """
        for i_order, level in enumerate(self.order):
            higher = self.order[0:i_order+1]
            higher = list(flatten(higher))
            mask = np.isin(self.args, higher, invert=True)
            for arg in level:
                i_arg = self.args.index(arg)
                # print(self.args)
                # print(self.order)
                # print(i_order)
                # print(arg)
                # print(higher)
                # print(mask)
                # print()
                self.mat[:, i_arg][mask] = 0
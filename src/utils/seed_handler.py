"""Handle the reproducibility via seed setting functions."""


import os

import random

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence


class SeedHandler():
    """An object designed to handle reproducibility."""

    def __init__(self, seed: int = 0):
        """Initialize a SeedHandler instance.

        Parameters
        ----------
        seed : int, default=0
            Seed to use everywhere for reproducibility.

        Attributes
        ----------
        global_seed : int
            The global seed linked to this instance of SeedHandler.

        is_init_seed : bool
            If True, it means that the global seed of this instance has been initialized to fix
            randomness.
        """
        self.global_seed = seed  # choose an arbitrary seed if none specified
        self.is_init_seed = False  # safeguard
        self.global_rs = None

    def get_seed(self) -> int:
        """Return the global seed.

        Returns
        -------
        seed : int
            The global seed linked to this instance.
        """
        if not self.is_init_seed:  # print warning
            print("Warning: This seed has not been initiated yet!")

        seed = self.global_seed

        return seed

    def get_rs(self) -> RandomState:
        """Return the global `RandomState` instance.

        Returns
        -------
        rs : RandomState
            The global `RandomSate` instance.
        """
        if not self.is_init_seed:  # print warning
            print("Warning: This seed has not been initiated yet!")

        rs = self.global_rs

        return rs

    def set_seed(self, seed: int) -> None:
        """Set the seed for this instance.

        Parameters
        ----------
        seed : int
            A new seed to use everywhere for reproducibility. Note that this seed is not
            initialized yet.
        """
        self.global_seed = seed
        self.is_init_seed = False

    def init_seed(self) -> None:
        """Fix the seed.

        This method will change the PYTHONHASHSEED variable and the random states of *random*,
        *numpy* and *tensorflow* packages.
        """
        os.environ["PYTHONHASHSEED"] = str(self.global_seed)
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)  # pandas seed is numpy seed

        self.global_rs = RandomState(MT19937(SeedSequence(self.global_seed)))

        self.is_init_seed = True

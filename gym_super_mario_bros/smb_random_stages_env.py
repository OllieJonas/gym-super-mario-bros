"""
An OpenAI Gym Super Mario Bros. environment that randomly selects levels.

Original version took ages to load because it was initialising every single environment (even if not needed!). New
version initialises environments lazily.

Original version can be found here:
https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_random_stages_env.py
"""
from typing import Optional

import gym
import numpy as np
from .smb_env import SuperMarioBrosEnv

from itertools import chain


class SuperMarioBrosRandomStagesEnv(gym.Env):
    """A Super Mario Bros. environment that randomly selects levels."""

    # relevant meta-data about the environment
    metadata = SuperMarioBrosEnv.metadata

    # the legal range of rewards for each step
    reward_range = SuperMarioBrosEnv.reward_range

    # observation space for the environment is static across all instances
    observation_space = SuperMarioBrosEnv.observation_space

    # action space is a bitmap of button press values for the 8 NES buttons
    action_space = SuperMarioBrosEnv.action_space

    def __init__(self, rom_mode='vanilla', stages=None, render_mode: Optional[str] = None):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            stages (list): select stages at random from a specific subset

        Returns:
            None

        """
        # create a dedicated random number generator for the environment
        self.np_random = np.random.RandomState()

        # set rom_mode
        self.rom_mode = rom_mode

        # set render_mode
        self.render_mode = render_mode

        # setup the environments as a 1d arr
        self.envs = self._make_envs(stages)
        # create a placeholder for the current environment
        self.env = self.envs[0]
        # create a placeholder for the image viewer to render the screen
        self.viewer = None
        # create a placeholder for the subset of stages to choose
        self.stages = stages

    def _make_envs(self, stages):

        stage_dict = {}

        # if stages is None, then assume select from any stage - return all stages (worlds 1 to 8, stages 1 to 4)
        if not stages:
            stage_dict = {str(i): list(map(lambda x: str(x), range(1, 5))) for i in range(1, 9)}

        else:
            # convert stages from str to tuple. e.g.: ['1-1', '1-2'] -> [(1, 1), (1,2)]
            stages = list(map(lambda x: tuple(x.split("-")), stages))

            stage_dict = {}

            # convert list of tuples to dict for world: [stages]. e.g.: [(1, 1), (1, 2), (2, 3)] -> {1: [1, 2], 2: [3]}
            for k, v in stages:
                stage_dict.setdefault(k, []).append(v)

        # convert integers to envs
        stage_dict = {k: [SuperMarioBrosEnv(
            target=(k, i),
            rom_mode=self.rom_mode,
            render_mode=self.render_mode) for i in v]
                      for k, v in stage_dict.items()}

        # flatten dict to 1d arr (easier to work with)
        stages_list = [v for _, v in stage_dict.items()]
        stages_list = list(chain.from_iterable(stages_list))

        return stages_list

    @property
    def screen(self):
        """Return the screen from the underlying environment"""
        return self.env.screen

    def _seed(self, seed=None):
        """
        Set the seed for this environment's random number generator.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.

        """
        # if there is no seed, return an empty list
        if seed is None:
            return []
        # set the random number seed for the NumPy random number generator
        self.np_random.seed(seed)
        # return the list of seeds used by RNG(s) in the environment
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.

        Args:
            seed (int): an optional random number seed for the next episode
            options (dict): An optional options for resetting the environment.
                Can include the key 'stages' to override the random set of
                stages to sample from.

        Returns:
            state (np.ndarray): next frame as a result of the given action

        """
        # Seed the RNG for this environment.
        self._seed(seed)

        # choose random environment
        self.env = self.np_random.choice(self.envs)
        # reset the environment
        return self.env.reset(
            seed=seed,
            options=options,
        )

    def step(self, action):
        """
        Run one frame of the NES and return the relevant observation data.

        Args:
            action (byte): the bitmap determining which buttons to press

        Returns:
            a tuple of:
            - state (np.ndarray): next frame as a result of the given action
            - reward (float) : amount of reward returned after given action
            - done (boolean): whether the episode has ended
            - info (dict): contains auxiliary diagnostic information

        """
        return self.env.step(action)

    def close(self):
        """Close the environment."""
        # make sure the environment hasn't already been closed
        if self.env is None:
            raise ValueError('env has already been closed.')
        # iterate over each list of stages
        for stage_lists in self.envs:
            # iterate over each stage
            for stage in stage_lists:
                # close the environment
                stage.close()
        # close the environment permanently
        self.env = None
        # if there is an image viewer open, delete it
        if self.viewer is not None:
            self.viewer.close()

    def render(self):
        """
        Render the environment.

        Args:
            mode (str): the mode to render with:
            - human: render to the current display
            - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
              representing RGB values for an x-by-y pixel image

        Returns:
            a numpy array if mode is 'rgb_array', None otherwise

        """
        return SuperMarioBrosEnv.render(self)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        return self.env.get_keys_to_action()

    def get_action_meanings(self):
        """Return the list of strings describing the action space actions."""
        return self.env.get_action_meanings()


# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosRandomStagesEnv.__name__]

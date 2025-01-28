# This code is from the pyboy example for how to create a gymnasium environment

from pyboy import PyBoy

# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# TODO will need to update this to inclue ALL possible combos for metroid
# for example, to screw attack, you must press right AND jump at the same time

# Removed "start" since we don't ever want to or need to pause
# Each "action" is a list of buttons to press at once
# VERY important difference from
# A is jump, B is shoot
actions = [
        '',
        ['a'], 
        ['b'], 
        ['left'],
        ['right'],
        ['up'],
        ['down'],
        # never need to press select with anything else, unless you REALLY
        # wanted to be able to switch weapons while moving, but that shouldn't
        # be necessary
        ['select'],  

        # Move and shoot combos, very necessary
        ['left', 'b'],       
        ['right', 'b'],       
        ['up', 'b'],       
        ['up', 'right', 'b'],  # aim up, walk right
        ['up', 'left',  'b'],  # aim up, walk left

        # Spin jump, for screw attack
        ['a', 'right'],
        ['a', 'left']
]

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class MetroidEnv(gym.Env):


    def __init__(self, rom_path, emulation_speed_factor=0, debug=False, render_mode=None):
        super().__init__()
        win = 'SDL2' if render_mode is not None else 'null'
        self.pyboy = PyBoy(rom_path, window=win)

        self._fitness=0
        self._previous_fitness=0
        self.debug = debug

        # This was their sample code
        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.pyboy.set_emulation_speed(emulation_speed_factor)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        # the game_wrapper is a part of PyBoy
        self.pyboy.game_wrapper.start_game()

        self.is_render_mode_human = True if render_mode == 'human' else False


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            # This only works for single buttons!
            # self.pyboy.button(actions[action])
            for a in actions[action]:
                self.pyboy.button(a)

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        self.pyboy.tick(1, self.is_render_mode_human)

        # done = self.pyboy.game_wrapper.game_over
        done = 0

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        # Sprites on the screen
        observation=self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        # self._fitness=self.pyboy.game_wrapper.score
        self._fitness = 0

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area()
        info = {}
        return observation, info


    def render(self, mode='human'):
        if mode == 'human':
            self.is_render_mode_human = True
        else:
            self.is_render_mode_human = False

    def close(self):
        self.pyboy.stop()


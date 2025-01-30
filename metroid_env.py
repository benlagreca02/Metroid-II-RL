# This code is from the pyboy example for how to create a gymnasium environment

from pyboy import PyBoy

# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
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


# observation_space = spaces.Box(low=0, high=254, shape=(144,160), dtype=np.int8)
observation_space = spaces.Box(low=0, high=255, shape=(144,160), dtype=np.uint8)



class MetroidEnv(gym.Env):

    # emulation_speed_factor overrides the "debug" emulation speed
    def __init__(self, rom_path, emulation_speed_factor=0, debug=False, render_mode=None):
        super().__init__()

        self.metadata = {'render_modes': ['human']}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        win = 'SDL2' if render_mode is not None else 'null'
        self.pyboy = PyBoy(rom_path, window=win, debug=debug)

        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.pyboy.set_emulation_speed(emulation_speed_factor)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = observation_space

        # the game_wrapper is a part of PyBoy
        self.pyboy.game_wrapper.start_game()

        self.is_render_mode_human = True if render_mode == 'human' else False

        self.game_state_old = self._get_current_mem_state_dict()

        # we have obviously explored the current area, we're there right now!
        self.explored = [self.pyboy.game_area()]


    def _get_observation(self):
        # Used to fetch an observation, this will save work down the line when
        # converting from pixels to the more compilcated sprited method
        # Also reduces duplicate code in step() and reset()

        # 160x144 grayscale image for now, but should eventually be 16x20 tile
        # data
        
        # returns RGBA
        rgb = self.pyboy.screen.ndarray[:, :, :3]   
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray

    
    def _get_current_mem_state_dict(self):
        # GMC is global metroid count
        vals_of_interest = {
                'hp': self.pyboy.game_wrapper.current_hp,
                # May be wrong, so ignoring e-tanks for now
                # 'e_tanks': self.pyboy.game_wrapper.current_e_tanks
                'missiles': self.pyboy.game_wrapper.current_missiles,
                'missile_capacity': self.pyboy.game_wrapper.current_missile_capacity,
                'upgrades': self.pyboy.game_wrapper.current_major_upgrades,
                'gmc': self.pyboy.game_wrapper.global_metroid_count,
        }
        return vals_of_interest


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # DO THE ACTION
        if action == 0:
            pass
        else:
            # So we can press more than one button at once
            for a in actions[action]:
                self.pyboy.button(a)

        # Change 1 if you only want to "play" every few frames
        # I think...
        self.pyboy.tick(1, self.is_render_mode_human)

        # Getting tiles on the screen is a little too complicated for now
        # Due to the way the GameBoy shuffles out what tiles are currently
        # available, there are lots of difficulties with this method
        # (Tile "135" won't always be the same thing for example)
        # Tiles on the screen
        # tile_idxs = self.pyboy.game_area()

        observation = self._get_observation()

        # PyBoy Cython weirdness makes "game_over()" an int 
        done = False if self.pyboy.game_wrapper.game_over() == 0 else True

        info = {}
        truncated = False

        # fetch new mem state for reward calculation
        curr_game_state = self._get_current_mem_state_dict()

        reward = self.calculate_reward(observation, curr_game_state)

        self.game_state_old = curr_game_state
        return observation, reward, done, truncated, info

    # TODO calculate exploration reward somehow
    def _calculate_exploration_reward(self, obs):
        return 0




    def calculate_reward(self, obs, mem_state):
        # TODO do something with 'obs' (the current observation
        # and do some kind of exploration reward
        reward = self._calculate_exploration_reward(obs)

        # for HP and missiles, new is smaller -> BAD, so + weight
        # delta = new - old
        # reward = weight * delta
        delta_reward_weights= {
                'hp': 10, # losing health is really bad!
                'missiles': 10,
                'missile_capacity': 100,
                'upgrades': 10000,
                'gmc': -5000,  # if GMC decreases, delta is negative
        }

        # for the "steady state" ?? I have no idea but I think this is silly
        current_reward_weights= {
                'hp': 1, 
                'missiles': 1,
                'missile_capacity': 0,
                'upgrades': 0,
                'gmc': 0,  # if GMC decreases, delta is negative
        }

        for k,v in mem_state.items():
            delta = v - self.game_state_old[k]
            reward += delta_reward_weights[k] * delta
            reward += current_reward_weights[k] * v


        return reward



    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()

        observation = self._get_observation()


        info = {}
        return observation, info



    def render(self, mode='human'):
        if mode == 'human':
            self.is_render_mode_human = True
        else:
            self.is_render_mode_human = False

    def close(self):
        self.pyboy.stop()


# This code is from the pyboy example for how to create a gymnasium environment
import sys
from pyboy import PyBoy

# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import gymnasium as gym
from gymnasium import spaces

import pufferlib

import numpy as np
import cv2

# Removed "start" since we don't ever want to or need to pause
# Each "action" is a list of buttons to press at once. VERY important to be able
# to press more than one button at once for metroid

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


# This would be observation space if using whole screen
# observation_space = spaces.Box(low=0, high=254, shape=(144,160, 1), dtype=np.int8)

# Will eventually update this to use game area (would be much smaller)
observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8)


# How many frames to advance every action
DEFAULT_NUM_TO_TICK = 4
ROM_PATH = "MetroidII.gb"
# This was used when using SB3 and gymnasium registering
# MAX_ENV_STEPS = 400000


# class MetroidEnv(pufferlib.emulation.GymnasiumPufferEnv):
class MetroidEnv(gym.Env):

    # emulation_speed_factor overrides the "debug" emulation speed
    def __init__(self,  
            # PyBoy options
            rom_path=ROM_PATH,
            emulation_speed_factor=0,
            debug=False,
            # TODO should probably change to use "rgb_array" render mode, since
            # thats what its doing, then later implement the tile-based approach
            render_mode='rgb_array',

            # Pufferlib options
            num_to_tick=DEFAULT_NUM_TO_TICK,
            buf=None): 

        self.metadata = {'render_modes': ['human', 'rgb_array']}

        # Emulator doesn't support "half speed" unfortunately
        assert type(emulation_speed_factor) is int, "Must use integer speed factor. Pyboy doesn't support fractional speeds!"
        assert type(num_to_tick) is int, "Must use integer frame-tick! You can't tick by half a frame you goober!"
        assert render_mode in self.metadata["render_modes"], "Invalid render mode!"

        if self.render_mode == 'human':
            self.is_render_mode_human = True
        else:
            self.is_render_mode_human = False

        # PyBoy emulator configuration
        self.num_to_tick = num_to_tick
        win = 'SDL2' if render_mode == 'human' else 'null'
        self.pyboy = PyBoy(rom_path, window=win, debug=debug)
        self.pyboy.set_emulation_speed(emulation_speed_factor)
        
        # Pufferlib configuration
        # self.single_observation_space = observation_space 
        # self.single_action_space = spaces.Discrete(len(actions))
        # self.num_agents = num_agents

        # Normal (Gymnasium) config
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(len(actions))

        # the game_wrapper is a part of PyBoy
        # PyBoy auto-detects game, and determines which wrapper to use
        self.pyboy.game_wrapper.start_game()
        self.is_render_mode_human = True if render_mode == 'human' else False
        self.explored = set()
        self.explored.add(self.getAllCoordData())

    def _get_obs(self):
        # Get an observation from environment
        # Used in step, and reset, so it reduces code and makes it much cleaner
        # to do this in its own function
        
        # returns RGBA
        rgb = self.pyboy.screen.ndarray[:, :, :3]

        # reduce by 50%, less input data -> faster training
        h, w = rgb.shape[:2]
        smaller = cv2.resize(rgb, (w//2, h//2))

        # cast to grayscale
        gray = cv2.cvtColor(smaller, cv2.COLOR_RGB2GRAY)

        # To make Gymnasium happy, must be 3d with 1 val in z dim
        gray = np.reshape(gray, gray.shape + (1,))

        return gray


    def getAllCoordData(self):
        return self.getCoordinatesArea(), self.getCoordinatesPixels()


    def getCoordinatesArea(self):
        x = self.pyboy.game_wrapper.x_pos_area
        y = self.pyboy.game_wrapper.y_pos_area
        return x,y


    def getCoordinatesPixels(self):
        x = self.pyboy.game_wrapper.x_pos_pixels
        y = self.pyboy.game_wrapper.y_pos_pixels
        return x,y
    

    def _get_mem_state_dict(self):
        '''
        Get values from emulator that are relevent to the game
        As game_wrapper changes in PyBoy, this can be changed as well
        '''
        # GMC is global metroid count
        vals_of_interest = {
                'hp': self.pyboy.game_wrapper.hp,
                # May be wrong, so ignoring e-tanks for now
                # 'e_tanks': self.pyboy.game_wrapper.e_tanks
                'missiles': self.pyboy.game_wrapper.missiles,
                'missile_capacity': self.pyboy.game_wrapper.missile_capacity,
                'upgrades': self.pyboy.game_wrapper.major_upgrades,
                'gmc': self.pyboy.game_wrapper.global_metroid_count,
        }
        return vals_of_interest



    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # DO THE ACTION
        if action == 0:
            pass
        else:
            # Press the button(s) for this action
            for a in actions[action]:
                self.pyboy.button(a)

        self.pyboy.tick(self.num_to_tick, self.is_render_mode_human)

        # Cache, so we don't fetch twice in render() function
        self.obs = self._get_obs()

        # PyBoy Cython weirdness makes "game_over()" an int 
        done = False if self.pyboy.game_wrapper.game_over() == 0 else True

        info = {}
        truncated = False

        # Calculate the reward
        reward = self._calculate_reward()

        return self.obs, reward, done, truncated, info



    def _calculate_reward(self):
        '''
        Calculates the reward based on the current state of the emulator.
        Includes the calculation of exploration reward 
        '''
        missileWeight = 2
        healthWeight = 1

        mem_state = self._get_mem_state_dict()
        reward = self._calc_and_update_exploration()
        # iterate through observations, and "weights"

        # TODO rewrite missile award, agent may avoid missile tanks
        # given 3 missiles, and 30 capacity, currently have 10%
        # if a tank is picked up and 30 incresases, percent missiles drops,
        # which is interpreted as a punishment

        # reward penalized as num missles missing
        # If all missiles present, this is 0
        # If no missiles present, this is 1
        missingMissilePercent = 1 - (mem_state['missiles'] / mem_state['missile_capacity'])
        reward -= (missileWeight*missingMissilePercent)

        # TODO Implement a "percent health" in game wrapper, and use that
        # instead. HP in metroid goes from 99 -> 0, then wraps around to 99
        # again.
        missingHealth = (99 - mem_state['hp'])
        reward -= healthWeight * missingHealth

        # NO CHANCE the agent gets this far yet, so I'll implement these later.


        # TODO implement metroid killing reward
        # reward += weight * metroidsKilled (?)

        # TODO implement upgrade award.
        # give some reward for number of upgrades?
        # reward = weight * numBits(upgrades)

        return reward

    def _calc_and_update_exploration(self):

        # Reward multiplier for hitting a new coordinate
        # reward = factor * len(explored)
        exploration_reward_factor = 2

        # Reward for NOT hitting a new coordinate
        # Very small, but non-zero
        no_exploration_reward = -0.005
        
        # Pixel value is 8 bit (0-255), This factor divides pixel value to make
        # reward more sparse, and so we don't cache as many values. 
        pixel_exploration_skip = 50


        # if these coordinates are new, cache them, give reward, and move on
        pixX, pixY = self.getCoordinatesPixels()
        # pixels move very quickly, this makes it so the reward only triggers
        # after going a direction for a while
        pixX = pixX // pixel_exploration_skip
        pixY = pixY // pixel_exploration_skip

        coordData = ((pixX, pixY), self.getCoordinatesArea())

        if coordData in self.explored:
            # We've been here before
            return no_exploration_reward
        self.explored.add(coordData)
        return exploration_reward_factor * len(self.explored)


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self.obs = self._get_obs()
        info = {}
        return self.obs, info


    def render(self):
        if self.render_mode == 'human':
            # We are already showing the screen!
            pass
        elif self.render_mode == 'rgb_array':
            if self.obs == None:
                self.obs = self._get_obs()
            return self.obs

    def close(self):
        self.pyboy.stop()


    def game_area(self):
        return self.pyboy.game_area()


# REGISTER THE ENVIRONMENT WITH GYMNASIUM
# Will I still do this with pufferlib?
'''
gym.register(
        id="MetroidII", 
        entry_point=MetroidEnv,
        nondeterministic=True,    # randomness is present in the game
        max_episode_steps=MAX_ENV_STEPS,
)
'''

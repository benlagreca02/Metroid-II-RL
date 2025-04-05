# This code is from the pyboy example for how to create a gymnasium environment
import sys
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import gymnasium as gym
from gymnasium import spaces

# import pufferlib

import numpy as np
import cv2

# All of the button constants
from actions_lists import *

import os
import random

# whole screen black and white observation space example
# observation_space = spaces.Box(low=0, high=254, shape=(144,160, 1), dtype=np.int8)

# the -8 is to remove the bottom banner of the screen
SCALE_FACTOR = 2
quarter_res_screen_obs_space = spaces.Box(low=0, high=255, shape=((144-8)//SCALE_FACTOR, (160//SCALE_FACTOR), 1), dtype=np.uint8)

SCREEN_OBS = "screen"
HEALTH_OBS = "health"
MISSILE_OBS = "missile"

# From wram.asm in M2 decomp linked on Datacrystal
# 00000001 ; 01: Bombs
# 00000010 ; 02: Hi-jump
# 00000100 ; 04: Screw attack
# 00001000 ; 08: Space jump
# 00010000 ; 10: Spring ball
# 00100000 ; 20: Spider ball
# 01000000 ; 40: Varia suit
# 10000000 ; 80: Unused
MAJOR_UPGRADES_OBS = "major_upgrades"

#        0: Normal
#        1: Ice
#        2: Wave
#        3: Spazer
#        4: Plasma
#        7: Bomb beam? (see $2:52C6)
#        8: Missile
BEAM_OBS = "beam"

# TODO FIX HACK
SAVE_STATE_DICT = '../Metroid-II-RL/checkpoints'

observation_space = spaces.Dict({
    SCREEN_OBS: quarter_res_screen_obs_space,
    # TODO change to be "total health percent" so its a flaot like missiles (?)
    HEALTH_OBS: spaces.Box(low=0, high=99, dtype=np.uint8),
    MISSILE_OBS: spaces.Box(low=0, high=100, dtype=np.float32),
    # TODO may be worth changing this to multibinary?
    MAJOR_UPGRADES_OBS: spaces.Box(low=0, high=127, dtype=np.uint8), # MSB not used
    BEAM_OBS: spaces.Box(low=0, high=255, dtype=np.uint8),
})


# How many frames to advance every action
# 1 = every single frame
# Frame skipping
DEFAULT_NUM_TO_TICK = 4

# Note if using pufferlib, must be in whatever folder you're running from.
# example: if using demo.py, must put rom in foler alongside demo.py script
ROM_PATH = "MetroidII.gb"

# Used when registering with gymnasium
# episode ends forcefully after this many steps
# MAX_EPISODE_STEPS = 400000

# class MetroidEnv(pufferlib.emulation.GymnasiumPufferEnv):
class MetroidEnv(gym.Env):
    # I think this is about 2 hours of "real life playing"
    # DEFAULT_EPISODE_LENGTH = 400000

    #  Was 75_000, should decrease eval time if shorter, which should increase
    #  SPS (?)
    DEFAULT_EPISODE_LENGTH = 40_000

    # emulation_speed_factor overrides the "debug" emulation speed
    def __init__(self,  
            # PyBoy options
            rom_path=ROM_PATH,
            emulation_speed_factor=0,
            debug=False,
            render_mode='rgb_array',
            num_to_tick=DEFAULT_NUM_TO_TICK,
            # training params
            random_state_load_freq = 0,
            stale_truncate_limit=5000,  # End game after this many stale steps
            lack_of_exploration_threshold=0,  # Wait this many steps before we start punishment
            # being pretty effecient, ship to "shaft" area is 90ish 
            reset_exploration_count=200, # reset the exploration cache after this many explored coordinates

            # Pufferlib options
            buf=None): 

        # self.metadata = {'render_modes': ['human', 'rgb_array', 'tiles']}
        self.metadata = {'render_modes': ['human', 'rgb_array']}

        # Emulator doesn't support "half speed" unfortunately
        assert type(emulation_speed_factor) is int, "Must use integer speed factor. Pyboy doesn't support fractional speeds!"
        assert type(num_to_tick) is int, "Must use integer frame-tick! You can't tick by half a frame you goober!"
        assert render_mode in self.metadata["render_modes"], "Invalid render mode!"

        # PyBoy emulator configuration
        self.num_to_tick = num_to_tick
        win = 'SDL2' if render_mode == 'human' else 'null'
        self.pyboy = PyBoy(rom_path, window=win, debug=debug)
        self.pyboy.set_emulation_speed(emulation_speed_factor)

        # Normal (Gymnasium) config
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(len(ACTIONS))

        # See PyBoy for game_wrapper details
        self.pyboy.game_wrapper.start_game()
        self.is_render_mode_human = True if render_mode == 'human' else False

        # dict of buttons, and True/False for if they're being held or not
        self._currently_held = {button: False for button in BUTTONS}

        # Cache the current coordinate
        self.explored = set()
        self._calc_and_update_exploration()
        self.old_mem_state = self._get_mem_state_dict()

        # For counting how long we've been in a spot
        self.stale_exploration_count = 0

        # Should be a large number. Environment will truncate if it hasn't hit a
        # new exploration in this many 'steps'. If 0, it won't truncate
        self.stale_truncate_limit = stale_truncate_limit
        self.reset_exploration_count = reset_exploration_count
        self.lack_of_exploration_threshold = lack_of_exploration_threshold

        self.random_state_load_freq = random_state_load_freq

        # For random point env starting
        if not os.path.exists(SAVE_STATE_DICT):
            raise ModuleNotFoundError("Couldn't find checkpoints folder!")
        self.state_files = [os.path.join(SAVE_STATE_DICT, name) for name in os.listdir(SAVE_STATE_DICT)]
        print(f"Found state files: {self.state_files}")



    def _get_screen_obs(self): 
        # -8 is to remove the "bar" from the bottom of the screen
        # 3 at end to only get RGB, we don't want alpha channel
        rgb = self.pyboy.screen.ndarray[:-8, :, :3]

        h, w = rgb.shape[:2]
        # less input data -> faster training

        # MAY BE REALLY SLOW
        # smaller = cv2.resize(rgb, (w//SCREEN_FACTOR, h//SCREEN_FACTOR))

        # THIS is the way pufferlib does it, get every N pixels
        smaller = rgb[::SCALE_FACTOR, ::SCALE_FACTOR]

        # cast to grayscale, way faster than Pillow's grayscale function
        # gray = cv2.cvtColor(smaller, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(smaller, cv2.COLOR_RGB2GRAY)

        # To make Gymnasium happy, must be 3d with 1 val in z dim
        gray = np.reshape(gray, gray.shape + (1,))

        return gray
    

    def _get_obs(self):
        # Get an observation from environment
        # Used in step, and reset, so it reduces code and makes it much cleaner
        # to do this in its own function
        screen = self._get_screen_obs()

        mem_vals = self._get_mem_state_dict()

        # note this is accessing the dict made by _get_mem_state_dict, NOT the
        # actual observation space, hence *_OBS isn't used

        # TODO eventually switch health to be percent given all e-tanks
        # I'll worry about that when the agent happens to get close to the first
        # E tank. Could also give huge reward for picking one up?
        health = np.array( [np.uint8(mem_vals['hp'])] )
        missiles = np.float32(mem_vals['missiles'] / mem_vals['missile_capacity'])
        missiles = np.array([missiles])

        upgrades = np.array([mem_vals['upgrades']], dtype=np.uint8)
        beam = np.array([mem_vals['beam']], dtype=np.uint8)

        # potential code for multibinary (?)
        # upgrades = np.unpackbits(np.array([mem_vals['upgrades']], dtype=np.uint8), bitorder='little')[:7]
        # beam = np.unpackbits(np.array([mem_vals['beam']], dtype=np.uint8), bitorder='little')[:8]
        # upgrades = np.uint8(mem_vals['upgrades'])
        # beam = np.uint8(mem_vals['beam'])
        
        dict_obs = {SCREEN_OBS: screen,
            HEALTH_OBS: health,  
            MISSILE_OBS: missiles,
            MAJOR_UPGRADES_OBS: upgrades,
            BEAM_OBS: beam,
        }

        return dict_obs



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
        # GMC is Global Metroid Count
        vals_of_interest = {
                'hp': self.pyboy.game_wrapper.hp,
                # Eventually, add E tanks and do some math with them
                'missiles': self.pyboy.game_wrapper.missiles,
                'missile_capacity': self.pyboy.game_wrapper.missile_capacity,
                'upgrades': self.pyboy.game_wrapper.major_upgrades,
                'beam': self.pyboy.game_wrapper.beam,
                'gmc': self.pyboy.game_wrapper.global_metroid_count,
        }
        return vals_of_interest


    def do_action_on_emulator(self, action):
        # get buttons currently being held
        holding = [b for b in self._currently_held if self._currently_held[b]]

        # Release buttons we don't need to press anymore
        for held in holding:
            # If the new action doesn't want us to press the button anymore
            if held not in action:
                release = RELEASE_BUTTON_LOOKUP[held]
                self.pyboy.send_input(release)
                self._currently_held[held] = False

        # Press buttons we need to press now
        for button in action:
            # NOP is a list, continaing just the PASS action, which isn't a
            # button we can "press"
            if button in NOP:
                continue
            # Press the button
            self.pyboy.send_input(button)
            # mark it as "pressed"
            self._currently_held[button] = True


    def step(self, action_index):
        # Would this slow down performance? I doubt it but I really don't NEED
        # to check this
        # assert self.action_space.contains(action_index), "%r (%s) invalid" % (action_index, type(action_index))

        action = ACTIONS[action_index]

        self.do_action_on_emulator(action)

        self.pyboy.tick(self.num_to_tick, self.is_render_mode_human)

        obs = self._get_obs()

        # PyBoy Cython weirdness makes "game_over()" an int 
        done = False if self.pyboy.game_wrapper.game_over() == 0 else True

        # For time limits, I'm opting to assume that the user will wrap with a
        # gymnasium time limit
        truncated = False

        # If we're getting stale, reset exploration, hopefully agent will do
        # something wacky here, rather than avoiding every place its already
        # been (some backtracking is ok)
        should_check_exploration_reset = self.reset_exploration_count > 0
        if should_check_exploration_reset and len(self.explored) > self.reset_exploration_count:
            self.explored = set()
            # add current point to set
            # doesn't give reward, we're already here
            self._calc_and_update_exploration()

        # Stale checking
        should_check_limit = self.stale_truncate_limit > 0
        truncated = should_check_limit and self.stale_exploration_count >= self.stale_truncate_limit

        info = {}

        # Calculate the reward
        reward = self._calculate_reward()

        return obs, reward, done, truncated, info



    def _calculate_reward(self):
        '''
        Calculates the reward based on the current state of the emulator.
        Includes the calculation of exploration reward 
        '''
        # TODO could convert this to a dictionary at some point?
        missileWeight = -0.01

        # Losing health is pretty bad
        healthWeight = 0.1

        # AMAZING, so make it giant
        gmcWeight = 10

        mem_state = self._get_mem_state_dict()
        # calculate "deltas" of memory values
        deltas = dict()
        # print("OLD MEM: {self.old_mem_state}")
        # print("CUR MEM: {mem_state}")
        for k,v in mem_state.items():
            deltas[k] = v - self.old_mem_state[k]

        reward = self._calc_and_update_exploration()
        # iterate through observations, and "weights"

        # TODO Implement a "percent health" in game wrapper, and use that
        # instead. HP in metroid goes from 99 -> 0, then wraps around to 99
        # again. This needs to be double checked at some point. But agent gets
        # nowhere near getting an E tank, so I'll worry about that later.
        # missingHealth = (99 - mem_state['hp'])
        # reward -= healthWeight * missingHealth


        # small punishment for every missle shot, and reward missiles gained 
        # Could change to give bigger reward for gained? i.e. -0.05 when
        # shooting
        reward += missileWeight * deltas['missiles']

        # health lost is bad
        # Could do something where losing your first few health isn't bad,
        # but losing your last bits of health is worse (?) Could incentivise
        # defensive behaivor with low health
        reward += healthWeight * deltas['hp']

        # Good to shrink this, so its negative
        reward += gmcWeight * -deltas['gmc']

        # NO CHANCE the agent gets this far yet, so I'll implement these later.
        # TODO implement upgrade award.
        # give some reward for number of upgrades?
        # reward = weight * numBits(upgrades)

        self.old_mem_state = mem_state

        return reward



    def _calc_and_update_exploration(self):

        # Reward multiplier for hitting a new coordinate
        # reward = factor * len(explored)
        # May become oversaturated at some point...
        exploration_reward = 0.005

        # Punish this much every step after threshold
        # Arbitrary, but very very small
        lack_of_exploration_punishment = -0.25


        # Pixel value is 8 bit (0-255)
        # Reward more frequently for vertical than horizontal
        # because jumping is hard and walking is easy

        # Rewarding every pixel would give way too many rewards
        # Note: it should probably be divisible evenly by 256
        # If you pick something not divisible evenly (say 100)
        # you will get bursts of rewards (0, 100, 200). When 255 rolls over to
        # 0, the reward is given over a delta of 55 pixels if that makes sense?
        # really should be 1,2,4,8,16,32,64, or 128
        pixel_exploration_skip_x = 32
        pixel_exploration_skip_y = 16

        pixX, pixY = self.getCoordinatesPixels()

        # pixels move very quickly, this makes it so the reward only triggers
        # after going a direction for a while
        pixX = pixX // pixel_exploration_skip_x
        pixY = pixY // pixel_exploration_skip_y

        coordData = ((pixX, pixY), self.getCoordinatesArea())

        if coordData not in self.explored:
            self.explored.add(coordData)
            self.stale_exploration_count = 0
            return exploration_reward
        
        self.stale_exploration_count += 1 
        if self.lack_of_exploration_threshold == 0:
            should_punish = False
        else:
            should_punish = self.stale_exploration_count > self.lack_of_exploration_threshold 

        return lack_of_exploration_punishment if should_punish else 0


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()

        # OCCASIONALLY LOAD OTHER START POINTS
        random_num = random.random()
        if(random_num < self.random_state_load_freq):
            # Load the save state
            with open(random.choice(self.state_files), "rb") as f:
                self.pyboy.load_state(f)



        info = {}
        self.explored = set()
        self._calc_and_update_exploration()
        return self._get_obs(), info


    def render(self):
        # TODO pretty sure this isn't how "render" is supposed to work
        if self.render_mode == 'human':
            # We are already showing the screen!
            pass
        elif self.render_mode =='rgb_array':
            # Should be easy enough
            raise NotImplementedError

    def close(self):
        self.pyboy.stop()

    # Mainly for debugging
    # def game_area(self):
        # return self.pyboy.game_area()


# REGISTER THE ENVIRONMENT WITH GYMNASIUM
# I don't need to do this if using pufferlib
'''
gym.register(
        id="MetroidII", 
        entry_point=MetroidEnv,
        nondeterministic=True,    # randomness is present in the game
        max_episode_steps=MAX_EPISODE_STEPS,
)
'''

# This code is from the pyboy example for how to create a gymnasium environment
import sys
from pyboy import PyBoy
from pyboy.utils import dec_to_bcd
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

# for saving the set of exploration data
import pickle


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
# 1 = every single frame, a decision is made
# Frame skipping
DEFAULT_NUM_TO_TICK = 8

# Note if using pufferlib, must be in whatever folder you're running from.
# example: if using demo.py, must put rom in foler alongside demo.py script
ROM_PATH = "MetroidII.gb"

# Used when registering with gymnasium
# episode ends forcefully after this many steps
# MAX_EPISODE_STEPS = 400000

# class MetroidEnv(pufferlib.emulation.GymnasiumPufferEnv):
class MetroidEnv(gym.Env):
    # I think this is about 2 hours of "real life playing"
    # DEFAULT_EPISODE_LENGTH = 400_000

    DEFAULT_EPISODE_LENGTH = 30_000

    # For doing checkpoint based training
    # CHECKPOINT_DIR = '../Metroid-II-RL/checkpoints/'
    CHECKPOINT_DIR = './checkpoints/'

    # ==== WEIGHTS ====
    # need to incorporate weight
    missileWeight = 0.025

    # Health is pretty important
    healthWeight = 0.1

    # AMAZING, so make it giant, this will break things if this really works
    # But I'll also see a giant spike
    gmcWeight = 10

    # Reward multiplier for hitting a new coordinate
    # reward = factor * len(explored)
    # May become oversaturated at some point...
    exploration_reward = 0.05

    # Give a decent reward when you hit a checkpoint
    progress_increase_reward = 1

    # NOT REALLY USED ANYMORE
    # Punish this much every step after threshold
    # Arbitrary, but very very small
    lack_of_exploration_punishment = -0.25


    # emulation_speed_factor overrides the "debug" emulation speed
    def __init__(self,  
            # PyBoy options
            rom_path=ROM_PATH,
            emulation_speed_factor=0,
            debug=False,
            render_mode='rgb_array',
            num_to_tick=DEFAULT_NUM_TO_TICK,
            # training params
            stale_truncate_limit=3000,  # End game after this many stale steps
            lack_of_exploration_threshold=0,  # Wait this many steps before we start punishment
            reset_exploration_count=0, # reset the exploration cache after this many explored coordinates
            invincibility=False,

            progress_checkpoints=False,
            progress_rewards=False,

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

        # Needs to be set before get mem state dict called
        self.invincibility = invincibility

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


        # flags for keeping track of progress
        # increments as we make progress through the game
        # 1: made it into the cave
        # 2: made it past the enemies
        # 3: made it to the corner of doom
        # By setting to -1, checks never happen
        self.progress = 0 if progress_checkpoints else -1
        self.progress_rewards = progress_rewards

        # RESET ENV to load 0th checkpoint, with the left portion of the map
        # "greyed out" in the exploration buffer
        self.reset()

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

        if self.invincibility:
            # TODO FIXME HACK, should do this though the game wrapper, but I just want
            # to get this going fast
            self.pyboy.memory[0xD051] = dec_to_bcd(99)
            vals_of_interest['hp'] = 99
            # TODO FIXME when missile tanks get involved, this will be different
            self.pyboy.memory[0xD053] = dec_to_bcd(30)
            vals_of_interest['missiles'] = 30


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

        # update progress (very rough)
        (ax, ay), (px, py) = self.getAllCoordData()

        # axe: area x equal (equal, to differentiate from ax)
        # aye: area y equal
        # pxl: Pixel X low
        # pxh: Pixel X high
        # pyl: Pixel Y low
        # pyh: Pixel Y high
        BOXES = [
                # ax, ay, pxl, pxh, pyl, pyh
                (10, 6, 77, 163, 10, 84),  # 1_shaft, the first shaft
                (12, 6, 53, 180, 60, 132), # 2_shaft2, the second downshaft
                (3, 0, 95, 155, 10, 40),   # 3_fork, after the third shaft
                (1, 5, 170, 231, 55, 135), # 4_left, before big "s" downwards
                (5, 1, 180, 220, 90, 135), # 5_left2, almost to boss
                (1, 1, 180, 245, 90, 135), # 6_metroid
            ]

        # based on current progress, check if we are in the NEXT "box" 
        axe, aye, pxl, pxh, pyl, pyh = BOXES[self.progress]

        if ax == axe and ay == aye and pxl < px < pxh and pyl <= py <= pyh:
            self.progress += 1
            if self.progress_reward:
                reward += self.progress_increase_reward
            print("Hit checkpoint {self.progress}!")

        

        return obs, reward, done, truncated, info



    def _calculate_reward(self):
        '''
        Calculates the reward based on the current state of the emulator.
        Includes the calculation of exploration reward 
        '''

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
        reward += self.missileWeight * deltas['missiles']

        # health lost is bad
        # Could do something where losing your first few health isn't bad,
        # but losing your last bits of health is worse (?) Could incentivise
        # defensive behaivor with low health
        reward += self.healthWeight * deltas['hp']

        # Good to shrink this, so its negative
        reward += self.gmcWeight * -deltas['gmc']

        # NO CHANCE the agent gets this far yet, so I'll implement these later.
        # TODO implement upgrade award.
        # give some reward for number of upgrades?
        # reward = weight * numBits(upgrades)

        self.old_mem_state = mem_state

        return reward

    def load_checkpoint(self, base_filename):
        state_filename = base_filename + '.state'
        set_filename = base_filename + '.set'
        # Load the 
        with open(state_filename, "rb") as f:
            print(f"loading file @ {state_filename}\tf:{f}")
            self.pyboy.load_state(f)

        with open(set_filename, 'rb') as file:
            self.explored = pickle.load(file)


    # really only called by view script
    def save_checkpoint(self, base_filename):
        state_filename = base_filename + '.state'
        set_filename = base_filename + '.set'
        with open(state_filename, "wb") as f:
            self.pyboy.save_state(f)

        # dump explored 
        with open(set_filename, 'wb') as file:
            pickle.dump(self.explored, file)


    def _calc_and_update_exploration(self):



        # Pixel value is 8 bit (0-255)
        # Reward more frequently for vertical than horizontal
        # because jumping is hard and walking is easy

        # Rewarding every pixel would give way too many rewards
        # Note: it should probably be divisible evenly by 256
        # If you pick something not divisible evenly (say 100)
        # you will get bursts of rewards (0, 100, 200). When 255 rolls over to
        # 0, the reward is given over a delta of 55 pixels if that makes sense?
        # really should be 1,2,4,8,16,32,64, or 128
        pixel_exploration_skip_x = 64
        pixel_exploration_skip_y = 64

        pixX, pixY = self.getCoordinatesPixels()

        # pixels move very quickly, this makes it so the reward only triggers
        # after going a direction for a while
        pixX = pixX // pixel_exploration_skip_x
        pixY = pixY // pixel_exploration_skip_y

        coordData = ((pixX, pixY), self.getCoordinatesArea())

        if coordData not in self.explored:
            self.explored.add(coordData)
            self.stale_exploration_count = 0
            return self.exploration_reward
        
        self.stale_exploration_count += 1 
        if self.lack_of_exploration_threshold == 0:
            should_punish = False
        else:
            should_punish = self.stale_exploration_count > self.lack_of_exploration_threshold 

        return self.lack_of_exploration_punishment if should_punish else 0


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()

        info = {}
        self.explored = set()
        self._calc_and_update_exploration()

        # TODO clean to not use so many magic constants
        if self.progress <= 0:
            # Even if we aren't doing checkpointing, start from here.
            # All it is, is a filled in exploration set so left by ship is
            # worthless
            self.load_checkpoint(self.CHECKPOINT_DIR + "0_start")
        elif self.progress == 1:
            self.load_checkpoint(self.CHECKPOINT_DIR + "1_shaft")
        elif self.progress == 2:
            self.load_checkpoint(self.CHECKPOINT_DIR + "2_shaft2")
        elif self.progress == 3:
            self.load_checkpoint(self.CHECKPOINT_DIR + "3_fork")
        elif self.progress == 4:
            self.load_checkpoint(self.CHECKPOINT_DIR + "4_left")
        elif self.progress == 5:
            self.load_checkpoint(self.CHECKPOINT_DIR + "5_left2")
        elif self.progress == 6:
            self.load_checkpoint(self.CHECKPOINT_DIR + "6_metroid")

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

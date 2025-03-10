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

# Removed "start" since we don't ever want to or need to pause
# Each "action" is a list of buttons to press at once. VERY important to be able
# to press more than one button at once for metroid

# A is jump, B is shoot

# Thanks to capnspacehook for action space
NOP = [WindowEvent.PASS]
SHOOT = [WindowEvent.PRESS_BUTTON_A]
JUMP = [WindowEvent.PRESS_BUTTON_B]

UP = [WindowEvent.PRESS_ARROW_UP]
DOWN = [WindowEvent.PRESS_ARROW_DOWN]
LEFT = [WindowEvent.PRESS_ARROW_LEFT]
RIGHT = [WindowEvent.PRESS_ARROW_RIGHT]

SWITCH = [WindowEvent.PRESS_BUTTON_SELECT]

# "Screw attack", jump and move to the side
JUMP_LEFT = [*JUMP, *LEFT]
JUMP_RIGHT = [*JUMP, *RIGHT]

# Run and shoot
SHOOT_LEFT = [*SHOOT, *LEFT]
SHOOT_RIGHT = [*SHOOT, *RIGHT]

# shoot upwards, and move while shooting upwards
SHOOT_UP = [*SHOOT, *UP]
SHOOT_UP_RIGHT = [*SHOOT, *UP, *RIGHT]
SHOOT_UP_LEFT = [*SHOOT, *UP, *LEFT]

#  VERY necessary for getting through vertical shafts where tiles need to be
#  shot
SHOOT_DOWN = [*SHOOT, *DOWN]

ACTIONS = [
        NOP,
        SHOOT,
        JUMP,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        SWITCH,
        JUMP_LEFT,
        JUMP_RIGHT,
        SHOOT_LEFT,
        SHOOT_RIGHT,
        SHOOT_UP,
        SHOOT_UP_RIGHT,
        SHOOT_UP_LEFT,
        SHOOT_DOWN
]

# All possible buttons "on hardware"
BUTTONS = [
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_SELECT
]

RELEASE_BUTTONS = [
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_SELECT
]

# Given a button press, get the "release" version of it
# ex: release_a = RELEASE_BUTTON_LOOKUP[press_a]
RELEASE_BUTTON_LOOKUP = {button: r_button for button, r_button in zip(BUTTONS, RELEASE_BUTTONS)}


# Each tile is 16 bytes: 8x8 with 2 bits/pixel
# Could probably remove this code eventually, but I may re-explore it eventually
TILE_NUM_BYTES = 16
DIGEST_SIZE_BYTES = 2
DIGEST_DTYPE = np.uint16
# tile based
# observation_space = spaces.Box(low=0, high=65535, shape=(17,20,1), dtype=DIGEST_DTYPE)

# TODO add health, and missles to the observation space
# Will probably be a tuple of Box and a few values

# whole screen black and white
# observation_space = spaces.Box(low=0, high=254, shape=(144,160, 1), dtype=np.int8)

# Was 2, divides screen resolution down, less detail but faster performance
SCREEN_FACTOR = 4
# the -8 is to remove the bottom banner of the screen
observation_space = spaces.Box(low=0, high=255, shape=((144-8)//SCREEN_FACTOR, (160)//SCREEN_FACTOR, 1), dtype=np.uint8)

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
    DEFAULT_EPISODE_LENGTH = 400000

    # emulation_speed_factor overrides the "debug" emulation speed
    def __init__(self,  
            # PyBoy options
            rom_path=ROM_PATH,
            emulation_speed_factor=0,
            debug=False,
            render_mode='rgb_array',

            # Pufferlib options
            num_to_tick=DEFAULT_NUM_TO_TICK,
            buf=None): 

        # self.metadata = {'render_modes': ['human', 'rgb_array', 'tiles']}
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
        self.action_space = spaces.Discrete(len(ACTIONS))

        # the game_wrapper is a part of PyBoy
        # PyBoy auto-detects game, and determines which wrapper to use
        self.pyboy.game_wrapper.start_game()
        self.is_render_mode_human = True if render_mode == 'human' else False

        self.explored = set()
        # dict of buttons, and True/False for if they're being held or not
        self._currently_held = {button: False for button in BUTTONS}

        self._calc_and_update_exploration()

    def _get_obs(self):
        # Get an observation from environment
        # Used in step, and reset, so it reduces code and makes it much cleaner
        # to do this in its own function
        
        # returns RGBA, we don't need "A" channel
        # -8 is to remove the "bar" from the bottom of the screen
        rgb = self.pyboy.screen.ndarray[:-8, :, :3]

        h, w = rgb.shape[:2]
        # less input data -> faster training
        smaller = cv2.resize(rgb, (w//SCREEN_FACTOR, h//SCREEN_FACTOR))

        # cast to grayscale
        gray = cv2.cvtColor(smaller, cv2.COLOR_RGB2GRAY)

        # To make Gymnasium happy, must be 3d with 1 val in z dim
        gray = np.reshape(gray, gray.shape + (1,))

        return gray

    # May eventually use this
    def _get_obs_tiles(self):
        # Get an observation from environment
        # Used in step, and reset
        
        # Get the ID of all the tiles
        # 17x20 IDs which can be used to get VRAM addresses
        tile_ids = self.pyboy.game_area()
         
        # 17x20x1 !!! must be x1!!!!!
        digest_array = np.zeros((*tile_ids.shape, 1), dtype=DIGEST_DTYPE)

        # FOR EACH TILE ON THE SCREEN
        for i in range(tile_ids.shape[0]):
            for j in range(tile_ids.shape[1]):
                # What address in VRAM is the tile stored at
                vram_addr = self.pyboy.get_tile(tile_ids[i,j]).data_address

                # Read the 16 bytes of data for the tile
                tile_byte_arr = self.pyboy.memory[vram_addr:vram_addr+16]
                
                # Each 8x8 tile is 2 bits per pixel (four possible color
                # selectinos) That gives 128 bits per tile

                # 128 bits per tile -> 16 bytes

                # Crush the 16 byte value into a 2 byte value using XORs
                # i.e. crushing 128 bits to 16 bits
                
                # 16 bytes per tile
                for d in range(TILE_NUM_BYTES//DIGEST_SIZE_BYTES):
                    curr = d*DIGEST_SIZE_BYTES
                    # convert from python ints to array of np.uint8's
                    np_byte_arr = np.array(tile_byte_arr[curr:curr+DIGEST_SIZE_BYTES], dtype=np.uint8)

                    digest_array[i,j] ^= np_byte_arr.view(DIGEST_DTYPE)
                    # digest_array[i,j] ^= np.frombuffer(np_byte_arr, dtype=DIGEST_DTYPE)

        return digest_array


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



    def step(self, action_index):
        assert self.action_space.contains(action_index), "%r (%s) invalid" % (action_index, type(action_index))

        action = ACTIONS[action_index]

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


        self.pyboy.tick(self.num_to_tick, self.is_render_mode_human)

        # Cache, so we don't fetch twice in render() function
        self.obs = self._get_obs()

        # PyBoy Cython weirdness makes "game_over()" an int 
        done = False if self.pyboy.game_wrapper.game_over() == 0 else True

        # For time limits, I'm opting to assume that the user will wrap with a
        # gymnasium time limit
        truncated = False

        info = {}

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
        # which is interpreted as a punishment.
        # Agent gets nowhere near a missle tank yet, so I'm not worrying about
        # it yet.

        # reward penalized as num missles missing
        # If all missiles present, this is 0
        # If no missiles present, this is 1
        missingMissilePercent = 1 - (mem_state['missiles'] / mem_state['missile_capacity'])
        reward -= (missileWeight*missingMissilePercent)

        # TODO Implement a "percent health" in game wrapper, and use that
        # instead. HP in metroid goes from 99 -> 0, then wraps around to 99
        # again. This needs to be double checked at some point. But agent gets
        # nowhere near getting an E tank, so I'll worry about that later.
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
        # May become oversaturated at some point...
        exploration_reward_factor = 4

        # Reward for NOT hitting a new coordinate
        # Very small, but non-zero
        # no_exploration_reward = -0.005
        no_exploration_reward = 0.0
        
        # Pixel value is 8 bit (0-255)
        # Reward more frequently for vertical than horizontal
        # jumping is hard!  walking is easy
        # Rewarding every pixel would give way too many rewards
        pixel_exploration_skip_x = 65
        pixel_exploration_skip_y = 30

        # if these coordinates are new, cache them, give reward, and move on
        pixX, pixY = self.getCoordinatesPixels()
        # pixels move very quickly, this makes it so the reward only triggers
        # after going a direction for a while
        pixX = pixX // pixel_exploration_skip_x
        pixY = pixY // pixel_exploration_skip_y

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

        self.explored = set()
        self._calc_and_update_exploration()
        return self.obs, info


    def render(self):
        # TODO pretty sure this isn't how "render" is supposed to work
        if self.render_mode == 'human':
            # We are already showing the screen!
            pass
        elif self.render_mode =='rgb_array':
            # Should be easy enough
            raise NotImplementedError
        '''
        elif self.render_mode == 'tiles':
            if self.obs == None:
                self.obs = self._get_obs()
            return self.obs
        '''

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

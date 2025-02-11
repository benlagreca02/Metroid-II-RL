# This code is from the pyboy example for how to create a gymnasium environment
import sys
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


# This would be observation space if using whole screen
# observation_space = spaces.Box(low=0, high=254, shape=(144,160, 1), dtype=np.int8)
observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8)


# BY DEFAULT USES THESE PARAMETERS 
DEFAULT_NUM_TO_TICK = 4
ROM_PATH = "MetroidII.gb"
# emulation_speed_factor=0 (emulate as fast as you can)
# render_mode=None (Don't show the GUI for the game

# When registering, usees
# max_episode_steps=100000


class MetroidEnv(gym.Env):

    # Reward for hitting a new coordinate
    exploration_reward_factor = 2

    # Reward for NOT hitting a new coordinate
    # Very small, but non-zero
    no_exploration_reward = -0.005
    
    # all pixels vals are integer divided by this value to make the reward
    # slightly more sparse, and decrese the amount of unique pixel values we
    # cache. This may become an issue in the future, with thosuands of
    # coordinates cached, checking "if we've been here before" shouldn't get
    # more expensive becuase I'm using a set (Set has 0(1) lookup time)
    # Pixels are from 0-255
    pixel_exploration_skip = 51

    # for HP and missiles, new is smaller -> BAD, so + weight
    # delta = new - old
    # reward = weight * delta

    # emulation_speed_factor overrides the "debug" emulation speed
    def __init__(self, rom_path=ROM_PATH, emulation_speed_factor=0, debug=False,
            render_mode=None, num_to_tick=DEFAULT_NUM_TO_TICK):
        super().__init__()

        # Emulator doesn't support "half speed" unfortunately
        assert type(num_to_tick) is int
        self.num_to_tick = num_to_tick

        self.metadata = {'render_modes': ['human']}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        win = 'SDL2' if render_mode is not None else 'null'
        self.pyboy = PyBoy(rom_path, window=win, debug=debug)

        assert type(emulation_speed_factor) is int
        self.pyboy.set_emulation_speed(emulation_speed_factor)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = observation_space

        # the game_wrapper is a part of PyBoy
        self.pyboy.game_wrapper.start_game()

        self.is_render_mode_human = True if render_mode == 'human' else False

        # self.game_state_old = self._get_mem_state_dict()

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
            # So we can press more than one button at once
            for a in actions[action]:
                self.pyboy.button(a)

        self.pyboy.tick(self.num_to_tick, self.is_render_mode_human)

        # Getting tiles on the screen is a little too complicated for now
        # Due to the way the GameBoy shuffles out what tiles are currently
        # available, there are lots of difficulties with this method
        # (Tile "135" won't always be the same thing for example)
        # Tiles on the screen
        # tile_idxs = self.pyboy.game_area()

        observation = self._get_obs()

        # PyBoy Cython weirdness makes "game_over()" an int 
        done = False if self.pyboy.game_wrapper.game_over() == 0 else True

        info = {}
        truncated = False

        # fetch new mem state for reward calculation
        curr_game_state = self._get_mem_state_dict() 
        reward = self._calculate_reward(curr_game_state)

        self.game_state_old = curr_game_state
        return observation, reward, done, truncated, info

    # Includes exploration reward caluclation ,which will fetch the current
    # coordinates and check if we've been there before
    def _calculate_reward(self, mem_state):
        reward = self._calc_and_update_exploration()
        # iterate through observations, and "weights"

        # TODO I could break these into smaller functions
        missileWeight = 2
        healthWeight = 1

        # reward penalized as num missles missing
        # If all missiles present, this is 0
        # If no missiles present, this is 1
        missingMissilePercent = 1 - (mem_state['missiles'] / mem_state['missile_capacity'])
        reward -= (missileWeight*missingMissilePercent)

        # TODO fix this to use real max health, this is a temp hack fix just so
        # I can train something tonight
        missingHealth = (99 - mem_state['hp'])
        reward -= healthWeight*missingHealth

        # TODO implement metroid killing and upgrade reward
        # There's no chance the agent gets that far as is though, so I'm not
        # worried about that yet

        return reward

    def _calc_and_update_exploration(self):

        # Could make this a dict and do some kind of time thing
        # If we were JUST here, its not that great, but if its been a while
        # since we've been somewhere, going back is somewhat beneficial in case
        # we ahve a new ability.
        # I.E. agent can have little reward for exploring somewhere "stale"

        # if these coordinates are new, cache them, give reward, and move on
        pixX, pixY = self.getCoordinatesPixels()
        # pixels move very quickly, this makes it so the reward only triggers
        # after going a direction for a while
        pixX = pixX // self.pixel_exploration_skip
        pixY = pixY // self.pixel_exploration_skip

        coordData = ((pixX, pixY), self.getCoordinatesArea())

        if coordData in self.explored:
            # We've been here before
            return self.no_exploration_reward
        self.explored.add(coordData)
        return self.exploration_reward_factor * len(self.explored)


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        observation = self._get_obs()
        info = {}
        return observation, info


    def render(self, mode='human'):
        if mode == 'human':
            self.is_render_mode_human = True
        else:
            self.is_render_mode_human = False

    def close(self):
        self.pyboy.stop()


    def game_area(self):
        return self.pyboy.game_area()


# REGISTER THE ENVIRONMENT WITH GYMNASIUM

# From my math, 200_000 is about a hour in real time
# This was done with a 2 timestep

gym.register(
        id="MetroidII", 
        entry_point=MetroidEnv,
        nondeterministic=True,    # randomness is present in the game
        max_episode_steps=100_000,
)


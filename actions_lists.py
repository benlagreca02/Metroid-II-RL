from pyboy.utils import WindowEvent
# Removed "start" since we don't ever want to or need to pause
# Each "action" is a list of buttons to press at once. VERY important to be able
# to press more than one button at once for metroid

# A is jump, B is shoot, select switches between missile and 

NOP = [WindowEvent.PASS]
SHOOT = [WindowEvent.PRESS_BUTTON_A]
JUMP = [WindowEvent.PRESS_BUTTON_B]

UP = [WindowEvent.PRESS_ARROW_UP]
DOWN = [WindowEvent.PRESS_ARROW_DOWN]
LEFT = [WindowEvent.PRESS_ARROW_LEFT]
RIGHT = [WindowEvent.PRESS_ARROW_RIGHT]

SWITCH = [WindowEvent.PRESS_BUTTON_SELECT]

SHOOT_UP = [*SHOOT, *UP]
SHOOT_DOWN = [*SHOOT, *DOWN]
SHOOT_LEFT = [*SHOOT, *LEFT]
SHOOT_RIGHT = [*SHOOT, *RIGHT]

SHOOT_UP_LEFT = [*SHOOT, *UP, *LEFT]
SHOOT_UP_RIGHT = [*SHOOT, *UP, *RIGHT]

JUMP_UP = [*JUMP, *UP]
JUMP_DOWN = [*JUMP, *DOWN]
JUMP_LEFT = [*JUMP, *LEFT]
JUMP_RIGHT = [*JUMP, *RIGHT]

JUMP_SHOOT_UP = [*SHOOT, *JUMP, *UP]
JUMP_SHOOT_DOWN = [*SHOOT, *JUMP, *DOWN]
JUMP_SHOOT_LEFT = [*SHOOT, *JUMP, *LEFT]
JUMP_SHOOT_RIGHT = [*SHOOT, *JUMP, *RIGHT]

ACTIONS = [
        NOP,
        SHOOT,
        JUMP,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        SWITCH,
        SHOOT_UP,
        SHOOT_DOWN,
        SHOOT_LEFT,
        SHOOT_RIGHT,
        SHOOT_UP_LEFT,
        SHOOT_UP_RIGHT,
        JUMP_UP,
        JUMP_DOWN,
        JUMP_LEFT,
        JUMP_RIGHT,
        JUMP_SHOOT_UP
        JUMP_SHOOT_DOWN,
        JUMP_SHOOT_LEFT
        JUMP_SHOOT_RIGHT,
]

# All possible buttons "on hardware"
# These two lists are for determining how we want to press and release buttons 
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

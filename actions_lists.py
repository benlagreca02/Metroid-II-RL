from pyboy.utils import WindowEvent
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

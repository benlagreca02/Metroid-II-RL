# Metroid-II-RL

Reinforcement learning plays Metroid II. 'Nuff said.

## Current state

PyBoy provides "game wrappers" for various games to make AI work easier. I am
working on implementing one for Metroid II and will eventually get my code
pulled into the project.

Currently, a pixel-based observation approach is going to be used. Due to the
environment, a tile-based approach could be used, and that may much faster,
however it has a lot of its own issues adn those may not be explored here.

`main.py` is going to mostly be used for testing purposes.

`train.py` is the script used to start a training run.

`view.py` is the script used to view a training result.

The `view.py` script has a lot of great args for testing modifications currently

## Tangent about timing info

Through testing and some math, roughly 216,000 iterations correspond to an hour
of "real life" game time. This was calculated by measuring 1000 iteration's
time. The output of the script was as follows. This assumes 1000 iterations

```
Human Time: 16.650566339492798
Machine time: 0.2655789852142334
Machine is 62.69534589139785 times faster
time per step HUMAN: 0.016650566339492797
time per step FAST: 0.0002655789852142334
One hour of human gameplay = 216208.8620049702
```
the math for this was simply
`time / (measured_time / iterations) --> 3600 / (16.65/1000)`


## TODO

I'm focusing much more on the model, as well as the "real AI" portion of the
project. I'm currently tweaking and tuning my reward function a ton.

Currently, the agent makes it pretty reliably out of the main portion of the
level, however it runs straight through enemies, and has learned to completely
not shoot at all.

At this point, a pull request has been made for PyBoy, and many changes need to
be made to that repository, but before I do that, I want to focus more on the
training and AI portion of things.

### Agent Milestones
- [x] explore starting area
- [x] Stop shooting randomly and "spazzing out"
- [x] Get out of starting area relatively quickly
- [ ] Avoid enemies just outside starting area
- [ ] Kill enemies just outside starting area
- [ ] Drop down through first major shaft (requires downward jump shooting) 
- [ ] Find first Metroid
- [ ] Kill first Metroid

### Model
- [ ] get wandb integration working
- [ ] Tweak exploration to include a "staleness" i.e. punish not moving, or not getting an exploration reward
- [ ] Potentially implement Random Network Distillation
- [ ] figure out a way to incentivise killing (sprite based?)
- [x] Add missle count, "gun state", and health to observation space
- [x] Re-write reward functions to punish when _losing_ health and missles, rather than when below max 
- [x] Add frame stacking to observation space (Added LSTM instead)
- [x] improve  observations of environments to be tiles, rather than pixels
- [x] Change action space so agent can hold buttons (SML code has simmilar)
- [x] Define a baseline test reward function
- [x] make observations of environments pixels of screen
- [x] Write simple exploration function using game coordinate hashing
- [x] Do some kind of extremely bare-bones training to just explore
- [ ] Change to Pufferlib native environment, rather than gym wrapper (?)


### Custom Environment
Environment is pretty much set. Its plenty good enough to start training
- [x] Verify ROM integrity (the rom works)
- [x] Take random actions in the environment (veryify pyboy interface working)
- [x] Implement a `game_wrapper` for Metroid II (makes AI stuff easier)
    - [ ] Potentially use RAM mappings to calculate more useful info (particularly health)
    - [ ] Improve `game_over` to check health, may be slightly faster than using "GAME OVER" screen
    - [x] Make a pull request for PyBoy to merge my code in
        - [ ] Fix code to comply with PyBoy coding standards (see PR in PyBoy)
    - [x] Implement the `start_game` function to skip though the menu
    - [x] Implement `game_over` function to check if agent is dead
    - [x] Integrate and verify the RAM mappings 
- [x] Determine and implement all possible button combos for "actions" (may
  need minor improvements)

### Misc
- [ ] Containerize the program to make running on other machines easy (either
  with conda, or docker potentially, or a setup script)
- [ ] Overall, clean up the code a LOT, including but not limited to complete
  restructuring for the sake of pufferlib. (I'm using three different repos!
  my Pufferlib fork, PyBoy fork, and my gymnasium environment.

## History
A majority of this projects initial work (January - February) was dedicated to
many many iterations of the PyBoy wrapper, action space, and overall
implementation, wrappers, libraries oh my! At this point, I'm switching gears to
90/10 doing AI and "backend" development. Up to this point, most of the
training was "proof of concept". And many bugs and kinks were worked out along
the way before switching to a more AI focused development.

The first real model was using simply the observation of the screen this worked
great for getting past the  initial

### Comedy of errors
A list of extremely silly mistakes
- agent couldn't actually jump! My original implementation of the button presses
  was simply single frame tapping the button. Needed to re-implement the
  emulator interfacing to check next action, and release buttons not in the next
  action.
- By shrinking the screen by four, after CNN layers, the screen was too small,
  and got reduced to one single feature! This is sub-optimal for AI.
- Gymnasium can't handle (width, height) for images even if its black and white.
  It needs to be (width, height, 1) even though its grayscale.


# Metroid-II-RL

This is my attempt at having reinforcemnt learning play Metroid II

Many other simmilar projects are out there, particularly with pokemon red

I am implementing a "game wrapper" for Metroid II in the PyBoy emulator, which
is being implemented in a fork of the emulator, but once I get my edits pulled
into main, hopefully you can just use pip to install pyboy and get my metroid
code.

For now I'm not going to stress too much about documentation.


## Current state

Currently, a pixel-based observation approach is going to be used. A tile-based
approach would be much more effecinet, and train much faster, however currently

Some incredibly simple training has been done, but its terrible unsurprisingly.

The two primary tasks of the project are optimizing the training, and the
training itsself. These will be worked on in parallel, so training runs can be
done 

## TODO

Lots to do! The first big milestone will be implementing a game wrapper for
Metroid II in PyBoy and potentially making a pull request with my new custom
wrapper.

### Custom Environment
It's not 100% complete, but its good enough to move on with getting my first
models trained
- [x] Verify ROM integrity (the rom works)
- [x] Take random actions in the environment (veryify pyboy interface working)
- [x] Implement a `game_wrapper` for Metroid II (makes AI stuff easier)
    - [ ] Potentially use RAM mappings to calculate more useful info (particularly health)
    - [ ] Improve `game_over` to check health, may be slightly faster than using "GAME OVER" screen
    - [ ] Make a pull request for PyBoy to merge my code in
    - [x] Implement the `start_game` function to skip though the menu
    - [x] Implement `game_over` function to check if agent is dead
    - [x] Integrate and verify the RAM mappings 
- [x] Determine and implement all possible button combos for "actions" (may
  need minor improvements)


### Model Training
Some simple training has been done, but the reward functions need to heavily be
tweaked
- [x] Define a baseline test reward function
- [x] make observations of environments pixels of screen
- [x] Write simple exploration function using game coordinate hashing
- [x] Do some kind of extremely bare-bones training to just explore

#### Milestones
- [ ] Stop shooting randomly and "spazzing out"
- [ ] Get out of starting area relatively quickly
- [ ] Avoid enemies/kill them
- [ ] Drop down through first major shaft (requires downward jump shooting) 
- [ ] Kill first metroid


### Optimization
- [ ] improve  observations of environments to be tiles (will train faster,
  but large undertaking)
- [ ] Rewrite to use Pufferlib, instead of vanilla SB3

### Misc
- [ ] Containerize the program to make running on other machines easy (either
  with conda, or docker)

# Metroid-II-RL

Reinforcement learning plays Metroid II. 'Nuff said.

## Current state

PyBoy provides "game wrappers" for various games to make AI work easier. I am
working on implementing one for Metroid II and will eventually get my code
pulled into the project.

Currently, a pixel-based observation approach is going to be used. A tile-based
approach would be much more effecinet, and train much faster, however currently

Some incredibly simple training has been done, but its terrible unsurprisingly.
This was mostly done to prove that the code environment worked, and some
learning could be done.

`main.py` is going to mostly be used for testing purposes.

`train.py` is the script used to start a training run.

`view.py` is the script used to view a training result.

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

At this point, a pull request has been made for PyBoy, and many changes need to
be made to that repository, but before I do that, I want to focus more on the
training and AI portion of things. I'll be modifying this code to use Pufferlib
and CleanRL soon. I'll also be heavily modifying the environment to give better
observations like missiles, health, etc.

### Agent Milestones
- [ ] Stop shooting randomly and "spazzing out"
- [ ] Get out of starting area relatively quickly
- [ ] Avoid/kill enemies in starting area
- [ ] Drop down through first major shaft (requires downward jump shooting) 
- [ ] Find first Metroid
- [ ] Kill first Metroid

### Model Training
- [ ] Change to CleanRL/Pufferlib approach
- [ ] improve  observations of environments to be tiles (will train faster, but large undertaking)
- [x] Define a baseline test reward function
- [x] make observations of environments pixels of screen
- [x] Write simple exploration function using game coordinate hashing
- [x] Do some kind of extremely bare-bones training to just explore


### Custom Environment
Environment is pretty much set. Its plenty good enough to start training
- [x] Verify ROM integrity (the rom works)
- [x] Take random actions in the environment (veryify pyboy interface working)
- [x] Implement a `game_wrapper` for Metroid II (makes AI stuff easier)
    - [ ] Potentially use RAM mappings to calculate more useful info (particularly health)
    - [ ] Improve `game_over` to check health, may be slightly faster than using "GAME OVER" screen
    - [x] Make a pull request for PyBoy to merge my code in
        - [ ] Fix code to comply with PyBoy coding standards
    - [x] Implement the `start_game` function to skip though the menu
    - [x] Implement `game_over` function to check if agent is dead
    - [x] Integrate and verify the RAM mappings 
- [x] Determine and implement all possible button combos for "actions" (may
  need minor improvements)

### Misc
- [ ] Containerize the program to make running on other machines easy (either
  with conda, or docker)

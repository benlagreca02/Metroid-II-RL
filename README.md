# Metroid-II-RL

This is my attempt at having reinforcemnt learning algo play Metroid II

I am implementing a "game wrapper" for Metroid II in the PyBoy emulator.

For now, I'm using a custom fork of the emulator, but once I get my edits pulled
into main, hopefully you can just use pip to install pyboy and get my metroid
code.

## TODO

Lots to do! The first big milestone will be implementing a game wrapper for
Metroid II in PyBoy and potentially making a pull request with my new custom
wrapper.

- [ ] Make and verify a custom environment with PyBoy
    - [x] Verify ROM integrity (the rom works)
    - [x] Take random actions in the environment (veryify pyboy interface working)
    - [ ] Implement a `game_wrapper` for Metroid II (makes AI stuff easier)
        - [x] Implement the `start_game` function to skip though the menu
        - [ ] Use RAM mappings to calculate more useful info (particularly health)
        - [x] Implement `game_over` function to check if agent is dead
        - [ ] Improve `game_over` to check health, may be slightly faster
        - [ ] implement `game_area_mapping` potentially, so that all "samus"
          tiles are treated the same, or all "floor" tiles are treated the same
          (maps tiles, to other tiles)
        - [x] Integrate and verify the RAM mappings 
        - [ ] Make a pull request for PyBoy
    - [ ] Determine and implement all possible button combos for "actions"


- [ ] Model Training
    - [ ] Define some reward functions including rewards for
        - [ ] Killing metroids (primary goal)
        - [ ] exploring (secondary goal)
    - [ ] Do some kind of extremely bare-bones training to just explore
    - [ ] Research, research, research...
    - [ ] Sektch out some rough reward functions


- [ ] Containerize the program to make running on other machines easy

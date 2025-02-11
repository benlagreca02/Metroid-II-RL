# TODO write code to load and watch a model perform here
import argparse

def get_action_from_model(model, obs):
    action, _states = model.predict(obs)
    return action, _states
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', type=str, help='Path to the model to be loaded')
    parser.add_argument('-a', '--area', action='store_true', help='Print game area')
    parser.add_argument('-d', '--debug', action='store_true', help='show pyboy debug windows')
    parser.add_argument('-c', '--coords', action='store_true', help='Print coordinate values (Pixels and Area)')
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("No model given, using human control!")

    if args.area:
        print("Printing game area from pyboy")

    if args.coords:
        print("Printing coordinates of samus")


    # TODO move these back up to the top when done prototyping with path loading
    import gymnasium as gym
    from stable_baselines3 import PPO
    from metroid_env import MetroidEnv, DEFAULT_NUM_TO_TICK

    env = MetroidEnv(
            render_mode='human',
            emulation_speed_factor=1, # Because we want to watch it, run in "real time"
            num_to_tick= 1 if not args.model_path else DEFAULT_NUM_TO_TICK,
            debug=args.debug)

    if args.model_path:
        print("Loading model from %s" % args.model_path)
        model= PPO("CnnPolicy", env, verbose=1) 
        model = PPO.load(args.model_path)
        action_getter = lambda obs: get_action_from_model(model, obs)
    else:
        print("Using human player mode")
        action_getter = lambda obs: (0,None)


    obs, info = env.reset()
    # Try catch is a little redundant, but this makes it quit a lot faster and
    # more reliably

    while True:
        # action, _states = model.predict(obs)
        action, _states = action_getter(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        if args.area:
            print(f"AREA: \n{env.game_area()}")
        if args.coords:
            print(f"{env.getAllCoordData()}")



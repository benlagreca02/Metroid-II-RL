# from pyboy import PyBoy

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metroid_env import MetroidEnv

# Determines length of environment vecotrs
# Also multiplies timesteps
NUM_ENVS = 6

# Roughly an hour of "gameplay" if doing real time
# Timestep count for one environment
TIMESTEPS = 200000
LEARNING_RATE = 2.5e-4
ENT_COEF = 0.01
# realisticly it should be a lot smaller
# This makes sure training doesn't stop in the middle of the night
EPOCH_COUNT = 100
GAMMA = 0.997

# across all environments 
TOTAL_TIMESTEPS = TIMESTEPS * NUM_ENVS * 100

CHECKPOINT_FREQUENCY = TIMESTEPS//4
EVAL_FREQUENCY = TIMESTEPS//5

# How many steps until we do a learning update
TRAIN_STEPS_BATCH = TIMESTEPS // 15
# True batch size to use when running multiple envs
BATCH_SIZE = int(TRAIN_STEPS_BATCH * NUM_ENVS)

LOG_DIR =  './log/'

def main():
    def make_env():
        # in case we want to complicate the env creation later
        return gym.make("MetroidII")


    

    env = SubprocVecEnv([make_env for n in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env for n in range(NUM_ENVS)])
    # env = make_env()
    # eval_env = make_env()

    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQUENCY,
                                             save_path=LOG_DIR,
                                             name_prefix="metroid")


    eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQUENCY,
                             deterministic=False, render=False)

    # callbacks = [checkpoint_callback, tb_callback, eval_callback]
    callbacks = [checkpoint_callback, eval_callback]



    # model = PPO("MultiInputPolicy",
    model = PPO("CnnPolicy",
            # normalize_images=False,
            policy_kwargs=dict(normalize_images=False),
            env=env,
            learning_rate=LEARNING_RATE,

            # Num steps to run for each env per update
            # i.e. batch size is n_steps * n_env
            n_steps=TRAIN_STEPS_BATCH,

            # Discount factor
            gamma=GAMMA,
            # Entropy coefficient for loss calculation
            ent_coef=ENT_COEF,
            tensorboard_log=LOG_DIR,
            verbose=1,

            batch_size=BATCH_SIZE,
            n_epochs=EPOCH_COUNT)


    # model = PPO('CnnPolicy', env, tensorboard_log='./ppo_tensorboard/', verbose=1)
    new_logger = configure(LOG_DIR, ["stdout", 'csv', "tensorboard"])
    model.set_logger(new_logger)

    model.learn( total_timesteps=TOTAL_TIMESTEPS,
                 callback=CallbackList(callbacks),
                 progress_bar=True,
                 log_interval=10,
                 tb_log_name="metroid_ppo")




if __name__ == "__main__":
    main()

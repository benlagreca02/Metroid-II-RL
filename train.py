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
NUM_ENVS = 8

TIMESTEPS = 1000 
TRAIN_STEPS_BATCH = TIMESTEPS // 10

LOG_DIR = 

def main():
    def make_env():
        # in case we want to complicate the env creation later
        return gym.make("MetroidII")



    env = SubprocVecEnv([make_env for n in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env for n in range(NUM_ENVS)])
    # env = make_env()
    # eval_env = make_env()

    checkpoint_callback = CheckpointCallback(save_freq=TIMESTEPS/4,
                                             save_path='./logs/',
                                             name_prefix="metroid")


    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/bestModel/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=False, render=False)

    # callbacks = [checkpoint_callback, tb_callback, eval_callback]
    callbacks = [checkpoint_callback, eval_callback]



    # model = PPO("MultiInputPolicy",
    model = PPO("MlpPolicy",
            env=env,
            verbose=1,
            n_steps=int(TRAIN_STEPS_BATCH),
            batch_size=int((TRAIN_STEPS_BATCH*NUM_ENVS)//4),
            n_epochs=1,
            gamma=0.997,
            ent_coef=0.01,
            tensorboard_log='./logs/')


    # model = PPO('CnnPolicy', env, tensorboard_log='./ppo_tensorboard/', verbose=1)
    new_logger = configure('./logs/', ["stdout", 'csv', "tensorboard"])
    model.set_logger(new_logger)

    model.learn( total_timesteps=(TIMESTEPS)*NUM_ENVS*100,
                 callback=CallbackList(callbacks),
                 progress_bar=True,
                 log_interval=10,
                 tb_log_name="metroid_ppo")




if __name__ == "__main__":
    main()

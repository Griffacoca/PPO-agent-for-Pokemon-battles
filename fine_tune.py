from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.player import SimpleHeuristicsPlayer

from pokemon_rl_common import create_env, MaskedActorCriticPolicy
from functools import partial

FINE_TUNE_TIMESTEPS = 500_000
NUM_ENVS = 2

def fine_tune():
    env = SubprocVecEnv([partial(create_env, opponent_cls=SimpleHeuristicsPlayer) for _ in range(NUM_ENVS)])

    # Load your existing best model (the random-only 1M run) and swap in the new env
    model = PPO.load("pokemon_ppo_model", env=env, custom_objects={"learning_rate": 5e-5})

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_ENVS, 1),
        save_path="./checkpoints_finetune/",
        name_prefix="pokemon_ppo_ft",
    )

    model.learn(total_timesteps=FINE_TUNE_TIMESTEPS, tb_log_name="ppo_pokemon_finetune", callback=checkpoint_callback, reset_num_timesteps=False)
    model.save("pokemon_ppo_model_finetuned")
    env.close()

if __name__ == "__main__":
    fine_tune()
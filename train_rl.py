import asyncio
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from pokemon_rl_common import create_env, MaskedActorCriticPolicy

TOTAL_TIMESTEPS = 2_000_000
MODEL_NAME = "pokemon_ppo_model"
TENSORBOARD_LOG_DIR = "./tb_logs"

# One opponent type per parallel env, so every policy update is trained
# against a mix rather than overfitting to one bot's pattern.
OPPONENTS = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
NUM_ENVS = len(OPPONENTS)


def train():
    env_fns = [partial(create_env, opponent_cls=opp) for opp in OPPONENTS]
    env = SubprocVecEnv(env_fns)

    print(f"Initializing RL Agent (opponent mix: {[o.__name__ for o in OPPONENTS]})...")
    model = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-4,
        n_steps=2048 // NUM_ENVS,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="pokemon_ppo",
    )

    print(f"Starting training loop over {TOTAL_TIMESTEPS:,} steps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="ppo_pokemon",
        callback=checkpoint_callback,
    )

    print("\nTraining complete! Saving model to disk...")
    model.save(MODEL_NAME)
    env.close()

    print(f"\nDone. To view training curves, run:\n  tensorboard --logdir {TENSORBOARD_LOG_DIR}")


if __name__ == "__main__":
    train()
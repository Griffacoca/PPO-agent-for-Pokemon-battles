from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from pokemon_rl_common import create_env

# Same opponent mix and learning rate as train_rl.py - this is a
# continuation of that exact run, not a specialization phase like
# fine_tune.py (which trains vs SimpleHeuristics only, at a lower lr,
# and previously made things worse when used this way).
OPPONENTS = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
NUM_ENVS = len(OPPONENTS)

ADDITIONAL_TIMESTEPS = 2_000_000  # takes total from ~2M to ~4M
MODEL_NAME = "pokemon_ppo_model"
TENSORBOARD_LOG_DIR = "./tb_logs"


def continue_training():
    env_fns = [partial(create_env, opponent_cls=opp) for opp in OPPONENTS]
    env = SubprocVecEnv(env_fns)

    print(f"Loading existing model '{MODEL_NAME}' and continuing training...")
    model = PPO.load(MODEL_NAME, env=env)
    # Explicitly keep the same learning rate used to produce the current
    # model, rather than relying on whatever was serialized (belt and
    # braces - PPO.load usually restores this correctly, but this matches
    # the caution used in fine_tune.py after PPO.load's custom_objects
    # learning-rate override proved unreliable there).
    model.learning_rate = 1e-4

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="pokemon_ppo",
    )

    print(f"Training {ADDITIONAL_TIMESTEPS:,} more steps (opponent mix: {[o.__name__ for o in OPPONENTS]})...")
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        tb_log_name="ppo_pokemon",
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    print("\nTraining complete! Saving model to disk...")
    model.save(MODEL_NAME)
    env.close()

    print(f"\nDone. To view training curves, run:\n  tensorboard --logdir {TENSORBOARD_LOG_DIR}")


if __name__ == "__main__":
    continue_training()
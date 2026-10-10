from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from pokemon_rl_common import create_env, SelfPlayOpponent

MODEL_NAME = "pokemon_ppo_model"
TENSORBOARD_LOG_DIR = "./tb_logs"
ADDITIONAL_TIMESTEPS = 2_000_000

# 3 scripted bots + 1 self-play opponent (a FROZEN snapshot of the current
# model, taken once at the start of this run - it does not update as
# training progresses). This adds a genuinely different kind of opponent
# behavior beyond the 3 scripted heuristics, without the complexity of
# periodically refreshing the snapshot mid-run.
ENV_SPECS = [
    (RandomPlayer, None),
    (MaxBasePowerPlayer, None),
    (SimpleHeuristicsPlayer, None),
    (SelfPlayOpponent, {"model_path": MODEL_NAME}),
]
NUM_ENVS = len(ENV_SPECS)


def train():
    env_fns = [
        partial(create_env, opponent_cls=cls, opponent_kwargs=kwargs)
        for cls, kwargs in ENV_SPECS
    ]
    env = SubprocVecEnv(env_fns)

    print(f"Loading '{MODEL_NAME}' and continuing training "
          f"(opponents: RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer, SelfPlay)...")
    model = PPO.load(MODEL_NAME, env=env)
    model.learning_rate = 1e-4

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="pokemon_ppo",
    )

    print(f"Training {ADDITIONAL_TIMESTEPS:,} more steps...")
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
    train()
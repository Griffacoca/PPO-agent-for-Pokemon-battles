import asyncio
import torch
import numpy as np
from typing import Any, Awaitable

from stable_baselines3 import PPO

from poke_env.environment import SinglesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from pokemon_rl_common import BATTLE_FORMAT, embed_battle

MODEL_NAME = "pokemon_ppo_model"
N_BATTLES = 100


class PolicyPlayer(Player):
    def __init__(self, policy, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()

        obs = embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))

        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict)

        return SinglesEnv.action_to_order(action.cpu().numpy()[0], battle)


async def main():
    print(f"Loading model '{MODEL_NAME}'...")
    ppo = PPO.load(MODEL_NAME)

    agent = PolicyPlayer(
        policy=ppo.policy,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
    )

    opponents: list[Player] = [
        cls(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
        for cls in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]

    print(f"Playing {N_BATTLES} battles against each of {len(opponents)} opponents...")
    await agent.battle_against(*opponents, n_battles=N_BATTLES)

    print("\n--- Win rates vs bots ---")
    for opp in opponents:
        win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
        print(f"vs {opp.username} ({opp.__class__.__name__}): {win_rate}%")


if __name__ == "__main__":
    asyncio.run(main())
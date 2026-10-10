"""
Shared code for the Pokemon PPO project (poke-env 0.10.0 compatible).
v2: adds type_2, bench-pokemon visibility (HP/fainted/matchup quality),
and a bigger [128, 128] network.
"""
import numpy as np
import torch
from gymnasium.spaces import Box

from poke_env.data import GenData
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

BATTLE_FORMAT = "gen8randombattle"
MAX_BENCH = 5  # team size 6 - 1 active = 5 possible bench slots

ALL_TYPES = [
    "normal", "fire", "water", "grass", "electric", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "steel", "dark", "fairy"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TYPES)}

# Feature breakdown:
# 4 move base powers + 4 move damage multipliers + 4 move accuracies = 12
# + 2 fainted fractions + 2 remaining fractions                    =  4
# + 2 active HP fractions + 2 status flags                         =  4
# + 3 speed features (our_norm, opp_norm, ratio)                   =  3
# + 4 dynamax features (can/is dynamaxed, both sides)              =  4
# + 18 user active type_1 + 18 opp active type_1                   = 36
# + 18 user active type_2 + 18 opp active type_2                   = 36
# + 5 bench slots x 4 feats (hp, fainted, offense, defense) x 2 sides = 40
N_FEATURES = 12 + 4 + 4 + 3 + 4 + 36 + 36 + (MAX_BENCH * 4 * 2)


def encode_type(type_enum) -> np.ndarray:
    """One-hot encodes a Pokemon type (or an all-zero vector if None/typeless)."""
    vec = np.zeros(len(ALL_TYPES), dtype=np.float32)
    if type_enum is not None:
        type_str = type_enum.name.lower()
        if type_str in TYPE_TO_IDX:
            vec[TYPE_TO_IDX[type_str]] = 1.0
    return vec


def _type_chart(battle):
    return GenData.from_gen(battle.gen).type_chart


# Abilities that grant full immunity to (and usually benefit from) a specific
# attacking type. The raw type chart has no concept of abilities, so without
# this, a move like Surf into a Water Absorb user looks identical to Surf
# into anything else - the model has no way to see the absorption coming.
# Not exhaustive (e.g. Wonder Guard's "only super-effective hits" rule and
# Dry Skin's extra Fire weakness aren't modeled), but covers the common
# single-type absorption/immunity abilities.
ABILITY_TYPE_IMMUNITIES = {
    "voltabsorb": "electric",
    "lightningrod": "electric",
    "motordrive": "electric",
    "waterabsorb": "water",
    "stormdrain": "water",
    "dryskin": "water",
    "flashfire": "fire",
    "wellbakedbody": "fire",
    "sapsipper": "grass",
    "levitate": "ground",
    "eartheater": "ground",
}


def _normalize_ability(ability) -> str:
    if not ability:
        return ""
    return ability.lower().replace(" ", "").replace("-", "").replace("'", "")


def _stab_multiplier_vs(attacker_type, defender, battle) -> float:
    """Type-effectiveness multiplier of an attack of `attacker_type` against
    `defender`, zeroed out if defender's ability grants immunity to that
    exact type (Water Absorb, Levitate, Flash Fire, etc.)."""
    if attacker_type is None or defender is None:
        return 1.0

    immune_type = ABILITY_TYPE_IMMUNITIES.get(_normalize_ability(getattr(defender, "ability", None)))
    if immune_type is not None and attacker_type.name.lower() == immune_type:
        return 0.0

    return attacker_type.damage_multiplier(
        defender.type_1, defender.type_2, type_chart=_type_chart(battle)
    )


def _bench_features(bench_mon, reference_mon, battle, mon_is_attacker: bool) -> list:
    """4 features for one bench slot: hp_fraction, fainted flag, offensive
    matchup, defensive risk.

    mon_is_attacker=True  -> our bench mon: offensive = "how good would
                              switching TO this mon be" (its types attacking
                              the opposing active); defensive = "how hard
                              would the opposing active hit THIS mon if I
                              switched it in"
    mon_is_attacker=False -> their bench mon: offensive = "how dangerous is
                              this opposing bench mon to our current active";
                              defensive = "how well could WE hit it if it
                              came in"
    Both directions are always computed as bench_mon's types vs
    reference_mon (offensive) and reference_mon's types vs bench_mon
    (defensive) - mon_is_attacker only affects which side's bench and which
    reference mon get passed in by the caller, not the formula itself.
    Missing/unknown slots return a neutral [0.0, 0.0, 1.0, 1.0] (no info).
    """
    if bench_mon is None:
        return [0.0, 0.0, 1.0, 1.0]

    hp = bench_mon.current_hp_fraction if bench_mon.current_hp_fraction else 0.0
    fainted = 1.0 if bench_mon.fainted else 0.0

    if reference_mon is None:
        offense = 1.0
        defense_risk = 1.0
    else:
        mult1 = _stab_multiplier_vs(bench_mon.type_1, reference_mon, battle)
        if bench_mon.type_2 is not None:
            mult2 = _stab_multiplier_vs(bench_mon.type_2, reference_mon, battle)
            offense = max(mult1, mult2)
        else:
            offense = mult1

        # The reverse direction: reference_mon's types attacking bench_mon.
        # This is the piece that was missing before - a bench mon could
        # look like a fine offensive pick while still being one-shot by
        # the opponent's current attacker (see: Terrakion/Crawdaunt both
        # switching into a Fairy-type sweeper and dying instantly despite
        # having a workable attack of their own).
        dmult1 = _stab_multiplier_vs(reference_mon.type_1, bench_mon, battle)
        if reference_mon.type_2 is not None:
            dmult2 = _stab_multiplier_vs(reference_mon.type_2, bench_mon, battle)
            defense_risk = max(dmult1, dmult2)
        else:
            defense_risk = dmult1

    return [hp, fainted, offense, defense_risk]


def _boost_multiplier(stage: int) -> float:
    """Standard Pokemon stat-stage multiplier, e.g. +1 = 1.5x, -1 = 0.67x."""
    stage = max(-6, min(6, stage or 0))
    if stage >= 0:
        return (2 + stage) / 2
    return 2 / (2 - stage)


def _is_paralyzed(mon) -> bool:
    status = getattr(mon, "status", None)
    return status is not None and "par" in str(status).lower()


def _effective_speed(mon) -> float:
    """Estimated in-battle Speed, accounting for stat stage boosts and
    paralysis (which halves Speed). For our own team, poke-env knows the
    exact calculated stat; for the opponent, exact EVs/IVs/nature usually
    aren't known, so this falls back to the raw base Speed stat as an
    approximation - still far better than no speed signal at all."""
    if mon is None:
        return 0.0

    stats = getattr(mon, "stats", None)
    base_speed = stats.get("spe") if stats else None
    if base_speed is None:
        base_stats = getattr(mon, "base_stats", None)
        base_speed = base_stats["spe"] if base_stats else 100.0

    boosts = getattr(mon, "boosts", None)
    stage = boosts.get("spe", 0) if boosts else 0
    speed = base_speed * _boost_multiplier(stage)

    if _is_paralyzed(mon):
        speed *= 0.5

    return speed


def _speed_features(our_active, opp_active) -> list:
    """3 features: our normalized effective speed, opponent's, and a ratio
    in (0, 1) where >0.5 means we're faster (0.5 = tied/no information)."""
    our_speed = _effective_speed(our_active)
    opp_speed = _effective_speed(opp_active)

    our_norm = float(np.clip(our_speed / 100.0, 0.0, 6.0))
    opp_norm = float(np.clip(opp_speed / 100.0, 0.0, 6.0))

    total = our_speed + opp_speed
    ratio = our_speed / total if total > 0 else 0.5

    return [our_norm, opp_norm, ratio]


def _move_accuracy(move) -> float:
    """Normalized 0-1 accuracy. `True` in poke-env means a move that never
    misses (e.g. Swift, Aerial Ace-style effects), so that maps to 1.0."""
    acc = getattr(move, "accuracy", None)
    if acc is True or acc is None:
        return 1.0
    return acc / 100.0


def _dynamax_features(battle, our_active, opp_active) -> list:
    """4 features: can we still Dynamax, are we currently Dynamaxed, can the
    opponent still Dynamax, are they currently Dynamaxed. Previously there
    was no Dynamax-related signal in the observation at all, so the model
    had no way to reason about WHEN to Dynamax (e.g. save it for a better
    matchup) - it could only ever react to the immediate damage number,
    which likely explains the consistent turn-1 Dynamax habit observed
    across multiple replays regardless of matchup quality."""
    our_can_dynamax = 1.0 if getattr(battle, "can_dynamax", False) else 0.0
    opp_can_dynamax = 1.0 if getattr(battle, "opponent_can_dynamax", False) else 0.0
    our_is_dynamaxed = 1.0 if our_active and getattr(our_active, "is_dynamaxed", False) else 0.0
    opp_is_dynamaxed = 1.0 if opp_active and getattr(opp_active, "is_dynamaxed", False) else 0.0
    return [our_can_dynamax, our_is_dynamaxed, opp_can_dynamax, opp_is_dynamaxed]


def embed_battle(battle) -> np.ndarray:
    """Turns a Battle object into a flat feature vector (see N_FEATURES above)."""
    moves_base_power = -np.ones(4, dtype=np.float32)
    moves_dmg_multiplier = np.ones(4, dtype=np.float32)
    moves_accuracy = -np.ones(4, dtype=np.float32)

    opp_active = battle.opponent_active_pokemon
    our_active = battle.active_pokemon
    opp_is_statused = opp_active.status is not None if opp_active else False

    for i, move in enumerate(battle.available_moves[:4]):
        moves_base_power[i] = move.base_power / 100.0
        moves_accuracy[i] = _move_accuracy(move)
        if move.base_power == 0 and opp_is_statused:
            moves_dmg_multiplier[i] = 0.0
        elif opp_active is not None:
            moves_dmg_multiplier[i] = _stab_multiplier_vs(move.type, opp_active, battle)

    team_size = max(len(battle.team), 1)
    opp_team_size = max(len(battle.opponent_team), 1)

    fainted_team = len([m for m in battle.team.values() if m.fainted]) / team_size
    fainted_opp = len([m for m in battle.opponent_team.values() if m.fainted]) / opp_team_size
    remaining_team = 1.0 - fainted_team
    remaining_opp = 1.0 - fainted_opp

    our_hp = our_active.current_hp_fraction if our_active else 0.0
    opp_hp = opp_active.current_hp_fraction if opp_active else 0.0
    our_status = 1.0 if our_active and our_active.status else 0.0
    opp_status = 1.0 if opp_is_statused else 0.0

    our_type1_vec = encode_type(our_active.type_1 if our_active else None)
    opp_type1_vec = encode_type(opp_active.type_1 if opp_active else None)
    our_type2_vec = encode_type(our_active.type_2 if our_active else None)
    opp_type2_vec = encode_type(opp_active.type_2 if opp_active else None)

    # Bench: everyone on the team who isn't the current active mon, padded/
    # truncated to a fixed MAX_BENCH slots so the feature vector is constant
    # size regardless of how many mons have fainted or been revealed.
    our_bench = [m for m in battle.team.values() if not m.active][:MAX_BENCH]
    opp_bench = [m for m in battle.opponent_team.values() if not m.active][:MAX_BENCH]

    our_bench_feats = []
    for i in range(MAX_BENCH):
        mon = our_bench[i] if i < len(our_bench) else None
        our_bench_feats.extend(_bench_features(mon, opp_active, battle, mon_is_attacker=True))

    opp_bench_feats = []
    for i in range(MAX_BENCH):
        mon = opp_bench[i] if i < len(opp_bench) else None
        opp_bench_feats.extend(_bench_features(mon, our_active, battle, mon_is_attacker=False))

    speed_feats = _speed_features(our_active, opp_active)
    dynamax_feats = _dynamax_features(battle, our_active, opp_active)

    return np.concatenate(
        [
            moves_base_power,
            moves_dmg_multiplier,
            moves_accuracy,
            [fainted_team, fainted_opp],
            [remaining_team, remaining_opp],
            [our_hp, opp_hp],
            [our_status, opp_status],
            speed_feats,
            dynamax_feats,
            our_type1_vec,
            opp_type1_vec,
            our_type2_vec,
            opp_type2_vec,
            our_bench_feats,
            opp_bench_feats,
        ],
        dtype=np.float32,
    )


class CustomPokemonEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        box_space = Box(low=-1.0, high=6.0, shape=(N_FEATURES,), dtype=np.float32)
        self.observation_spaces = {agent: box_space for agent in self.possible_agents}

    def embed_battle(self, battle) -> np.ndarray:
        return embed_battle(battle)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )


def create_env(opponent_cls=RandomPlayer):
    """opponent_cls lets different parallel envs train against different
    bots at once (see train_rl.py), so the policy doesn't overfit to any
    single opponent's quirks."""
    raw_env = CustomPokemonEnv(
        battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None, strict=False
    )
    opponent = opponent_cls(start_listening=False)
    return Monitor(SingleAgentWrapper(raw_env, opponent))


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[128, 128],  # bumped up from [64, 64]
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)
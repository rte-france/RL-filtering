from grid2op.Action import PlayableAction, DontAct
from grid2op.Observation import CompleteObservation
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import ChangeNothing
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import NeverAttackBudget

try:
    # TODO change that !
    from grid2op.l2rpn_utils import ActionWCCI2022, ObservationWCCI2022
except ImportError:
    from grid2op.Action import PlayableAction
    from grid2op.Observation import CompleteObservation
    import warnings
    warnings.warn("The grid2op version you are trying to use is too old for this environment. Please upgrade it.")
    ActionWCCI2022 = PlayableAction
    ObservationWCCI2022 = CompleteObservation

config = {
    "backend": PandaPowerBackend,
    "action_class": ActionWCCI2022,
    "observation_class": ObservationWCCI2022,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    # TODO change that too
    "chronics_class": ChangeNothing,
    # "chronics_class": Multifolder,
    # "grid_value_class": GridStageFromBlablabla,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "opponent_attack_cooldown": 100000,
    "opponent_attack_duration": 0,
    "opponent_budget_per_ts": 0.,
    "opponent_init_budget": 0.,
    "opponent_action_class": DontAct,
    "opponent_budget_class": NeverAttackBudget,
    # TODO change this (read from file instead, put back the opponent etc.)
    # "opponent_class": GeometricOpponent,
    # "kwargs_opponent": {
    #     "lines_attacked": lines_attacked,
    #     "attack_every_xxx_hour": 24,
    #     "average_attack_duration_hour": 4,
    #     "minimum_attack_duration_hour": 1,
    # },
}

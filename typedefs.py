import enum
from typing import Callable, Dict, List, Tuple


class Action(enum.Enum):
    """Actions that an agent can take in a gridworld."""

    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()


# position on a grid
GridState = Tuple[int, int]
# (s, a, s') triple
StateActionSuccessorState = Tuple[GridState, Action, GridState]
# probabilities of taking an action from a particular grid state
Policy = Dict[GridState, Dict[Action, float]]
# transition probabilities
TransitionProbabilities = Dict[StateActionSuccessorState, float]
# rewards
Rewards = Dict[StateActionSuccessorState, float]
# value function
ValueFunction = Dict[GridState, float]
# state evaluator
StateEvaluator = Callable[
    [
        GridState,
        ValueFunction,
        Policy,
        Rewards,
        TransitionProbabilities,
        int,
        float,
    ],
    float,
]

from typing import Set, Tuple

from policy_iteration import (
    calculate_action_value,
    compute_greedy_policy,
    compute_value_function,
    get_equiprobable_policy,
)
from typedefs import (
    Action,
    GridState,
    Policy,
    Rewards,
    StateActionSuccessorState,
    TransitionProbabilities,
    ValueFunction,
)


# Constants

MAX_ITERATIONS = 10000
GAMMA = 1
EPSILON = 1e-5


def calculate_max_state_value(
    state: GridState,
    previous_values: ValueFunction,
    policy: Policy,
    rewards: Rewards,
    transition_probabilities: TransitionProbabilities,
    grid_size: int,
    gamma: float,
) -> float:
    state_value = -1e9
    for action, action_probability in policy[state].items():
        action_value = calculate_action_value(
            state,
            previous_values,
            action,
            rewards,
            transition_probabilities,
            grid_size,
            gamma,
        )
        if action_value >= state_value:
            state_value = action_value

    return state_value


def run_value_iteration(
    actions: Set[Action],
    rewards: Rewards,
    transition_probabilities: TransitionProbabilities,
    grid_size: int,
    terminal_states: Set[GridState],
    gamma: float = GAMMA,
    update_inplace: bool = True,
    max_k: int = None,
    epsilon: float = EPSILON,
    round_end_values: int = None,
) -> Tuple[Policy, ValueFunction]:
    """Iteratively improve policy using value iteration."""
    # no starting policy is needed for value iteration, this is just necessary for compute_value_function API
    policy = get_equiprobable_policy(grid_size, actions)
    values = compute_value_function(
        policy,
        rewards,
        transition_probabilities,
        grid_size,
        terminal_states,
        state_evaluator=calculate_max_state_value,
        gamma=gamma,
        update_inplace=update_inplace,
        max_k=max_k,
        epsilon=epsilon,
        round_end_values=round_end_values,
    )
    updated_policy = compute_greedy_policy(
        values,
        policy,
        rewards,
        transition_probabilities,
        grid_size,
        terminal_states,
        gamma,
    )
    return updated_policy, values

"""This is an example of policy evaluation & iteration for a simple grid world problem in RL.

Author: Vadym Barda vadim.barda@gmail.com
"""

from collections import defaultdict
import enum
from typing import Dict, List, Tuple, Set

import funcy


# Constants

MAX_ITERATIONS = 10000
GAMMA = 1
EPSILON = 1e-5


class Action(enum.Enum):
    """Actions that an agent can take in a gridworld."""

    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()


# Types

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


def get_states(grid_size: int) -> List[GridState]:
    states = [(m, n) for m in range(grid_size) for n in range(grid_size)]
    return states


# policy


def get_equiprobable_policy(grid_size: int, actions: Set[Action]) -> Policy:
    """Return policy where actions have equal probabilities for each state."""
    policy: Policy = {}
    probability = 1 / len(actions)
    for state in get_states(grid_size):
        policy[state] = {action: probability for action in actions}
    return policy


def get_state_agnostic_policy(
    grid_size: int, action_to_probability: Dict[Action, float]
) -> Policy:
    """Return policy where specified action probabilities are the same for each state."""
    if sum(action_to_probability.values()) != 1:
        raise ValueError("All action probabilities should add up to 1.")

    policy: Policy = {}
    for state in get_states(grid_size):
        policy[state] = action_to_probability
    return policy


# transitions


def get_successor_states(
    current_state: GridState, grid_size: int, action: Action
) -> List[GridState]:
    """Get a list of possible successor state from a current state given an action."""
    current_x, current_y = current_state
    if action == Action.UP:
        successor_x = max(current_x - 1, 0)
        successor_y = current_y
    elif action == Action.DOWN:
        successor_x = min(current_x + 1, grid_size - 1)
        successor_y = current_y
    elif action == Action.LEFT:
        successor_y = max(current_y - 1, 0)
        successor_x = current_x
    elif action == Action.RIGHT:
        successor_y = min(current_y + 1, grid_size - 1)
        successor_x = current_x
    else:
        raise ValueError(f"Action {action} is not supported")

    successor_state = (successor_x, successor_y)
    return [successor_state]


def get_transition_probabilities(
    grid_size: int, actions: Set[Action]
) -> TransitionProbabilities:
    """Get transition probabilities for each (s, a, s') triple."""
    transition_probabilities: TransitionProbabilities = {}
    states = get_states(grid_size)
    for current_state in states:
        for action in actions:
            # we can only transition to neighboring positions (and stay in the same location if we're off grid)
            # note: there could theoretically be multiple successor states given current state & action
            # but in our case we basically deal with a single successor state
            successor_states = get_successor_states(current_state, grid_size, action)
            for successor_state in successor_states:
                state_action_successor_state = (current_state, action, successor_state)
                transition_probabilities[state_action_successor_state] = 1.0

    return transition_probabilities


# rewards


def get_constant_rewards(
    grid_size: int, reward: float, actions: Set[Action]
) -> Rewards:
    """Get constant rewards for each (s, a, s') triple."""
    rewards: Rewards = {}
    states = get_states(grid_size)
    for current_state in states:
        for action in actions:
            successor_states = get_successor_states(current_state, grid_size, action)
            for successor_state in successor_states:
                state_action_successor_state = (current_state, action, successor_state)
                rewards[state_action_successor_state] = reward

    return rewards


# Bellman equation calculations


def calculate_action_value(
    state: GridState,
    previous_values: ValueFunction,
    action: Action,
    rewards: Rewards,
    transition_probabilities: TransitionProbabilities,
    grid_size: int,
    gamma: float,
) -> float:
    """Calculate action value from a current state."""
    action_value = 0.0
    # note that there could possibly be multiple successor states, and just in this case there is only
    # one successor state given a particular action
    successor_states = get_successor_states(state, grid_size, action)
    for successor_state in successor_states:
        state_action_successor_state = (state, action, successor_state)
        transition_probability = transition_probabilities[state_action_successor_state]
        immediate_reward = rewards[state_action_successor_state]
        previous_value_at_successor_state = previous_values[successor_state]
        action_value += transition_probability * (
            immediate_reward + gamma * previous_value_at_successor_state
        )

    return action_value


def calculate_state_value(
    state: GridState,
    previous_values: ValueFunction,
    policy: Policy,
    rewards: Rewards,
    transition_probabilities: TransitionProbabilities,
    grid_size: int,
    gamma: float,
) -> float:
    """Calculate updated state value from a current state & previous state values."""
    state_value = 0.0
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
        state_value += action_probability * action_value

    return state_value


def evaluate_policy(
    policy: Policy,
    rewards: Rewards,
    transition_probabilities: TransitionProbabilities,
    grid_size: int,
    terminal_states: Set[GridState],
    gamma: float = GAMMA,
    update_inplace: bool = True,
    max_k: int = None,
    epsilon: float = EPSILON,
    round_end_values: int = None,
) -> ValueFunction:
    """Evaluate a policy & produce a value function."""
    states = get_states(grid_size)

    # initialize value function with 0 value for each state
    previous_values = {state: 0.0 for state in states}
    # if we update value function in-place, re-use initialized values
    if update_inplace:
        values = previous_values

    k = 0  # iteration counter
    converged = False

    while True:
        if not update_inplace:
            # if we're not updating in-place, we'll need to create a new value function
            # and populate it using values from the old value function (previous_values)
            values = {}

        delta = 0.0
        for state in states:
            # don't touch terminal states
            if state in terminal_states:
                values[state] = 0.0
                continue

            # decide which values to use during the backup calculation of
            # an updated state value
            if update_inplace:
                values_from_previous_iteration = values
            else:
                values_from_previous_iteration = previous_values

            previous_state_value = values_from_previous_iteration[state]
            state_value = calculate_state_value(
                state,
                values_from_previous_iteration,
                policy,
                rewards,
                transition_probabilities,
                grid_size,
                gamma,
            )

            values[state] = state_value
            delta = max(delta, abs(state_value - previous_state_value))
            converged = delta < epsilon

        if converged:
            break

        k += 1
        if max_k is not None and k >= max_k:
            break

        if max_k is None and k >= MAX_ITERATIONS:
            raise RuntimeError(
                "Reached max allowed # of iterations during policy evaluation."
            )

        if not update_inplace:
            previous_values = values

    if converged:
        print(f"Took {k} steps to reach convergence.")

    if round_end_values is not None:
        return funcy.walk_values(funcy.partial(round, ndigits=round_end_values), values)

    return values


def compute_greedy_policy(
    values: ValueFunction,
    policy: Policy,
    rewards: Rewards,
    transition_probabilities: TransitionProbabilities,
    grid_size: int,
    terminal_states: Set[GridState],
    gamma: float,
) -> Policy:
    """Compute a greedy policy wrt a given value function & current policy."""
    updated_policy: Policy = defaultdict(dict)
    for state in get_states(grid_size):
        # don't touch terminal states
        if state in terminal_states:
            updated_policy[state] = {action: 0.0 for action in policy[state].keys()}
            continue

        action_to_action_value = {
            action: calculate_action_value(
                state,
                values,
                action,
                rewards,
                transition_probabilities,
                grid_size,
                gamma,
            )
            for action in policy[state].keys()
        }

        max_action_value = max(action_to_action_value.values())

        # if we have multiple actions that result in max value, set probability of 1 / n argmax actions
        updated_policy_for_state = {
            action: 1.0 if action_value == max_action_value else 0.0
            for action, action_value in action_to_action_value.items()
        }
        n_argmax_actions = sum(updated_policy_for_state.values())
        updated_policy_for_state = funcy.walk_values(
            lambda is_action_argmax: is_action_argmax / n_argmax_actions,
            updated_policy_for_state,
        )
        updated_policy[state] = updated_policy_for_state

    return updated_policy


# policy iteration


def improve_policy(
    policy: Policy,
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
    """Iteratively improve policy."""
    policy_stable = False
    n_iterations = 0
    while not policy_stable:
        values = evaluate_policy(
            policy,
            rewards,
            transition_probabilities,
            grid_size,
            terminal_states,
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

        policy_stable = updated_policy == policy
        policy = updated_policy

        n_iterations += 1
        if n_iterations >= MAX_ITERATIONS:
            raise RuntimeError(
                "Reached max allowed # of iterations during policy iteration."
            )

    return updated_policy, values

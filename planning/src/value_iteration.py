import numpy as np


def episodic_value_iteration(state_space, instance):

    # Initialize the value function with 0 for any s in stae_space
    V = {frozenset(state.as_atoms()): 0 for state in state_space}

    # Set error tolerance (since were are dealing with integer values
    # and no discount factor this will stop when error = 0).
    tolerance = 0.1
    diff = tolerance + 0.1
    iterations = 0

    while diff > tolerance:

        diff = 0
        V_old = V.copy()
        iterations += 1

        for state in state_space:

            # Episodic value iteration -> value function at goal states is 0
            if state[instance.problem.goal]:
                continue

            neighbors = instance.search_model.successors(state)

            neighbors_v = [V[frozenset(n.as_atoms())] for _, n in neighbors]

            V[frozenset(state.as_atoms())] = -1 + np.max(neighbors_v)

            state_as_atoms = frozenset(state.as_atoms())

            diff = np.max(
                [
                    diff,
                    np.abs(V_old[state_as_atoms] - V[state_as_atoms]),
                ],
            )

    return V, diff, iterations
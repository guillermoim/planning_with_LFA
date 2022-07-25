
import pickle
import os
import numpy as np

from src import *
import sys


def _load_env(domain, instance_filepath):
    instance = InstanceData(instance_filepath, domain)

    return instance


def _read_value_function(filepath):
    
    V = {}

    with open(filepath) as file:
        lines = file.readlines()[1:]
        for line in lines:
            line = line.strip()
            state, value = line.split(";")
            state, value = eval(state), int(value)
            V[state] = value

    return V


def _get_state_index():
    pass


def _base_features():

    return [ "n_concept_distance(c_primitive(clear,0),r_primitive(on,0,1),c_primitive(clear_g,0))",
             "n_count(c_primitive(holding,0))",]

def td_learning(W, domain, instance_filepath, value_function_filepath, matrix_filepath):
    
    # 
    V = _read_value_function(value_function_filepath)

    X, y, feature_names, _ = data_loader(matrix_filepath)
    
    states_ordered = list(V.keys())
    del V

    indices = [feature_names.index(x) for x in _base_features()]

    instance = _load_env(domain, instance_filepath)

    X = X[:, indices]
    X = np.hstack([X, np.zeros((X.shape[0], 1))])
    
    state = instance.problem.init

    ALPHA = 0.2
    LAMBDA = 1
    R = -1
    EPS = 0.3
    
    for _ in range(100):

        while not state[instance.problem.goal]:

            state_str = frozenset([str(e) for e in state.as_atoms()])
            idx = states_ordered.index(state_str)

            succesors = [s for _, s in instance.search_model.successors(state)]

            if np.random.rand() < 1 - EPS:
                # double list comprehension and index extraction for succesors
                succesors_as_atoms = [
                    frozenset([str(e) for e in s.as_atoms()]) for s in succesors
                ]
                succesors_indices = list(
                    map(states_ordered.index, succesors_as_atoms))

                succesors_values = [np.dot(W, x)
                                    for x in X[succesors_indices, :]]
                succesor = succesors[np.argmax(succesors_values)]
                v_ = np.max(succesors_values)


            else:
                succesor = succesors[np.random.randint(0, len(succesors))]
                x_ = X[
                    states_ordered.index(
                        frozenset([str(e) for e in succesor.as_atoms()])
                    ),
                    :,
                ]
                v_ = np.dot(W, x_)

            x = X[idx, :]
            v = np.dot(W, x)

            W = W + ALPHA * (R + LAMBDA * v_ - v) * x
            state = succesor
        print(W)
        EPS *= 0.9
        state = instance.problem.init
    
    return W

if __name__ == "__main__":
    
    domain = DomainData("pddl/blocksworld/domain.pddl")

    W = np.zeros(3)

    instance_1 = "pddl/blocksworld/p-clear-3blocks-0.pddl"
    instance_2 = "pddl/blocksworld/p-clear-4blocks-0.pddl"
    W = _td_learning(W, domain, instance_1, "results/blocksworld/p-clear-3blocks-0-flag/value_function.csv", "results/blocksworld/p-clear-3blocks-0-flag/feat_matrix_extended.csv")
    W = _td_learning(W, domain, instance_2, "results/blocksworld/p-clear-4blocks-0-flag/value_function.csv", "results/blocksworld/p-clear-4blocks-0-flag/feat_matrix_extended.csv")


    #W = np.array([-2, 1, -1])
    print(f"WEIGHTS: {W}")

    for i, x in enumerate(X):
        print(x, states_ordered[i], np.dot(W, x), y[i])

    print(y)
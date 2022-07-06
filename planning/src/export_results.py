import numpy as np
import os
import pickle as pkl


def save_features_matrix(
    basepath,
    feat_names,
    feat_values,
    comp_values,
):

    # line 0 are the headers
    # lines 1 up to n-1 contain the evalautions of the features being each line a state plus the value function
    # line n contains the complexities of the features

    outpath = os.path.join(basepath, "features_matrix.csv")

    X = np.asarray(feat_values)

    lines = []
    l = ";".join(feat_names)
    l+= "\n"
    lines.append(l)

    for row in X:
        line = str(";".join(map(str, row.tolist())))
        line += "\n"
        lines.append(line)

    l = ";".join(map(str, comp_values))
    l += "\n"
    lines.append(l)


    with open(outpath, "w") as file:
        file.writelines(lines)

    return outpath


def _parse_predicate(predicate):

    init_idx = predicate.find("(")
    end_idx = predicate.find(")")

    symbol = predicate[:init_idx]
    objects = predicate[init_idx + 1 : end_idx].split(",")

    objects = " ".join(objects)
    if objects != "":
        objects = " " + objects

    return f"({symbol}{objects})"


def save_value_function(V, basepath):

    # Two column csv representation of the value function.

    outpath = os.path.join(basepath, "value_function.csv")

    with open(outpath, "w") as file:
        file.write("state;V*\n")
        for state in V:
            # rep_state = " ".join([_parse_predicate(str(e)) for e in state])" ".join([_parse_predicate(str(e)) for e in state])
            rep_state = frozenset([str(e) for e in state])
            file.write(f"{rep_state};{str(V[state])}\n")
    return outpath

def save_states_pddl(states, domain_name, basepath):
    '''
        expects a collection of states in frozenset format
    '''
    outpath = os.path.join(basepath, "pddl-states.pddl")

    with open(outpath, "w") as file:
        file.write(f"(define (problem {domain_name})\n")
        for state in states:
            rep_state = " ".join([_parse_predicate(str(e)) for e in state])
            file.write(f"(:STATE {rep_state})\n")
        file.write(")")

    return outpath

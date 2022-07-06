import argparse 
import os
import sys
import pandas as pd
from timeit import default_timer as timer
import subprocess
from pathlib import Path

sys.path.append('../formula_interpreter')

from formula_interpreter import *

from src import *

def _load_instance(domain, instance_filename):
    
    instance_file = f"pddl/{domain_name}/{instance_filename}"
    instance = InstanceData(instance_file, domain)

    return instance


def main(domain_name, multipath, formula_path, feat_complexity,  flag=True):

    LFA_INITIAL_DIR=os.environ['LFA_INITIAL_DIR']
    LFA_TRANSLATOR_DIR=os.environ['LFA_TRANSLATOR_DIR']
    LFA_UNIVERSAL_PDDL_PARSER=os.environ['LFA_UNIVERSAL_PDDL_PARSER']
    LFA_PLANNING_DIR=os.environ['LFA_PLANNING_DIR']
    LFA_FORMULA_INTERPRETER_DIR=os.environ['LFA_FORMULA_INTERPRETER_DIR']

    FORMULA_PATH = os.path.abspath(formula_path)
    
    DOMAIN_PATH = f'pddl/{domain_name}/domain.pddl'

    assert LFA_INITIAL_DIR != "", "LFA_INITIAL_DIR env. variable must be defined."
    assert LFA_TRANSLATOR_DIR != "", "LFA_TRANSLATOR_DIR env. variable must be defined."
    assert LFA_UNIVERSAL_PDDL_PARSER != "", "LFA_UNIVERSAL_PDDL_PARSER env. variable must be defined."
    assert LFA_PLANNING_DIR != "", "LFA_PLANNING_DIR env. variable must be defined."
    assert LFA_FORMULA_INTERPRETER_DIR != "", "LFA_FORMULA_INTERPRETER_DIR env. variable must be defined."

    # Read the domain
    domain_file = f"pddl/{domain_name}/domain.pddl"
    domain = DomainData(domain_file)

    # Read multiple instances
    states_by_instance = {}
    with open(multipath) as file:
        df = pd.read_csv(file) 
        files = list(df.itertuples(index=False, name=None))

    print(f'\nDOMAIN: {domain_name}')
    print(20*'-')
    print('-Expanding states')

    # For each instance, expand the state space and store them both in dlplan and tarski format
    for instance_filename, MAX_NUM_STATES in files:
        print(f'\tExpanding states for {instance_filename}')
        task = domain_name + instance_filename
        instance = _load_instance(domain, instance_filename)
        tarski_states = expand_states(instance, MAX_NUM_STATES)
        dl_states = [tarski_to_dl_state(instance, state) for state in tarski_states]
        states_by_instance[instance_filename] = {}
        states_by_instance[instance_filename].update({'dl_states': dl_states})
        states_by_instance[instance_filename].update({'tarski_states': tarski_states})
    
    # Retrieve the key features to compose the value function.
    KEY_FEATURES, WEIGHTS, BIAS = get_key_features(task, regression=False)
    # Store all the states in dlplan format in one object and use
    # it to extract the features. So features extracted are inter
    # instance.
    data = []
    
    for k in states_by_instance:
        data.extend(states_by_instance[k]['dl_states'])
    
    print('-Extracting features')
    start = timer()
    features, complexities, F = extract_features(
        domain, data, feat_complexity, KEY_FEATURES, flag
    )
    end = timer()
    elapsed = end - start
    print(f"\t {len(features)} features extracted (in {elapsed:.4f}s)")

    #print(features)

    KEY_FEATURES_IDXS = list(map(features.index, KEY_FEATURES))

    # Per instance, extact Simon's features (by first translating to appropiate format)
    # and then compose the value function.
    for instance in states_by_instance: 
        print(f'-Creating dataset (feature matrix) for {instance}')
        dl_states = states_by_instance[instance]['dl_states']
        tarski_states = states_by_instance[instance]['tarski_states']
        features_evaluations = [evaluate_features_state(F, state, features) for state in dl_states]
        valid_states = [frozenset(state.as_atoms()) for state in tarski_states]
        
        # build value function
        V = {}

        for i, state in enumerate(valid_states):
            feat_vals = list(map(lambda x: features_evaluations[i][x], KEY_FEATURES_IDXS))
            feat_vals = np.array(feat_vals)
            feat_vals[np.where(feat_vals > 2500)] = 0
            V[state] = WEIGHTS.T @ feat_vals + BIAS

        outfilename = instance_filename.split(".")[0]
        outfilename += "-flag" if flag else ""

        outpath = f"./results/{domain_name}/{outfilename}"
        os.makedirs(outpath, exist_ok=True)

        FEAT_MATRIX_PATH = save_features_matrix(
            outpath,
            features,
            features_evaluations,
            complexities,
        )

        VALUE_FUNCTION_PATH = save_value_function(V, outpath)
        PDDL_STATES_PATH = save_states_pddl(V, domain_name, outpath)

        INSTANCE_PATH = f"pddl/{domain_name}/{instance}"
        OUT_STATES = open(f"{outpath}/states.txt", "w")

        print(f'-Translating states for {instance}.')

        subprocess.run([f"{LFA_TRANSLATOR_DIR}/convert", DOMAIN_PATH, INSTANCE_PATH,
                         PDDL_STATES_PATH], stdout=OUT_STATES)

        print(f'-Extracting extended features for {instance}.')
        dataset, predicates = load_datasets(Path(outpath))
        (features2, value_function, vector_function) = parse_formula(Path(FORMULA_PATH), predicates)
        custom_evaluation(dataset, features2, FEAT_MATRIX_PATH, value_function,
                        vector_function, False)


if __name__ == "__main__":
    domain_name = 'blocksworld'
    multipath = 'multiples/blocksworld-clear.csv'
    formula_path = '../formula_interpreter/blocks-clear.txt'


    parser = argparse.ArgumentParser(description="LFA for symbolic features")
    parser.add_argument("--domain_name", type=str)
    parser.add_argument("--multipath", type=str)
    parser.add_argument("--formula_path", type=str)
    parser.add_argument("--flag_features", action="store_true")
    parser.add_argument("--max_complexity", type=int, default=4)

    args = parser.parse_args()

    domain_name = args.domain_name
    multipath = args.multipath
    formula_path = args.formula_path
    flag_features = args.flag_features
    max_complexity = args.max_complexity


    main(domain_name, multipath, formula_path, max_complexity, flag_features)
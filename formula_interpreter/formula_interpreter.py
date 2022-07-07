import argparse
import re
import os

from collections import deque
from datasets.supervised.optimal import ValueDataset, ExtendedDataset
from pathlib import Path
from typing import List, Tuple


def load_datasets(path: Path):
    files = list(path.glob('*states.txt'))
    decoded_datasets = [ValueDataset(d, decode=True) for d in files]
    predicates = decoded_datasets[0].predicates
    # return (ExtendedDataset(decoded_datasets, 1), predicates)
    return decoded_datasets[0], predicates


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formula', required=True, type=Path,
                        help='Path to file with formulas')
    parser.add_argument('--data', required=True, type=Path,
                        help='Path to directory of state values')
    parser.add_argument('--feat_path', required=True, type=str,
                        help='Path to file with features matrix.')
    parser.add_argument('--verbose', action='store_true', help='Print values')
    args = parser.parse_args()
    return args


def _predicate(predicate, variables, first, second, invert):
    def evaluation(state: dict, context: dict):
        if predicate in state:
            objects = state[predicate]
            if invert:
                objects = [(y, x) for (x, y) in objects]
            if first:
                objects = [(x,) for x, _ in objects]
            if second:
                objects = [(y,) for _, y in objects]
            values = set(objects)
            arguments = tuple([context['scope'][variable]
                              for variable in variables])
            return arguments in values
        return False
    return evaluation


def _find_corresponding_predicate(name: str, predicates):
    tokens = name.lower().split(':')
    first = False
    second = False
    filter = None
    invert = False
    for predicate, arity in predicates:
        if predicate.lower() == tokens[-1]:
            for token in tokens[:-1]:
                first |= token.startswith('first')
                second |= token.startswith('second')
                invert |= token.startswith('invert')
                if '|' in token:
                    filter, _, _, _, _, _ = _find_corresponding_predicate(
                        token.split('|')[-1], predicates)
            return (predicate, arity, first, second, filter, invert)
    return (None, None, None, None, None, None)


def _and(first_body, second_body):
    return lambda state, context: first_body(state, context) and second_body(state, context)


def _exists(body, variable):
    def evaluation(state: dict, context: dict):
        previous_value = context['scope'][variable] if variable in context['scope'] else None
        result = False
        objects = context['objects']
        for obj in objects:
            context['scope'][variable] = obj
            result = body(state, context)
            if result:
                break
        # Restore previous value, if any.
        if previous_value:
            context['scope'][variable] = previous_value
        else:
            del context['scope'][variable]
        return result
    return evaluation


def _not(body):
    return lambda state, context: not body(state, context)


def _shortest_path(goal_body: str, relation_description: str, aux, variable, max_aux: str, predicates):
    (relation, _, _, _, _, relation_invert) = _find_corresponding_predicate(
        relation_description, predicates)

    def evaluation(state, context):
        obj = context['scope'][variable]

        if relation not in state:
            context['cache'][aux] = 0
            return goal_body(state, context)

        edges = {}
        relations = state[relation]
        if relation_invert:
            relations = [(y, x) for (x, y) in relations]
        for lhs, rhs in relations:
            edges.setdefault(lhs, []).append(rhs)
        max_depth = context['cache'][max_aux] if max_aux in context['cache'] else 10000
        previous_value = context['scope'][variable] if variable in context['scope'] else None
        found_depth = None

        queue = deque([(obj, 0)])
        closed = set()
        while len(queue) > 0:
            (x, d) = queue.popleft()
            if (d <= max_depth) and (x not in closed):
                closed.add(x)
                context['scope'][variable] = x
                if goal_body(state, context):
                    found_depth = d
                    break
                if x in edges:
                    for y in edges[x]:
                        queue.append((y, d + 1))

        # Restore previous value, if any.
        if previous_value:
            context['scope'][variable] = previous_value
        else:
            del context['scope'][variable]

        if found_depth is not None:
            context['cache'][aux] = found_depth
            return True
        else:
            context['cache'][aux] = 0
            return False

    return evaluation


def _multiplication(first_body, second_body):
    def evaluation(state, context):
        lhs = first_body(state, context)
        if abs(lhs) < 0.01:
            return lhs
        return lhs * second_body(state, context)
    return evaluation


def _addition(first_body, second_body):
    return lambda state, context: first_body(state, context) + second_body(state, context)


def _subtraction(first_body, second_body):
    return lambda state, context: first_body(state, context) - second_body(state, context)


def _constant_or_feature(feature: str):
    def evaluation(state, context):
        if feature.isdigit():
            return int(feature)
        features = context['features']
        cache = context['cache']
        # Cache evaluations, some features might be used in multiple places.
        if feature in cache:
            result = cache[feature]
            if isinstance(result, bool):
                result = 1 if result else 0
            return result
        if feature in features:
            result = 1 if features[feature](state, context) else 0
        elif feature in features:
            result = 1 if features[feature](state, context) else 0
        else:
            raise Exception(
                'Error: {} is not defined. Always place variables last in the term.'.format(feature))
        cache[feature] = result
        return result
    return evaluation


DEBUG = False
DEBUG_INDENT = 0


def _parse_feature_expression(index, expression, predicates: List[Tuple[str, int]], boolean_features: dict):
    operator = expression[index]

    global DEBUG
    global DEBUG_INDENT
    if 'DEBUG' in globals() and DEBUG:
        print('{}{}'.format(' ' * (2 * DEBUG_INDENT), operator))
    DEBUG_INDENT += 1

    if operator == 'NOT':
        index, body = _parse_feature_expression(
            index + 1, expression, predicates, boolean_features)
        DEBUG_INDENT -= 1
        return index, _not(body)
    elif operator == 'EXISTS':
        variable = expression[index + 1]
        index, body = _parse_feature_expression(
            index + 2, expression, predicates, boolean_features)
        DEBUG_INDENT -= 1
        return index, _exists(body, variable)
    elif operator == 'AND':
        index, first_body = _parse_feature_expression(
            index + 1, expression, predicates, boolean_features)
        index, second_body = _parse_feature_expression(
            index, expression, predicates, boolean_features)
        DEBUG_INDENT -= 1
        return index, _and(first_body, second_body)
    elif operator == 'SHORTEST_PATH':
        index, goal_body = _parse_feature_expression(
            index + 1, expression, predicates, boolean_features)
        relation = expression[index]
        aux = expression[index + 1]
        max_aux = expression[index + 2]
        variable = expression[index + 3]
        DEBUG_INDENT -= 1
        return index + 4, _shortest_path(goal_body, relation, aux, variable, max_aux, predicates)
    elif operator in boolean_features:
        DEBUG_INDENT -= 1
        return index + 1, boolean_features[operator]
    else:
        (predicate, arity, first, second, filter,
         invert) = _find_corresponding_predicate(operator, predicates)
        if first:
            arity = 1
        if second:
            arity = 1
        if filter:
            raise NotImplementedError()
        if predicate is not None:
            variables = expression[index + 1: index + arity + 1]
            DEBUG_INDENT -= 1
            return index + arity + 1, _predicate(predicate, variables, first, second, invert)
        raise Exception('Not a predicate: {}'.format(operator))


def _parse_value_expression(index, expression):
    operator = expression[index]
    if operator == '*':
        index, first_body = _parse_value_expression(index + 1, expression)
        index, second_body = _parse_value_expression(index, expression)
        return index, _multiplication(first_body, second_body)
    elif operator == '+':
        index, first_body = _parse_value_expression(index + 1, expression)
        index, second_body = _parse_value_expression(index, expression)
        return index, _addition(first_body, second_body)
    elif operator == '-':
        index, first_body = _parse_value_expression(index + 1, expression)
        index, second_body = _parse_value_expression(index, expression)
        return index, _subtraction(first_body, second_body)
    else:
        return index + 1, _constant_or_feature(operator)


def parse_formula(file: Path, predicates: list):
    features = {}
    with file.open('r') as stream:
        lines = stream.readlines()
    for line in lines:
        if len(line.strip()) <= 0:
            continue
        lhs, rhs = line.split('=')
        lhs = lhs.strip()
        rhs = rhs.strip()
        rhs = rhs.replace('\n', '')
        rhs = [token for token in re.split(' |,|\(|\)', rhs) if len(token) > 0]
        if lhs.startswith('FEATURE'):
            name = lhs.strip().split(' ')[1]
            features[name] = _parse_feature_expression(
                0, rhs, predicates, features)[1]
        elif lhs.startswith('VALUE'):
            value_function = _parse_value_expression(0, rhs)[1]
        elif lhs.startswith('VECTOR'):
            if 'PREREQUISITES' in rhs:
                idx = rhs.index('PREREQUISITES')
                vector_tokens = rhs[0:idx]
                prereq_tokens = rhs[idx + 1:]
            else:
                vector_tokens = rhs
                prereq_tokens = []
            
            def vector_function_closure(state, context):
                p_vector = []
                for token in prereq_tokens:
                    features[token](state, context)
                    p_vector.append(context['cache'][token])
                feature_vector = []
                index = 0
                num_tokens = len(vector_tokens)
                while index < num_tokens:
                    token = vector_tokens[index]
                    feature_value = 0.0
                    if token == '+':
                        lhs_token = vector_tokens[index + 1]
                        rhs_token = vector_tokens[index + 2]
                        if lhs_token in features:
                            feature_value += 1.0 if features[lhs_token](
                                state, context) else 0.0
                        elif lhs_token in context['cache']:
                            feature_value += context['cache'][lhs_token]
                        if rhs_token in features:
                            feature_value += 1.0 if features[rhs_token](
                                state, context) else 0.0
                        elif rhs_token in context['cache']:
                            feature_value += context['cache'][rhs_token]
                        index += 2
                    elif token in features:
                        feature_value = 1.0 if features[token](
                            state, context) else 0.0
                    elif token in context['cache']:
                        feature_value = context['cache'][token]
                    feature_vector.append(feature_value)
                    index += 1
                return feature_vector, p_vector
            vector_function = vector_function_closure
        else:
            raise Exception('Format error')
    return (features, value_function, vector_function)


def _evaluate_function(dataset, features, value_function, vector_function, verbose):
    for state, label_value in dataset:
        context = {
            'objects': set([obj for arguments in [arguments for argument_instance in state.values() for arguments in argument_instance] for obj in arguments]),
            'features': features,
            'scope': {},
            'cache': {}
        }
        # print(state)
        function_value = value_function(state, context)
        vector, _ = vector_function(state, context)
        if verbose:
            print('Function value: {}; Label value: {}; Vector: {}'.format(
                function_value, int(label_value), vector))
        if int(label_value) != function_value:
            print('Function output does not match label_value: {} != {}'.format(
                function_value, int(label_value)))
            # return


def custom_evaluation(dataset, features, features_matrix_path, value_function, vector_function, verbose):
    """
        This function extends the feature_matrix with the features build by the work of Simon.
    """


    with open(features_matrix_path, "r") as file:
        lines = file.readlines()

    complexities = lines[-1]
    feat_names = lines[0].strip('\n')
    features_lines = lines[1:-1]
    new_features_lines = []
    
    new_headings = list(features_lines)

    V = []

    for ix, (state, _) in enumerate(dataset):

        context = {
            'objects': set([obj for arguments in [arguments for argument_instance in state.values() for arguments in argument_instance] for obj in arguments]),
            'features': features,
            'scope': {},
            'cache': {}
        }

        function_value = value_function(state, context)
        V.append(-function_value)

        f_vector, p_vector = vector_function(state, context)

        vector = f_vector + p_vector
        
        if ix == 0:
            new_headings = [f'F_{i}' for i in range(len(vector))]
            heading = feat_names.split(';')
            final_headings = heading + new_headings + ['V*\n']
            final_heading = ";".join(final_headings)
            new_features_lines.append(final_heading)

        line = features_lines[ix].strip('\n').split(";")
        line = line[:] + [str(int(x)) for x in vector] + [str(-function_value)+'\n']

        line = ";".join(line)
        new_features_lines.append(line)

        if verbose:
            print('Function value: {}; Vector: {}'.format(
                function_value, vector))
    
    new_features_lines.append(complexities)

    dirname = os.path.dirname(features_matrix_path)
    
    with open(os.path.join(dirname, "feat_matrix_extended.csv"), "w") as file:
        file.writelines(new_features_lines)

    return V


if __name__ == "__main__":
    args = _parse_arguments()
    #print('Loading datasets...', flush=True)
    dataset, predicates = load_datasets(args.data)
    #print('Parsing formula...', flush=True)
    (features, value_function, vector_function) = parse_formula(
        args.formula, predicates)
    #print('Evaluating function...', flush=True)
    custom_evaluation(dataset, features, args.feat_path, value_function,
                       vector_function, args.verbose)

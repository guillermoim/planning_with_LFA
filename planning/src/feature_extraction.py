import dlplan

def _filter_features(features):

    return set(
        [
            feature
            for feature in features
            if feature.startswith("b_") or feature.startswith("n_")
        ]
    )


def extract_features(domain, states, feature_complexity, base_features, flag):

    F = construct_syntactic_element_factory(domain)
    generator = construct_feature_generator(flag)
    features = _filter_features(
        generator.generate(F, feature_complexity, 180, 100000, 1, states)
    )

    features = features.union(set(base_features))

    complexities = []

    for f in features:
        feature = None
        if f.startswith("n_"):
            feature = F.parse_numerical(f)
            complexities.append(feature.compute_complexity())
        elif f.startswith("b_"):
            feature = F.parse_boolean(f)
            complexities.append(feature.compute_complexity())

    return list(features), complexities, F


def evaluate_features_state(F, state, features):

    evaluations = []

    for f in features:
        if f.startswith("n_"):
            feature = F.parse_numerical(f)
            val = feature.evaluate(state)
            evaluations.append(val)
        elif f.startswith("b_"):
            feature = F.parse_boolean(f)
            val = int(feature.evaluate(state))
            evaluations.append(val)

    return evaluations


def get_complexities(F, features):
    feature = None
    complexities = []
    for f in features:
        feature = None
        if f.startswith("n_"):
            feature = F.parse_numerical(f)
        elif f.startswith("b_"):
            feature = F.parse_boolean(f)
        complexities.append(feature.compute_complexity())

    return complexities

def construct_syntactic_element_factory(domain_data):
    """Constructs an empty factory for constructing elements."""
    return dlplan.SyntacticElementFactory(domain_data.vocabulary_info)


def construct_feature_generator(flag):

    generator = dlplan.FeatureGenerator()

    generator.set_generate_inclusion_boolean(flag)
    generator.set_generate_diff_concept(flag)
    generator.set_generate_or_concept(flag)
    generator.set_generate_subset_concept(flag)
    generator.set_generate_sum_concept_distance_numerical(flag)
    generator.set_generate_role_distance_numerical(flag)
    generator.set_generate_sum_role_distance_numerical(flag)
    generator.set_generate_and_role(True)
    generator.set_generate_compose_role(flag)
    generator.set_generate_identity_role(flag)
    generator.set_generate_diff_role(flag)
    generator.set_generate_not_role(True)
    generator.set_generate_or_role(flag)
    generator.set_generate_transitive_reflexive_closure_role(flag)
    generator.set_generate_top_role(flag)

    return generator
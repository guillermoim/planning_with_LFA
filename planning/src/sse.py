
def _sse(problem, search_model, max_num_states):
    """Run DFS to explore state space.
       sse: (S)tate (S)pace (E)xploration.
    """
    frontier = []
    frontier.append(problem.init)
    generated = set()
    generated.add(problem.init)
    num_states = 0
    while frontier:
        cur = frontier.pop()
        successors = search_model.successors(cur)
        for action, succ in successors:
            if succ in generated:
                continue
            frontier.append(succ)
            generated.add(succ)
            num_states += 1
            if num_states == max_num_states:
                return generated
    return generated


def expand_states(instance, max_num_states=1000):
    states = _sse(instance.problem, instance.search_model, max_num_states)
    return states

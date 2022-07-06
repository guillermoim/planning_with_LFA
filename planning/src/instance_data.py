import dlplan
import tarski
from tarski.io import PDDLReader
from tarski.grounding import LPGroundingStrategy
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.search import GroundForwardSearchModel


class State:
    def __init__(self, index, tarski_state, dlplan_state):
        self.index = index
        self.tarski_state = tarski_state
        self.dlplan_state = dlplan_state

    def __str__(self):
        return str(self.dlplan_state)


class InstanceData:
    """Store data related to a single instance."""

    def __init__(self, instance_file, domain_data):
        self.instance_file = instance_file
        self.domain_data = domain_data

        self.problem = parse_instance_file(domain_data.domain_file, instance_file)
        grounder = LPGroundingStrategy(self.problem)
        (
            self.tarski_dynamic_atoms,
            self.tarski_static_atoms_textual,
            self.tarski_sorts_textual,
        ) = grounder.ground_atoms()
        self.tarski_goal_atoms = parse_conjunctive_formula(self.problem.goal)
        self.instance_info, self.tarski_atom_to_dlplan_atom = construct_instance_info(
            domain_data, self
        )
        self.search_model = GroundForwardSearchModel(
            self.problem, ground_problem_schemas_into_plain_operators(self.problem)
        )

    def map_tarski_atoms_to_dlplan_state(self, tarski_atoms):
        return dlplan.State(
            self.instance_info,
            [
                self.tarski_atom_to_dlplan_atom[tarski_atom]
                for tarski_atom in tarski_atoms
                if tarski_atom in self.tarski_dynamic_atoms
            ],
        )


def parse_instance_file(domain_file, instance_file):
    """Parses the PDDL instance file using Tarski."""
    reader = PDDLReader()
    reader.parse_domain(domain_file)
    reader.parse_instance(instance_file)
    return reader.problem


def parse_conjunctive_formula(goal):
    """Compute all tarski Atoms from a tarski ComboundFormula."""
    if isinstance(goal, tarski.syntax.formulas.CompoundFormula):
        if goal.connective == tarski.syntax.formulas.Connective.And:
            atom_names = []
            for subformula in goal.subformulas:
                atom_names.extend(parse_conjunctive_formula(subformula))
            return atom_names
        else:
            raise Exception(
                f"Unsupported connective {goal.connective} in goal description."
            )
    elif isinstance(goal, tarski.syntax.formulas.Atom):
        return [goal]



def construct_instance_info(domain_data, instance_data):
    """Constructs an InstanceInfo from a problem description."""
    instance_info = dlplan.InstanceInfo(domain_data.vocabulary_info)
    # Add dynamic atoms
    tarski_atom_to_dlplan_atom = dict()
    for tarski_atom in instance_data.tarski_dynamic_atoms:
        dlplan_atom = instance_info.add_atom(
            tarski_atom.predicate.name, [obj.name for obj in tarski_atom.subterms]
        )
        assert tarski_atom not in tarski_atom_to_dlplan_atom
        tarski_atom_to_dlplan_atom[tarski_atom] = dlplan_atom
    # Add other static atoms
    for static_atom in instance_data.tarski_static_atoms_textual:
        instance_info.add_static_atom(static_atom[0], static_atom[1])
    # Add sorts
    for static_atom in instance_data.tarski_sorts_textual:
        instance_info.add_static_atom(static_atom[0], static_atom[1])
    # Add static goal atoms
    for tarski_atom in instance_data.tarski_goal_atoms:
        predicate_name = tarski_atom.predicate.name
        object_names = []
        for obj in tarski_atom.subterms:
            object_names.append(obj.name)
        # add atom as goal version of the predicate
        instance_info.add_static_atom(predicate_name + "_g", object_names)
    return instance_info, tarski_atom_to_dlplan_atom

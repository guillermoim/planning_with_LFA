#include "Instance.h"
#include "Lifted.h"

using namespace parser::pddl;

void encode(GroundVec &vec, Domain &d)
{
	std::set<IntVec> s;
	for (unsigned i = 0; i < vec.size(); i++)
	{
		IntVec v(1, d.preds.index(vec[i]->lifted->name));
		for (unsigned j = 0; j < vec[i]->params.size(); j++)
			v.push_back(vec[i]->params[j]);
		s.insert(v);
	}

	for (std::set<IntVec>::iterator it = s.begin(); it != s.end(); it++)
	{
		std::cout << (*it)[0];
		for (unsigned i = 1; i < it->size(); i++)
			std::cout << " " << (*it)[i];
		std::cout << "\n";
	}
}

// const char read_pddl_string()

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		std::cout << "Usage: ./convert <domain.pddl> <task.pddl> <states.pddl>\n";
		exit(1);
	}

	// Read multiagent domain and instance
	Domain domain(argv[1]);
	Instance instance(domain, argv[2]);

	std::vector<GroundVec> states;

	instance.parseStates(argv[3], states);

	std::cout << "BEGIN_OBJECTS\n";
	for (unsigned i = 0; i < domain.types[0]->objects.size(); i++)
		std::cout << i << " " << domain.types[0]->objects[i] << "\n";
	std::cout << "END_OBJECTS\n";

	std::cout << "BEGIN_PREDICATES\n";
	for (unsigned i = 0; i < domain.preds.size(); i++)
		std::cout << i << " " << domain.preds[i]->name << "\n";
	std::cout << "END_PREDICATES\n";

	std::cout << "BEGIN_FACT_LIST\n";
	std::cout << "END_FACT_LIST\n";

	std::cout << "BEGIN_GOAL_LIST\n";
	encode(instance.goal, domain);
	std::cout << "END_GOAL_LIST\n";

	std::cout << "BEGIN_STATE_LIST\n";

	for (GroundVec state : states)
	{
		std::cout << "BEGIN_LABELED_STATE\n";
		std::cout << "1\n";
		std::cout << "BEGIN_STATE\n";
		encode(state, domain);
		std::cout << "END_STATE\n";
		std::cout << "END_LABELED_STATE\n";
	}

	std::cout << "END_STATE_LIST\n";
}

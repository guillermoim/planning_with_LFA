import dlplan

def tarski_to_dl_state(instance, tarski_state):

    return dlplan.State(
        instance.instance_info,
        [
            instance.tarski_atom_to_dlplan_atom[atom]
            for atom in tarski_state.as_atoms()
            if atom in instance.tarski_dynamic_atoms
        ],
    )

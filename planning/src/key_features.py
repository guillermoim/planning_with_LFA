import numpy as np


def get_key_features(task, regression=0):

    feats = []
    weights = None
    bias = 0

    if "blocksworld" in task and "clear" in task:
        weights = np.array([-2, 1])
        bias = -1
        
        if not regression:
            feats = [
                "n_concept_distance(c_primitive(clear,0),r_primitive(on,0,1),c_primitive(clear_g,0))",
                "b_nullary(arm-empty)",
            ]


        elif regression:
            feats = [
                "n_concept_distance(c_primitive(clear,0),r_primitive(on,0,1),c_primitive(clear_g,0))",
                "b_nullary(arm-empty)",
            ]


    elif "blocksworld" in task and "on" in task:

        block_x = "c_primitive(on_g,0)"
        block_y = "c_primitive(on_g,1)"

        above_x_no_x = f"c_some(r_transitive_closure(r_primitive(on,0,1)),{block_x})"

        above_y_inc_y = (
            f"c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),{block_y})"
        )

        above_y_no_y = f"c_some(r_transitive_closure(r_primitive(on,0,1)),{block_y})"
        above_x_inc_x = (
            f"c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),{block_x})"
        )

        A_x_on_y = f"b_inclusion(c_primitive(on_g,0),{above_y_no_y})"
        B_y_on_x = f"b_inclusion(c_primitive(on_g,1),{above_x_no_x})"

        E_arm_empty = "b_empty(c_primitive(holding,0))"

        D_goal = "n_count(r_and(r_primitive(on,0,1),r_primitive(on_g,0,1)))"

        Z_holding_x_clear_y = "n_concept_distance(c_primitive(holding,0),r_primitive(on_g,0,1),c_primitive(clear,0))"

        N_blocks_above_x_below_y_no_y = (
            f"n_count(c_and({above_x_no_x},c_not({above_y_inc_y})))"
        )

        N2_blocks_above_x_below_y_no_y = (
            f"n_count(c_and({above_x_no_x},c_not({A_x_on_y})))"
        )

        M_blocks_above_y_below_x_no_x = (
            f"n_count(c_and({above_y_no_y},c_not({above_x_inc_x})))"
        )

        M2_blocks_above_y_below_x_no_x = (
            f"n_count(c_and({above_y_no_y},c_not({B_y_on_x})))"
        )

        if not regression:
            feats = [
                A_x_on_y,
                B_y_on_x,
                E_arm_empty,
                D_goal,
                Z_holding_x_clear_y,
                N_blocks_above_x_below_y_no_y,
                M_blocks_above_y_below_x_no_x,
            ]

        if regression:

            feats = [
                'n_count(r_and(r_primitive(on,0,1),r_primitive(on_g,0,1)))',
                'n_concept_distance(c_primitive(holding,0),r_primitive(on_g,0,1),c_primitive(clear,0))',
                'b_empty(c_primitive(holding,0))',
                'F_1',
                'F_3']

            bias = 3.
            weights = np.array([-2.0, -2.0, -1.0, 2.0, 2.0])

    elif "transport" in task:
        if not regression:
            feats = []
        if regression:
            feats = ['F_0', 'F_1', 'F_2', 'F_3', 'F_4', 'F_5']
    elif "gripper" in task:
        if not regression:
            feats = []
        if regression:
            feats = ['F_0', 'F_1', 'F_2', 'F_3', 'F_4']
            weights = np.array([-1.0, -3.0, -1.0, -2.0, -1.0])

    elif "visitall" in task:
        if not regression:
            feats = []
        if regression:
            feats = ['F_0', 'F_1']

    return feats, weights, bias

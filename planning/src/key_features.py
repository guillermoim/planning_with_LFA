import numpy as np


def get_key_features(task, regression=0):

    feats = []
    weights = None
    bias = 0

    if "blocksworld-clear" == task:
        
        weights = np.array([-2, -1])
        bias = 0
        
        if not regression:
            feats = [
                "n_concept_distance(c_primitive(clear,0),r_primitive(on,0,1),c_primitive(clear_g,0))",
                "n_count(c_primitive(holding,0))",
            ]


        elif regression:
            feats = [
                "n_concept_distance(c_primitive(clear,0),r_primitive(on,0,1),c_primitive(clear_g,0))",
                "n_count(c_primitive(holding,0))",
            ]


    elif "blocksworld-on" == task:

        if not regression:
            feats = [
                'n_count(r_and(r_primitive(on,0,1),r_primitive(on_g,0,1)))',
                'n_concept_distance(c_primitive(holding,0),r_primitive(on_g,0,1),c_primitive(clear,0))',
                'b_empty(c_primitive(holding,0))',
            ]
            bias = 3.
            weights = np.array([-2.0, -2.0, -1.0, 2.0, 2.0])

        if regression:
        
            feats = [
                'n_count(r_and(r_primitive(on,0,1),r_primitive(on_g,0,1)))',
                'n_concept_distance(c_primitive(holding,0),r_primitive(on_g,0,1),c_primitive(clear,0))',
                'b_empty(c_primitive(holding,0))',
                'F_1',
                'F_3']

            bias = 3.
            weights = np.array([-2.0, -2.0, -1.0, 2.0, 2.0])

    elif "transport" == task:
        if not regression:
            feats = []
        if regression:
            feats = ['F_0', 'F_1', 'F_2', 'F_3', 'F_4', 'F_5']
    
    elif "gripper" == task:
        
        if regression:
            feats = ['F_0', 'F_1', 'F_2', 'F_3', 'F_4']
            weights = np.array([-1.0, -3.0, -1.0, -2.0, -1.0])
            bias = 0
        else:
            feats = []
            weights = None #np.array([-1.0, -3.0, -1.0, -2.0, -1.0])
            bias = 0 

    elif "visitall" == task:
        if not regression:
            feats = []
        if regression:
            feats = ['F_0', 'F_1']

    return feats, weights, bias

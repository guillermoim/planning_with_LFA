#!/bin/zsh 
python sse.py --domain gripper --max_complexity 4 --max_num_states 10000 --flag  --instance prob01_0.pddl
python sse.py --domain gripper --max_complexity 4 --max_num_states 10000 --flag  --instance prob01_1.pddl
python sse.py --domain gripper --max_complexity 4 --max_num_states 10000 --flag  --instance prob02_0.pddl
python sse.py --domain gripper --max_complexity 4 --max_num_states 20000 --flag  --instance prob02_1.pddl
#!/bin/zsh 
python sse.py --domain blocksworld --max_complexity 4 --max_num_states 10000 --flag  --instance p-clear-3blocks-0.pddl 
python sse.py --domain blocksworld --max_complexity 4 --max_num_states 10000 --flag  --instance p-clear-3blocks-1.pddl 
python sse.py --domain blocksworld --max_complexity 4 --max_num_states 10000 --flag  --instance p-clear-4blocks-0.pddl 
python sse.py --domain blocksworld --max_complexity 4 --max_num_states 10000 --flag  --instance p-clear-4blocks-1.pddl 
#python sse.py --domain blocksworld --max_complexity 4 --max_num_states 10000 --flag  --instance p-clear-4blocks-0.pddl 
#python sse.py --domain blocksworld --max_complexity 4 --max_num_states 10000 --flag  --instance p-clear-5blocks-0.pddl  
#python sse.py --domain blocksworld --max_complexity 4 --max_num_states 20000 --flag  --instance p-clear-6blocks-0.pddl  
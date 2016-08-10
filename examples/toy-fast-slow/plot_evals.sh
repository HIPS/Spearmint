#!/bin/bash

cd ../../spearmint/visualizations

python cumulative_function_evals_wall_time.py ../../examples/toy-fast-slow/coupled $1
python cumulative_function_evals_wall_time.py ../../examples/toy-fast-slow/dec-fast $1
python cumulative_function_evals_wall_time.py ../../examples/toy-fast-slow/dec-slow $1
python cumulative_function_evals_wall_time.py ../../examples/toy-fast-slow/dec-fast-gamma $1
python cumulative_function_evals_wall_time.py ../../examples/toy-fast-slow/dec-fast-gamma0 $1

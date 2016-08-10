#!/bin/bash

cd ../../spearmint/visualizations

python progress_curve.py ../../examples/toy-fast-slow/coupled ../../examples/toy-fast-slow/dec-slow ../../examples/toy-fast-slow/dec-fast ../../examples/toy-fast-slow/dec-fast-gamma ../../examples/toy-fast-slow/dec-fast-gamma0 --repeat=$1 --logscale --violation-value=2.0 --mainfile=toy --wall-time --labels="coupled;decoupled, $\gamma=\infty$;decoupled, $\gamma=1$;decoupled, $\gamma=0.1$;decoupled, $\gamma=0$"

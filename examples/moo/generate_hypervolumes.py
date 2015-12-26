import os
import sys
import importlib
import imp
import pdb
import numpy             as np
import numpy.random      as npr
import numpy.linalg      as npla
import matplotlib        as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from spearmint.visualizations         import plots_2d
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.parsing          import get_objectives_and_constraints
from spearmint.utils.parsing          import DEFAULT_TASK_NAME
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.tasks.input_space      import paramify_no_types
from spearmint.main                   import load_jobs
from spearmint.utils.moop             import MOOP_basis_functions
from spearmint.utils.moop             import average_min_distance
import os
import sys
from spearmint.grids                 import sobol_grid
import scipy.optimize as spo
from DIRECT import solve
from PyGMO import *

def main(expt_dir):

	os.chdir(expt_dir)
	sys.path.append(expt_dir)

	options         = parse_config_file(expt_dir, 'config.json')
	experiment_name = options["experiment-name"]

	main_file = options['main_file']
	if main_file[-3:] == '.py':
		main_file = main_file[:-3]
	module  = __import__(main_file)

	input_space     = InputSpace(options["variables"])
	chooser_module  = importlib.import_module('spearmint.choosers.' + options['chooser'])
	chooser         = chooser_module.init(input_space, options)
	db              = MongoDB(database_address=options['database']['address'])
	jobs            = load_jobs(db, experiment_name)
	hypers          = db.load(experiment_name, 'hypers')
	tasks           = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)

	if len(tasks) < 2:
		print 'Not a multi-objective problem!'
		return -1

	if options['language'] != "PYTHON":
		print 'Only python programs supported!'
		return -1

	for task in tasks:
		if tasks[ task ].type != 'objective':
			print 'Not a multi-objective problem!'
			return -1

	def create_fun(task):
		def fun(params, gradient = False):

			if len(params.shape) > 1 and params.shape[ 1 ] > 1:
				params = params.flatten()

			params = input_space.from_unit(np.array([ params ])).flatten()

			return module.main(0, paramify_no_types(input_space.paramify(params)))[ task ]

		return fun

	funs = [ create_fun(task) for task in tasks ]

	moop = MOOP_basis_functions(funs, input_space.num_dims)

	grid = sobol_grid.generate(input_space.num_dims, grid_size = 1000 * input_space.num_dims)

	moop.solve_using_grid(grid)

	reference = np.ones(len(tasks)) * 1e3

	hyper_volume_solution = moop.get_hypervolume(reference.tolist())

	result = moop.compute_pareto_front_and_set()
	front = result['frontier']
	pareto_set = result['pareto_set']

	with open('hypervolume_solution.txt', 'a') as f:
		print >> f, "%lf" % (hyper_volume_solution)

	# We iterate through each recommendation made

	i = 0
	more_recommendations = True
	while more_recommendations:

                recommendation = db.load(experiment_name, 'recommendations', {'id' : i + 1})

		if recommendation == None:
			more_recommendations = False
		else:

			solution = input_space.to_unit(input_space.vectorify(recommendation[ 'params' ]))

			if len(solution.shape) == 1:
				solution = solution.reshape((1, len(solution)))
			
			# We compute the objective values associated to this recommendation
	
			values_solution = np.zeros((solution.shape[ 0 ], len(tasks)))
		
			for j in range(values_solution.shape[ 0 ]):
				for k in range(values_solution.shape[ 1 ]):
					values_solution[ j, k ] = funs[ k ](solution[ j : (j + 1), : ])

			moop = MOOP_basis_functions(funs, input_space.num_dims)

			moop.set_population(solution)

			hyper_volume = moop.get_hypervolume(reference.tolist())

			with open('hypervolumes.txt', 'a') as f:
				print >> f, "%lf" % (hyper_volume)

			with open('evaluations.txt','a') as f_handle: 
				np.savetxt(f_handle, np.array([recommendation['num_complete_tasks'].values()]), delimiter = ' ', newline = '\n')

		i += 1

if __name__ == '__main__':
	main(*sys.argv[1:])


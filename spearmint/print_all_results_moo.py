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

# This method finds the reference point Using multiobjective optimization and a grid of
# values

def find_reference_point_using_direct(tasks, module, input_space, grid_size = 20000):

	def create_fun_neg(task):
		def fun(params, gradient = False):

			if len(params.shape) > 1 and params.shape[ 1 ] > 1:
				params = params.flatten()

			params = input_space.from_unit(np.array([ params ])).flatten()

			return -1.0 * module.main(0, paramify_no_types(input_space.paramify(params)))[ task ]

		return fun

	funs_neg = [ create_fun_neg(task) for task in tasks ]

	reference_point = np.zeros(len(funs_neg))

	for i in range(len(funs_neg)):

                def f(x, user_data):
			if x.ndim == 1:
				x = x[None,:]
				value = funs_neg[ i ](x)

			return value, 0

                l = np.zeros(input_space.num_dims) * 1.0
                u = np.ones(input_space.num_dims) * 1.0
		
		x, y_opt, ierror = solve(f, l, u, maxf = 85000)

		reference_point[ i ] = -1.0 * y_opt + np.abs(-1.0 * y_opt * 0.01)

	return reference_point


def find_reference_point(tasks, module, input_space, grid_size = 20000):

	def create_fun_neg(task):
		def fun(params, gradient = False):

			if len(params.shape) > 1 and params.shape[ 1 ] > 1:
				params = params.flatten()

			params = input_space.from_unit(np.array([ params ])).flatten()

			return -1.0 * module.main(0, paramify_no_types(input_space.paramify(params)))[ task ]

		return fun

	funs_neg = [ create_fun_neg(task) for task in tasks ]


	moop_neg = MOOP_basis_functions(funs_neg, input_space.num_dims)
	moop_neg.evolve(400, 400)
	result = moop_neg.compute_pareto_front_and_set()
	front = result['frontier']
	pareto_set = result['pareto_set']

	grid = sobol_grid.generate(input_space.num_dims, grid_size = grid_size, grid_seed = npr.randint(0, grid_size))
	grid = np.vstack((grid, pareto_set))

	# We add the borders of the hyper-cube to the grid since there it is likely to be the maximum

	for i in range(2**input_space.num_dims):

		vector = np.zeros(input_space.num_dims)
		
		for j in range(input_space.num_dims):
			if bin(i & 2**j) != bin(0):
				vector[ j ] = 1.0

		grid = np.vstack((grid, vector.reshape((1, input_space.num_dims))))

	reference_point = np.zeros(len(funs_neg))

	for i in range(len(funs_neg)):

		grid_values = np.zeros(grid.shape[ 0 ])

		for j in range(grid.shape[ 0 ]):
			grid_values[ j ] = funs_neg[ i ](grid[ j, : ])

		best = grid[ np.argmin(grid_values), : ]

                def f(x):
			if x.ndim == 1:
				x = x[None,:]
				value = funs_neg[ i ](x)

			return (value)

                bounds = [ (0.0, 1.0) ] * input_space.num_dims

                x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, best, bounds = bounds, disp = 0, approx_grad = True)

		reference_point[ i ] = -1.0 * y_opt + np.abs(-1.0 * y_opt * 0.01)

	return reference_point

def main(expt_dir):

	os.chdir(expt_dir)
	sys.path.append(expt_dir)

	options         = parse_config_file(expt_dir, 'config.json')
	experiment_name = options["experiment-name"]
	options['main_file'] = 'prog_no_noisy'

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

#	moop.evolve(1, 8)

	grid = sobol_grid.generate(input_space.num_dims, grid_size = 1000 * input_space.num_dims)

	moop.solve_using_grid(grid)

#	reference = find_reference_point_using_direct(tasks, module, input_space)

#	reference = reference + np.abs(reference) * 0.1

	reference = np.ones(len(tasks)) * 7

	hyper_volume_solution = moop.get_hypervolume(reference.tolist())

	result = moop.compute_pareto_front_and_set()
	front = result['frontier']
	pareto_set = result['pareto_set']

#	os.remove('hypervolume_solution.txt')

	with open('hypervolume_solution.txt', 'a') as f:
		print >> f, "%lf" % (hyper_volume_solution)

#	os.remove('hypervolumes.txt')

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

			with open('mean_min_distance_to_frontier.txt', 'a') as f: 
				print >> f, "%lf" % (average_min_distance(values_solution, front))

			with open('mean_min_distance_from_frontier.txt', 'a') as f: 
				print >> f, "%lf" % (average_min_distance(front, values_solution))

			with open('mean_min_distance_to_pareto_set.txt', 'a') as f: 
				print >> f, "%lf" % (average_min_distance(input_space.from_unit(solution), \
				input_space.from_unit(pareto_set)))

			with open('mean_min_distance_from_pareto_set.txt', 'a') as f: 
				print >> f, "%lf" % (average_min_distance(input_space.from_unit(pareto_set), \
				input_space.from_unit(solution)))

			with open('evaluations.txt','a') as f_handle: 
				np.savetxt(f_handle, np.array([recommendation['num_complete_tasks'].values()]), delimiter = ' ', newline = '\n')

		i += 1

if __name__ == '__main__':
	main(*sys.argv[1:])


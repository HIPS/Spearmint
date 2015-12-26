
# This class implements a multi objective optimizatoin problem that is solved via NSGA-II
# and the asynchronous island model.

# This class could be extended in the future to consider other algorithms for solving the problem
# It should receive in the constructor the models employed

import numpy as np
from PyGMO.problem import base
from PyGMO import *
from PyGMO.util import hypervolume
from spearmint.grids import sobol_grid

from spearmint.models.abstract_model import function_over_hypers
from spearmint.models.gp             import GP
from spearmint                       import models
import copy
from scipy.spatial.distance import cdist

# This class is used in the best function of the default chooser

class MOOP(base):

    """
    This class is a multi-objective optimization problem that can be solved via NSGA-II or a grid.
    """

    def __init__(self, objective_tasks, models, input_space, avg_over_hypers = True):

        # We call the base constructor with info about the problem

	self.models = models 
	self.tasks = objective_tasks
	self.input_space = input_space
	self.avg_over_hypers = avg_over_hypers

        super(MOOP, self).__init__(input_space.num_dims, 0, len(objective_tasks))

	# We set the bounds of the problem (unit hyper-cube)

        self.set_bounds(0.0, 1.0)

	# We specify the algorithm and its characteristics 

	self.alg = algorithm.nsga_II(gen = 1)

    def __deepcopy__(self, memo):

	return MOOP(self.tasks, self.models, self.input_space, self.avg_over_hypers)

    # Reimplement the virtual method that defines the objective function

    def _objfun_impl(self, x):

	x = np.array(x)
	x = x[ None, : ]

        # Compute the GP mean over the hyper-params

	mean = np.zeros(len(self.tasks))

	n_task = 0
	for key in self.tasks:
		if self.avg_over_hypers:
			mean[ n_task ] = float(self.models[ key ].function_over_hypers(self.models[ key ].predict, x)[ 0 ])
		else:
			mean[ n_task ] = float(self.models[ key ].predict(x)[ 0 ])

		n_task += 1

        return tuple(mean)

    def initialize_population(self, pop_size = 200):

	self.pop = population(self, pop_size)

    # This method solves the problem using a grid

    def solve_using_grid(self, grid = None):

	if grid is None:
		grid = sobol_grid.generate(self.input_space.num_dims, 20000)

	values = np.ones((grid.shape[ 0 ], len(self.models.keys())))

	n_task = 0
	for key in self.tasks:

		if self.avg_over_hypers:
			values[ :, n_task ] = self.models[ key ].function_over_hypers(self.models[ key ].predict, grid)[ 0 ]
		else:
			values[ :, n_task ] = self.models[ key ].predict(grid)[ 0 ]

		n_task += 1

	pareto_indices = _cull_algorithm(values)

	values = values[ pareto_indices, : ]
	grid = grid[ pareto_indices, : ]

	# We remove repeated entries from the pareto front and set

	frontier = values

	X = frontier[ 0 : 1, : ]
	pareto_set = grid[ 0 : 1, : ]

	for i in range(frontier.shape[ 0 ]):
		if np.min(cdist(frontier[ i : (i + 1), : ], X)) > 1e-8:
			X = np.vstack((X, frontier[ i, ])) 
			pareto_set = np.vstack((pareto_set, grid[ i, ])) 

	grid = pareto_set
	frontier = X

	self.pop = population(self, 0)

	for i in range(grid.shape[ 0 ]):
		self.pop.push_back(grid[ i, : ].tolist())

    def evolve(self, epochs = 100, pop_size = 100):
	
	self.pop = population(self, pop_size)

	for i in xrange(epochs):
		self.pop = self.alg.evolve(self.pop)

    # Initializes the population to some given values

    def set_population(self, values):
		
	self.pop = population(self, values.shape[ 0 ])

	for i in range(values.shape[ 0 ]):
		self.pop.set_x(i, values[ i, : ])

    def set_population_at_i(self, i, values):
		
	self.pop.set_x(i, values)

    # This method returns the current pareto front

    def compute_pareto_front_and_set(self):

	fronts = self.pop.compute_pareto_fronts()[ 0 ]

	return { 'pareto_set': np.array([ self.pop[ key ].best_x for key in fronts ]), \
		'frontier': np.array([ self.pop[ key ].best_f for key in fronts ]) }

    def compute_pareto_front_and_set_summary(self, size):

	fronts = self.pop.compute_pareto_fronts()[ 0 ]

	frontier = np.array([ self.pop[ key ].best_f for key in fronts ])
	pareto_set = np.array([ self.pop[ key ].best_x for key in fronts ])

	return _compute_pareto_front_and_set_summary_y_space(frontier, pareto_set, size)

    def append_to_population(self, values):

	self.pop.push_back(values.tolist())

    def evolve_population_only(self, epochs = 200):
	
	for i in xrange(epochs):
		self.pop = self.alg.evolve(self.pop)

    # This method returns the non-dominated observations. If there
    # are missing objective observations for an input we use the gps to predict them

    def get_non_dominated_observations_predict_missing_observations(self):

	values_final = None
	inputs = copy.deepcopy(self.models[ self.models.keys()[ 0 ] ].inputs)

	for key in self.models:
		for i in range(self.models[ key ].inputs.shape[ 0 ]):
			if np.min(cdist(self.models[ key ].inputs[ i : (i + 1), : ], inputs)) > 0:
				inputs = np.vstack((inputs, self.models[ key ].inputs[ i : (i + 1), : ]))

	inputs_models = dict()
	values_models = dict()

	for key in self.models:
		inputs_models[ key ] = copy.deepcopy(self.models[ key ].inputs)
		values_models[ key ] = copy.deepcopy(self.models[ key ].values)

	for key in inputs_models:
		i = 0
		while i < inputs_models[ key ].shape[ 0 ]:
			if np.min(cdist(inputs_models[ key ][ i : (i + 1), : ], inputs)) > 0:
				inputs_models[ key ] = np.delete(inputs_models[ key ], i, axis = 0)
				values_models[ key ] = np.delete(values_models[ key ], i, axis = 0)
			else:
				i += 1
		
	# We obtain the values and inputs (we consider the average value in the case of repeated inputs for an objective)

	for i in range(inputs.shape[ 0 ]):

		values_to_add = None

		for key in self.models:	

			values_tmp = None

			for j in range(inputs_models[ key ].shape[ 0 ]):
				if np.max(cdist(inputs_models[ key ][ j : (j + 1), : ], inputs[ i : (i + 1), : ])) == 0:
					if values_tmp == None:
						values_tmp = np.array(values_models[ key ][ j ])
					else:
						values_tmp = np.append(values_tmp , values_models[ key ][ j ])
			
			# If we have not found the point we compute a prediction

			if values_tmp == None:
				values_tmp = float(self.models[ key ].function_over_hypers(self.models[ key ].predict, \
					inputs[ i : (i + 1), : ])[ 0 ])

			if values_to_add == None:
				values_to_add = np.array([np.mean(values_tmp)])
			else:
				values_to_add = np.append(values_to_add, np.mean(values_tmp))

		if values_final == None:
			values_final = values_to_add.reshape((1 , len(values_to_add)))
		else:
			values_final = np.vstack((values_final, values_to_add.reshape((1 , len(values_to_add)))))

	# Now we only obtain the non-dominated points

	pareto_indices = self._cull_algorithm(values_final)

	return { 'pareto_set': inputs[ pareto_indices, ], 'frontier': values_final[ pareto_indices, ] }

    # This method returns the non-dominated observations

    def get_non_dominated_observations(self):

	values_final = None
	inputs = copy.deepcopy(self.models[ self.models.keys()[ 0 ] ].inputs)

	inputs_models = dict()
	values_models = dict()

	for key in self.models:
		inputs_models[ key ] = copy.deepcopy(self.models[ key ].inputs)
		values_models[ key ] = copy.deepcopy(self.models[ key ].values)

	i = 0

	# We only consider the inputs with all outputs observed

	while i < inputs.shape[ 0 ]:

		input_in_all_models = True

		for key in self.models:
#			if inputs[ i, ] not in self.models[ key ].inputs:
			if np.min(cdist(inputs[ i : (i + 1), : ], self.models[ key ].inputs)) > 0:
				input_in_all_models = False

		if input_in_all_models == False:
			inputs = np.delete(inputs, i, axis = 0)
		else:
			i += 1
	
	for key in inputs_models:
		i = 0
		while i < inputs_models[ key ].shape[ 0 ]:
#			if not inputs_models[ key ][ i, : ] in inputs:
			if np.min(cdist(inputs_models[ key ][ i : (i + 1), : ], inputs)) > 0:
				inputs_models[ key ] = np.delete(inputs_models[ key ], i, axis = 0)
				values_models[ key ] = np.delete(values_models[ key ], i, axis = 0)
			else:
				i += 1
		
	# We obtain the values and inputs (we consider the average value in the case of repeated inputs for an objective)

	for i in range(inputs.shape[ 0 ]):

		values_to_add = None

		for key in self.models:	

			values_tmp = None

			for j in range(inputs_models[ key ].shape[ 0 ]):
#				if inputs_models[ key ][ j, : ] in inputs[ i, : ]:
				if np.max(cdist(inputs_models[ key ][ j : (j + 1), : ], inputs[ i : (i + 1), : ])) == 0:
					if values_tmp == None:
						values_tmp = np.array(values_models[ key ][ j ])
					else:
						values_tmp = np.append(values_tmp , values_models[ key ][ j ])

			if values_to_add == None:
				values_to_add = np.array([np.mean(values_tmp)])
			else:
				values_to_add = np.append(values_to_add, np.mean(values_tmp))

		if values_final == None:
			values_final = values_to_add.reshape((1 , len(values_to_add)))
		else:
			values_final = np.vstack((values_final, values_to_add.reshape((1 , len(values_to_add)))))

	# Now we only obtain the non-dominated points

	pareto_indices = self._cull_algorithm(values_final)

	return { 'pareto_set': inputs[ pareto_indices, ], 'frontier': values_final[ pareto_indices, ] }

    def get_hypervolume(self, reference = None):

	if reference == None:
		reference = [1000.0] * len(self.tasks)
	
	hv = hypervolume(self.pop)
	return hv.compute(r = reference)

    def _cull_algorithm(self, inputPoints):

        return _cull_algorithm(inputPoints)
			
    # Add some output to __repr__

    def human_readable_extra(self):
        return "\n\tMulti-Objective problem"

# This class is used in the PESM function

funs = None

class MOOP_basis_functions(base):

	"""
	This class is a multi-objective optimization problem that can be solved via NSGA-II.
	"""

	def __init__(self, funs, num_dims):

		# We call the base constructor with info about the problem

		self.funs = funs
		self.num_dims = num_dims

		super(MOOP_basis_functions, self).__init__(num_dims, 0, len(funs))

		# We set the bounds of the problem (unit hyper-cube)

		self.set_bounds(0.0, 1.0)

		# We specify the algorithm and its characteristics (islands and archipelago)

		self.alg = algorithm.nsga_II(gen = 1)

	def __copy__(self):

		return MOOP_basis_functions(self.funs, self.num_dims)

	def __deepcopy__(self, memo):

		return MOOP_basis_functions(self.funs, self.num_dims)

	def __getinitargs__(self):

	        return (self.funs, self.num_dims)

	# Initializes the population to some given values

	def set_population(self, values):
		
		self.pop = population(self, values.shape[ 0 ])

		for i in range(values.shape[ 0 ]):
			self.pop.set_x(i, values[ i , : ])

	def get_population(self):
		
		values = np.zeros((len(self.pop), len(self.pop[ 0 ].cur_x)))

		for i in range(len(self.pop)):
			values[ i, ] = np.array(self.pop[ i ].cur_x)

		return values

	# Reimplement the virtual method that defines the objective function

	def _objfun_impl(self, x):

		x = np.array(x)
		x = x[ None, : ]

		# Compute the GP mean over the hyper-params

		values = np.zeros(len(self.funs))

		for i in range(len(self.funs)):
			values[ i ] = self.funs[ i ](x, gradient = False)

		return tuple(values)

	# This method evolves the population for a number of epocs to solve the problem

	def evolve(self, epochs = 200, pop_size = 200):
	
		self.pop = population(self, pop_size)

		for i in xrange(epochs):
			self.pop = self.alg.evolve(self.pop)

	# This method solves the problem using a grid

	def solve_using_grid(self, grid = None):

		if grid is None:
			grid = sobol_grid.generate(self.num_dims, 20000)

		values = np.ones((grid.shape[ 0 ], len(self.funs)))

		for i in range(len(self.funs)):
			values[ :, i ] = self.funs[ i ](grid, gradient = False)

		pareto_indices = _cull_algorithm(values)

		grid_to_add = grid[ pareto_indices, : ]
		values = values[ pareto_indices, : ]

		self.pop = population(self, 0)

		for i in range(grid_to_add.shape[ 0 ]):
			self.pop.push_back(grid_to_add[ i, : ].tolist())

	def initialize_population(self, pop_size = 200):

		self.pop = population(self, pop_size)

	def set_population_at_i(self, i, values):
		
		self.pop.set_x(i, values)

	def append_to_population(self, values):

		self.pop.push_back(values.tolist())

	# This method evolves the population for a number of epocs to solve the problem

	def evolve_population_only(self, epochs = 200):
	
		for i in xrange(epochs):
			self.pop = self.alg.evolve(self.pop)

	# This method returns the current pareto front

	def compute_pareto_front_and_set(self):

		fronts = self.pop.compute_pareto_fronts()[ 0 ]

		return { 'pareto_set': np.array([ self.pop[ key ].best_x for key in fronts ]), \
			'frontier': np.array([ self.pop[ key ].best_f for key in fronts ]) }

	def compute_pareto_front_and_set_summary(self, size):

		fronts = self.pop.compute_pareto_fronts()[ 0 ]

		frontier = np.array([ self.pop[ key ].best_f for key in fronts ])
		pareto_set = np.array([ self.pop[ key ].best_x for key in fronts ])

		return _compute_pareto_front_and_set_summary_y_space(frontier, pareto_set, size)

	def get_hypervolume(self, reference = None):

		if reference == None:
			reference = [ 1000.0 ] * len(self.funs)
	
		hv = hypervolume(self.pop)
		return hv.compute(r = reference)

	# Add some output to __repr__

	def human_readable_extra(self):
		return "\n\tMulti-Objective problem"


def _compute_pareto_front_and_set_summary_x_space(frontier, pareto_set, size):

	assert size > 0

	if size >= pareto_set.shape[ 0 ]:
		return { 'pareto_set': pareto_set, 'frontier': frontier }

	n_total = frontier.shape[ 0 ]

#	distances = np.zeros((n_total, n_total))

	distances = cdist(pareto_set, pareto_set)

#	for i in range(n_total):
#		for j in range(n_total):
#			distances[ i, j ] = np.sqrt(np.sum((pareto_set[ i, : ] - pareto_set[ j, : ])**2))

	subset = [ -1 for i in range(size) ]
	subset[ 0 ] = int(np.random.choice(range(n_total), 1)[ 0 ])

	n_chosen = 1

	while n_chosen < size:

		dist_tmp = np.zeros( n_total )
		
		for i in range(n_total):
			if i not in subset:
				dist_tmp[ i ] = np.min(distances[ subset, i ]) 

		to_include = 0
		dist_current = 0 

		for i in range(n_total):
			if i not in subset:
				if dist_tmp[ i ] > dist_current:
					to_include = i
					dist_current = dist_tmp[ i ]

		subset[ n_chosen ] = to_include
		n_chosen += 1

	return { 'pareto_set': pareto_set[ subset, : ], 'frontier': frontier[ subset, : ] }

def _compute_pareto_front_and_set_summary_y_space(frontier, pareto_set, size):

	assert size > 0

	if size >= pareto_set.shape[ 0 ]:
		return { 'pareto_set': pareto_set, 'frontier': frontier }

	n_total = frontier.shape[ 0 ]

#	distances = np.zeros((n_total, n_total))

	distances = cdist(frontier, frontier)

#	for i in range(n_total):
#		for j in range(n_total):
#			distances[ i, j ] = np.sqrt(np.sum((frontier[ i, : ] - frontier[ j, : ])**2))

	# Useful hack we add the best observations of each objective first

	subset = [ -1 for i in range(size) ]

	for i in range(frontier.shape[ 1 ]):
		subset[ i ] = np.argmin(frontier[ :, i ])

	n_chosen = frontier.shape[ 1 ]

	while n_chosen < size:

		dist_tmp = np.zeros( n_total )
		
		for i in range(n_total):
			if i not in subset:
				dist_tmp[ i ] = np.min(distances[ subset, i ]) 

		to_include = 0
		dist_current = 0 

		for i in range(n_total):
			if i not in subset:
				if dist_tmp[ i ] > dist_current:
					to_include = i
					dist_current = dist_tmp[ i ]

		subset[ n_chosen ] = to_include
		n_chosen += 1

	return { 'pareto_set': pareto_set[ subset, : ], 'frontier': frontier[ subset, : ] }



def _cull_algorithm_slow(inputPoints):

    # Simple cull algorithm
    # Reference http://www.es.ele.tue.nl/pareto/papers/date2007_paretocalculator_final.pdf

    def dominates(row, anotherRow):
        return all(row <= anotherRow) 

    original_inputs = copy.deepcopy(inputPoints)

    paretoPoints = np.array([])

    while len(inputPoints):

        candidateRow = inputPoints[ 0, : ]
        inputPoints = np.delete(inputPoints, (0), axis = 0)
        index = 0
        dominated = False

        while index < paretoPoints.shape[ 0 ]:

            row = paretoPoints[ index, : ]

            if dominates(candidateRow, row):
		paretoPoints = np.delete(paretoPoints, (index), axis = 0)
            elif dominates(row, candidateRow):
                dominated = True
                index += 1
            else:
                index += 1
                
        if not dominated:
	    if paretoPoints.shape[ 0 ] == 0:
		paretoPoints = np.hstack((np.array([]), candidateRow)).reshape((1, original_inputs.shape[ 1 ]))
	    else:
		paretoPoints = np.vstack((paretoPoints, candidateRow))

    indices = np.ones(original_inputs.shape[ 0 ]) == 1

    for i in range(original_inputs.shape[ 0 ]):
	if original_inputs[ i, : ] in paretoPoints:
		indices[ i ] = True
	else:
		indices[ i ] = False

    return indices 

##
# This function computes the average distance of one point to the closest point of a subset of the pareto front
#

def average_min_distance(approx_set, exact_set):
	
	n_approx = approx_set.shape[ 0 ]
	n_exact = exact_set.shape[ 0 ]

	mean_distance = 0.0

	for i in range(n_approx):

		min_distance = 1e300

		for j in range(n_exact):

			distance = np.sqrt(np.sum((approx_set[ i, : ] - exact_set[ j, : ])**2))

			if distance < min_distance:
				min_distance = distance

		mean_distance += min_distance

	return mean_distance / n_approx


def dominates(row, rowCandidate):
    return all(r <= rc for r, rc in zip(row, rowCandidate))

#def cull_fast(pts):

def _cull_algorithm(pts):
    pts = pts.tolist()
    dominated = []
    cleared = []
    remaining = pts
    while remaining:
        candidate = remaining[0]
        new_remaining = []
        for other in remaining[1:]:
            [new_remaining, dominated][dominates(candidate, other)].append(other)
        if not any(dominates(other, candidate) for other in new_remaining):
            cleared.append(candidate)
        else:
            dominated.append(candidate)
        remaining = new_remaining
    
    original_inputs = np.array(pts)

    indices = np.ones(original_inputs.shape[ 0 ]) == 1
    paretoPoints = np.array(cleared)

    for i in range(original_inputs.shape[ 0 ]):
	if original_inputs[ i, : ] in paretoPoints:
		indices[ i ] = True
	else:
		indices[ i ] = False

    return indices


# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.

import numpy          as np
import numpy.random   as npr
import scipy.stats    as sps
import scipy.linalg   as spla
import numpy.linalg   as npla
import scipy.optimize as spo
import copy
import traceback
import warnings
import sys

from spearmint.utils.moop            import MOOP
from collections import defaultdict
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.utils.numerics import logcdf_robust
from spearmint.models.gp import GP
from spearmint.utils.moop            import MOOP_basis_functions
from spearmint.utils.moop            import _cull_algorithm
import importlib
from spearmint.tasks.task         import Task
from spearmint.utils.hv import HyperVolume
from scipy.spatial.distance import cdist
from spearmint.grids                 import sobol_grid
from spearmint.utils.pbivnorm_interface import pbivnorm_easy_numpy_vectors
from spearmint.utils.pbivnorm_interface import pbivnorm_easy_numpy_floats

from spearmint.utils.moop import _cull_algorithm
from spearmint.models.abstract_model import function_over_hypers
import logging

NUM_POINTS_FRONTIER = 5
N_SAMPLES = 10
USE_GRID_ONLY = False
GRID_SIZE = 1000
NSGA_POP = 100
NSGA_EPOCHS = 100

SMSEGO_OPTION_DEFAULTS  = {
    'sur_pareto_set_size'      : 5,
    'sur_grid_size'      : 1000,
    'sur_nsga_epochs'      : 100,
    'sur_nsga_pop'      : 100,
    'sur_use_grid_only_to_solve_problem' : False,
    }



class SUR(AbstractAcquisitionFunction):

	def __init__(self, num_dims, verbose=True, input_space=None, grid=None, opt = None):

		global USE_GRID_ONLY
		global NUM_POINTS_FRONTIER
		global NSGA_EPOCHS
		global NSGA_POP

		# we want to cache these. we use a dict indexed by the state integer

		self.cached_information = dict()
		self.has_gradients = False
		self.num_dims = num_dims
		self.input_space = input_space
		self.options = opt

		self.options = SMSEGO_OPTION_DEFAULTS.copy()
		self.options.update(opt)

		NUM_POINTS_FRONTIER = self.options['sur_pareto_set_size']
		GRID_SIZE = self.options['sur_grid_size'] 
		USE_GRID_ONLY = self.options['sur_use_grid_only_to_solve_problem'] 
		NSGA_POP = self.options['sur_nsga_pop'] 
		NSGA_EPOCHS = self.options['sur_nsga_epochs'] 

	def acquisition(self, obj_model_dict, con_models_dict, cand, current_best, compute_grad, minimize=True, tasks=None):

		models = obj_model_dict.values()

		# make sure all models are at the same state

		assert len({model.state for model in models}) == 1, "Models are not all at the same state"

		assert not compute_grad 

		# We check if we have already computed the cells associated to the model and other stuff

		key = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ])

		if not key in self.cached_information:
			if len(obj_model_dict) == 2:
				self.cached_information[ key ] = self.compute_cell_information_2_objectives(obj_model_dict, cand)
			else:
				self.cached_information[ key ] = self.compute_cell_information(obj_model_dict)

		# We use the chooser to compute the expected improvement on the scalarized task

		if len(obj_model_dict) == 2:
			return self.compute_acquisition_2_objectives(cand, obj_model_dict, self.cached_information[ key ])
		else:
			return self.compute_acquisition(cand, obj_model_dict, self.cached_information[ key ])


	def find_optimum_gp(self, obj_model, grid = None):

		if grid is None:
			grid = self.grid

		# Compute the GP mean
	
		obj_mean, obj_var = obj_model.predict(grid, compute_grad = False)

		# find the min and argmin of the GP mean

		current_best_location = grid[np.argmin(obj_mean),:]
		best_ind = np.argmin(obj_mean)
		current_best_value = obj_mean[best_ind]

		def f(x):
			if x.ndim == 1:
				x = x[None,:]

			mn, var, mn_grad, var_grad = obj_model.predict(x, compute_grad = True)

			return (mn.flatten(), mn_grad.flatten())

		bounds = [(0.0,1.0)]*self.num_dims

		x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, current_best_location.copy(), bounds=bounds, disp=0)

		# make sure bounds were respected

		x_opt[x_opt > 1.0] = 1.0
		x_opt[x_opt < 0.0] = 0.0

		return x_opt


	# This method computes the cached information. It is based on the paper:
	# Hypervolume-based Expected Improvement: Monotonicity Properties and Exact Computation
	# Michael T.M. Emmerich Andr ́e H. Deutz Jan Willem Klinkenberg

	def compute_cell_information(self, obj_model_dict, cand):

		cached_information = dict()

		# First we obtain a sample from the Pareto Frontier of NUM_POINTS_FRONTIER

		moop = MOOP(obj_model_dict, obj_model_dict, self.input_space, False)
		
		grid = sobol_grid.generate(self.input_space.num_dims, self.input_space.num_dims * GRID_SIZE)

		if USE_GRID_ONLY == True:

			moop.solve_using_grid(grid)

			for i in range(len(obj_model_dict.keys())):
				result = self.find_optimum_gp(obj_model_dict[ obj_model_dict.keys()[ i ] ], grid)
				moop.append_to_population(result)
		else:

			assert NSGA_POP > len(obj_model_dict.keys()) + 1

			moop.solve_using_grid(grid)

			for i in range(len(obj_model_dict.keys())):
				result = self.find_optimum_gp(obj_model_dict[ obj_model_dict.keys()[ i ] ], grid)
				moop.append_to_population(result)

			pareto_set = moop.compute_pareto_front_and_set_summary(NSGA_POP)['pareto_set']

			moop.initialize_population(np.maximum(NSGA_POP - pareto_set.shape[ 0 ], 0))

			for i in range(pareto_set.shape[ 0 ]):
				moop.append_to_population(pareto_set[ i, : ])

			moop.evolve_population_only(NSGA_EPOCHS)

			for i in range(pareto_set.shape[ 0 ]):
				moop.append_to_population(pareto_set[ i, : ])

		result = moop.compute_pareto_front_and_set_summary(NUM_POINTS_FRONTIER)

		means_objectives = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].inputs.shape[ 0 ], len(obj_model_dict)))

		k = 0
		for obj in obj_model_dict:
			means_objectives[ :, k ] = obj_model_dict[ obj ].predict(obj_model_dict[ obj ].inputs)[ 0 ]
			k += 1

		# We add the non-dominated observations

		v_inf = np.ones((1, len(obj_model_dict))) * 1e2

		# We obtain the pareto frontier approximation which is added the observations

		frontier = result['frontier']
		frontier = np.vstack((frontier, means_objectives))
		frontier = frontier[ _cull_algorithm(frontier), ]

		# We remove repeated entries from the pareto front

		X = frontier[ 0 : 1, : ]

		for i in range(frontier.shape[ 0 ]):
			if np.min(cdist(frontier[ i : (i + 1), : ], X)) > 1e-8:
			    	X = np.vstack((X, frontier[ i, ]))
	
		frontier = X

		cached_information['frontier'] = frontier

		# We sort the entries in the pareto frontier

		frontier_sorted = np.vstack((-v_inf, cached_information['frontier'], v_inf))

		for i in range(len(obj_model_dict)):
			frontier_sorted[ :, i ] = np.sort(frontier_sorted[ :, i ])

		# Now we build the info associated to each cell

		n_repeat = (frontier_sorted.shape[ 0 ] - 1) ** frontier_sorted.shape[ 1 ]

		cached_information['cells'] = dict()

		added_cells = 0
		for i in range(n_repeat):

			cell = dict()

			indices = np.zeros(len(obj_model_dict))

			j = i

			for k in range(len(obj_model_dict)):
				indices[ k ] = j % (frontier_sorted.shape[ 0 ] - 1)
				j = np.floor(j / (frontier_sorted.shape[ 0 ] - 1))

			u = np.zeros(len(obj_model_dict))

			for k in range(len(obj_model_dict)):
				u[ k ] = frontier_sorted[ indices[ k ] + 1, k ]

			l = np.zeros(len(obj_model_dict))
				
			for k in range(len(obj_model_dict)):
				l[ k ] = frontier_sorted[ indices[ k ], k ]

			# If the cell is dominated we mark it

			is_dominated = False
			for k in range(frontier.shape[ 0 ]):
				if np.all(l >= frontier[ k, : ]):
					is_dominated = True

			cell['u'] = u
			cell['l'] = l
			cell['is_dominated'] = is_dominated
			cached_information['cells'][ str(added_cells) ] = cell
			added_cells += 1
			
		n_cells = added_cells

		cached_information['n_cells'] = n_cells
		cached_information['n_objectives'] = len(obj_model_dict)

		# We compute a sobol grid to approximate the integral over x. By default we use
		# 100 points per dimension (too much?)
		
		if not 'sur_points_per_dimension' in self.options.keys():
#	            	cached_information['grid'] = np.array(sobol_grid.generate(self.num_dims, 100 * self.num_dims))
	            	cached_information['grid'] = np.random.uniform(size = (100 * self.num_dims, self.num_dims))
		else:
#	            	cached_information['grid'] = np.array(sobol_grid.generate(self.num_dims, \
#				int(self.options['sur_points_per_dimension']) * self.num_dims))
	            	cached_information['grid'] = np.random.uniform(size = (int(self.options['sur_points_per_dimension']) \
				* self.num_dims, self.num_dims))

		# We obtain the samples needed for a Monte Carlo approximation of the objective
		# This is not used in practice because it is too innacurate

		cached_information['gauss_samples_grid'] = np.random.normal(size = (cached_information['grid'].shape[ 0 ], \
			len(obj_model_dict), N_SAMPLES))

		cached_information['gauss_sample_cand'] = np.random.normal(size = N_SAMPLES)

#		self.print_cell_info(cached_information)

		return cached_information

	# This method computes the cached information. 
	# This does the cell grouping

	def compute_cell_information_2_objectives(self, obj_model_dict, cand):

		assert len(obj_model_dict) == 2

		cached_information = dict()

		# First we obtain a sample from the Pareto Frontier of NUM_POINTS_FRONTIER

		moop = MOOP(obj_model_dict, obj_model_dict, self.input_space, False)
		
		grid = sobol_grid.generate(self.input_space.num_dims, self.input_space.num_dims * GRID_SIZE)

		if USE_GRID_ONLY == True:

			moop.solve_using_grid(grid)

			for i in range(len(obj_model_dict.keys())):
				result = self.find_optimum_gp(obj_model_dict[ obj_model_dict.keys()[ i ] ], grid)
				moop.append_to_population(result)
		else:

			assert NSGA_POP > len(obj_model_dict.keys()) + 1

			moop.solve_using_grid(grid)

			for i in range(len(obj_model_dict.keys())):
				result = self.find_optimum_gp(obj_model_dict[ obj_model_dict.keys()[ i ] ], grid)
				moop.append_to_population(result)

			pareto_set = moop.compute_pareto_front_and_set_summary(NSGA_POP)['pareto_set']

			moop.initialize_population(np.maximum(NSGA_POP - pareto_set.shape[ 0 ], 0))

			for i in range(pareto_set.shape[ 0 ]):
				moop.append_to_population(pareto_set[ i, : ])

			moop.evolve_population_only(NSGA_EPOCHS)

			for i in range(pareto_set.shape[ 0 ]):
				moop.append_to_population(pareto_set[ i, : ])

		result = moop.compute_pareto_front_and_set_summary(NUM_POINTS_FRONTIER)

		# First we obtain a sample from the Pareto Frontier of NUM_POINTS_FRONTIER

		means_objectives = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].inputs.shape[ 0 ], len(obj_model_dict)))

		k = 0
		for obj in obj_model_dict:
			means_objectives[ :, k ] = obj_model_dict[ obj ].predict(obj_model_dict[ obj ].inputs)[ 0 ]
			k += 1


		v_inf = np.ones((1, len(obj_model_dict))) * 1e3

		# We obtain the pareto frontier approximation which is added the observations

		frontier = result['frontier']
		frontier = np.vstack((frontier, means_objectives))
		frontier = frontier[ _cull_algorithm(frontier), ]

		# We remove repeated entries from the pareto front

		X = frontier[ 0 : 1, : ]


		for i in range(frontier.shape[ 0 ]):
			if np.min(cdist(frontier[ i : (i + 1), : ], X)) > 1e-8:
			    	X = np.vstack((X, frontier[ i, ])) 
	
		frontier = X

		cached_information['frontier'] = frontier

		# We sort the entries in the pareto frontier

		frontier_sorted = np.vstack((-v_inf, cached_information['frontier'], v_inf))

		frontier_sorted[ :, 0 ] = np.sort(frontier_sorted[ :, 0 ])
		frontier_sorted[ :, 1 ] = np.sort(frontier_sorted[ :, 1 ] * -1.0) * -1.0

		# Now we build the info associated to each cell

		n_cells = (frontier_sorted.shape[ 0 ] - 1) + (frontier_sorted.shape[ 0 ] - 2)

		# We add first the non dominated cells

		cached_information['cells'] = dict()
		added_cells = 0

		for i in range(frontier_sorted.shape[ 0 ] - 1):

			cell = dict()
			cell['l'] = np.array([ frontier_sorted[ i, 0 ], frontier_sorted[ 0, 0 ] ])
			cell['u'] = np.array([ frontier_sorted[ i + 1, 0 ], frontier_sorted[ i, 1 ] ])
			cell['is_dominated'] = False

			cached_information['cells'][ str(added_cells) ] = cell
			added_cells += 1
	
#		# Now the dominated cells
#
#		for i in range(frontier_sorted.shape[ 0 ] - 2):
#
#			cell = dict()
#			cell['l'] = np.array([ frontier_sorted[ i + 1, 0 ], frontier_sorted[ i + 1, 1 ] ])
#			cell['u'] = np.array([ frontier_sorted[ frontier_sorted.shape[ 0 ] - 1, 0 ], frontier_sorted[ i, 1 ] ])
#			cell['is_dominated'] = True
#			cached_information['cells'][ str(added_cells) ] = cell
#			added_cells += 1

#		cached_information['n_cells'] = n_cells

		cached_information['n_cells'] = added_cells
		cached_information['n_objectives'] = len(obj_model_dict)

		# We compute a sobol grid to approximate the integral over x. By default we use
		# 100 points per dimension (too much?)
		
		if not 'sur_points_per_dimension' in self.options.keys():
#	            	cached_information['grid'] = np.array(sobol_grid.generate(self.num_dims, 100 * self.num_dims))
	            	cached_information['grid'] = np.random.uniform(size = (100 * self.num_dims, self.num_dims))
		else:
#	            	cached_information['grid'] = np.array(sobol_grid.generate(self.num_dims, \
#				int(self.options['sur_points_per_dimension']) * self.num_dims))
	            	cached_information['grid'] = np.random.uniform(size = (int(self.options['sur_points_per_dimension']) \
				 * self.num_dims, self.num_dims))

		# We obtain the samples needed for a Monte Carlo approximation of the objective

		cached_information['gauss_samples_grid'] = np.random.normal(size = (cached_information['grid'].shape[ 0 ], \
			len(obj_model_dict), N_SAMPLES))

		cached_information['gauss_sample_cand'] = np.random.normal(size = N_SAMPLES)

#		self.print_cell_info(cached_information)

		return cached_information

	# This method is the one that actually does the computation of the acquisition_function
	
	def compute_acquisition(self, cand, obj_model_dict, information):

		n_objectives = information['n_objectives']

		# We obtain the predictive means and variances for the candiate points and the sobol_grid

		Xgrid = information['grid']

		meanXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		varXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		meanCand = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		varCand = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		covCandXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))

		k = 0
		for obj in obj_model_dict:

			meanXgrid_k, varXgrid_k = obj_model_dict[ obj ].predict(Xgrid)
			meanCand_k, varCand_k = obj_model_dict[ obj ].predict(cand)

			meanXgrid[ :, :, k ] = np.tile(meanXgrid_k, cand.shape[ 0 ]).reshape((cand.shape[ 0 ], Xgrid.shape[ 0 ]))
			varXgrid[ :, :, k ] = np.tile(varXgrid_k, cand.shape[ 0 ]).reshape((cand.shape[ 0 ], Xgrid.shape[ 0 ]))
			meanCand[ :, :, k ] = np.tile(meanCand_k, Xgrid.shape[ 0 ]).reshape((Xgrid.shape[ 0 ], cand.shape[ 0 ])).T
			varCand[ :, :, k ] = np.tile(varCand_k, Xgrid.shape[ 0 ]).reshape((Xgrid.shape[ 0 ], cand.shape[ 0 ])).T

			# We get the covariances in an efficient way 

			cholKstarstar   = spla.cholesky(obj_model_dict[ obj ].kernel.cov(obj_model_dict[ obj ].inputs))
			Kstarstar = obj_model_dict[ obj ].noiseless_kernel.cross_cov(cand, Xgrid)
			Kstar1 = obj_model_dict[ obj ].noiseless_kernel.cross_cov(cand, obj_model_dict[ obj ].inputs)
			Kstar2 = obj_model_dict[ obj ].noiseless_kernel.cross_cov(Xgrid, obj_model_dict[ obj ].inputs)
			aux1 = spla.solve_triangular(cholKstarstar.T, Kstar1.T, lower=True)
			aux2 = spla.solve_triangular(cholKstarstar.T, Kstar2.T, lower=True)
			covCandXgrid[ :, :, k ] = Kstarstar - np.dot(aux1.T, aux2)

			k += 1

		n_cells = information['n_cells']

		total_acquisition = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ]))

		for j in range(n_cells):

			cell_j = information['cells'][ str(j) ]

			if cell_j['is_dominated'] == True:
				continue

			for i in range(n_cells):

				cell_i = information['cells'][ str(i) ]

				l_i = cell_i['l']
				l_j = cell_j['l']
				u_i = cell_i['u']
				u_j = cell_j['u']

				# If the cell_i dominates the cell_j we do nothing

				if np.all(l_i < l_j):
					continue
	
				elif np.any(l_i <= l_j):

					# We look for partial domination

					acq_cell_1 = np.ones((cand.shape[ 0 ], Xgrid.shape[ 0 ]))

					for k in range(n_objectives):
						acq_cell_1 *= self.bij(k, cell_i, cell_j, meanXgrid[ :, :, k ], varXgrid[ :, :, k ], \
							meanCand[ :, :, k ], varCand[ :, :, k ], covCandXgrid[ :, :, k ])

					acq_cell_2 = np.ones((cand.shape[ 0 ], Xgrid.shape[ 0 ]))

					for k in range(n_objectives):
						acq_cell_2 *= self.deltaij(k, cell_i, cell_j, meanXgrid[ :, :, k ], varXgrid[ :, :, k ], \
							meanCand[ :, :, k ], varCand[ :, :, k ], covCandXgrid[ :, :, k ])

					acq_cell = acq_cell_1 - acq_cell_2

				else:
					# We look for non domination

					acq_cell = np.ones((cand.shape[ 0 ], Xgrid.shape[ 0 ]))

					for k in range(n_objectives):
						acq_cell *= self.bij(k, cell_i, cell_j, meanXgrid[ :, :, k ], varXgrid[ :, :, k ], \
							meanCand[ :, :, k ], varCand[ :, :, k ], covCandXgrid[ :, :, k ])

				total_acquisition += acq_cell
				sys.stdout.write(".")
				sys.stdout.flush()
		print ''	

		return -1.0 * np.mean(total_acquisition, axis = 1)

	# This method is the one that actually does the computation of the acquisition_function using monte_carlo methods
	
	def compute_acquisition_monte_carlo(self, cand, obj_model_dict, information):

		n_objectives = information['n_objectives']

		# We obtain the predictive means and variances for the candiate points and the sobol_grid

		Xgrid = information['grid']

		meanXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		varXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		meanCand = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		varCand = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		covCandXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))

		k = 0
		for obj in obj_model_dict:

			meanXgrid_k, varXgrid_k = obj_model_dict[ obj ].predict(Xgrid)
			meanCand_k, varCand_k = obj_model_dict[ obj ].predict(cand)

			meanXgrid[ :, :, k ] = np.tile(meanXgrid_k, cand.shape[ 0 ]).reshape((cand.shape[ 0 ], Xgrid.shape[ 0 ]))
			varXgrid[ :, :, k ] = np.tile(varXgrid_k, cand.shape[ 0 ]).reshape((cand.shape[ 0 ], Xgrid.shape[ 0 ]))
			meanCand[ :, :, k ] = np.tile(meanCand_k, Xgrid.shape[ 0 ]).reshape((Xgrid.shape[ 0 ], cand.shape[ 0 ])).T
			varCand[ :, :, k ] = np.tile(varCand_k, Xgrid.shape[ 0 ]).reshape((Xgrid.shape[ 0 ], cand.shape[ 0 ])).T

			# We get the covariances in an efficient way 

			cholKstarstar   = spla.cholesky(obj_model_dict[ obj ].kernel.cov(obj_model_dict[ obj ].inputs))
			Kstarstar = obj_model_dict[ obj ].noiseless_kernel.cross_cov(cand, Xgrid)
			Kstar1 = obj_model_dict[ obj ].noiseless_kernel.cross_cov(cand, obj_model_dict[ obj ].inputs)
			Kstar2 = obj_model_dict[ obj ].noiseless_kernel.cross_cov(Xgrid, obj_model_dict[ obj ].inputs)
			aux1 = spla.solve_triangular(cholKstarstar.T, Kstar1.T, lower=True)
			aux2 = spla.solve_triangular(cholKstarstar.T, Kstar2.T, lower=True)
			covCandXgrid[ :, :, k ] = Kstarstar - np.dot(aux1.T, aux2)

			k += 1

		frontier = information['frontier']
		acq_to_return = np.zeros(cand.shape[ 0 ])

		for s in range(N_SAMPLES):

			total_acquisition = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ])) != 0

			observations_grid = np.tile(information['gauss_samples_grid'][ :, :, s ], cand.shape[ 0 ]).reshape((cand.shape[ 0 ], \
				Xgrid.shape[ 0 ], len(obj_model_dict)))
			observations_grid = observations_grid * np.sqrt(varXgrid) + meanXgrid
			observations_cand = information['gauss_sample_cand'][ s ] * np.ones(observations_grid.shape)

			# We sample from the conditional distribution

			observations_cand = np.sqrt(varCand - covCandXgrid**2 * 1.0 / varXgrid) + \
					(meanCand + covCandXgrid * 1.0 / varXgrid * (observations_grid - meanXgrid))

			# We look for improvements

			for j in range(frontier.shape[ 0 ]):
				for k in range(frontier.shape[ 1 ]):
					total_acquisition = np.logical_or(total_acquisition, (observations_grid[ :, :, k ] < frontier[ j, k ]))

			acq_to_return += -1.0 * np.mean(total_acquisition, axis = 1)

		acq_to_return /= N_SAMPLES

		return acq_to_return

	# This method is the one that actually does the computation of the acquisition_function
	
	def compute_acquisition_2_objectives(self, cand, obj_model_dict, information):

		n_objectives = information['n_objectives']

		# We obtain the predictive means and variances for the candiate points and the sobol_grid

		Xgrid = information['grid']

		meanXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		varXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		meanCand = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		varCand = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))
		covCandXgrid = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ], n_objectives))

		k = 0
		for obj in obj_model_dict:

			meanXgrid_k, varXgrid_k = obj_model_dict[ obj ].predict(Xgrid)
			meanCand_k, varCand_k = obj_model_dict[ obj ].predict(cand)

			meanXgrid[ :, :, k ] = np.tile(meanXgrid_k, cand.shape[ 0 ]).reshape((cand.shape[ 0 ], Xgrid.shape[ 0 ]))
			varXgrid[ :, :, k ] = np.tile(varXgrid_k, cand.shape[ 0 ]).reshape((cand.shape[ 0 ], Xgrid.shape[ 0 ]))
			meanCand[ :, :, k ] = np.tile(meanCand_k, Xgrid.shape[ 0 ]).reshape((Xgrid.shape[ 0 ], cand.shape[ 0 ])).T
			varCand[ :, :, k ] = np.tile(varCand_k, Xgrid.shape[ 0 ]).reshape((Xgrid.shape[ 0 ], cand.shape[ 0 ])).T

			# We get the covariances in an efficient way 

			cholKstarstar   = spla.cholesky(obj_model_dict[ obj ].kernel.cov(obj_model_dict[ obj ].inputs))
			Kstarstar = obj_model_dict[ obj ].noiseless_kernel.cross_cov(cand, Xgrid)
			Kstar1 = obj_model_dict[ obj ].noiseless_kernel.cross_cov(cand, obj_model_dict[ obj ].inputs)
			Kstar2 = obj_model_dict[ obj ].noiseless_kernel.cross_cov(Xgrid, obj_model_dict[ obj ].inputs)
			aux1 = spla.solve_triangular(cholKstarstar.T, Kstar1.T, lower=True)
			aux2 = spla.solve_triangular(cholKstarstar.T, Kstar2.T, lower=True)
			covCandXgrid[ :, :, k ] = Kstarstar - np.dot(aux1.T, aux2)

			k += 1

		n_cells = information['n_cells']

		total_acquisition = np.zeros((cand.shape[ 0 ], Xgrid.shape[ 0 ]))

		frontier_sorted = information['frontier']
		frontier_sorted[ :, 0 ] = np.sort(frontier_sorted[ :, 0 ])
		frontier_sorted[ :, 1 ] = np.sort(frontier_sorted[ :, 1 ] * -1.0) * - 1.0

		rho = covCandXgrid / np.sqrt(varXgrid * varCand)
		eta = (meanCand - meanXgrid) / np.sqrt(varCand + varXgrid - 2 * covCandXgrid)
		nu = (covCandXgrid - varCand) / (np.sqrt(varCand) * np.sqrt(varXgrid + varCand - 2 * covCandXgrid))

		for i in range(frontier_sorted.shape[ 0 ] + 1):

			# We compute required quantities

			if i == 0:

				y_overline_1 = (frontier_sorted[ i , 0 ] - meanCand[ :, :, 0 ]) / np.sqrt(varCand[ :, :, 0])
				y_tilde_1 = (frontier_sorted[ i , 0 ] - meanXgrid[ :, :, 0 ]) / np.sqrt(varXgrid[ :, :, 0])

				total_acquisition += (pbivnorm_easy_numpy_vectors(y_overline_1, y_tilde_1, rho[ :, :, 0 ]) - \
					pbivnorm_easy_numpy_vectors(y_overline_1, eta[ :, :, 0 ], nu[ :, :, 0 ])) * \
					(np.exp(sps.norm.logcdf(eta[ :, :, 1 ])) - 1) + np.exp(sps.norm.logcdf(y_tilde_1))

			elif i > 0 and i < frontier_sorted.shape[ 0 ]:

				y_overline_1 = (frontier_sorted[ i - 1 , 0 ] - meanCand[ :, :, 0 ]) / np.sqrt(varCand[ :, :, 0])
				y_overline_2 = (frontier_sorted[ i - 1 , 1 ] - meanCand[ :, :, 1 ]) / np.sqrt(varCand[ :, :, 1])
				y_tilde_1 = (frontier_sorted[ i - 1, 0 ] - meanXgrid[ :, :, 0 ]) / np.sqrt(varXgrid[ :, :, 0])
				y_tilde_2 = (frontier_sorted[ i - 1, 1 ] - meanXgrid[ :, :, 1 ]) / np.sqrt(varXgrid[ :, :, 1])
				y_overline_1_plus = (frontier_sorted[ i , 0 ] - meanCand[ :, :, 0 ]) / np.sqrt(varCand[ :, :, 0])
				y_tilde_1_plus = (frontier_sorted[ i, 0 ] - meanXgrid[ :, :, 0 ]) / np.sqrt(varXgrid[ :, :, 0])

				total_acquisition += (pbivnorm_easy_numpy_vectors(y_overline_1_plus, y_tilde_1_plus, rho[ :, :, 0 ]) - \
					pbivnorm_easy_numpy_vectors(y_overline_1_plus, eta[ :, :, 0 ], nu[ :, :, 0 ]) + \
					pbivnorm_easy_numpy_vectors(y_overline_1, eta[ :, :, 0 ], nu[ :, :, 0 ]) - \
					pbivnorm_easy_numpy_vectors(y_overline_1, y_tilde_1, rho[ :, :, 0 ])) * \
					(pbivnorm_easy_numpy_vectors(y_overline_2, eta[ :, :, 1 ], nu[ :, :, 1 ]) - \
					pbivnorm_easy_numpy_vectors(y_overline_2, y_tilde_2, rho[ :, :, 1 ])) + \
					(np.exp(sps.norm.logcdf(y_tilde_1_plus)) - np.exp(sps.norm.logcdf(y_tilde_1))) * \
					np.exp(sps.norm.logcdf(y_tilde_2))
			else:
				y_overline_1 = (frontier_sorted[ i - 1 , 0 ] - meanCand[ :, :, 0 ]) / np.sqrt(varCand[ :, :, 0])
				y_overline_2 = (frontier_sorted[ i - 1 , 1 ] - meanCand[ :, :, 1 ]) / np.sqrt(varCand[ :, :, 1])
				y_tilde_1 = (frontier_sorted[ i - 1, 0 ] - meanXgrid[ :, :, 0 ]) / np.sqrt(varXgrid[ :, :, 0])
				y_tilde_2 = (frontier_sorted[ i - 1, 1 ] - meanXgrid[ :, :, 1 ]) / np.sqrt(varXgrid[ :, :, 1])

				total_acquisition += (1 - np.exp(sps.norm.logcdf(eta[ :, :, 0 ])) + \
					pbivnorm_easy_numpy_vectors(y_overline_1, eta[ :, :, 0 ], nu[ :, :, 0 ]) - \
					pbivnorm_easy_numpy_vectors(y_overline_1, y_tilde_1, rho[ :, :, 0 ])) * (\
					pbivnorm_easy_numpy_vectors(y_overline_2, eta[ :, :, 1 ], nu[ :, :, 1 ]) - \
					pbivnorm_easy_numpy_vectors(y_overline_2, y_tilde_2, rho[ :, :, 1 ])) + \
					(1 - np.exp(sps.norm.logcdf(y_tilde_1))) * np.exp(sps.norm.logcdf(y_tilde_2)) # DHL: This line is wrong
					# in the paper !!!!

			sys.stdout.write(".")
			sys.stdout.flush()

		print ''	

#		value = self.approximate_probability_via_monte_carlo(information['frontier'], meanCand, meanXgrid, varCand, \
#			varXgrid, covCandXgrid, 15607, 0, n_objectives, n_samples = 1e6)

#		value = self.approximate_probability_via_monte_carlo_conditional(information['frontier'], meanCand, meanXgrid, varCand, \
#			varXgrid, covCandXgrid, 15607, 0, n_objectives, n_samples = 1e6)

		return -1.0 * np.mean(total_acquisition, axis = 1)

	def bij(self, k, cell_i, cell_j, meanXgrid, varXgrid, meanCand, varCand, covCandXgrid):

		rho = covCandXgrid / np.sqrt(varXgrid * varCand)

		y_overline_i_plus = (cell_i['u'][ k ] - meanCand) / np.sqrt(varCand)
		y_overline_i_minus = (cell_i['l'][ k ] - meanCand) / np.sqrt(varCand)
		y_tilde_j_plus = (cell_j['u'][ k ] - meanXgrid) / np.sqrt(varXgrid)
		y_tilde_j_minus = (cell_j['l'][ k ] - meanXgrid) / np.sqrt(varXgrid)

		return 	pbivnorm_easy_numpy_vectors(y_overline_i_plus, y_tilde_j_plus, rho) - \
			pbivnorm_easy_numpy_vectors(y_overline_i_plus, y_tilde_j_minus, rho) - \
			pbivnorm_easy_numpy_vectors(y_overline_i_minus, y_tilde_j_plus, rho) + \
			pbivnorm_easy_numpy_vectors(y_overline_i_minus, y_tilde_j_minus, rho)  

	def deltaij(self, k, cell_i, cell_j, meanXgrid, varXgrid, meanCand, varCand, covCandXgrid):

		# We check if it is always the case that Yn+1 < Yn 

		if cell_i[ 'l' ][ k ] <  cell_j[ 'l' ][ k ]:
			return self.bij(k, cell_i, cell_j, meanXgrid, varXgrid, meanCand, varCand, covCandXgrid)

		rho = covCandXgrid / np.sqrt(varXgrid * varCand)
		nu = (covCandXgrid - varCand) / (np.sqrt(varCand) * np.sqrt(varXgrid + varCand - 2 * covCandXgrid))
		eta = (meanCand - meanXgrid) / np.sqrt(varCand + varXgrid - 2 * covCandXgrid)
		
		y_overline_i_plus = (cell_i['u'][ k ] - meanCand) / np.sqrt(varCand)
		y_overline_i_minus = (cell_i['l'][ k ] - meanCand) / np.sqrt(varCand)
		y_tilde_j_plus = (cell_j['u'][ k ] - meanXgrid) / np.sqrt(varXgrid)

		# We check if it is actually possible that Yn+1 (cell i) < Yn  (cell j) on this objective. If not
		# we return 0

		if cell_j['l'][ k ] < cell_i['l'][ k ]:

			return np.zeros(y_overline_i_minus.shape)

		else:
			return  (pbivnorm_easy_numpy_vectors(y_overline_i_plus, y_tilde_j_plus, rho) - \
				pbivnorm_easy_numpy_vectors(y_overline_i_plus, eta, nu) + \
				pbivnorm_easy_numpy_vectors(y_overline_i_minus, eta, nu) - \
				pbivnorm_easy_numpy_vectors(y_overline_i_minus, y_tilde_j_plus, rho)) 

	def print_cell_info(self, information):
	
		import matplotlib.pyplot as plt
		import matplotlib.patches as patches
		from matplotlib.pyplot import text

		assert information['n_objectives'] == 2

		fig = plt.figure()
		plt.plot(information['frontier'][ :,0 ], information['frontier'][ :,1 ], color='black', 
			marker='x', markersize=10, linestyle='None')

		for i in range(information['n_cells']):
			cell = information['cells'][str(i)]
			u = cell['u']
			l = cell['l']
		
			# We draw the points

			plt.plot(u[ 0 ], u[ 1 ], color='blue', marker='o', markersize=10, linestyle='None')
			plt.plot(l[ 0 ], l[ 1 ], color='red', marker='o', markersize=10, linestyle='None')

			# We draw the lines

			plt.plot(np.linspace(l[ 0 ], u[ 0 ], 100), np.ones(100) * l[ 1 ], color = 'b', marker='.', markersize=1)
			plt.plot(np.linspace(l[ 0 ], u[ 0 ], 100), np.ones(100) * u[ 1 ], color = 'b', marker='.', markersize=1)
			plt.plot(np.ones(100) * l[ 0 ], np.linspace(l[ 1 ], u[ 1 ], 100), color = 'b', marker='.', markersize=1)
			plt.plot(np.ones(100) * u[ 0 ], np.linspace(l[ 1 ], u[ 1 ], 100), color = 'b', marker='.', markersize=1)

			# We draw the cell number

			text((l[ 0 ] + u[ 0 ]) / 2, (l[ 1 ] + u[ 1 ]) / 2, str(i))

		plt.show()

	# This computes the probability of improving at a point (obtaining a non-dominated observations) given 
	# that we are evaluating at another. It is used to check that the method is OK.

	def approximate_probability_via_monte_carlo(self, frontier, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		for k in range(n_objectives):

			C = np.zeros((2, 2))
			m = np.zeros(2)
		
			C[ 0, 0 ] = varCand[ i_cand, i_grid, k ]
			C[ 1, 1 ] = varXgrid[ i_cand, i_grid, k ]
			C[ 1, 0 ] = covCandXgrid[ i_cand, i_grid, k ]
			C[ 0, 1 ] = covCandXgrid[ i_cand, i_grid, k ]
			m[ 0 ] = meanCand[ i_cand, i_grid, k ]
			m[ 1 ] = meanXgrid[ i_cand, i_grid, k ]
			L = spla.cholesky(C)
			
			samples = np.dot(np.random.normal(size = 2 * n_samples).reshape((n_samples, 2)), L) + \
				np.tile(m, n_samples).reshape((n_samples, 2))

			samples_cand[ k, : ] = samples[ :, 0 ]
			samples_grid[ k, : ] = samples[ :, 1 ]

		counts = 0.0

		for i in range(int(n_samples)):

			dominates = True
			for j in range(frontier.shape[ 0 ]):
				if np.all(samples_grid[ :, i ] > frontier[ j, : ]):
					dominates = False

			if np.all(samples_grid[ :, i ] > samples_cand[ :,  i ]):
				dominates = False
				
			if dominates == True:
				counts += 1.0

		return counts / n_samples

	def approximate_probability_via_monte_carlo_conditional(self, frontier, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		for k in range(n_objectives):

			samples_cand[ k, : ]= np.random.normal(size = n_samples) * np.sqrt(varCand[ i_cand, i_grid, k ]) + \
				meanCand[ i_cand, i_grid, k ]

		for k in range(n_objectives):

			samples_grid[ k, : ] = np.random.normal(size = n_samples) * np.sqrt(varXgrid[ i_cand, i_grid, k ] - 
				covCandXgrid[ i_cand, i_grid, k ]**2 * 1.0 / varCand[ i_cand, i_grid, k ]) + \
				(meanXgrid[ i_cand, i_grid, k ] + covCandXgrid[ i_cand, i_grid, k ] * \
				1.0 / varCand[ i_cand, i_grid, k ] * (samples_cand[ k, : ] - meanCand[ i_cand, i_grid, k ]))

		counts = 0.0

		for i in range(int(n_samples)):

			dominates = True
			for j in range(frontier.shape[ 0 ]):
				if np.all(samples_grid[ :, i ] > frontier[ j, : ]):
					dominates = False

			if np.all(samples_grid[ :, i ] > samples_cand[ :,  i ]):
				dominates = False
				
			if dominates == True:
				counts += 1.0

		return counts / n_samples

	def approximate_bij_via_monte_carlo(self, k, cell_i, cell_j, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		C = np.zeros((2, 2))
		m = np.zeros(2)
	
		C[ 0, 0 ] = varCand[ i_cand, i_grid, k ]
		C[ 1, 1 ] = varXgrid[ i_cand, i_grid, k ]
		C[ 1, 0 ] = covCandXgrid[ i_cand, i_grid, k ]
		C[ 0, 1 ] = covCandXgrid[ i_cand, i_grid, k ]
		m[ 0 ] = meanCand[ i_cand, i_grid, k ]
		m[ 1 ] = meanXgrid[ i_cand, i_grid, k ]
		L = spla.cholesky(C)
		
		samples = np.dot(np.random.normal(size = 2 * n_samples).reshape((n_samples, 2)), L) + \
			np.tile(m, n_samples).reshape((n_samples, 2))

		samples_cand[ k, : ] = samples[ :, 0 ]
		samples_grid[ k, : ] = samples[ :, 1 ]

		counts = 0.0

		valid_samples = np.ones(n_samples) == 1.0

		valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] <= cell_j['u'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] <= cell_i['u'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] >= cell_j['l'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] >= cell_i['l'][ k ])

		return np.mean(valid_samples)

	def approximate_deltaij_via_monte_carlo(self, k, cell_i, cell_j, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		C = np.zeros((2, 2))
		m = np.zeros(2)
	
		C[ 0, 0 ] = varCand[ i_cand, i_grid, k ]
		C[ 1, 1 ] = varXgrid[ i_cand, i_grid, k ]
		C[ 1, 0 ] = covCandXgrid[ i_cand, i_grid, k ]
		C[ 0, 1 ] = covCandXgrid[ i_cand, i_grid, k ]
		m[ 0 ] = meanCand[ i_cand, i_grid, k ]
		m[ 1 ] = meanXgrid[ i_cand, i_grid, k ]
		L = spla.cholesky(C)
		
		samples = np.dot(np.random.normal(size = 2 * n_samples).reshape((n_samples, 2)), L) + \
			np.tile(m, n_samples).reshape((n_samples, 2))

		samples_cand[ k, : ] = samples[ :, 0 ]
		samples_grid[ k, : ] = samples[ :, 1 ]

		counts = 0.0

		valid_samples = np.ones(n_samples) == 1.0

		valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] <= cell_j['u'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] <= cell_i['u'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] >= cell_j['l'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] >= cell_i['l'][ k ])
		valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] <= samples_grid[ k, : ])

		return np.mean(valid_samples)

	def approximate_neg_deltaij_via_monte_carlo(self, cell_i, cell_j, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		for k in range(n_objectives):

			C = np.zeros((2, 2))
			m = np.zeros(2)
		
			C[ 0, 0 ] = varCand[ i_cand, i_grid, k ]
			C[ 1, 1 ] = varXgrid[ i_cand, i_grid, k ]
			C[ 1, 0 ] = covCandXgrid[ i_cand, i_grid, k ]
			C[ 0, 1 ] = covCandXgrid[ i_cand, i_grid, k ]
			m[ 0 ] = meanCand[ i_cand, i_grid, k ]
			m[ 1 ] = meanXgrid[ i_cand, i_grid, k ]
			L = spla.cholesky(C)
			
			samples = np.dot(np.random.normal(size = 2 * n_samples).reshape((n_samples, 2)), L) + \
				np.tile(m, n_samples).reshape((n_samples, 2))

			samples_cand[ k, : ] = samples[ :, 0 ]
			samples_grid[ k, : ] = samples[ :, 1 ]

		counts = 0.0

		valid_samples = np.ones(n_samples) == 1.0

		for k in range(n_objectives):
			valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] <= cell_j['u'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] <= cell_i['u'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] >= cell_j['l'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] >= cell_i['l'][ k ])

		valid_samples_delta  = np.ones(n_samples) != 1.0

		for k in range(n_objectives):
			valid_samples_delta = np.logical_or(valid_samples_delta, samples_cand[ k, : ] >= samples_grid[ k, : ])

		valid_samples = np.logical_and(valid_samples, valid_samples_delta)

		return np.mean(valid_samples)

	def approximate_cell_prob_via_monte_carlo(self, cell_i, cell_j, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		for k in range(n_objectives):

			C = np.zeros((2, 2))
			m = np.zeros(2)
		
			C[ 0, 0 ] = varCand[ i_cand, i_grid, k ]
			C[ 1, 1 ] = varXgrid[ i_cand, i_grid, k ]
			C[ 1, 0 ] = covCandXgrid[ i_cand, i_grid, k ]
			C[ 0, 1 ] = covCandXgrid[ i_cand, i_grid, k ]
			m[ 0 ] = meanCand[ i_cand, i_grid, k ]
			m[ 1 ] = meanXgrid[ i_cand, i_grid, k ]
			L = spla.cholesky(C)
			
			samples = np.dot(np.random.normal(size = 2 * n_samples).reshape((n_samples, 2)), L) + \
				np.tile(m, n_samples).reshape((n_samples, 2))

			samples_cand[ k, : ] = samples[ :, 0 ]
			samples_grid[ k, : ] = samples[ :, 1 ]

		counts = 0.0

		valid_samples = np.ones(n_samples) == 1.0

		for k in range(n_objectives):
			valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] <= cell_j['u'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] <= cell_i['u'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] >= cell_j['l'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_cand[ k, : ] >= cell_i['l'][ k ])

		return np.mean(valid_samples)

	def approximate_cell_j_imp_prob_via_monte_carlo(self, cell_j, meanCand, meanXgrid, varCand, \
		varXgrid, covCandXgrid, i_cand, i_grid, n_objectives, n_samples = 1e3):

		samples_cand = np.zeros((n_objectives, n_samples))
		samples_grid = np.zeros((n_objectives, n_samples))
		
		for k in range(n_objectives):

			C = np.zeros((2, 2))
			m = np.zeros(2)
		
			C[ 0, 0 ] = varCand[ i_cand, i_grid, k ]
			C[ 1, 1 ] = varXgrid[ i_cand, i_grid, k ]
			C[ 1, 0 ] = covCandXgrid[ i_cand, i_grid, k ]
			C[ 0, 1 ] = covCandXgrid[ i_cand, i_grid, k ]
			m[ 0 ] = meanCand[ i_cand, i_grid, k ]
			m[ 1 ] = meanXgrid[ i_cand, i_grid, k ]
			L = spla.cholesky(C)
			
			samples = np.dot(np.random.normal(size = 2 * n_samples).reshape((n_samples, 2)), L) + \
				np.tile(m, n_samples).reshape((n_samples, 2))

			samples_cand[ k, : ] = samples[ :, 0 ]
			samples_grid[ k, : ] = samples[ :, 1 ]

		counts = 0.0

		valid_samples = np.ones(n_samples) == 1.0

		for k in range(n_objectives):
			valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] <= cell_j['u'][ k ])
			valid_samples = np.logical_and(valid_samples, samples_grid[ k, : ] >= cell_j['l'][ k ])

		valid_samples_delta  = np.ones(n_samples) != 1.0

		for k in range(n_objectives):
			valid_samples_delta = np.logical_or(valid_samples_delta, samples_cand[ k, : ] >= samples_grid[ k, : ])

		valid_samples = np.logical_and(valid_samples, valid_samples_delta)

		return np.mean(valid_samples)





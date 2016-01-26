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
from spearmint.grids import sobol_grid
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.utils.numerics import logcdf_robust
from spearmint.models.gp import GP
from spearmint.utils.moop            import MOOP_basis_functions
from spearmint.utils.moop            import _cull_algorithm
import importlib
from spearmint.tasks.task         import Task
from spearmint.utils.hv import HyperVolume
from scipy.spatial.distance import cdist
from spearmint.utils.moop import _cull_algorithm

from spearmint.models.abstract_model import function_over_hypers
import logging

NUM_POINTS_FRONTIER = 10
USE_GRID_ONLY = False
GRID_SIZE = 1000
NSGA_POP = 100
NSGA_EPOCHS = 100

SMSEGO_OPTION_DEFAULTS  = {
    'ehi_pareto_set_size'      : 10,
    'ehi_grid_size'      : 1000,
    'ehi_nsga_epochs'      : 100,
    'ehi_nsga_pop'      : 100,
    'ehi_use_grid_only_to_solve_problem' : False,
    }

class EHI(AbstractAcquisitionFunction):

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

		self.options = SMSEGO_OPTION_DEFAULTS.copy()
		self.options.update(opt)

		NUM_POINTS_FRONTIER = self.options['ehi_pareto_set_size']
		GRID_SIZE = self.options['ehi_grid_size'] 
		USE_GRID_ONLY = self.options['ehi_use_grid_only_to_solve_problem'] 
		NSGA_POP = self.options['ehi_nsga_pop'] 
		NSGA_EPOCHS = self.options['ehi_nsga_epochs'] 

	def acquisition(self, obj_model_dict, con_models_dict, cand, current_best, compute_grad, minimize=True, tasks=None):

		models = obj_model_dict.values()

		# make sure all models are at the same state

		assert len({model.state for model in models}) == 1, "Models are not all at the same state"

		assert not compute_grad 

		# We check if we have already computed the cells associated to the model and other stuff

		key = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ])

		if not key in self.cached_information:
			self.cached_information[ key ] = self.compute_cell_information(obj_model_dict)

		# We use the chooser to compute the expected improvement on the scalarized task

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

	def compute_cell_information(self, obj_model_dict):

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

		print 'Inner multi-objective problem solved!'

		means_objectives = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].inputs.shape[ 0 ], len(obj_model_dict)))

		k = 0
		for obj in obj_model_dict:
			means_objectives[ :, k ] = obj_model_dict[ obj ].predict(obj_model_dict[ obj ].inputs)[ 0 ]
			k += 1

		v_inf = np.ones((1, len(obj_model_dict))) * np.inf
		v_ref = np.ones((1, len(obj_model_dict))) * 1e3

		# We add the non-dominated prediction and the observed inputs to the frontier

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

		frontier_sorted = np.vstack((-v_inf, cached_information['frontier'], v_ref, v_inf))

		for i in range(len(obj_model_dict)):
			frontier_sorted[ :, i ] = np.sort(frontier_sorted[ :, i ])

		# Now we build the info associated to each cell

		n_repeat = (frontier_sorted.shape[ 0 ] - 2) ** frontier_sorted.shape[ 1 ]

		cached_information['cells'] = dict()

		added_cells = 0
		for i in range(n_repeat):

			cell = dict()

			indices = np.zeros(len(obj_model_dict)).astype(int)

			j = i

			for k in range(len(obj_model_dict)):
				indices[ k ] = int(j % (frontier_sorted.shape[ 0 ] - 2))
				j = np.floor(j / (frontier_sorted.shape[ 0 ] - 2))

			u = np.zeros(len(obj_model_dict))

			for k in range(len(obj_model_dict)):
				u[ k ] = frontier_sorted[ int(indices[ k ] + 1), k ]

			l = np.zeros(len(obj_model_dict))
				
			for k in range(len(obj_model_dict)):
				l[ k ] = frontier_sorted[ indices[ k ], k ]

			# If the cell is dominated we discard it

			is_dominated = False
			for k in range(frontier.shape[ 0 ]):
				if np.all(l >= frontier[ k, : ]):
					is_dominated = True

			if is_dominated:
				continue

			# We find the vector v

			v = np.zeros(len(obj_model_dict))

			for k in range(len(obj_model_dict)):

				l_tmp = np.copy(l)

				for j in range(int(frontier_sorted.shape[ 0 ] - indices[ k ] - 1)):
					l_tmp[ k ] = frontier_sorted[ indices[ k ] + j, k ]

					dominates_all = True
					for h in range(frontier.shape[ 0 ]):
						if np.all(frontier[ h, : ] <= l_tmp):
							dominates_all = False
							break

					if dominates_all == False:
						break
					
				if dominates_all == False:
					v[ k ] = l_tmp[ k ]
				else:
					v[ k ] = v_ref[ 0, k ]

			# We compute the quantities required for evaluating the gain in hyper-volume

			# We find the points dominated by u

			dominated_by_u = frontier
			h = 0
			while (h < dominated_by_u.shape[ 0 ]):
				if (not np.any(u < dominated_by_u[ h, : ])) and (not np.all(u == dominated_by_u[ h, : ])):
					dominated_by_u = np.delete(dominated_by_u, (h), axis = 0)
				else:
					h+= 1

			# The value of minusQ2plusQ3 is given by the hypervolume of the dominated points with reference v

			if dominated_by_u.shape[ 0 ] == 0:
				minusQ2plusQ3 = 0.0
			else:
				hv = HyperVolume(v.tolist())
				minusQ2plusQ3 = -hv.compute(dominated_by_u.tolist())
			
			cell['u'] = u
			cell['l'] = l
			cell['v'] = v
			cell['dominated_by_u'] = dominated_by_u
			cell['minusQ2plusQ3'] = minusQ2plusQ3
			
			cached_information['cells'][ str(added_cells) ] = cell
			added_cells += 1
			
		n_cells = added_cells

		cached_information['n_cells'] = n_cells
		cached_information['v_ref'] = v_ref[ 0, : ]
		cached_information['n_objectives'] = len(obj_model_dict)

#		self.print_cell_info(cached_information)

		return cached_information


	# This method is the one that actually does the computation of the acquisition_function
	
	def compute_acquisition(self, cand, obj_model_dict, information):

		n_cells = information['n_cells']
		n_objectives = information['n_objectives']

		# We compute the mean and the variances at each candidate point

		mean = np.zeros((cand.shape[ 0 ], n_objectives))
		var = np.zeros((cand.shape[ 0 ], n_objectives))

		n_objective = 0
		for obj in obj_model_dict:
			mean[ :, n_objective ], var[ :, n_objective ] = obj_model_dict[ obj ].predict(cand) 	
			n_objective += 1

		# We loop over the non-dominated cells

		total_acquisition = np.zeros(cand.shape[ 0 ])

		for i in range(n_cells):

			cell = information['cells'][ str(i) ]

			int_cell = np.ones(cand.shape[ 0 ])
			for k in range(n_objectives):
				v_u = sps.norm.cdf((cell['u'][ k ] - mean[ :, k ]) / np.sqrt(var[ : , k ]))
				v_l = sps.norm.cdf((cell['l'][ k ] - mean[ :, k ]) / np.sqrt(var[ : , k ]))
				int_cell *=  (v_u - v_l)

			Q1 = np.ones(cand.shape[ 0 ])
			for k in range(n_objectives):
				Q1 *= (self.psi(cell['v'][ k ], cell['u'][ k ], mean[ :, k ], np.sqrt(var[ :, k ])) - \
					self.psi(cell['v'][ k ], cell['l'][ k ], mean[ :, k ], np.sqrt(var[ :, k ])))

			cell_cont = Q1 + cell['minusQ2plusQ3'] * int_cell

			total_acquisition += cell_cont

		# We substract the hyper-volume of the frontier

		return total_acquisition

	def psi(self, a, b, mu, sigma):

		return sigma * sps.norm.pdf((b - mu) / sigma) + (a - mu) * sps.norm.cdf((b - mu) / sigma)

	def print_cell_info(self, information):
	
		import matplotlib.pyplot as plt
		import matplotlib.patches as patches
		from matplotlib.pyplot import text

		assert information['n_objectives'] == 2

		fig = plt.figure()
		plt.plot(information['frontier'][ :,0 ], information['frontier'][ :,1 ], color='black', 
			marker='x', markersize=10, linestyle='None')
		plt.plot(information['frontier'][ :,0 ], information['frontier'][ :,1 ], color='black', 
			marker='x', markersize=10, linestyle='None')
		plt.plot(information['v_ref'][ 0 ], information['v_ref'][ 1 ], color='brown', 
			marker='x', markersize=10, linestyle='None')

		plt.plot(information['v_ref'][ 0 ] + 100, information['v_ref'][ 1 ] + 100, color='white', 
			marker='x', markersize=10, linestyle='None')

		plt.plot(-information['v_ref'][ 0 ] - 100, -information['v_ref'][ 1 ] - 100, color='white', 
			marker='x', markersize=10, linestyle='None')

		for i in range(information['n_cells']):
			cell = information['cells'][str(i)]
			u = cell['u']
			l = cell['l']
			l[ np.where(l == -np.inf) ] = -information['v_ref'][ 0 ]
			v = cell['v']
			minusQ2plusQ3 = cell['minusQ2plusQ3']
		
			# We draw the points

			plt.plot(u[ 0 ], u[ 1 ], color='blue', marker='o', markersize=10, linestyle='None')
			plt.plot(l[ 0 ], l[ 1 ], color='red', marker='o', markersize=10, linestyle='None')
			plt.plot(v[ 0 ], v[ 1 ], color='green', marker='o', markersize=10, linestyle='None')

			# We draw the lines

			plt.plot(np.linspace(l[ 0 ], u[ 0 ], 100), np.ones(100) * l[ 1 ], color = 'b', marker='.', markersize=1)
			plt.plot(np.linspace(l[ 0 ], u[ 0 ], 100), np.ones(100) * u[ 1 ], color = 'b', marker='.', markersize=1)
			plt.plot(np.ones(100) * l[ 0 ], np.linspace(l[ 1 ], u[ 1 ], 100), color = 'b', marker='.', markersize=1)
			plt.plot(np.ones(100) * u[ 0 ], np.linspace(l[ 1 ], u[ 1 ], 100), color = 'b', marker='.', markersize=1)

			# We draw the cell number

			text((l[ 0 ] + u[ 0 ]) / 2, (l[ 1 ] + u[ 1 ]) / 2, str(i))

			print(minusQ2plusQ3)

		plt.show()

	def compute_acquisition_mc(self, cand, obj_model_dict, information, n_samples = 100):

		n_objectives = information['n_objectives']

		samples = np.random.normal(size = (n_samples, n_objectives))

		mean = np.zeros((cand.shape[ 0 ], n_objectives))
		var = np.zeros((cand.shape[ 0 ], n_objectives))

		n_objective = 0
		for obj in obj_model_dict:
			mean[ :, n_objective ], var[ :, n_objective ] = obj_model_dict[ obj ].predict(cand) 	
			n_objective += 1

		frontier = information['frontier']

		hv = HyperVolume(information['v_ref'].tolist())
		hv_frontier = hv.compute(frontier.tolist())

		acquisition_values = np.zeros(cand.shape[ 0 ])

		for i in range(cand.shape[ 0 ]):

			value = 0.0			

			for j in range(n_samples):

				new_point_frontier = samples[ j, : ] * np.sqrt(var[ i, : ]) + mean[ i, : ]
				new_frontier = np.vstack((frontier, new_point_frontier.reshape(1, len(obj_model_dict))))
				new_hv_frontier = hv.compute(new_frontier.tolist())
	
				value += new_hv_frontier - hv_frontier
	
			value /= n_samples

			acquisition_values[ i ] = value

		return acquisition_values


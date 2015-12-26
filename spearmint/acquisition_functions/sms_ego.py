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

from spearmint.grids import sobol_grid
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

from spearmint.models.abstract_model import function_over_hypers
import logging

NUM_POINTS_FRONTIER = 10
USE_GRID_ONLY = False
GRID_SIZE = 1000
NSGA_POP = 100
NSGA_EPOCHS = 100

SMSEGO_OPTION_DEFAULTS  = {
    'smsego_pareto_set_size'      : 10,
    'smsego_grid_size'      : 1000,
    'smsego_nsga_epochs'      : 100,
    'smsego_nsga_pop'      : 100,
    'smsego_use_grid_only_to_solve_problem' : False,
    }


epsilon = 1e-6 			# Based on the implementation of the R package GPareto

class SMSego(AbstractAcquisitionFunction):

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

		NUM_POINTS_FRONTIER = self.options['smsego_pareto_set_size']
		GRID_SIZE = self.options['smsego_grid_size'] 
		USE_GRID_ONLY = self.options['smsego_use_grid_only_to_solve_problem'] 
		NSGA_POP = self.options['smsego_nsga_pop'] 
		NSGA_EPOCHS = self.options['smsego_nsga_epochs'] 

	def acquisition(self, obj_model_dict, con_models_dict, cand, current_best, compute_grad, minimize=True, tasks=None):

		models = obj_model_dict.values()

		# make sure all models are at the same state

		assert len({model.state for model in models}) == 1, "Models are not all at the same state"

		assert not compute_grad 

		# We check if we have already computed the information associated to the model and other stuff

		key = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ])

		if not key in self.cached_information:
			self.cached_information[ key ] = self.compute_information(obj_model_dict)

		# We use the chooser to compute the expected improvement on the scalarized task

		return self.compute_acquisition(cand, obj_model_dict, self.cached_information[ key ])

	# We compute the required information for carrying out the method

	def compute_information(self, obj_model_dict):

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

		print 'Internal optimization finished'

		# We remove repeated entries from the pareto front

		frontier = result['frontier']

		X = frontier[ 0 : 1, : ]

		for i in range(frontier.shape[ 0 ]):
			if np.min(cdist(frontier[ i : (i + 1), : ], X)) > 1e-8:
			    	X = np.vstack((X, frontier[ i, ])) 
	
		frontier = X

		cached_information['frontier'] = frontier
		cached_information['v_ref'] = np.ones(len(obj_model_dict)) 

		for k in range(frontier.shape[ 1 ]):
			cached_information['v_ref'][ k ] = np.max(frontier[ :, k ]) + 1.0

		return cached_information

	# This functions optimizes a GP. 

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

	# This method is the one that actually does the computation of the acquisition_function
	# We follow the implementation considered in the R package GPareto
	
	def compute_acquisition(self, cand, obj_model_dict, information):

		n_objectives = len(obj_model_dict)

		# We compute the mean and the variances at each candidate point

		mean = np.zeros((cand.shape[ 0 ], n_objectives))
		var = np.zeros((cand.shape[ 0 ], n_objectives))
		potSol = np.zeros((cand.shape[ 0 ], n_objectives))

		n_objective = 0
		for obj in obj_model_dict:
			mean[ :, n_objective ], var[ :, n_objective ] = obj_model_dict[ obj ].predict(cand) 	

			# We set gain = 3 which seems to give good results

			potSol[ :, n_objective ] = mean[ :, n_objective ] - 3 * np.sqrt(var[ :, n_objective ])
			n_objective += 1

		frontier = information['frontier']
		hv = HyperVolume(information['v_ref'].tolist())
		hv_frontier = hv.compute(frontier.tolist())

		total_acquisition = np.zeros(cand.shape[ 0 ])

		# We loop over the candidate points

		for i in range(cand.shape[ 0 ]):

			penalty = 0.0

			# We look for epsilon dominance

			for k in range(frontier.shape[ 0 ]):
				if np.all(frontier[ k , : ] <= potSol[ i, : ] + epsilon):
					p = -1 + np.prod(1 + np.maximum(potSol[ i, : ] - frontier[ k, : ], np.zeros(n_objectives)))
					penalty = np.maximum(penalty, p)
				
			if penalty == 0.0:
				
				newFront = np.vstack((frontier, np.array(potSol[ i, : ])))
				hv_new = hv.compute(newFront.tolist())
	
				total_acquisition[ i ] = hv_frontier - hv_new
			else:
				total_acquisition[ i ] = penalty

		return -1.0 * total_acquisition


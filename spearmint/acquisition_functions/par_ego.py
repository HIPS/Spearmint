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

from collections import defaultdict
from spearmint.grids import sobol_grid
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.utils.numerics import logcdf_robust
from spearmint.models.gp import GP
from spearmint.utils.moop            import MOOP_basis_functions
from spearmint.utils.moop            import _cull_algorithm
import importlib
from spearmint.tasks.task         import Task

from spearmint.models.abstract_model import function_over_hypers
import logging

try:
    import nlopt
except:
    nlopt_imported = False
else:
    nlopt_imported = True
# see http://ab-initio.mit.edu/wiki/index.php/NLopt_Python_Reference


# Compute log of the normal CDF of x in a robust way
# Based on the fact that log(cdf(x)) = log(1-cdf(-x))
# and log(1-z) ~ -z when z is small, so  this is approximately
# -cdf(-x), which is just the same as -sf(x) in scipy

def logcdf_robust(x):
    
    if isinstance(x, np.ndarray):
        ret = sps.norm.logcdf(x)
        ret[x > 5] = -sps.norm.sf(x[x > 5])
    elif x > 5:
        ret = -sps.norm.sf(x)
    else:
        ret = sps.norm.logcdf(x)

    return ret

"""
FOR GP MODELS ONLY
"""

class ParEGO(AbstractAcquisitionFunction):

	def __init__(self, num_dims, verbose=True, input_space=None, grid=None, opt = None):

		# we want to cache these. we use a dict indexed by the state integer

		self.cached_scalarization = dict()
		self.cached_tasks = dict()
		self.has_gradients = False
		self.num_dims = num_dims
		self.input_space = input_space
	        self.weight_vector = None
		self.hypers = None
		self.recommendation = None
		self.acq = None

		# Load up the chooser used for expected improvement computation.

		chooser_module = importlib.import_module('spearmint.choosers.default_chooser')

		# We use the default options for the chooser

		options = dict()

		options['acquisition'] = 'ExpectedImprovement'
		options['tolerance'] = None
		options['always_sample'] = True
		options['grid_subset'] = 1
		options['grid_seed'] = 0
		options['opt_acq_maxeval'] = 500
		options['optimize_acq'] = True
		options['num_spray'] = 10
		options['regenerate_grid'] = True
		options['check_grad'] = False
		options['spray_std'] = 0.0001
		options['scale-duration'] = False
		options['optimize_best'] = True
		options['unit_tolerance'] = 0.0001

		self.chooser = chooser_module.init(input_space, options)

	def acquisition(self, obj_model_dict, con_models_dict, cand, current_best, compute_grad, minimize=True, tasks=None):

		obj_models = obj_model_dict.values()

		models = obj_models

		# make sure all models are at the same state

		assert len({model.state for model in models}) == 1, "Models are not all at the same state"

		assert not compute_grad 

		# We check if we have already computed the scalarization (for a particular number of observations in each model). 
		# If so, we reuse the result obtained.

		key = tuple([obj_model_dict[ obj ].inputs.shape[0] for obj in obj_model_dict])

		if not key in self.cached_scalarization:

	        	self.weight_vector = np.random.dirichlet(np.ones(len(obj_model_dict)), 1)
			scalarization = self.scalarization(obj_model_dict, tasks)
			self.cached_scalarization[ key ] = scalarization
			self.hypers = self.chooser.fit(scalarization[ 'tasks' ], self.hypers)
			recommendation = self.chooser.best()
                        self.current_best_value = (recommendation['model_model_value'] - self.chooser.objective.standardization_mean) / \
				self.chooser.objective.standardization_variance
        		self.acq = self.chooser.acquisition_function(self.num_dims, grid = None, input_space = self.input_space)

#			print_images(self.chooser, recommendation['model_model_value'])

		# We use the chooser to compute the expected improvement on the scalarized task

		return  function_over_hypers(self.chooser.models.values(), self.acq.acquisition, \
			self.chooser.objective_model_dict, self.chooser.constraint_models_dict, \
			cand, self.current_best_value, compute_grad = False, tasks = self.chooser.tasks)

	# This computes the model for the scalarization

	def scalarization(self, obj_model_dict, tasks):

		# We assume the same observations for all the objectives

		inputs = copy.deepcopy(self.input_space.from_unit(obj_model_dict[ obj_model_dict.keys()[ 0 ] ].inputs))
		values = None
		

		for objective in obj_model_dict:
			if values == None:
				values = np.array([ obj_model_dict[ objective ].values ]).T
			else:
				values = np.hstack((values, np.array([obj_model_dict[ objective ].values]).T))

		new_values = np.zeros(inputs.shape[ 0 ])

		for i in range(inputs.shape[ 0 ]):
			new_values[ i ] = np.max(self.weight_vector * values[ i, : ]) +  0.05 * np.sum(self.weight_vector * values[ i, : ])

		# We look for noisy observations. If one task is noisy we consider a noisy model

		noisy = False
		options = dict()

		for t in obj_model_dict:
			if obj_model_dict[ t ].options['likelihood'].lower() != 'noiseless':
				noisy = True
				break

		# We set the options for the task to be built

		if noisy == True:
			options['likelihood'] = 'GAUSSIAN'
		else:
			options['likelihood'] = 'NOISELESS'

		options['acquisition'] = 'ExpectedImprovement'
		options['main_file'] = None
		options['language'] = None
		options['type'] = 'OBJECTIVE'
		options['group'] = 0
		options['cost'] = 1.0
		options['experiment-name'] = None
		options['config'] = None
		options["fit_mean"] = False
		options['model'] = 'GP'
		options['max_finished_jobs'] = None
		options['scale-duration'] = False
		options['main_file_path'] = None
		trans = [ dict() ]
		trans[ 0 ]['BetaWarp'] = dict()
		options['transformations'] = []

		scalarization = dict()

		# We create the corresponding task that corresponds to the scalarized objective

		task = Task('ParEGO', options, inputs.shape[ 1 ])
		task.inputs = inputs
		task.values = new_values

		tasks = dict()
		tasks[ task.name ] = task
		
		scalarization['tasks'] = tasks

		return scalarization


def print_images(chooser, current_best_value_unormalized):

	import matplotlib.pyplot as plt
        spacing = np.linspace(0,1,1000)[:,None]

	inputs = chooser.input_space.from_unit(chooser.models['ParEGO'].inputs)
	values = chooser.tasks[ 'ParEGO' ].unstandardize_mean(chooser.tasks[ 'ParEGO' ].unstandardize_variance(\
		chooser.models[ 'ParEGO'].values))

	mean, var = chooser.models[ 'ParEGO' ].function_over_hypers(chooser.models[ 'ParEGO' ].predict, spacing)
	mean = chooser.tasks[ 'ParEGO' ].unstandardize_mean(chooser.tasks[ 'ParEGO' ].unstandardize_variance(mean))
	var = chooser.tasks[ 'ParEGO' ].unstandardize_variance(var)

	current_best_value = current_best_value_unormalized

	fig = plt.figure()
	plt.plot(inputs, values, color='black', marker='x', markersize=10, linestyle='None')
	plt.plot(chooser.input_space.from_unit(spacing), mean, color = 'black', marker = '.')
	plt.plot(chooser.input_space.from_unit(spacing), current_best_value * np.ones(1000), color = 'red', marker = '.')
	plt.plot(chooser.input_space.from_unit(spacing), mean - np.sqrt(var), color = 'black', marker = '.', markersize = 1)
	plt.plot(chooser.input_space.from_unit(spacing), mean + np.sqrt(var), color = 'black', marker = '.', markersize = 1)
	plt.savefig('./figures/' + str(inputs.shape[ 0 ]) + '-scalarization.pdf', format='pdf', dpi=1000)
	plt.close(fig)




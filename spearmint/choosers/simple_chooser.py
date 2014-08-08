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
import scipy.optimize as spo
import multiprocessing
import logging

from .acquisition_functions  import compute_ei
from ..models                import GP, GPClassifier, MixedGP
from ..utils.grad_check      import check_grad
from ..grids                 import sobol_grid
from ..models.abstract_model import function_over_hypers

log = logging.getLogger(__name__)

DEFAULT_GRIDSIZE  = 20000
DEFAULT_GRIDSEED  = 0
DEFAULT_NUMDESIGN = 2


def init(options):
    return SimpleChooser(options)


class SimpleChooser(object):
    def __init__(self, options):
        self.grid_size  = options.get('grid_size', DEFAULT_GRIDSIZE)
        self.grid_seed  = options.get('grid_seed', DEFAULT_GRIDSEED)
        self.num_spray  = 10
        self.spray_std  = 1e-3
        self.check_grad = options.get('check-grad', False)

        # Grab options for the objective and the constraint
        self.objective_options  = dict([(key.split('.')[1],value) for key,value in options.iteritems() if key.startswith("objective.")])
        self.constraint_options = dict([(key.split('.')[1],value) for key,value in options.iteritems() if key.startswith("constraint.")])
        log.debug('Initializing objective with options: %s' % (self.objective_options))
        log.debug('Initializing constraint with options: %s' % (self.constraint_options))

        self.grid_subset = 20

        if 'chooser-args' in options:
            self.parallel_opt = bool(options['chooser-args'].get('parallel-opt', False))
        else:
            self.parallel_opt = False

        self.objective_task  = None
        self.constraint_task = None

        self.objective_data  = None
        self.constraint_data = None

        self.objective_model  = None
        self.constraint_model = None

        self.grid             = None
        self.original_inputs  = None

        self.task             = None

    def has_constraint_violations(self):
        return (self.constraint_task is not None) and (np.any(self.constraint_task.values <= 0.0))

    def fit(self, objective_task, constraint_task=None, objective_hypers=None, constraint_hypers=None):
        self.objective_task  = objective_task
        self.constraint_task = constraint_task

        self.objective_data  = objective_task.normalized_data_dict
        self.constraint_data = constraint_task.normalized_data_dict if constraint_task else None

        self.num_dims        = self.objective_task.num_dims

        new_objective_hypers  = {}
        new_constraint_hypers = {}

        # Create the grid of optimization initializers
        # Need to do it here because it's used in many places e.g. best
        self.grid = sobol_grid.generate(self.num_dims)
        assert(np.all(self.grid <= 1.0))
        assert(np.all(self.grid >= 0.0))

        # A fallback in case you don't fit -- just submit this
        # index off the grid
        self.design_index = objective_task.inputs.shape[0] + objective_task.pending.shape[0]

        num_valid = objective_task.valid_values.shape[0]

        objective_hypers  = objective_hypers if objective_hypers and num_valid > 1 else {}
        constraint_hypers = constraint_hypers if constraint_hypers else {}

        groups = objective_task.get_categorical_groups()
        if groups:
            self.objective_model = MixedGP(self.num_dims, groups, **self.objective_options)
        else:
            self.objective_model = GP(self.num_dims, **self.objective_options)

        new_objective_hypers = None
        new_constraint_hypers = None
        inputs = self.objective_data['inputs'].copy()
        values = self.objective_data['values'].copy()
        pending = self.objective_data['pending']
        if pending is not None:
            pending = pending.copy()

        if num_valid > 1:
            new_objective_hypers = self.objective_model.fit(inputs, values, pending=pending,
                    hypers=objective_hypers)

        # Don't bother fitting the constraint GP if we don't have violations
        if constraint_task and num_valid >= 1 and self.has_constraint_violations():
            log.debug('Sampling constraint GP')
            self.constraint_model = GPClassifier(self.num_dims, **self.constraint_options)
            new_constraint_hypers = self.constraint_model.fit(
                self.constraint_data['inputs'].copy(),
                self.constraint_data['counts'].copy(),
                pending = self.constraint_data['pending'],
                    hypers=constraint_hypers)

        return new_objective_hypers, new_constraint_hypers

    def suggest(self):
        num_valid = self.objective_task.valid_values.shape[0]
        # Cases in which to spit out grid points:
        # 1) We have 0 to N constraint violations and no valid observations.
        # 2) We have no constraint violations and fewer valid observations than design points
        # Otherwise we should use the constraint value(s) to guide the next suggestion.
        if (self.constraint_task and num_valid == 0) or (num_valid < DEFAULT_NUMDESIGN):
            return self.objective_task.from_unit(self.grid[self.design_index])

        # Add some extra candidates around the min so far (a useful hack)
        best_ind     = self.objective_data['values'].argmin()
        spray_points = npr.randn(self.num_spray, self.num_dims)*self.spray_std + self.objective_data['inputs'][best_ind]
        spray_points[spray_points < 0.0] = 0.0
        spray_points[spray_points > 1.0] = 1.0

        # Compute the current best
        current_best, current_best_location = self.best()

        # Compute EI on the grid
        grid_pred = np.vstack((self.grid, spray_points))
        grid_ei   = self.acquisition_function_over_hypers(grid_pred, current_best, compute_grad=False)
        log.debug('Computed grid EI.')

        order = np.argsort(grid_ei)
        best_grid_inds = []
        
        # Only add points that haven't already been observed
        for i in xrange(order.shape[0]-1, 0, -1):
            if not self.is_duplicate(grid_pred[order[i]]):
                best_grid_inds.append(order[i])
            else:
                grid_ei[order[i]] = -1
            if len(best_grid_inds) == self.grid_subset:
                break

        log.debug('Filtered grid points that have already been evaluated.')

        # Find the points on the grid with highest EI
        # The index and value of the top grid point
        best_grid_ind = np.argmax(grid_ei)
        best_grid_ei  = grid_ei[best_grid_ind]
        log.debug('Best EI before optimization: %f' % best_grid_ei)

        best_grid_pred = grid_pred[best_grid_inds]

        if self.check_grad:
            check_grad(lambda x: self.acq_optimize_wrapper(x, current_best, True), 
                best_grid_pred[0], verbose=True)

        # Optimize the top points from the grid to get better points
        cand2 = best_grid_pred
        cand = []
        b = [(0,1)]*cand2.shape[1]# optimization bounds

        if self.parallel_opt:
            # Optimize each point in parallel
            pool = multiprocessing.Pool(self.grid_subset)
            results = [pool.apply_async(self.optimize_pt,args=(
                    c,b,current_best,True)) for c in cand2]

            for res in results:
                cand.append(res.get(1e8))
            pool.close()
        else: 
            # Optimize in series
            for c in cand2:
                cand.append(self.optimize_pt(c,b,current_best,compute_grad=True))
        # Cand now stores the optimized points

        # Compute one more time (re-computed is unnecessary, oh well... TODO)
        cand = np.vstack(cand)
        opt_ei = self.acquisition_function_over_hypers(cand, current_best, compute_grad=False)

        # The index and value of the top optimized point
        best_opt_ind = np.argmax(opt_ei)
        best_opt_ei  = opt_ei[best_opt_ind]

        # Check to make sure that point hasn't already been evaluated!
        # Note that this should happen less than n log n times
        while self.is_duplicate(cand[best_opt_ind]):
            log.debug('Suggested input is duplicate: %s' % cand[best_opt_ind])
            opt_ei[best_opt_ind] = -1.0 # Make sure we don't choose it again
            best_opt_ind = np.argmax(opt_ei)
            best_opt_ei  = opt_ei[best_opt_ind]
            if best_opt_ei == -1: # They're all duplicates! (very unlikely)
                break

        log.debug('Best EI after  optimization: %f' % best_opt_ei)

        # Optimization should always be better unless the optimization
        # breaks in some way.
        log.debug('Suggested input %s' % cand[best_opt_ind])
        if best_opt_ei >= best_grid_ei:
            return self.objective_task.from_unit(cand[best_opt_ind])
        else:
            return self.objective_task.from_unit(grid_pred[best_grid_ind])

    # The chooser might round to an experiment that has already been run
    # and we don't want to rerun things
    def is_duplicate(self, cand):
        cand = np.reshape(cand.copy(),(1,self.grid.shape[1]))
        if self.original_inputs is None:
            data = self.objective_data['inputs']
            if self.objective_data['pending'] is not None:
                data = np.vstack((data, self.objective_data['pending']))
            self.original_inputs = np.reshape(data,(-1,self.grid.shape[1]))
            self.original_inputs = self.objective_task.from_unit(self.original_inputs)

        # First convert to original format (e.g. integer, non-unit, etc.)
        original_cand = self.objective_task.from_unit(cand)

        # Treat extremely close as equal due to numerical precision
        if np.any(np.sum((self.original_inputs-original_cand)**2.0, axis=1) < 1e-08):
            return True
        else:
            return False

    # TODO: add optimization in here
    def best(self):
        best_ind              = self.objective_task.values.argmin()
        current_best_value    = self.objective_task.valid_values[best_ind]
        current_best_location = self.objective_task.valid_inputs[best_ind]

        log.debug('Best location = %s' % current_best_location)
        log.debug('Best value = %f' % current_best_value)

        return current_best_value, current_best_location

    def numConstraints(self):
        return len(self.constraints)

    # The confidence that the constraint is satisfied
    def probability_satisfied(self, inputs, compute_grad=False):
        return self.constraint_model.function_over_hypers(self.constraint_model.pi, inputs, compute_grad=compute_grad)

    # Returns a boolean array of size pred.shape[0] indicating whether the prob con-constraint is satisfied there
    def probabilistic_constraint(self, inputs):
        return self.probability_satisfied(inputs) >= self.constraint_task.options.get('min-confidence', 0.99)

    def acquisition_function_over_hypers(self, *args, **kwargs):
        # If no invalid points, only do standard EI and forget the constraint task
        if not self.has_constraint_violations():            
            return self.objective_model.function_over_hypers(self.acquisition_function, *args, **kwargs)
        else:
            return function_over_hypers([self.objective_model, self.constraint_model], self.acquisition_function, *args, **kwargs)

    def acquisition_function(self, cand, current_best, compute_grad=True):

        # If unconstrained, just compute regular ei
        if not self.has_constraint_violations():
            return compute_ei(self.objective_model, cand, ei_target=current_best, compute_grad=compute_grad)

        if cand.ndim == 1:
            cand = cand[None,:]

        N_cand = cand.shape[0]

        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############   Part that depends on the objective     ############
        ##############                                          ############
        ############## ---------------------------------------- ############
        if current_best is None:
            ei = 1.
            ei_grad = 0.
        else:
            target = current_best
     
            # Compute the predictive mean and variance
            obj_model = self.objective_model
            if not compute_grad:
                ei = compute_ei(obj_model, cand, target, compute_grad=compute_grad)
            else:
                ei, ei_grad = compute_ei(obj_model, cand, target, compute_grad=compute_grad)

        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############  Part that depends on the constraints    ############
        ##############                                          ############
        ############## ---------------------------------------- ############
        # Compute p(valid) for ALL constraints
        # p_valid, p_grad = list(), list()
        # for c in self.constraints:
        if compute_grad:
            p_valid, p_grad = self.constraint_model.pi(cand, compute_grad=True)
        else:
            p_valid = self.constraint_model.pi(cand, compute_grad=False)
        
        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############    Combine the two parts (obj and con)   ############
        ##############                                          ############
        ############## ---------------------------------------- ############

        acq = ei * p_valid

        if not compute_grad:
            return acq
        else:
            return acq, ei_grad * p_valid + p_grad * ei

    # Flip the sign so that we are maximizing with BFGS instead of minimizing
    def acq_optimize_wrapper(self, cand, current_best, compute_grad):
        bad_inds_0 = cand < 0
        bad_inds_1 = cand > 1
        if np.any(bad_inds_0) or np.any(bad_inds_1):
            logging.debug('Points lying outside of the unit hypercube detected during LBFGS optimization:\n%s' % cand)
        cand[cand<0] = 0.0
        cand[cand>1] = 1.0
        ret = self.acquisition_function_over_hypers(cand, current_best, compute_grad=compute_grad)

        if isinstance(ret, tuple) or isinstance(ret, list):
            return (-ret[0],-ret[1].flatten())
        else:
            return -ret

    def optimize_pt(self, initializer, bounds, current_best, compute_grad=True):
        opt_x, opt_y, opt_info = spo.fmin_l_bfgs_b(self.acq_optimize_wrapper,
                initializer.flatten(), args=(current_best,compute_grad),
                bounds=bounds, disp=0, approx_grad=(not compute_grad))

        # Truncate points to ensure nothing lies outside of the bounds.
        bad_inds_0 = opt_x < 0
        bad_inds_1 = opt_x > 1
        if np.any(bad_inds_0) or np.any(bad_inds_1):
            logging.debug('Points lying outside of the unit hypercube detected after LBFGS optimization:\n%s' % opt_x)
        return opt_x

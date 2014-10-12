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


import sys
import numpy          as np
import numpy.random   as npr
import scipy.optimize as spo
import multiprocessing

from collections import defaultdict

from .acquisition_functions  import compute_ei
from ..utils.grad_check      import check_grad
from ..grids                 import sobol_grid
from ..models.abstract_model import function_over_hypers
from ..                      import models

DEFAULT_GRIDSIZE  = 20000
DEFAULT_GRIDSEED  = 0
DEFAULT_NUMDESIGN = 2

DEFAULT_NUMSPRAY  = 10
DEFAULT_SPRAYSTD  = 1e-3

VERBOSE = False


def init(options):
    return DefaultChooser(options)


class DefaultChooser(object):
    """class which which makes suggestions for new jobs
    
    Attributes
    ----------
    models : dict
        The keys are the tasks and the values are the models which are used 
        to make the suggestsions.  The default models are GP or GPClassifier
    objective : dict
        ?
    """
    def __init__(self, options):
        self.grid_size = options.get('grid_size', DEFAULT_GRIDSIZE)
        self.grid_seed = options.get('grid_seed', DEFAULT_GRIDSEED)
        self.num_spray = options.get('num-spray', DEFAULT_NUMSPRAY)
        self.spray_std = options.get('spray-std', DEFAULT_SPRAYSTD)
        self.check_grad = options.get('check-grad', False)

        self.grid_subset = 20

        if 'chooser-args' in options:
            self.parallel_opt = bool(options['chooser-args'].get('parallel-opt', False))
        else:
            self.parallel_opt = False

        self.models      = {}
        self.objective   = {}
        self.constraints = defaultdict(dict)
        self.grid        = None
        self.task_group  = None
        self.isFit = False

    def fit(self, task_group, hypers=None, options=None):
        """return a set of hyper parameters for the model fitted to the data
        
        Parameters
        ----------
        task_goup : object of type TaskGroup
        """
        self.task_group = task_group
        self.num_dims   = task_group.num_dims
        new_hypers      = {}

        # Create the grid of optimization initializers
        # Need to do it here because it's used in many places e.g. best
        self.grid = sobol_grid.generate(self.num_dims, grid_size=self.grid_size, 
                                        grid_seed=self.grid_seed)

        # A useful hack: add previously visited points to the grid
        for task_name, task in task_group.tasks.iteritems():
            if task.has_valid_inputs():
                self.grid = np.append(self.grid, task.valid_normalized_data_dict['inputs'], axis=0)
            if task.has_pending():
                self.grid = np.append(self.grid, task.valid_normalized_data_dict['pending'], axis=0)

        # A fallback in case you don't fit -- just submit this
        # index off the grid
        self.design_index = task_group.inputs.shape[0] + task_group.pending.shape[0]

        hypers = hypers if hypers is not None else defaultdict(dict)

        # print 'Fittings tasks: %s' % str(task_group.tasks.keys())

        for task_name, task in task_group.tasks.iteritems():
            if task.type.lower() == 'objective':
                data_dict = self.objective # confusing: this is how self.objective gets populated
            elif task.type.lower() == 'constraint':
                self.constraints[task_name] = {}
                data_dict = self.constraints[task_name]
            else:
                raise Exception('Unknown task type.')

            data_dict['num_dims'] = task_group.num_dims
            data_dict['name']     = task_name
            data_dict.update(task.valid_normalized_data_dict)

            # print 'Task %s (%s %s): found %d value%s' % (task_name, 
            #     task.options['likelihood'].lower(), task.type.lower(), 
            #     data_dict['inputs'].shape[0], 's' if data_dict['inputs'].shape[0] != 1 else '')

            if task.valid_values.shape[0] >= DEFAULT_NUMDESIGN:
                # Add the valid data from the task to the data_dict

                default_model = 'GP' if task.options['likelihood'].lower() in ['gaussian', 'noiseless'] else 'GPClassifier'
                model_class   = task.options.get('model', default_model)

                self.models[task_name] = getattr(models, model_class)(task_group.num_dims, **task.options)

                vals = data_dict['values'] if data_dict.has_key('values') else data_dict['counts']

                sys.stderr.write('Fitting %s for %s task...\n' % (model_class, task_name))
                new_hypers[task_name] = self.models[task_name].fit(
                    data_dict['inputs'],
                    vals,
                    pending=data_dict['pending'],
                    hypers=hypers.get(task_name, None)
                )

        self.isFit = True

        return new_hypers

    def suggest(self):
        sys.stderr.write('Getting suggestion...\n')
        assert not np.any(self.grid < 0)
        assert not np.any(self.grid > 1)

        if not self.isFit:
            raise Exception("You must call fit() before calling suggest()")

        if self.objective['inputs'].shape[0] < DEFAULT_NUMDESIGN:
            suggestion = self.task_group.from_unit(self.grid[self.design_index])
            sys.stderr.write("\nSuggestion:     ")
            self.task_group.paramify_and_print(suggestion.flatten(), left_indent=16)
            return suggestion

        # print 'inputs: %s' % self.objective['inputs']
        # if self.objective.has_key('pending'):
            # print 'pending: %s' % self.objective['pending']

        # Compute the current best
        current_best, current_best_location = self.best()

        # Add some extra candidates around the best so far (a useful hack)
        spray_points = npr.randn(self.num_spray, self.num_dims)*self.spray_std + current_best_location
        spray_points = np.minimum(np.maximum(spray_points,0.0),1.0)
        
        # Compute EI on the grid
        grid_pred = np.vstack((self.grid, spray_points))
        grid_ei = self.acquisition_function_over_hypers(grid_pred, current_best, compute_grad=False)

        # Find the points on the grid with highest EI
        best_grid_inds = np.argsort(grid_ei)[-self.grid_subset:]
        best_grid_pred = grid_pred[best_grid_inds]

        # The index and value of the top grid point
        best_grid_ind = np.argmax(grid_ei)
        best_grid_ei  = grid_ei[best_grid_ind]
        
        if VERBOSE:
            print 'Best EI before optimization: %f' % best_grid_ei

        if self.check_grad:
            check_grad(lambda x: self.acq_optimize_wrapper(x, current_best, True), 
                best_grid_pred[0], verbose=True)

        # Optimize the top points from the grid to get better points
        cand = []
        b = [(0,1)]*best_grid_pred.shape[1]# optimization bounds

        if self.parallel_opt:
            # Optimize each point in parallel
            pool = multiprocessing.Pool(self.grid_subset)
            results = [pool.apply_async(self.optimize_pt,args=(
                    c,b,current_best,True)) for c in best_grid_pred]

            for res in results:
                cand.append(res.get(1e8))
            pool.close()
        else: 
            # Optimize in series
            for c in best_grid_pred:
                cand.append(self.optimize_pt(c,b,current_best,compute_grad=True))
        # Cand now stores the optimized points

        # Compute one more time (re-computing is unnecessary, oh well... TODO)
        cand = np.vstack(cand)
        opt_ei = self.acquisition_function_over_hypers(cand, current_best, compute_grad=False)

        # The index and value of the top optimized point
        best_opt_ind  = np.argmax(opt_ei)
        best_opt_ei   = opt_ei[best_opt_ind]

        # Optimization should always be better unless the optimization
        # breaks in some way.
        if VERBOSE:
            print 'Best EI after  optimization: %f' % best_opt_ei
            print 'Suggested input %s' % cand[best_opt_ind]

        if best_opt_ei >= best_grid_ei:
            suggestion = cand[best_opt_ind]
        else:
            suggestion = grid_pred[best_grid_ind]

        # Make sure BFGS didn't do anything weird with the boudns
        suggestion[suggestion > 1] = 1.0
        suggestion[suggestion < 0] = 0.0

        suggestion = self.task_group.from_unit(suggestion)

        sys.stderr.write("\nSuggestion:     ")
        self.task_group.paramify_and_print(suggestion.flatten(), left_indent=16)
        return suggestion

    # TODO: add optimization in here
    def best(self):
        grid = self.grid
        obj_task = self.task_group.tasks[self.objective['name']]
        obj_model = self.models[self.objective['name']]

        # If unconstrained
        if self.numConstraints() == 0:
            # Compute the GP mean
            obj_mean, obj_var = obj_model.function_over_hypers(obj_model.predict, grid)

            # find the min and argmin of the GP mean
            current_best_location = grid[np.argmin(obj_mean),:][None]
            best_ind = np.argmin(obj_mean)
            current_best_value = obj_mean[best_ind]
            std_at_best = np.sqrt(obj_var[best_ind])

            # un-normalize the min of mean to original units
            unnormalized_best_value = obj_task.unstandardize_mean(obj_task.unstandardize_variance(current_best_value))
            unnormalized_std_at_best = obj_task.unstandardize_variance(std_at_best)

            # Print out the minimum according to the model
            sys.stderr.write('\nMinimum expected objective value under model '
                'is %.5f (+/- %.5f), at location:\n' % (unnormalized_best_value, unnormalized_std_at_best))
            self.task_group.paramify_and_print(self.task_group.from_unit(current_best_location).flatten(), 
                                               left_indent=16, indent_top_row=True)

            # Compute the best value seen so far
            vals = self.task_group.values[self.objective['name']]
            inps = self.task_group.inputs
            best_observed_value = np.min(vals)
            best_observed_location = inps[np.argmin(vals),:][None]

            # Don't need to un-normalize inputs here because these are the raw inputs
            sys.stderr.write('\nMinimum of observed values is %f, at location:\n' % best_observed_value)
            self.task_group.paramify_and_print(best_observed_location.flatten(), left_indent=16, indent_top_row=True)

        else:

            mc = self.probabilistic_constraint(grid)
            if not np.any(mc): 
                # P-con is violated everywhere
                # Compute the product of the probabilities, and return None for the current best value
                probs = reduce(lambda x,y:x*y, [self.confidence(c, grid) for c in self.constraints], np.ones(grid.shape[0]))
                best_probs_ind = np.argmax(probs)
                best_probs_location = grid[best_probs_ind,:][None]
                # TODO -- could use BFGS for this (unconstrained) optimization as well -- everytime for min of mean

                sys.stderr.write('\nNo feasible region found (yet).\n')
                sys.stderr.write('Maximum probability of satisfying constraints = %f\n' % np.max(probs))
                sys.stderr.write('At location:    ')
                self.task_group.paramify_and_print(self.task_group.from_unit(best_probs_location).flatten(), 
                                                   left_indent=16)
                
                return None, best_probs_location

            # A feasible region has been found

            # Compute GP mean and find minimum
            mean, var = obj_model.function_over_hypers(obj_model.predict, grid)
            valid_mean = mean[mc]
            valid_var = var[mc]
            best_ind = np.argmin(valid_mean)
            current_best_location = (grid[mc])[best_ind,:][None]
            ind = np.argmin(valid_mean)
            current_best_value = valid_mean[ind]
            std_at_best = np.sqrt(valid_var[ind])

            unnormalized_best = obj_task.unstandardize_mean(obj_task.unstandardize_variance(current_best_value))
            unnormalized_std_at_best = obj_task.unstandardize_variance(std_at_best) # not used -- not quite
            # right to report this -- i mean there is uncertainty in the constraints too
            # this is the variance at that location, not the standard deviation of the minimum... 
            # not sure if this distinction is a big deal

            sys.stderr.write('\nMinimum expected objective value satisfying constraints w/ high prob: %f\n' % unnormalized_best)
            sys.stderr.write('At location:    ')
            self.task_group.paramify_and_print(self.task_group.from_unit(current_best_location).flatten(), left_indent=16)

            # Compute the best value seen so far
            with np.errstate(invalid='ignore'):
                all_constraints_satisfied = np.all(np.greater(np.array([x.values for x in self.task_group.tasks.values()]), 0), axis=0)
            if not np.any(all_constraints_satisfied):
                sys.stderr.write('No observed result satisfied all constraints.\n')
            else:
                inps = self.task_group.inputs
                vals = self.task_group.values[self.objective['name']]
                # get rid of those that violate constraints
                vals[np.logical_not(all_constraints_satisfied)] = np.max(vals)            
                # get rid of NaNs -- set them to biggest not-nan value, then they won't be the minimum
                vals[np.isnan(vals)] = np.max(vals[np.logical_not(np.isnan(vals))])
                best_observed_value = np.min(vals)
                best_observed_location = inps[np.argmin(vals),:][None]
                # Don't need to un-normalize inputs here because these are the raw inputs
                sys.stderr.write('\nBest observed values satisfying constraints is %f, at location:\n' % best_observed_value)
                self.task_group.paramify_and_print(best_observed_location.flatten(), left_indent=16, indent_top_row=True)


        # Return according to model, not observed
        return current_best_value, current_best_location

    def numConstraints(self):
        return len(self.constraints)

    # The confidence that conststraint c is satisfied
    def confidence(self, c, grid, compute_grad=False):
        return self.models[c].function_over_hypers(self.models[c].pi, grid, compute_grad=compute_grad)

    # Returns a boolean array of size pred.shape[0] indicating whether the prob con-constraint is satisfied there
    def probabilistic_constraint(self, pred):
        return reduce(np.logical_and, 
            [self.confidence(c, pred) >= self.task_group.tasks[c].options.get('min-confidence', 0.99)
                for c in self.constraints], 
                np.ones(pred.shape[0], dtype=bool))

    def acquisition_function_over_hypers(self, *args, **kwargs):
        return function_over_hypers(self.models.values(), self.acquisition_function, *args, **kwargs)

    def acquisition_function(self, cand, current_best, compute_grad=True):
        obj_model = self.models[self.objective['name']]

        # If unconstrained, just compute regular ei
        if self.numConstraints() == 0:
            return compute_ei(obj_model, cand, ei_target=current_best, compute_grad=compute_grad)

        if cand.ndim == 1:
            cand = cand[None]

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
        p_valid, p_grad = list(), list()
        for c in self.constraints:
            if compute_grad:
                pv, pvg = self.models[c].pi(cand, compute_grad=True)
                p_valid.append(pv)
                p_grad.append(pvg)
            else:
                p_valid.append(self.models[c].pi(cand, compute_grad=False))
        
        p_valid_prod = reduce(np.multiply, p_valid, np.ones(N_cand))

        # To compute the gradient, need to do the chain rule for the product of N factors
        if compute_grad:
            p_grad_prod = np.zeros(p_grad[0].shape)
            for i in xrange(self.numConstraints()):
                pg = p_grad[i]
                for j in xrange(self.numConstraints()):
                    if j == i:
                        continue
                    pg *= p_valid[j]
                p_grad_prod += pg
            # multiply that gradient by all other pv's (this might be numerically disasterous if pv=0...)

        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############    Combine the two parts (obj and con)   ############
        ##############                                          ############
        ############## ---------------------------------------- ############

        acq = ei * p_valid_prod

        if not compute_grad:
            return acq
        else:
            return acq, ei_grad * p_valid_prod + p_grad_prod * ei

    # Flip the sign so that we are maximizing with BFGS instead of minimizing
    def acq_optimize_wrapper(self, cand, current_best, compute_grad):
        ret = self.acquisition_function_over_hypers(cand, current_best, compute_grad=compute_grad)

        if isinstance(ret, tuple) or isinstance(ret, list):
            return (-ret[0],-ret[1].flatten())
        else:
            return -ret

    def optimize_pt(self, initializer, bounds, current_best, compute_grad=True):
        opt_x, opt_y, opt_info = spo.fmin_l_bfgs_b(self.acq_optimize_wrapper,
                initializer.flatten(), args=(current_best,compute_grad),
                bounds=bounds, disp=0, approx_grad=(not compute_grad))
        return opt_x

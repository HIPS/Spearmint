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
import scipy.stats    as sps
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.acquisition_functions.constraints_helper_functions import total_constraint_confidence

def compute_ei(model, pred, ei_target=None, compute_grad=True):
    if model.pending:
        return compute_ei_pending(model, pred, ei_target, compute_grad)

    if pred.ndim == 1:
        pred = pred[None,:]

    if not compute_grad:
        mean, var = model.predict(pred)
    else:
        mean, var, grad_mean_x, grad_var_x = model.predict(pred, compute_grad=True)

    # Expected improvement
    sigma  = np.sqrt(var)
    z      = (ei_target - mean) / sigma
    ncdf   = sps.norm.cdf(z)
    npdf   = sps.norm.pdf(z)
    ei     = (ei_target - mean) * ncdf + sigma * npdf

    if not compute_grad:
        return ei

    # ei = np.sum(ei)

    # Gradients of ei w.r.t. mean and variance            
    g_ei_m = -ncdf
    g_ei_s2 = 0.5*npdf / sigma
    
    # Gradient of ei w.r.t. the inputs
    grad_ei_x = grad_mean_x*g_ei_m + grad_var_x*g_ei_s2

    return ei, grad_ei_x

def compute_ei_pending(model, pred, ei_target=None, compute_grad=True):
    # TODO: use ei_target!!
    if pred.ndim == 1:
        pred = pred[None,:]

    if not compute_grad:
        func_m, func_v = model.predict(pred)
    else:
        (func_m,
        func_v,
        grad_xp_m,
        grad_xp_v) = model.predict(pred, compute_grad=True)

    if func_m.ndim == 1:
        func_m = func_m[:,np.newaxis]
    if func_v.ndim == 1:
        func_v = func_v[:,np.newaxis]

    if compute_grad:
        if grad_xp_m.ndim == 2:
            grad_xp_m = grad_xp_m[:,:,np.newaxis]
        if grad_xp_v.ndim == 2:
            grad_xp_v = grad_xp_v[:,:,np.newaxis]

    ei_values = model.values.min(axis=0)

    ei_values = np.array(ei_values)
    if ei_values.ndim == 0:
        ei_values = np.array([[ei_values]])

    # Expected improvement
    func_s = np.sqrt(func_v)
    u      = (ei_values - func_m) / func_s
    ncdf   = sps.norm.cdf(u)
    npdf   = sps.norm.pdf(u)
    ei     = np.mean(func_s*( u*ncdf + npdf),axis=1)

    if not compute_grad:
        return ei

    ei = np.sum(ei)

    # Gradients of ei w.r.t. mean and variance            
    g_ei_m = -ncdf
    g_ei_s2 = 0.5*npdf / func_s
    
    # Gradient of ei w.r.t. the inputs
    grad_xp = (grad_xp_m*np.tile(g_ei_m,(pred.shape[1],1))).T + (grad_xp_v.T*g_ei_s2).T
    grad_xp = np.mean(grad_xp,axis=0)

    return ei, grad_xp.flatten()


def constraint_weighted_ei(obj_model, constraint_models, cand, current_best, compute_grad):

    numConstraints = len(constraint_models)

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
    if not compute_grad:
        p_valid_prod = total_constraint_confidence(constraint_models, cand, compute_grad=False)
    else:
        p_valid_prod, p_grad_prod = total_constraint_confidence(constraint_models, cand, compute_grad=True)

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


def ei_evaluate_constraint_only(obj_model, constraint_models, cand, current_best, compute_grad):

    improvement = lambda mean: np.maximum(0.0, current_best-mean)
    improvement_grad = lambda mean_grad: -mean_grad*np.less(mean, current_best).flatten()

    # If unconstrained, just compute the GP mean
    if len(constraint_models) == 0:
        if compute_grad:
            mean, var, grad_mean_x, grad_var_x = obj_model.predict(cand, compute_grad=True)
            return improvement(mean), improvement_grad(grad_mean_x)
        else:
            mean, var = obj_model.predict(cand, compute_grad=False)
            return improvement(mean)

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
            mean, var = obj_model.predict(cand, compute_grad=False)
        else:
            mean, var, grad_mean_x, grad_var_x = obj_model.predict(cand, compute_grad=True)
            ei_grad = improvement_grad(grad_mean_x)
        ei = improvement(mean)
    ############## ---------------------------------------- ############
    ##############                                          ############
    ##############  Part that depends on the constraints    ############
    ##############                                          ############
    ############## ---------------------------------------- ############
    # Compute p(valid) for ALL constraints
    if not compute_grad:
        p_valid_prod = total_constraint_confidence(constraint_models, cand, compute_grad=False)
    else:
        p_valid_prod, p_grad_prod = total_constraint_confidence(constraint_models, cand, compute_grad=True)

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




class ExpectedImprovement(AbstractAcquisitionFunction):
    """ This is regular expected improvement when there are no constraints,
        and the constraint-weighted EI when there are constraints. """
    def create_acquisition_function(self, objective_model_dict, constraint_models_dict, current_best, **kwargs):

        objective_model = objective_model_dict.values()[0]
        if len(constraint_models_dict) == 0:
            return lambda cand, compute_grad, **kwargs: compute_ei(objective_model, cand, 
                ei_target=current_best, compute_grad=compute_grad)
        else:
            return lambda cand, compute_grad, **kwargs: constraint_weighted_ei(objective_model, constraint_models_dict.values(), cand, current_best, compute_grad)


class ConstraintAndMean(AbstractAcquisitionFunction):
    """  This class is not useful in most contexts. I suggest never using it.
 It computes real the expected improvement assuming you only evaluate the constraints
 Probability of satisfying the constraints multiplied
 by the objective GP mean "improvement", 
 max(0,best-mean)
 rather than EI. the problem with it is that it 
 suffers from the chicken and egg pathology for decoupled constraints.
 It's good if the objective is known and the constraint is unknown  """
    def create_acquisition_function(self, objective_model, constraint_models_dict, current_best):
        return lambda cand, compute_grad, **kwargs: ei_evaluate_constraint_only(objective_model, constraint_models_dict.values(), cand, current_best, compute_grad)
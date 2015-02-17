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
import pdb
import copy
import traceback
import warnings

from collections import defaultdict
from spearmint.grids import sobol_grid
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.utils.numerics import logcdf_robust
from spearmint.models.gp import GP

import logging

try:
    import nlopt
except:
    nlopt_imported = False
else:
    nlopt_imported = True
# see http://ab-initio.mit.edu/wiki/index.php/NLopt_Python_Reference


NUM_RANDOM_FEATURES = 1000
GRID_SIZE = 1000 # used in a hacky way when chooser passes in the grid

"""
FOR GP MODELS ONLY
"""

"""
The thing is that for each sample of the hyper-parameters, you have to
draw a sample of x_\star, the global solution of the problem, from its
posterior distribution. However, it may happen that for some samples
of the hyper-parameters, the posterior probability of the constraint
being satisfied at least at one point of the input space can be very
small and close to zero (when you condition to the drawn
hyper-parameters). This may happen when for example you have collected
data only at two infeasible locations that have the same value of the
constraint (assuming zero noise). Then you may get some posterior
samples of the hyper-parameters for the constraint that say that the
constraint is a constant function. When this happens there are no
feasible points and you cannot sample x_\star. What to do?
"""
# get samples of the solution to the problem
def sample_solution(grid, num_dims, objective_gp, constraint_gps=[]):
    assert num_dims == grid.shape[1]

    # 1. The procedure is: sample f and all the constraints on the grid "cand" (or use a smaller grid???)
    # 2. Look for the best point on the grid. if none exists, goto 1
    # 3. Do an optimization given this best point as the initializer

    MAX_ATTEMPTS = 10
    num_attempts = 0

    while num_attempts < MAX_ATTEMPTS:

        gp_samples = dict()
        gp_samples['objective'] = sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES)
        gp_samples['constraints'] = [sample_gp_with_random_features(constraint_gp, \
            NUM_RANDOM_FEATURES) for constraint_gp in constraint_gps]

        # f = gp_samples['objective']
        # print 'max:%f, min:%f   %d' % (np.max(f(grid, False)), np.min(f(grid, False)), len(f(grid, False)))
        # for f in gp_samples['constraints']:
        #     print 'max:%f, min:%f   %d' % (np.max(f(grid, False)), np.min(f(grid, False)), len(f(grid, False)))

        x_star_sample = global_optimization_of_GP_approximation(gp_samples, num_dims, grid)

        if x_star_sample is not None: # success
            logging.debug('successfully sampled x* in %d attempt(s)' % (num_attempts+1))

            # obj_at_x_star = gp_samples['objective'](x_star_sample, gradient=False)
            # con_at_x_star = [sample(x_star_sample, gradient=False) for sample in gp_samples['constraints']]
            # print 'Sampled x* after %d attempts. obj = %s, con=%s' % (num_attempts, obj_at_x_star, con_at_x_star)
            # print 'obj mean = %+f, var = %f' % objective_gp.predict(x_star_sample)
            # print 'c1  mean = %+f, var = %f' % constraint_gps[0].predict(x_star_sample)
            # if len(constraint_gps) > 1:
            #     print 'c2  mean = %+f, var = %f' % constraint_gps[1].predict(x_star_sample)
            return x_star_sample

        num_attempts += 1
        # print x_star_sample

        # ok this gets messy for binomial constraints, where it's not just >= 0
        # damn. need to think about this in terms of gradients
        # will it be an issue?? ... hmmm

    logging.info('Failed to sample x*')

    return None

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

# Compute log(exp(a)+exp(b)) in a robust way. 
def logSumExp_scalar(a, b):

    if a > b:
        # compute log(exp(a)+exp(b)) 
        # this is just the log-sum-exp trick but with only 2 terms in the sum
        # we chooser to factor out the largest one
        # log(exp(a)+exp(b)) = log( exp(a) [1 + exp(b-a) ] )
        # = a + log(1 + exp(b-a))
        return a + log_1_plus_exp_x(b-a)
    else:
        return b + log_1_plus_exp_x(a-b)

def logSumExp(a,b):
    if (not isinstance(a, np.ndarray) or a.size==1) and (not isinstance(b, np.ndarray) or b.size==1):
        return logSumExp_scalar(a,b)

    result = np.zeros(a.shape)
    result[a>b] =  a[a>b]  + log_1_plus_exp_x(b[a>b] -a[a>b])
    result[a<=b] = b[a<=b] + log_1_plus_exp_x(a[a<=b]-b[a<=b])
    return result



# Compute log(1+exp(x)) in a robust way
def log_1_plus_exp_x_scalar(x):
    if x < np.log(1e-6):
        # if exp(x) is very small, i.e. less than 1e-6, then we can apply the taylor expansion:
        # log(1+x) approx equals x when x is small
        return np.exp(x)
    elif x > np.log(100):
        # if exp(x) is very large, i.e. greater than 100, then we say the 1 is negligible comared to it
        # so we just return log(exp(x))=x
        return x
    else:
        return np.log(1.0+np.exp(x))

def log_1_plus_exp_x(x):
    if not isinstance(x, np.ndarray) or x.size==1:
        return log_1_plus_exp_x_scalar(x)

    result = np.log(1.0+np.exp(x)) # case 3
    result[x < np.log(1e-6)] = np.exp(x[x < np.log(1e-6)])
    result[x > np.log(100) ] = x [x > np.log(100) ]
    return result


# Compute log(1-exp(x)) in a robust way, when exp(x) is between 0 and 1 
# well, exp(x) is always bigger than 0
# but it cannot be above 1 because then we have log of a negative number
def log_1_minus_exp_x_scalar(x):
    if x < np.log(1e-6): 
        # if exp(x) is very small, i.e. less than 1e-6, then we can apply the taylor expansion:
        # log(1-x) approx equals -x when x is small
        return -np.exp(x)
    elif x > -1e-6:
        # if x > -1e-6, i.e. exp(x) > exp(-1e-6), then we do the Taylor expansion of exp(x)=1+x+...
        # then the argument of the log, 1- exp(x), becomes, approximately, 1-(1+x) = -x
        # so we are left with log(-x)
        return np.log(-x)
    else:
        return np.log(1.0-np.exp(x))

def log_1_minus_exp_x(x):
    if not isinstance(x, np.ndarray) or x.size==1:
        return log_1_minus_exp_x_scalar(x)

    assert np.all(x <= 0)

    case1 = x < np.log(1e-6) # -13.8
    case2 = x > -1e-6
    case3 = np.logical_and(x >= np.log(1e-6), x <= -1e-6)
    assert np.all(case1+case2+case3 == 1)

    result = np.zeros(x.shape)
    result[case1] = -np.exp(x[case1])
    with np.errstate(divide='ignore'): # if x is exactly 0, give -inf without complaining
        result[case2] = np.log(-x[case2])
    result[case3] = np.log(1.0-np.exp(x[case3]))

    return result

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))


def matrixInverse(M):
    return chol2inv(spla.cholesky(M, lower=False))


def ep(obj_model, con_models, x_star, minimize=True):
    # We construct the Vpred matrices and the mPred vectors

    n = obj_model.observed_values.size
    obj = 'objective'
    con = con_models.keys()
    all_tasks = con_models.copy()
    all_tasks[obj] = obj_model

    """ X contains X_star """
    X = np.append(obj_model.observed_inputs, x_star, axis=0)

    mPred         = dict()        
    Vpred         = dict()
    cholVpred     = dict()
    VpredInv      = dict()
    cholKstarstar = dict()

    for t in all_tasks:
        mPred[t], Vpred[t] = all_tasks[t].predict(X, full_cov=True)
        cholVpred[t]       = spla.cholesky(Vpred[t])
        VpredInv[t]        = chol2inv(cholVpred[t])
        # Perform a redundant computation of this thing because predict() doesn't return it...
        cholKstarstar[t]   = spla.cholesky(all_tasks[t].noiseless_kernel.cov(X))
    
    jitter = dict()
    jitter[obj] = obj_model.jitter_value()
    for c in con:
        jitter[c] = con_models[c].jitter_value()

    # We create the posterior approximation
    a = {
        'obj'      : obj,
        'constraints': con,
        'Ahfhat'   : np.zeros((n, 2, 2)), # intiialize approximate factors to 0
        'bhfhat'   : np.zeros((n, 2)), 
        'ahchat'   : defaultdict(lambda: np.zeros(n)),
        'bhchat'   : defaultdict(lambda: np.zeros(n)),
        'agchat'   : defaultdict(lambda: np.zeros(1)),
        'bgchat'   : defaultdict(lambda: np.zeros(1)),
        'm'        : defaultdict(lambda: np.zeros(n+1)),  # marginals
        'V'        : defaultdict(lambda: np.zeros((n+1, n+1))),
        'cholV'    : dict(),
        'mc'       : dict(),
        'Vc'       : dict(),
        'cholVc'   : dict(),
        'n'        : n,
        'mPred'    : mPred,
        'Vpred'    : Vpred,
        'VpredInv' : VpredInv,
        'cholKstarstar'  : cholKstarstar,
        'cholKstarstarc' : dict(),
        'jitter'   : jitter
    }

    # We update the marginals
    a = updateMarginals(a)

    # We start the main loop of EP
    convergence = False
    damping     = 1.0
    iteration   = 1
    while not convergence and n>0:

        aOld = copy.deepcopy(a)
        
        # We update the factors

        while True:

            try:
                aNew = copy.deepcopy(a)

                # We update the factors Ahfhat, bhfhat, ahchat, bhchat, agchat, bgchat
                aNew = updateFactors(aNew, damping, minimize=minimize)

                # We update the marginals V and m
                aNew = updateMarginals(aNew) 

                # We verify that we can update the factors with an update of size 0
                updateFactors(aNew, 0, minimize=minimize)

                # This is also a testing step
                checkConstraintsPSD(con_models, aNew, X)

            except npla.linalg.LinAlgError as e:

                a = aOld
                damping *= 0.5

                # print 'reducing damping to %f' % damping

                if damping < 1e-5:
                    # print 'giving up'
                    aNew = aOld
                    break  # things failed, you are done...??
            else:
                # print "success"
                break # things worked, you are done

        # We check for convergence
        a = aNew

        # print np.sum(a['m'][obj])
        # print np.sum(a['m']['c1'])
        # print np.sum(a['V'][obj])
        # print np.sum(a['V']['c1'])
        # print '-'
        # print np.sum(a['Ahfhat'])
        # print np.sum(a['bhfhat'])
        # print np.sum(a['ahchat']['c1']) # a bit off 
        # print np.sum(a['bhchat']['c1']) # a bit off
        # print np.sum(a['agchat']['c1'])
        # print np.sum(a['bgchat']['c1'])
        # print '---'

        change = 0.0
        for t in all_tasks:
            change = max(change, np.max(np.abs(a['m'][t] - aOld['m'][t])))
            change = max(change, np.max(np.abs(a['V'][t] - aOld['V'][t])))
        # print 'change=%f' % change

        if change < 1e-4 and iteration > 2:
            convergence = True

        damping   *= 0.99
        iteration += 1

    # print "**done EP"



    # We update the means and covariance matrices for the constraint functions
    for c in con_models:
        X_all                  = np.append(X, con_models[c].observed_inputs, axis=0)
        noise                  = con_models[c].noise_value()
        Kstarstar              = con_models[c].noiseless_kernel.cov(X_all)
        a['cholKstarstarc'][c] = spla.cholesky(Kstarstar)
        mTilde                 = np.concatenate((a['bhchat'][c], np.array(a['bgchat'][c]), con_models[c].observed_values / noise))
        vTilde                 = np.concatenate((a['ahchat'][c], np.array(a['agchat'][c]), np.tile(1.0 / noise, con_models[c].observed_values.size)))
        Vc_inv                 = chol2inv(a['cholKstarstarc'][c]) + np.diag(vTilde)
        if np.any(npla.eigvalsh(Vc_inv) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")
        # Vc_inv += np.eye(Vc_inv.shape[0])*jitter[c] # added by mike
        chol_Vc_inv            = spla.cholesky(Vc_inv)
        Vc                     = chol2inv(chol_Vc_inv) # a['Vc'][c]
        a['cholVc'][c]         = spla.cholesky(Vc)
        # a['mc'][c]             = np.dot(Vc, mTilde)
        a['mc'][c]             = spla.cho_solve((chol_Vc_inv, False), mTilde)
    
        # We compute the cholesky factorization of the posterior covariance functions
        a['cholV'][c] = spla.cholesky(a['V'][c])
    a['cholV'][obj] = spla.cholesky(a['V'][obj])

    return a
    # fields: cholKstarstar, cholKstarstarc, V, cholV, Vc, cholVc, m, mc
    # (Vc not actually used, but include it for consistency)


# This checks that things are PSD. We want this to trigger failure in the EP loop 
# so that damping can be reduced if needed
def checkConstraintsPSD(con_models, aNew, X):
    for c in con_models:
        X_all = np.append(X, con_models[c].observed_inputs, axis=0)
        noise   = con_models[c].noise_value()
        Kstarstar  = con_models[c].noiseless_kernel.cov(X_all)
        aNew['cholKstarstarc'][c] = spla.cholesky(Kstarstar)
        vTilde = np.concatenate((aNew['ahchat'][c], np.array(aNew['agchat'][c]),
            np.tile(1.0 / noise, con_models[c].observed_values.size)))
        Vc_inv = chol2inv(aNew['cholKstarstarc'][c]) + np.diag(vTilde)
        if np.any(npla.eigvalsh(Vc_inv) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")

# Updated a['V'] and a['m']
def updateMarginals(a):

    n = a['n']
    obj = a['obj']
    constraints = a['constraints']
    all_tasks = [obj] + constraints

    # for the objective
    vTilde = np.zeros((n+1,n+1))
    vTilde[np.eye(n+1).astype('bool')] = np.append(a['Ahfhat'][:, 0, 0], np.sum(a['Ahfhat'][: , 1, 1]))
    vTilde[:n, -1] = a['Ahfhat'][:, 0, 1]
    vTilde[-1, :n] = a['Ahfhat'][:, 0, 1]
    if np.any(npla.eigvalsh(a['VpredInv'][obj] + vTilde) < 1e-6):
        raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")    

    a['V'][obj] = matrixInverse(a['VpredInv'][obj] + vTilde)
    mTilde = np.append(a['bhfhat'][:, 0], np.sum(a['bhfhat'][:, 1]))
    a['m'][obj] = np.dot(a['V'][obj], np.dot(a['VpredInv'][obj], a['mPred'][obj]) + mTilde)

    # for the constraints
    for c in constraints:
        vTilde = np.diag(np.append(a['ahchat'][c], a['agchat'][c]))
        if np.any(npla.eigvalsh(a['VpredInv'][c] + vTilde) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")    

        a['V'][c] = matrixInverse(a['VpredInv'][c] + vTilde)
        mTilde = np.append(a['bhchat'][c], np.sum(a['bgchat'][c]))
        a['m'][c] = np.dot(a['V'][c], np.dot(a['VpredInv'][c], a['mPred'][c]) + mTilde)


    # Before returning, we verify that the variances of the cavities are positive
    for i in xrange(n):
        # We obtain the cavities
        Vfinv = matrixInverse(a['V'][obj][np.ix_([i,n],[i,n])])
        if np.any(npla.eigvalsh(Vfinv - a['Ahfhat'][i,:,:]) < 1e-6):
            raise npla.linalg.LinAlgError("Covariance matrix is not PSD!")
        for c in constraints:
            if ((1.0 / a['V'][c][i, i] - a['ahchat'][c][i]) < 1e-6):
                raise npla.linalg.LinAlgError("Negative variance in cavity!")
    for c in constraints:
        if np.any(1.0 / a['V'][c][-1, -1] - a['agchat'][c] < 1e-6):
            raise npla.linalg.LinAlgError("Negative variance in cavity!")

    return a


def updateFactors(a, damping, minimize=True):

    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    # We update the h factors
    obj = a['obj']
    constraints = a['constraints']
    all_tasks = [obj] + constraints

    k = len(constraints)
    n = a['n']

    for i in xrange(n):

        # We obtain the cavities

        cholVf = spla.cholesky(a['V'][obj][np.ix_([i,n],[i,n])])
        Vfinv = chol2inv(cholVf)
        # Vfinv = matrixInverse(a['V'][obj][np.ix_([i,n],[i,n])])
        
        VfOldinv = Vfinv - a['Ahfhat'][i,:,:]
        cholVfOldinv = spla.cholesky(VfOldinv)
        VfOld = chol2inv(cholVfOldinv)

        # VfOld = matrixInverse(Vfinv - a['Ahfhat'][i,:,:])
        # mfOld = np.dot(VfOld, np.dot(Vfinv, a['m'][obj][[i, n]]) - a['bhfhat'][i,:])
        mfOld = spla.cho_solve((cholVfOldinv,False), spla.cho_solve((cholVf,False), a['m'][obj][[i, n]]) - a['bhfhat'][i,:])

        vcOld = np.zeros(k)
        mcOld = np.zeros(k)
        for j,c in enumerate(constraints):
            vcOld[j] = 1.0 / (1.0 / a['V'][c][i,i] - a['ahchat'][c][i])
            mcOld[j] = vcOld[j] * (a['m'][c][i] / a['V'][c][i,i] - a['bhchat'][c][i])

        # We compute the updates
        alphac     = mcOld / np.sqrt(vcOld)
        s           = VfOld[0, 0] - 2.0 * VfOld[1, 0] + VfOld[1, 1]
        alpha       = (mfOld[1] - mfOld[0]) / np.sqrt(s) * sgn
        logProdPhis = np.sum(logcdf_robust(alphac)) 
        logZ        = logSumExp(logProdPhis + logcdf_robust(alpha),  log_1_minus_exp_x(logProdPhis))
        ratio       = np.exp(sps.norm.logpdf(alphac) - logZ - logcdf_robust(alphac)) * (np.exp(logZ) - 1.0)
        # dlogZdmcOld = ratio / np.sqrt(vcOld)
        d2logZdmcOld2 = -ratio * (alphac + ratio) / vcOld
        ahchatNew = -1.0 / (1.0 / d2logZdmcOld2 + vcOld)
        # bhchatNew = (mcOld + np.sqrt(vcOld) / (alphac + ratio)) * ahchatNew
        bhchatNew = -(mcOld / (1.0 / (-ratio * (alphac + ratio) / vcOld) + vcOld) + np.sqrt(vcOld) / (-vcOld / ratio + (alphac + ratio) * vcOld))
        # above: the bottom way of computing bhchatNew is more stable when (alphac + ratio) = 0

        ratio = np.exp(logProdPhis + sps.norm.logpdf(alpha) - logZ)
        dlogZdmfOld = ratio / np.sqrt(s) * np.array([-1.0, 1.0]) * sgn
        dlogZdVfOld = -0.5 * ratio * alpha / s * np.array([[1.0,-1.0],[-1.0,1.0]])
        mfNew = mfOld + spla.cho_solve((cholVfOldinv,False), dlogZdmfOld)

        VfNew = VfOld - np.dot(spla.cho_solve((cholVfOldinv,False), np.dot(dlogZdmfOld[:,None], dlogZdmfOld[None]) - 2.0 * dlogZdVfOld), VfOld)
        
        # EXTRA_JITTER = np.eye(VfNew.shape[0])*a['jitter'][obj]
        # VfNew += EXTRA_JITTER

        # this is where the linalg error gets thrown, causing damping to be reduced
        cholVfNew = spla.cholesky(VfNew)
        vfNewInv = chol2inv(cholVfNew)
        # vfNewInv = matrixInverse(VfNew)
        
        AhfHatNew = vfNewInv - (Vfinv - a['Ahfhat'][i,:,:])
        # bhfHatNew = np.dot(vfNewInv, mfNew) - (np.dot(Vfinv, a['m'][obj][[i, n]]) - a['bhfhat'][i,:])
        bhfHatNew = spla.cho_solve((cholVfNew,False), mfNew) - (spla.cho_solve((cholVf,False), a['m'][obj][[i, n]]) - a['bhfhat'][i,:])

        # We do damping
        a['Ahfhat'][i,:,:] = damping * AhfHatNew + (1.0 - damping) * a['Ahfhat'][i,:,:]
        a['bhfhat'][i,:]   = damping * bhfHatNew + (1.0 - damping) * a['bhfhat'][i,:]

        for j,c in enumerate(constraints):
            a['ahchat'][c][i] = damping * ahchatNew[j] + (1.0 - damping) * a['ahchat'][c][i]
            a['bhchat'][c][i] = damping * bhchatNew[j] + (1.0 - damping) * a['bhchat'][c][i]


    # ***
    # here we have slight deviation in the R code for some elments of Ahfhat and bhfhat
    # that are zero in one and nonzero but small in the other. I am not sure yet if this
    # is a bug or just a numerical difference in how things are computed...

    # We update the g factors
    # We obtain the cavities
    for j,c in enumerate(constraints):
        vcOld[j] = 1.0 / (1.0 / a['V'][c][n, n] - a['agchat'][c])
        mcOld[j] = vcOld[j] * (a['m'][c][n] / a['V'][c][n, n] - a['bgchat'][c])
    
    # We compute the updates
    alpha = mcOld / np.sqrt(vcOld)
    ratio = np.exp(sps.norm.logpdf(alpha) - logcdf_robust(alpha))
    # dlogZdmcOld = ratio / np.sqrt(vcOld)
    d2logZdmcOld2 = -ratio / vcOld * (alpha + ratio)
    agchatNew = -1 / (1.0 / d2logZdmcOld2 + vcOld)
    # bgchatNew = (mcOld + np.sqrt(vcOld) / (alpha + ratio)) * agchatNew
    bgchatNew = -(mcOld / (1.0 / (-ratio * (alpha + ratio) / vcOld) + vcOld) + np.sqrt(vcOld) / (-vcOld / ratio + (alpha + ratio) * vcOld))
    # above: the bottom way of computing bhchatNew is more stable when (alphac + ratio) = 0

    # We do damping
    for j,c in enumerate(constraints):
        a['agchat'][c] = damping * agchatNew[j] + (1.0 - damping) *  a['agchat'][c]
        a['bgchat'][c] = damping * bgchatNew[j] + (1.0 - damping) *  a['bgchat'][c]

    # We are done
    return a

def gp_prediction_given_chol_K(X, Xtest, chol_star, cholV, m, model, jitter):
# computes the predictive distributions. but the chol of the kernel matrix and the
# chol of the test matrix are already provided. 
    
    Kstar = model.noiseless_kernel.cross_cov(X, Xtest)
    mf = np.dot(Kstar.T, spla.cho_solve((chol_star, False), m))
    aux = spla.cho_solve((chol_star, False), Kstar)
    # vf = model.params['amp2'].value * (1.0 + jitter) - \
    #     np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
    #     np.sum(np.dot(cholV, aux)**2, axis=0)
    vf = model.params['amp2'].value - \
        np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
        np.sum(np.dot(cholV, aux)**2, axis=0) + \
        jitter


    if np.any(vf < 0.0):
        raise Exception("Encountered negative variance: %f" % np.min(vf))

    return Kstar, mf, vf

# Method that approximates the predictive distribution at a particular location.
def predictEP(obj_model, con_models, a, x_star, Xtest, minimize=True):

    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    obj = a['obj']
    constraints = con_models.keys()
    all_tasks = [obj] + constraints

    X = np.append(obj_model.observed_inputs, x_star, axis=0) 

    Kstar, mf, vf = gp_prediction_given_chol_K(X, Xtest, 
        a['cholKstarstar'][obj], a['cholV'][obj], a['m'][obj], obj_model, a['jitter'][obj])

    # We compute the covariance between the test point and the optimum
    KstarXstar = obj_model.noiseless_kernel.cross_cov(X, x_star)
    aux1       = spla.solve_triangular(a['cholKstarstar'][obj].T, Kstar, lower=True) 
    aux2       = spla.solve_triangular(a['cholKstarstar'][obj].T, KstarXstar, lower=True)
    aux11      = np.dot(a['cholV'][obj], spla.solve_triangular(a['cholKstarstar'][obj], aux1, lower=False))
    aux12      = np.dot(a['cholV'][obj], spla.solve_triangular(a['cholKstarstar'][obj], aux2, lower=False))
    cov        = Kstar[-1,:] - np.sum(aux2 * aux1, axis=0) + np.sum(aux12 * aux11, axis=0)
    # Above: in computing "cov" we use broadcasting, so we deviate a bit from the R code

    assert Kstar.shape[0] == X.shape[0]

    # We obtain the posterior mean and variance at the optimum    
    mOpt = a['m'][obj][-1]
    vOpt = a['V'][obj][-1, -1]

    # We compute the predictive distribution for the constraints
    mc = np.zeros((Xtest.shape[0], len(constraints)))
    vc = np.zeros((Xtest.shape[0], len(constraints)))
    for i,c in enumerate(constraints):
        Xc = np.append(X, con_models[c].observed_inputs, axis=0)

        Kstar, mc[:,i], vc[:,i] = gp_prediction_given_chol_K(Xc, Xtest, 
            a['cholKstarstarc'][c], a['cholVc'][c], a['mc'][c], con_models[c], a['jitter'][c])

    # scale things for stability
    scale = 1.0 - 1e-4
    while np.any(vf - 2.0 * scale * cov + vOpt < 1e-10):
        scale = scale**2
    cov = scale * cov

    # We update the predictive distribution for f to take into account that it has to be smaller than the optimum
    s       = vf - 2.0 * cov + vOpt
    alpha   = (mOpt - mf) / np.sqrt(s) * sgn
    alphac = mc / np.sqrt(vc)
    logProdPhis = np.sum(logcdf_robust(alphac), axis=1) # sum over constraints
    logZ    = logSumExp(logProdPhis + logcdf_robust(alpha), log_1_minus_exp_x(logProdPhis))
    ratio   = np.exp(logProdPhis + sps.norm.logpdf(alpha) - logZ)
    # mfNew   = mf + (cov - vf) *  ratio / np.sqrt(s) * sgn
    # above: not used in the acquisition function
    vfNew   = vf - ratio / s * (ratio + alpha) * (vf - cov)**2

    logZ = logZ[:,None] # normalization constant for the product of gaussian and factor that ensures x star is the solution # supplement line 8
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            ratio = np.exp(sps.norm.logpdf(alphac) - logZ - logcdf_robust(alphac)) * (np.exp(logZ) - 1.0)
        except RuntimeWarning:
            print ("Capped ratio!  "*50)
            ratio = np.exp(50) # cap ratio to avoid overflow issues

    # dlogZdmcOld = ratio / np.sqrt(vc)
    d2logZdmcOld2 = -ratio * (alphac + ratio) / vc

    # if -np.inf in d2logZdmcOld2:
    #     pdb.set_trace()
    #     obj_model.observed_inputs
    #     obj_model.observed_values
    #     obj_model.params['ls'].value
    #     obj_model.params['amp2'].value

    #     for con_model in con_models.values():
    #         con_model.params['ls'].value
    #         con_model.params['amp2'].value            

        # raise Exception("d2logZdmcOld2 contains -inf. this will cause vcNew to be inf")

    ahchatNew = -1.0 / (1.0 / d2logZdmcOld2 + vc)
    # bhchatNew = (mc + np.sqrt(vc) / (alphac + ratio)) * ahchatNew
    bhchatNew = -(mc / (1.0 / (-ratio * (alphac + ratio) / vc) + vc) + np.sqrt(vc) / (-vc / ratio + (alphac + ratio) * vc))
    # above: the bottom way of computing bhchatNew is more stable when (alphac + ratio) = 0
    vcNew = 1.0 / (1.0 / vc + ahchatNew)
    # mcNew = vcNew *  (mc / vc + bhchatNew)
    # mcNew not actually used

    # make sure variances are not negative, by replacing with old values
    vfNew[vfNew < 0] = vf[vfNew < 0]
    vcNew[vcNew < 0] = vc[vcNew < 0]


    if np.any(vfNew <= 0):
        raise Exception("vfnew is negative: %g at index %d" % (np.min(vfNew), np.argmin(vfNew)))
    if np.any(vcNew <= 0):
        raise Exception("vcnew is negative: %g at index %d" % (np.min(vcNew), np.argmin(vcNew)))
    # if np.any(np.isnan(mfNew)):
    #     raise Exception("mfnew contains nan at index %s" % str(np.nonzero(np.isnan(mfNew))))
    # if np.any(np.isnan(mcNew)):
    #     raise Exception("mcNew contains nan at index %s" % str(np.nonzero(np.isnan(mcNew))))
    if np.any(np.isnan(vcNew)):
        raise Exception("vcnew constrains nan")
    if np.any(np.isnan(vfNew)):
        raise Exception("vfnew constrains nan")

    return {'mf':None, 'vf':vfNew, 'mc':None, 'vc':vcNew} 
    # don't both computing mc since it's not used in the acquisition function
    # m = mean, v = var, f = objective, c = constraint



"""
See Miguel's paper (http://arxiv.org/pdf/1406.2541v1.pdf) section 2.1 and Appendix A

Returns a function the samples from the approximation...

if testing=True, it does not return the result but instead the random cosine for testing only

We express the kernel as an expectation. But then we approximate the expectation with a weighted sum
theta are the coefficients for this weighted sum. that is why we take the dot product of theta at the end
we also need to scale at the end so that it's an average of the random features. 

if use_woodbury_if_faster is False, it never uses the woodbury version
"""
def sample_gp_with_random_features(gp, nFeatures, testing=False, use_woodbury_if_faster=True):

    d = gp.num_dims
    N_data = gp.observed_values.size

    nu2 = gp.noise_value()

    sigma2 = gp.params['amp2'].value  # the kernel amplitude

    # We draw the random features
    if gp.options['kernel'] == "SquaredExp":
        W = npr.randn(nFeatures, d) / gp.params['ls'].value
    elif gp.options['kernel'] == "Matern52":
        m = 5.0/2.0
        W = npr.randn(nFeatures, d) / gp.params['ls'].value / np.sqrt(npr.gamma(shape=m, scale=1.0/m, size=(nFeatures,1)))
    else:
        raise Exception('This random feature sampling is for the squared exp or Matern5/2 kernels and you are using the %s' % gp.options['kernel'])
    b = npr.uniform(low=0, high=2*np.pi, size=nFeatures)[:,None]


    # Just for testing the  random features in W and b... doesn't test the weights theta
    if testing:
        return lambda x: np.sqrt(2 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b)

    randomness = npr.randn(nFeatures)

    # W has size nFeatures by d
    # tDesignMatrix has size Nfeatures by Ndata
    # woodbury has size Ndata by Ndata
    # z is a vector of length nFeatures

    if gp.has_data:
        tDesignMatrix = np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, gp.observed_inputs.T) + b)

        if use_woodbury_if_faster and N_data < nFeatures:
            # you can do things in cost N^2d instead of d^3 by doing this woodbury thing

            # We obtain the posterior on the coefficients
            woodbury = np.dot(tDesignMatrix.T, tDesignMatrix) + nu2*np.eye(N_data)
            chol_woodbury = spla.cholesky(woodbury)
            # inverseWoodbury = chol2inv(chol_woodbury)
            z = np.dot(tDesignMatrix, gp.observed_values / nu2)
            # m = z - np.dot(tDesignMatrix, np.dot(inverseWoodbury, np.dot(tDesignMatrix.T, z)))
            m = z - np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), np.dot(tDesignMatrix.T, z))) 
            # (above) alternative to original but with cho_solve
            
            # z = np.dot(tDesignMatrix, gp.observed_values / nu2)
            # m = np.dot(np.eye(nFeatures) - \
            # np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), tDesignMatrix.T)), z)
            
            # woodbury has size N_data by N_data
            D, U = npla.eigh(woodbury)
            # sort the eigenvalues (not sure if this matters)
            idx = D.argsort()[::-1] # in decreasing order instead of increasing
            D = D[idx]
            U = U[:,idx]
            R = 1.0 / (np.sqrt(D) * (np.sqrt(D) + np.sqrt(nu2)))
            # R = 1.0 / (D + np.sqrt(D*nu2))

            # We sample from the posterior of the coefficients
            theta = randomness - \
    np.dot(tDesignMatrix, np.dot(U, (R * np.dot(U.T, np.dot(tDesignMatrix.T, randomness))))) + m

        else:
            # all you are doing here is sampling from the posterior of the linear model
            # that approximates the GP
            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) / nu2 + np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values / nu2))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma, lower=False)).T

            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T

            chol_Sigma_inverse = spla.cholesky(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            Sigma = chol2inv(chol_Sigma_inverse)
            m = spla.cho_solve((chol_Sigma_inverse, False), np.dot(tDesignMatrix, gp.observed_values))
            theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T


    else:
        # We sample from the prior -- same for Matern
        theta = npr.randn(nFeatures)

    def wrapper(x, gradient): 
    # the argument "gradient" is 
    # not the usual compute_grad that computes BOTH when true
    # here it only computes the objective when true
        
        if x.ndim == 1:
            x = x[None,:]

        if not gradient:
            result = np.dot(theta.T, np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b))
            if result.size == 1:
                result = float(result) # if the answer is just a number, take it out of the numpy array wrapper
                # (failure to do so messed up NLopt and it only gives a cryptic error message)
            return result
        else:
            return np.dot(theta.T, -np.sqrt(2.0 * sigma2 / nFeatures) * np.sin(np.dot(W, x.T) + b) * W)
    
    return wrapper

"""
Given some approximations to the GP sample, find its minimum
We do that by first evaluating it on a grid, taking the best, and using that to
initialize an optimization. If nothing on the grid satisfies the constraint, then
we return None

wrapper_functions should be a dict with keys 'objective' and optionally 'constraints'
"""
# find MINIMUM if minimize=True, else find a maximum
def global_optimization_of_GP_approximation(funs, num_dims, grid, minimize=True):

    assert num_dims == grid.shape[1]

    num_con = len(funs['constraints'])

    # print 'evaluating on grid'
    # First, evaluate on a grid and see what you get
    obj_evals = funs['objective'](grid, gradient=False)
    con_evals = np.ones(grid.shape[0]).astype('bool')
    for con_fun in funs['constraints']:
        con_evals = np.logical_and(con_evals, con_fun(grid, gradient=False)>=0)
    # TODO-- deal with other stuff -- or make the binomial thing fit in somehow, this is becoming a mess...
    # can we give it the latent values somehow... or something?
    if not np.any(con_evals):
        return None

    if minimize:
        best_guess_index = np.argmin(obj_evals[con_evals])
        best_guess_value = np.min(obj_evals[con_evals])
    else:
        best_guess_index = np.argmax(obj_evals[con_evals])
        best_guess_value = np.max(obj_evals[con_evals])
    x_initial = grid[con_evals][best_guess_index]

    # print 'optimiszing'
    # for reference info
    # todo - use scipy optimizer as a backup

    fun_counter = defaultdict(int)

    if nlopt_imported:

        opt = nlopt.opt(nlopt.LD_MMA, num_dims)
        opt.set_lower_bounds(0.0)#np.zeros(num_dims))
        opt.set_upper_bounds(1.0)#np.ones(num_dims))

        def f(x, put_gradient_here):
            fun_counter['obj'] += 1
            if put_gradient_here.size > 0:
                # set grad to the gradient, here

                put_gradient_here[:] = funs['objective'](x, gradient=True)
            # return the value of f(x)
            return float(funs['objective'](x, gradient=False))

        if minimize:
            opt.set_min_objective(f)
        else:
            opt.set_max_objective(f)
        
        # constraints in NLopt are <= 0 constraint. So we want to take the negative of them...
        def g(put_result_here, x, put_gradient_here):
            fun_counter['con'] += 1
            for i,constraint_wrapper in enumerate(funs['constraints']):
                if put_gradient_here.size > 0:
                    put_gradient_here[i,:] = -constraint_wrapper(x, gradient=True)
                put_result_here[i] = -constraint_wrapper(x, gradient=False)

        # tol = [1e-8]*len(funs['constraints'])
        tol = np.zeros(len(funs['constraints']))
        opt.add_inequality_mconstraint(g, tol)
        opt.set_maxeval(10000)

        opt.set_xtol_abs(1e-6)

        # print 'Optimizing in %d dimensions with %s.' % (opt.get_dimension(), opt.get_algorithm_name())
        opt_x = opt.optimize(x_initial.copy())

        returncode = opt.last_optimize_result()
        y_opt = f(opt_x, np.array([]))

        # logging.debug('returncode=%d'%returncode)
        # logging.debug('Evaluated the objective %d times and constraints %d times' % (fun_counter['obj'], fun_counter['con']))
        nlopt_constraints_results = np.zeros(num_con)
        g(nlopt_constraints_results, opt_x, np.zeros(0))
        constraint_tol = 1e-8 # my tolerance, not the one I give to NLOPT
        # all_constraint_satisfied = np.all(nlopt_constraints_results <= constraint_tol)
        if (returncode > 0 or returncode==-4) and y_opt < best_guess_value:# and all_constraint_satisfied:
            return opt_x[None]
        # elif not all_constraint_satisfied:
            # logging.debug('NLOPT failed when optimizing x*: violated constraints: %g' % np.max(nlopt_constraints_results))
            # return x_initial[None]
        elif not (returncode > 0 or returncode==-4):
            logging.debug('NLOPT failed when optimizing x*: bad returncode')
            return x_initial[None]
        else:
            logging.debug('NLOPT failed when optimizing x*: objective got worse from %f to %f' %(best_guess_value, y_opt))
            return x_initial[None]


    else:
        assert minimize # todo - can fix later

        f       = lambda x: float(funs['objective'](x, gradient=False))
        f_prime = lambda x: funs['objective'](x, gradient=True).flatten()


        # with SLSQP in scipy, the constraints are written as c(x) >= 0
        def g(x):
            g_func = np.zeros(num_con)
            for i,constraint_wrapper in enumerate(funs['constraints']):
                g_func[i] = constraint_wrapper(x, gradient=False)
            return g_func

        def g_prime(x):
            g_grad_func = np.zeros((num_con, num_dims))
            for i,constraint_wrapper in enumerate(funs['constraints']):
                g_grad_func[i,:] = constraint_wrapper(x, gradient=True)
            return g_grad_func

        bounds = [(0.0,1.0)]*num_dims

        opt_x = spo.fmin_slsqp(f, x_initial.copy(), 
            bounds=bounds, disp=0, fprime=f_prime, f_ieqcons=g, fprime_ieqcons=g_prime)
        # make sure bounds are respected
        opt_x[opt_x > 1.0] = 1.0
        opt_x[opt_x < 0.0] = 0.0

        if f(opt_x) < best_guess_value and np.all(g(opt_x)>=0):
            return opt_x[None]
        else:
            logging.debug('SLSQP failed when optimizing x*')
            return x_initial[None]

    # return opt_x[None]


class PES(AbstractAcquisitionFunction):

    def __init__(self, num_dims, verbose=True, input_space=None, grid=None):
        # we want to cache these. we use a dict indexed by the state integer
        self.cached_EP_solutions = dict()
        self.cached_x_star = dict()

        self.has_gradients = False

        self.num_dims = num_dims

        # self.verbose = verbose

        if grid is None:
            self.xstar_grid = sobol_grid.generate(num_dims, grid_size=GRID_SIZE)
        else:
            # This is a total hack. We just do this to make sure we include
            # The observed points and spray points that are added on.
            # If you had more than GRID_SIZE observations this would be totally messed up...
            logging.debug('Note: grid passed in has size %d, truncating to size %d for PESC.' % (grid.shape[0], GRID_SIZE))
            self.xstar_grid = grid[-GRID_SIZE:]

        self.input_space = input_space

    # obj_models is a GP
    # con_models is a dict of named constraints and their GPs
    def acquisition(self, obj_model_dict, con_models_dict, cand, current_best, 
                    compute_grad, DEBUG_xstar=None, minimize=True, tasks=None):
        obj_model = obj_model_dict.values()[0]

        models = [obj_model] + list(con_models_dict.values())

        for model in models:
            if model.pending is not None:
                raise NotImplementedError("PES not implemented for pending stuff? Not sure. Should just impute the mean...")

            if not model.options['caching']:
                logging.error("Warning: caching is off while using PES!")

        # make sure all models are at the same state
        assert len({model.state for model in models}) == 1, "Models are not all at the same state"
        assert not compute_grad 

        N_cand = cand.shape[0]

        # If the epSolution is already saved, load it.
        # Otherwise, compute it with the ep() function and save it
        # print ''
        if obj_model.state in self.cached_EP_solutions:
            x_star     = self.cached_x_star[obj_model.state]
            epSolution = self.cached_EP_solutions[obj_model.state]
        else:
            #logging.debug('Sampling solution for hyper sample %d' % obj_model.state)
            if DEBUG_xstar is None:
                x_star = sample_solution(self.xstar_grid, self.num_dims, obj_model, con_models_dict.values())

                if x_star is None:
                    self.cached_x_star[obj_model.state] = None
                    self.cached_EP_solutions[obj_model.state] = None
                    return np.zeros(cand.shape[0])

                if self.input_space:
                    logging.debug('x* = %s' % self.input_space.from_unit(x_star))
                else:
                    logging.debug('x* = %s' % str(x_star))
            else:
                logging.debug('DEBUG MODE: using xstar value of %s' % str(DEBUG_xstar))
                x_star = DEBUG_xstar

            # logging.debug('Doing EP for hyper sample %d' % obj_model.state)
            with np.errstate(divide='ignore',over='ignore'):
                epSolution = ep(obj_model, con_models_dict, x_star, minimize=minimize)

            self.cached_x_star[obj_model.state] = x_star
            self.cached_EP_solutions[obj_model.state] = epSolution

        # if you failed to sample a solution, just return 0
        if x_star is None:
            return np.zeros(cand.shape[0])

        # use the EP solutions to compute the acquisition function 
        # (in the R code, this is the function evaluateAcquisitionFunction, which calls predictEP)
        acq_dict = evaluate_acquisition_function_given_EP_solution(obj_model_dict, con_models_dict, cand, epSolution, x_star, minimize=minimize)

        # by default, sum the PESC contribution for all tasks
        if tasks is None:
            tasks = acq_dict.keys()

        # Compute the total acquisition function for the tasks of interests
        total_acq = 0.0
        for task in tasks:
            total_acq += acq_dict[task]

        return total_acq


# Returns the PES(C) for each task given the EP solution and sampled x_star. 
def evaluate_acquisition_function_given_EP_solution(obj_model_dict, con_models, cand, epSolution, x_star, minimize=True):
    if cand.ndim == 1:
        cand = cand[None]

    N_cand = cand.shape[0]
    obj_name  = obj_model_dict.keys()[0]
    obj_model = obj_model_dict.values()[0]

    # unconstrainedVariances = np.zeros((N_cand, len(con_models)+1))
    # unconstrainedVariances[:,0] = obj_model.predict(cand)[1] + obj_model.noise_value()
    # for j, c in enumerate(con_models):
    #     unconstrainedVariances[:,j+1] = con_models[c].predict(cand)[1] + con_models[c].noise_value()
    unconstrainedVariances = dict()
    unconstrainedVariances[obj_name] = obj_model.predict(cand)[1] + obj_model.noise_value()
    for c, con_model in con_models.iteritems():
        unconstrainedVariances[c] = con_model.predict(cand)[1] + con_model.noise_value()

    # We then evaluate the constrained variances
    with np.errstate(divide='ignore', over='ignore'):
        predictionEP = predictEP(obj_model, con_models, epSolution, x_star, cand, minimize=minimize)

    # constrainedVariances = np.zeros((N_cand, len(con_models)+1))
    # constrainedVariances[:,0] = predictionEP['vf'] + obj_model.noise_value()
    # for j, c in enumerate(con_models):
    #     constrainedVariances[:, j+1] = predictionEP['vc'][:,j] + con_models[c].noise_value()
    constrainedVariances = dict()
    constrainedVariances[obj_name] = predictionEP['vf'] + obj_model.noise_value()
    for j, c in enumerate(con_models):
        constrainedVariances[c] = predictionEP['vc'][:,j] + con_models[c].noise_value()

    # if N_cand == 1:
    #     print 'cand=%s,  x*+%s' % (cand, x_star)
    #     print 'obj:%+g = %g - %g;  noise=%g' % (unconstrainedVariances[0,0]-constrainedVariances[0,0], unconstrainedVariances[0,0], constrainedVariances[0,0], obj_model.noise_value())
    #     print 'co1:%+g = %g - %g;  noise=%g' % (unconstrainedVariances[0,1]-constrainedVariances[0,1], unconstrainedVariances[0,1], constrainedVariances[0,1], con_models.values()[0].noise_value())
    #     print 'co2:%+g = %g - %g;  noise=%g' % (unconstrainedVariances[0,2]-constrainedVariances[0,2], unconstrainedVariances[0,2], constrainedVariances[0,2], con_models.values()[1].noise_value())
    #     print 'conLHS:%+g' % np.sum(np.log(2 * np.pi * np.e * unconstrainedVariances[0,1:]))
    #     print 'conRHS:%+g' % np.sum(np.log(2 * np.pi * np.e * constrainedVariances[0,1:]))

    # We only care about the variances because the means do not affect the entropy

    acq = dict()
    for t in unconstrainedVariances:
        acq[t] = 0.5 * np.log(2 * np.pi * np.e * unconstrainedVariances[t]) - \
                 0.5 * np.log(2 * np.pi * np.e * constrainedVariances[t])

    # acq = np.sum(0.5 * np.log(2 * np.pi * np.e * unconstrainedVariances), axis=1) - \
          # np.sum(0.5 * np.log(2 * np.pi * np.e * constrainedVariances)  , axis=1)
    # assert not np.any(np.isnan(acq)), "Acquisition function contains NaN"

    for t in acq:
        if np.any(np.isnan(acq[t])):
            raise Exception("Acquisition function containts NaN for task %s" % t)

    return acq


def test_random_features_sampling():
    D = 1
    N = 12
    
    inputs  = npr.rand(N,D)
    # W       = npr.randn(D,1)
    # vals    = np.dot(inputs**2, W).flatten() + np.sqrt(1e-3)*npr.randn(N)
    # vals = npr.randn(N)   
    vals = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
    gp = GP(D, kernel='SquaredExp', likelihood='noiseless', stability_jitter=1e-10)
    gp.fit(inputs, vals, fit_hypers=False)

    print 'ls=%s' % str(gp.params['ls'].value)
    print 'noise=%f' % float(gp.noise_value())
    print 'amp2=%f' % float(gp.params['amp2'].value)

    """
    Test the function sample_gp_with_random_features by taking the dot product
    of the random cosine functions and comparing them to the kernel...
    Right, because these are like the finite feature space, whereas the kernel is
    like an infinite feature space. So as the number of features grows the result
    should approach the kernel
    """
    num_test_inputs = 20
    test_input_1 = 5*npr.randn(num_test_inputs,D)
    test_input_2 = 5*npr.randn(num_test_inputs,D)
    # print test_input_1
    # print test_input_2
    K = gp.scaled_input_kernel.cross_cov(test_input_1, test_input_2)

    print 'Error between the real coveraiance matrix and the approximated covariance matrix'
    nmax = 5
    for log_nFeatures in np.arange(0,nmax+1):
        tst_fun = sample_gp_with_random_features(gp, nFeatures=10**log_nFeatures, testing=True)
        this_should_be_like_K = np.dot(tst_fun(test_input_1).T, tst_fun(test_input_2))
        # print '%f, %f' % (K, this_should_be_like_K)
        print 'nFeatures = 10^%d, average absolute error = %f' % (log_nFeatures, np.mean(np.abs(K-this_should_be_like_K)))

    # The above test is good for the random features. But we should also test theta somehow. 
    print 'difference between predicted mean at the inputs and the true values (should be 0 if noiseless): %f' % np.mean(np.abs(gp.predict(inputs)[0]-vals))
    print 'Error between the predicted mean using the GP approximation, and the true values'
    for log_nFeatures in np.arange(0,nmax+1):
        wrapper = sample_gp_with_random_features(gp, nFeatures=10**log_nFeatures)
        print 'nFeatures = 10^%d, error on true values = %f' % (log_nFeatures, np.mean(np.abs(vals-wrapper(inputs, gradient=False))))
        # print 'True values: %s' % str(vals)
        # print 'Approximated values: %s' % str(wrapper(inputs, gradient=False))

    # print 'at test, sampled val = %s' % wrapper(inputs[0][None], gradient=False)
    # print 'at test, mean=%f,var=%f' % gp.predict(inputs[0][None])



    # Now test the mean and covariance at some test points?
    test = npr.randn(2, D)
    # test[1,:] = test[0,:]+npr.randn(1,D)*0.2

    m, cv = gp.predict(test, full_cov=True)
    print 'true mean = %s' % m
    print 'true cov = \n%s' % cv

    n_samples = int(1e4)
    samples = gp.sample_from_posterior_given_hypers_and_data(test, n_samples=n_samples, joint=True)
    true_mean = np.mean(samples, axis=1)
    true_cov = np.cov(samples)
    print ''
    print 'mean of %d gp samples = %s' % (n_samples, true_mean)
    print 'cov of %d gp samples = \n%s' % (n_samples, true_cov)

    import sys
    approx_samples = 0.0*samples
    for i in xrange(n_samples):
        if i % (n_samples/100) == 0:
            sys.stdout.write('%02d%% ' % (i/((n_samples/100))))
            sys.stdout.flush()
        wrapper = sample_gp_with_random_features(gp, nFeatures=10000, use_woodbury_if_faster=True)
        samples[:,i] = np.array(wrapper(test, gradient=False)).T
    approx_mean = np.mean(samples, axis=1)
    approx_cov = np.cov(samples)
    print ''
    print 'mean of %d approx samples = %s' % (n_samples, approx_mean)
    print 'cov of %d approx samples = \n%s' % (n_samples, approx_cov)

    print ''
    print 'error of true means = %s' % np.sum(np.abs(true_mean-m))
    print 'error of true covs = %s' % np.sum(np.abs(true_cov-cv))
    print 'error of approx means = %s' % np.sum(np.abs(approx_mean-m))
    print 'error of approx covs = %s' % np.sum(np.abs(approx_cov-cv))


def test_x_star_sampling():
    D = 1
    N = 12
    
    inputs  = npr.rand(N,D)
    # W       = npr.randn(D,1)
    # vals    = np.dot(inputs**2, W).flatten() + np.sqrt(1e-3)*npr.randn(N)
    # vals = npr.randn(N)   
    vals = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
    objective = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
    objective.fit(inputs, vals, fit_hypers=False)

    print 'ls=%s' % str(objective.params['ls'].value)
    print 'noise=%f' % float(objective.params['noise'].value)
    print 'amp2=%f' % float(objective.params['amp2'].value)

    wrapper = sample_gp_with_random_features(objective, nFeatures=100) # test
    
    constraint_vals = npr.randn(N)
    constraint = GP(D)#, kernel='SquaredExp')
    constraint.fit(inputs, constraint_vals, fit_hypers=False)


    grid = sobol_grid.generate(D, grid_size=GRID_SIZE) 
    
    minimum = sample_solution(grid, D, objective, [constraint])
    print 'Constrained Minimum of a different samples = %s' % str(minimum)

    print 'plotting'
    if D == 1:
        import matplotlib.pyplot as plt
        spacing = np.linspace(0,1,1000)[:,None]

        approx = wrapper(spacing, gradient=False)
        approx_grad = wrapper(spacing, gradient=True)

        unconstrained_minimum = global_optimization_of_GP_approximation({'objective':wrapper, 'constraints':[]}, D, grid)
        print 'Unconstrained minimum = %s' % str(unconstrained_minimum)

        plt.figure()
        plt.plot(inputs, vals, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(spacing, approx, 'b.')
        plt.plot(spacing, approx_grad, 'g')
        plt.plot(unconstrained_minimum, wrapper(unconstrained_minimum, gradient=False), color='orange', marker='*', markersize=20)
        plt.show()


def test_acquisition_function():

    D = 2
    N = 4
    num_test_inputs = 20

    while True:

        inputs  = npr.rand(N,D)
        # vals = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
        
        # 2d
        vals = npr.randn(N)
        constraint_vals = npr.randn(N)
        test_input = 5*npr.randn(num_test_inputs,D)

        # print inputs
        # print vals
        # print constraint_vals

        # 2d
        # inputs = np.array([[ 0.206,  0.77], [ 0.421,  0.40], [ 0.50, 0.22], [ 0.69 , 0.149]])
        # vals = np.array([ 0.874,    1.3, -0.157, -0.77])
        # constraint_vals = np.array([ 0.74,  -1.16, -1.144,  0.94])
        # x_star = np.array([0.63, 0.0])[None]
        # test_input = np.array([0.854, 0.518])[None]

        # 1d
        # inputs = np.array([0.007, 0.72, 0.75, 0.4])[:,None]
        # vals = np.array([-0.045, -0.8, -0.5, 0.1])
        # constraint_vals = np.array([0.44, 0.85, 2.13, -0.2])
        # x_star = np.array([0.7])[:,None]
        
        x_star = None

        # test_input = np.linspace(0,1,num_test_inputs)[:,None]
        # test_input = npr.rand(N,D)
        # print test_input
        # test_input = np.array([[0,1],[1,1],[0,0],[1,0]])
        
        # the noise is 1e-6, the stability jitter is 1e-10
        STABILITY_JITTER = 1e-10

        objective = GP(D, 
            mcmc_iters=0, 
            likelihood='gaussian', 
            kernel="SquaredExp", 
            stability_jitter=STABILITY_JITTER,
            initial_noise=1e-6)
        objective.fit(inputs, vals, fit_hypers=False)
        constraint = GP(D, 
            mcmc_iters=0, 
            likelihood='gaussian', 
            kernel="SquaredExp", 
            stability_jitter=STABILITY_JITTER,
            initial_noise=1e-6)
        constraint.fit(inputs, constraint_vals, fit_hypers=False)

        c2 = GP(D, mcmc_iters=0)
        c2.fit(inputs, npr.randn(N), fit_hypers=False)

        cons = {'c1':constraint, 'c2':c2}
        # cons = {'c1':constraint}

        # for p in objective.params:
        #     print '%s: %f' % (p, objective.params[p].value)
        # for p in constraint.params:
        #     print '%s: %f' % (p, constraint.params[p].value)

        acq_f = PES(D)
        acq_val = acq_f.acquisition(objective, cons, test_input, None, False, DEBUG_xstar=x_star)
        print 'Acquisition value: %s' % (acq_val)


if __name__ == "__main__":
    test_random_features_sampling()
    # test_x_star_sampling()
    # test_acquisition_function()
    

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
import math

import numpy        as np
import numpy.random as npr
import scipy.linalg as spla

from .abstract_sampler import AbstractSampler
from ..utils import param as hyperparameter_utils


class EllipticalSliceSampler(AbstractSampler):

    # No prior explicitly done here, that is included in the sampling
    def logprob(self, x, model):
        hyperparameter_utils.set_params_from_array(self.params, x)
        return model.log_binomial_likelihood() # no contribution from priors-- Gaussian prior built in to sampler

    def sample(self, model):
        if not model.has_data:
            return np.zeros(0) # TODO this should be a sample from the prior...

        prior_cov      = model.noiseless_kernel.cov(model.inputs)
        prior_cov_chol = spla.cholesky(prior_cov, lower=True)
        # Here get the Cholesky from model

        params_array = hyperparameter_utils.params_to_array(self.params)
        for i in xrange(self.thinning + 1):
            params_array, current_ll = elliptical_slice(
                params_array,
                self.logprob,
                prior_cov_chol,
                model.mean.value,
                model,
                **self.sampler_options
            )
            hyperparameter_utils.set_params_from_array(self.params, params_array)
        self.current_ll = current_ll # for diagnostics


# xx: the initial point
# sample_nu: a function that samples from the multivariate Gaussian prior
# log_like_fn: a function that computes the log likelihood of an input
# cur_log_like (optional): the current log likelihood
# angle_range: not sure
def elliptical_slice(xx, log_like_fn, prior_chol, prior_mean, *log_like_args, **sampler_args):
    cur_log_like = sampler_args.get('cur_log_like', None)
    angle_range = sampler_args.get('angle_range', 0)

    if cur_log_like is None:
        cur_log_like = log_like_fn(xx, *log_like_args)

    if np.isneginf(cur_log_like):
        raise Exception("Elliptical Slice Sampler: initial logprob is -inf for inputs %s" % xx)
    if np.isnan(cur_log_like):
        raise Exception("Elliptical Slice Sampler: initial logprob is NaN for inputs %s" % xx)

    nu = np.dot(prior_chol, npr.randn(xx.shape[0])) # don't bother adding mean here, would just subtract it at update step
    hh = np.log(npr.rand()) + cur_log_like  
    # log likelihood threshold -- LESS THAN THE INITIAL LOG LIKELIHOOD

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range <= 0:
        # Bracket whole ellipse with both edges at first proposed point
        phi = npr.rand()*2*math.pi
        phi_min = phi - 2*math.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -angle_range*npr.rand();
        phi_max = phi_min + angle_range;
        phi = npr.rand()*(phi_max - phi_min) + phi_min;

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference 
        # and check if it's on the slice
        xx_prop = (xx-prior_mean)*np.cos(phi) + nu*np.sin(phi) + prior_mean

        cur_log_like = log_like_fn(xx_prop, *log_like_args)
        
        if cur_log_like > hh:
            # New point is on slice, ** EXIT LOOP **
            return xx_prop, cur_log_like

        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            sys.stderr.write('Initial x: %s\n' % xx)
            # sys.stderr.write('initial log like = %f\n' % initial_log_like)
            sys.stderr.write('Proposed x: %s\n' % xx_prop)
            sys.stderr.write('ESS log lik = %f\n' % cur_log_like)
            raise Exception('BUG DETECTED: Shrunk to current position '
                            'and still not acceptable.');

        # Propose new angle difference
        phi = npr.rand()*(phi_max - phi_min) + phi_min



if __name__ == '__main__':

    from utils import priors
    import time

    print '2D Gaussian:'

    n = 1000000

    # Test on 2D Gaussian times another 2D Gaussian
    # one will be the "prior" and the other the "posterior"
    x_samples = np.zeros((2,n))
    x = np.zeros(2)

    prior_mu = np.array([-5, 2])
    prior_L = npr.randn(2,2)
    prior_L[0,1] = 0.0 # lower chol
    prior_cov = np.dot(prior_L,prior_L.T)
    
    like_mu = np.array([2, -1])
    like_L = npr.randn(2,2)
    like_L[0,1] = 0.0 # lower chol
    like_cov = np.dot(like_L, like_L.T)
    like = priors.MultivariateNormal(mu=like_mu, cov=like_cov)

    print 'Prior cov:'
    print prior_cov
    print 'Like cov:'
    print like_cov

    current_time = time.time()
    cur_ll = None
    for i in xrange(n):
        if i % 1000 == 0:
            print 'Elliptical Slice Sample %d/%d' % (i,n)
        x, cur_ll = elliptical_slice(x, like.logprob, prior_L, prior_mu, cur_log_like=cur_ll)
        x_samples[:,i] = x.copy()

    print 'Elliptical slice sampling took %f seconds' % (time.time() - current_time)

    # Formula for the actual mean and covariance matrix below came from
    # the wikipedia page on conjugate priors
    actual_cov = spla.inv(spla.inv(prior_cov) + spla.inv(like_cov))
    A = spla.cho_solve((prior_L, True), prior_mu)
    B = spla.cho_solve((like_L, True), like_mu)
    actual_mean = np.dot(actual_cov, A+B)

    print 'Actual mean:           %s' % actual_mean
    print 'Mean of ESS samples:   %s' % np.mean(x_samples,axis=1)

    print 'Actual Cov:'
    print actual_cov
    print 'Cov of ESS samples:'
    print np.cov(x_samples)

    # below: also compare with regular slice sampling (slower)

    # # Can also sample with regular slice sampling as another comparison
    # # Because I just compose them
    # from slice_sampler import slice_sample
    # pri = priors.MultivariateNormal(mu=prior_mu, cov=prior_cov)
    # post = priors.ProductOfPriors([pri, like])
    # xx_samples = np.zeros((2,n))
    # xx = np.zeros(2)

    # current_time = time.time()
    # for i in xrange(n):
    #     if i % 1000 == 0:
    #         print 'Slice Sample %d/%d' % (i,n)
    #     xx, cur_ll = slice_sample(xx, post.logprob)
    #     xx_samples[:,i] = xx.copy()

    # print 'Slice sampling took %f seconds' % (time.time() - current_time)
    # print ''


    # print 'Actual mean:           %s' % actual_mean
    # print 'Mean of ESS samples:   %s' % np.mean(x_samples,axis=1)
    # print 'Mean of slice samples: %s' % np.mean(xx_samples,axis=1)

    # print 'Actual Cov:'
    # print actual_cov
    # print 'Cov of ESS samples:'
    # print np.cov(x_samples)
    # print 'Cov of slice samples:'
    # print np.cov(xx_samples)



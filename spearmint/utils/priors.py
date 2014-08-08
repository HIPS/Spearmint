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

from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as npr
import scipy.stats as sps
from operator import add # same as lambda x,y:x+y I think
# import scipy.special.gammaln as log_gamma



class AbstractPrior(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def logprob(self, x):
        pass

    # Some of these are "improper priors" and I cannot sample from them
    # In this case the sample method will just return None
    # (or could raise an exception)
    # In any case the sampling should only be used for debugging
    # Unless we want to initialize the hypers by sampling from the prior?
    # def sample(self, n_samples):
    #     # raise Exception("Sampling not implemented for composed prior")
    #     return None


class Tophat(AbstractPrior):
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        if not (xmax > xmin):
            raise Exception("xmax must be greater than xmin")

    def logprob(self, x):
        if np.any(x < self.xmin) or np.any(x > self.xmax):
            return -np.inf
        else:
            return 0.  # More correct is -np.log(self.xmax-self.xmin), but constants don't matter

    def sample(self, n_samples):
        return self.xmin + npr.rand(n_samples) * (self.xmax-self.xmin)

# This is the Horseshoe prior for a scalar entity
# The multivariate Horseshoe distribution is not properly implemented right now
# None of these are, really. We should fix that up at some point, you might
# have to tell it the size in the constructor (e.g. with kwarg: dims=1)
# (Not that we ever really want the multivariate one as a prior, do we?)
# (I think more often we'd just want to vectorize the univariate one)
class Horseshoe(AbstractPrior):
    def __init__(self, scale):
        self.scale = scale

    # THIS IS INEXACT
    def logprob(self, x):
        if np.any(x == 0.0):
            return np.inf  # POSITIVE infinity (this is the "spike")
        # We don't actually have an analytical form for this
        # But we have a bound between 2 and 4, so I just use 3.....
        # (or am I wrong and for the univariate case we have it analytically?)
        return np.sum(np.log(np.log(1 + 3.0 * (self.scale/x)**2) ) )

    def sample(self, n_samples):
        # Sample from standard half-cauchy distribution
        lamda = np.abs(npr.standard_cauchy(size=n_samples))

        # I think scale is the thing called Tau^2 in the paper.
        return npr.randn() * lamda * self.scale
        # return npr.multivariate_normal()

class Lognormal(AbstractPrior):
    def __init__(self, scale, mean=0):
        self.scale = scale
        self.mean = mean

    def logprob(self, x):
        return np.sum(sps.lognorm.logpdf(x, self.scale, loc=self.mean))

    def sample(self, n_samples):
        return npr.lognormal(mean=self.mean, sigma=self.scale, size=n_samples)

class LognormalTophat(AbstractPrior):
    def __init__(self, scale, xmin, xmax, mean=0):
        self.scale = scale
        self.mean  = mean
        self.xmin  = xmin
        self.xmax  = xmax

        if not (xmax > xmin):
            raise Exception("xmax must be greater than xmin")

    def logprob(self, x):
        if np.any(x < self.xmin) or np.any(x > self.xmax):
            return -np.inf
        else:
            return np.sum(sps.lognorm.logpdf(x, self.scale, loc=self.mean))

    def sample(self, n_samples):
        raise Exception('Sampling of LognormalTophat is not implemented.')

# Let X~lognormal and Y=X^2. This is distribution of Y.
class LognormalOnSquare(Lognormal):
    def logprob(self, y):
        if np.any(y < 0): # Need this here or else sqrt(y) may occur with y < 0
            return -np.inf

        x = np.sqrt(y)
        dy_dx = 2*x  # this is the Jacobean or inverse Jacobean, whatever
        # p_y(y) = p_x(sqrt(x)) / (dy/dx)
        # log p_y(y) = log p_x(x) - log(dy/dx)
        return Lognormal.logprob(self, x) - np.log(dy_dx)

    def sample(self, n_samples):
        return Lognormal.sample(self, n_samples)**2

class LogLogistic(AbstractPrior):
    def __init__(self, shape, scale=1):
        self.shape = shape
        self.scale = scale

    def logprob(self, x):
        return np.sum(sps.fisk.logpdf(x, self.shape, scale=self.scale))

class Exponential(AbstractPrior):
    def __init__(self, mean):
        self.mean = mean

    def logprob(self, x):
        return np.sum(sps.expon.logpdf(x, scale=self.mean))

    def sample(self, n_samples):
        return npr.exponential(scale=self.mean, size=n_samples)



class Gaussian(AbstractPrior):
    def __init__(self, mu, sigma):
        self.mu    = mu
        self.sigma = sigma

    def logprob(self, x):
        return np.sum(sps.norm.logpdf(x, loc=self.mu, scale=self.sigma))

    def sample(self, n_samples):
        return self.mu + npr.randn(n_samples) * self.sigma

class MultivariateNormal(AbstractPrior):
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov

        if mu.size != cov.shape[0] or cov.shape[0] != cov.shape[1]:
            raise Exception("mu should be a vector and cov a matrix, of matching sizes")

    def logprob(self, x):
        return sps.multivariate_normal.logpdf(x, mean=self.mu, cov=self.cov)

    def sample(self, n_samples):
        return npr.multivariate_normal(self.mu, self.cov, size=n_samples).T.squeeze()

class NoPrior(AbstractPrior):
    def __init__(self):
        pass

    def logprob(self, x):
        return 0.0

# This class takes in another prior in its constructor
# And gives you the nonnegative version (actually the positive version, to be numerically safe)
class NonNegative(AbstractPrior):
    def __init__(self, prior):
        self.prior = prior

        if hasattr(prior, 'sample'):
            self.sample = lambda n_samples: np.abs(self.prior.sample(n_samples))

    def logprob(self, x):
        if np.any(x <= 0): 
            return -np.inf
        else:
            return self.prior.logprob(x)# + np.log(2.0)
        # Above: the log(2) makes it correct, but we don't ever care about it I think
        

# This class allows you to compose a list priors
# (meaning, take the product of their PDFs)
# The resulting distribution is "improper" -- i.e. not normalized
class ProductOfPriors(AbstractPrior):
    def __init__(self, priors):
        self.priors = priors

    def logprob(self, x):
        lp = 0.0
        for prior in self.priors:
            lp += prior.logprob(x)
        return lp

# class Binomial(AbstractPrior):
#     def __init__(self, p, n):
#         self.p = p
#         self.n = n

#     def logprob(self, k):
#         pos = k
#         neg = self.n-k

#         with np.errstate(divide='ignore'):  # suppress warnings about log(0)
#             return np.sum( pos[pos>0]*np.log(self.p[pos>0]) ) + np.sum( neg[neg>0]*np.log(1-self.p[neg>0]) )

#     def sample(self, n_samples):
#         return np.sum(npr.rand(n, n_samples) < p, axis=0)

# class Bernoulli(Binomial):
#     def __init__(self, p):
#         super(Bernoulli, self).__init__(p, 1)


def ParseFromOptions(options):
    parsed = dict()
    for p in options:
        prior_class = eval(options[p]['distribution'])
        args = options[p]['parameters']

        # If they give a list, just stick them in order
        # If they give something else (hopefully a dict of some sort), pass them in as kwargs
        if isinstance(args, list):
            parsed[p] = prior_class(*args)
        elif isinstance(args, dict): # use isinstance() not type() so that defaultdict, etc are allowed
            parsed[p] = prior_class(**args)
        else:
            raise Exception("Prior parameters must be list or dict type")

    return parsed




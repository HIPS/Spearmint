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


import copy
import sys, logging
import numpy             as np
import numpy.random      as npr
import scipy.linalg      as spla
import scipy.optimize    as spo
import scipy.io          as sio
import scipy.stats       as sps
import scipy.weave


from .gp                                     import GP
from ..utils.param                           import Param as Hyperparameter
from ..kernels                               import Matern52, Noise, Scale, SumKernel, TransformKernel
from ..sampling.slice_sampler                import SliceSampler
from ..sampling.whitened_prior_slice_sampler import WhitenedPriorSliceSampler
from ..sampling.elliptical_slice_sampler     import EllipticalSliceSampler
from ..utils                                 import priors
from ..transformations                       import BetaWarp, Transformer

try:
    module = sys.modules['__main__'].__file__
    log    = logging.getLogger(module)
except:
    log = logging.getLogger()
    print 'Not running from main.'

class GPClassifier(GP):
    def __init__(self, num_dims, **options):
        self.counts = None

        log.debug('GP Classifier initialized with options: %s' % (options))
        self.ess_thinning = int(options.get("ess-thinning", 10))

        self._set_likelihood(options)
    
        self.prior_whitening = options.get('prior-whitening', True)

        sigmoid = options.get("sigmoid", "probit")
        if not self.noiseless:
            if sigmoid == "probit":
                self.sigmoid            = sps.norm.cdf
                self.sigmoid_derivative = sps.norm.pdf   # not used
                self.sigmoid_inverse    = sps.norm.ppf
            elif sigmoid == "logistic":
                self.sigmoid            = sps.logistic.cdf
                self.sigmoid_derivative = sps.logistic.pdf
                self.sigmoid_inverse    = sps.logistic.ppf
            else:
                raise Exception("Only probit and logistic sigmoids are supported")
        else:
            # If no noise we use the step function and ignore the "sigmoid" argument.
            # (This is the step function likelihood)
            # assert options['likelihood'] == 'STEP'
            self.sigmoid            = lambda x: np.greater_equal(x, 0)
            self.sigmoid_derivative = lambda x: 0.
            self.sigmoid_inverse    = lambda x: 0.
        
        # The constraint is that p=s(f) > 1-epsilon
        # where s if the sigmoid and f is the latent function value, and p is the binomial probability
        # This is only in more complicated situations. The main situation where this is used
        # we want f>0. This is equivalent to epsilon=0.5 for the sigmoids we use
        # The point is: do not set epsilon unless you know what you are doing!
        # (and do not confuse it with delta, the min constraint confidence)
        self._one_minus_epsilon = 1.0 - float(options.get("epsilon", 0.5))

        self.latent_values_list = []

        super(GPClassifier, self).__init__(num_dims, **options)

    def _set_likelihood(self, options):
        self.likelihood = options.get('likelihood', 'binomial').lower()

        if self.likelihood.lower() == "binomial":
            self.noiseless = False
        elif self.likelihood.lower() == "step":
            self.noiseless = True
        else:
            raise Exception("GP classifier only supports step or binomial likelihood, not %s" % (options['likelihood']))

    def _reset(self):
        super(GPClassifier, self)._reset()

        # Reset the latent values
        if self.counts is not None:
            initial_latent_vals = self.counts - 0.5
        else:
            initial_latent_vals = np.zeros(0)

        self.latent_values.initial_value = initial_latent_vals
        self.latent_values.reset_value()

        self._latent_values_list = []

    def _set_latent_values_from_dict(self, latent_values_dict):
        # Read in the latent values. For pre-existing data, just load them in
        # For new data, set them to a default.
        default_latent_values = self.counts - 0.5

        latent_values = np.zeros(self._inputs.shape[0])
        for i in xrange(self._inputs.shape[0]):
            key = str(hash(self._inputs[i].tostring()))
            
            if key in latent_values_dict:
                latent_values[i] = latent_values_dict[key]
            else:
                latent_values[i] = default_latent_values[i]

        self.latent_values.value = latent_values

    def _burn_samples(self, num_samples):
        # sys.stderr.write('GPClassifer: burning %s: ' % ', '.join(self.params.keys()))
        # sys.stderr.write('%04d/%04d' % (0, num_samples))
        for i in xrange(num_samples):
            # sys.stderr.write('\b'*9+'%04d/%04d' % (i, num_samples))
            for sampler in self._samplers:
                sampler.sample(self)

            self.latent_values_sampler.sample(self)

            self.chain_length += 1
        # sys.stderr.write('\n')


    def _collect_samples(self, num_samples):
        # sys.stderr.write('GPClassifer: sampling %s: ' % ', '.join(self.params.keys()))
        # sys.stderr.write('%04d/%04d' % (0, num_samples))
        hypers_list        = []
        latent_values_list = []
        for i in xrange(num_samples):
            # sys.stderr.write('\b'*9+'%04d/%04d' % (i, num_samples))
            for sampler in self._samplers:
                sampler.sample(self)

            self.latent_values_sampler.sample(self)

            current_dict = self.to_dict()
            hypers_list.append(current_dict['hypers'])
            latent_values_list.append(current_dict['latent values'])

            self.chain_length += 1

        # sys.stderr.write('\n')
        return hypers_list, latent_values_list

    def _build(self):
        self.params        = {}
        self.latent_values = None

        # Build the transformer
        beta_warp                 = BetaWarp(self.num_dims)
        beta_alpha, beta_beta    = beta_warp.hypers
        self.params['beta_alpha'] = beta_alpha
        self.params['beta_beta']  = beta_beta

        transformer = Transformer(self.num_dims)
        transformer.add_layer(beta_warp)

        # Build the component kernels
        input_kernel      = Matern52(self.num_dims)
        ls                = input_kernel.hypers
        self.params['ls'] = ls

        # Now apply the transformation.
        transform_kernel = TransformKernel(input_kernel, transformer)

        # Add some perturbation for stability
        stability_noise = Noise(self.num_dims)

        # Finally make a noisy version if necessary
        # In a classifier GP the notion of "noise" is really just the scale.
        if self.noiseless:
            self._kernel = SumKernel(transform_kernel, stability_noise)
        else:
            scaled_kernel       = Scale(transform_kernel)
            self._kernel        = SumKernel(scaled_kernel, stability_noise)
            amp2                = scaled_kernel.hypers
            self.params['amp2'] = amp2

        # Build the mean function (just a constant mean for now)
        self.mean = Hyperparameter(
            initial_value = 0.0,
            prior         = priors.Gaussian(0.0,1.0),
            name          = 'mean'
        )
        self.params['mean'] = self.mean

        # Buld the latent values. Empty for now until the GP gets data.
        self.latent_values = Hyperparameter(
            initial_value  = np.array([]),
            name           = 'latent values'
        )

        # Build the samplers
        to_sample = [self.mean] if self.noiseless else [self.mean, amp2]
        self._samplers.append(SliceSampler(*to_sample, compwise=False, thinning=self.thinning))
        self._samplers.append(WhitenedPriorSliceSampler(ls, beta_alpha, beta_beta, compwise=True, thinning=self.thinning))
        self.latent_values_sampler = EllipticalSliceSampler(self.latent_values, thinning=self.ess_thinning)

    @property
    def values(self):
        if self.pending is None or len(self._fantasy_values_list) < self.num_states:
            return self.observed_values

        if self.num_fantasies == 1:
            return np.append(self.latent_values.value, self._fantasy_values_list[self.state].flatten(), axis=0)
        else:
            return np.append(np.tile(self.latent_values.value[:,None], (1,self.num_fantasies)), self._fantasy_values_list[self.state], axis=0)

    @property
    def observed_values(self):
        if self.latent_values is not None:
            return self.latent_values.value
        else:
            return np.array([])

    def set_state(self, state):
        self.state = state
        self._set_params_from_dict(self._hypers_list[state])
        self._set_latent_values_from_dict(self._latent_values_list[state])

    def pi(self, pred, compute_grad=False):
        return super(GPClassifier, self).pi( pred, compute_grad=compute_grad, 
            C=self.sigmoid_inverse(self._one_minus_epsilon) )

    def fit(self, inputs, counts, pending=None, hypers=None, reburn=False, fit_hypers=True):
        # Set the data for the GP
        self._inputs = inputs
        self.counts  = counts

        # Reset the GP
        self._reset()

        # Initialize the GP with hypers if provided
        if hypers:
            self.from_dict(hypers)

        if fit_hypers:
            # Burn samples (if needed)
            num_samples = self.burnin if reburn or self.chain_length < self.burnin else 0
            self._burn_samples(num_samples)

            # Now collect some samples
            self._hypers_list, self._latent_values_list = self._collect_samples(self.mcmc_iters)

            # Now we have more states
            self.num_states = self.mcmc_iters
        elif not self._hypers_list:
            # Just use the current hypers as the only state
            current_dict             = self.to_dict()
            self._hypers_list        = [current_dict['hypers']]
            self._latent_values_list = [current_dict['latent values']]
            self.num_states          = 1

        # Set pending data and generate corresponding fantasies
        if pending is not None:
            self.pending              = pending
            self._fantasy_values_list = self._collect_fantasies(pending)

        # Get caching ready
        if self.caching:
            self._prepare_cache()

        # Set the hypers to the final state of the chain
        self.set_state(len(self._hypers_list)-1)

        return self.to_dict()

    def log_binomial_likelihood(self, y=None):
        # If no data, don't do anything
        if not self.has_data:
            return 0.0

        if y is None:
            y = self.latent_values.value

        p = self.sigmoid(y)
        
        # Note on the below: the obvious implementation would be 
        #    return np.sum( pos*np.log(p) + neg*np.log(1-p) )
        # The problem is, if pos = 0, and p=0, we will get a 0*-Inf = nan
        # This messes things up. So we use the safer implementation below that ignores
        # the term entirely if the counts are 0.
        pos = self.counts # positive counts
        neg = 1 - pos

        with np.errstate(divide='ignore'):  # suppress warnings about log(0)
            return np.sum( pos[pos>0]*np.log(p[pos>0]) ) + np.sum( neg[neg>0]*np.log(1-p[neg>0]) )

    def to_dict(self):
        gp_dict = {}

        gp_dict['hypers'] = {}
        for name, hyper in self.params.iteritems():
            gp_dict['hypers'][name] = hyper.value

        # Save the latent values as a dict with keys as hashes of the data
        # so that each latent value is associated with its input
        # then when we load them in we know which ones are which
        gp_dict['latent values'] = {str(hash(self._inputs[i].tostring())) : self.latent_values.value[i] 
                for i in xrange(self._inputs.shape[0])}

        gp_dict['chain length'] = self.chain_length

        return gp_dict

    def from_dict(self, gp_dict):
        self._set_params_from_dict(gp_dict['hypers'])
        self._set_latent_values_from_dict(gp_dict['latent values'])
        self.chain_length = gp_dict['chain length']




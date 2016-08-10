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

import logging
import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats  as sps

from .abstract_model          import AbstractModel
from ..utils.param            import Param as Hyperparameter
import spearmint.kernels
from ..kernels                import *
from ..sampling               import *
from ..utils                  import priors
from ..                       import transformations
from ..transformations        import Transformer   
from ..transformations        import SimpleTransformer

# try:
#     module = sys.modules['__main__'].__file__
#     log    = logging.getLogger(module)
# except:
#     log    = logging.getLogger()
#     print 'Not running from main.'



class GP(AbstractModel):
    
    def __init__(self, num_dims, **options):
        
        # opts = OPTION_DEFAULTS.copy()
        # opts.update(options)
        # if hasattr(self, 'options'):
            # opts.update(self.options)
        # This is a bit of a mess. Basically to make it work with the GPClassifer --
        # but yes I know the GP shouldn't have code for the sake of those who inherit from it
        # TODO -- clean this up
        self.options = options

        self.num_dims = num_dims

        self.noiseless = self.options['likelihood'].lower() == "noiseless"

        self._inputs = None # Matrix of data inputs
        self._values = None # Vector of data values
        self.pending = None # Matrix of pending inputs
        # TODO: support meta-data

        self.params = None

        self._cache_list                 = [] # Cached computations for re-use.
        self._hypers_list                = [] # Hyperparameter dicts for each state.
        self._fantasy_values_list        = [] # Fantasy values generated from pending samples.
        self.state                       = None
        self._random_state               = npr.get_state()
        self._samplers                   = []

        # If you are only doing one fantasy of pending jobs, then don't even both sampling
        # it from the marginal gaussian posterior predictive and instead just take
        # the mean of this distribution. This only has an effect if num_fantasies is 1
        self._use_mean_if_single_fantasy = True
        
        # get the Kernel type from the options
        try:
            self.input_kernel_class = getattr(spearmint.kernels, self.options['kernel'])
        except NameError:
            raise Exception("Unknown kernel: %s" % self.options['kernel'])

        self._kernel            = None
        self._kernel_with_noise = None

        self.num_states   = 0
        self.chain_length = 0

        self.max_cache_bytes = self.options['max_cache_mb']*1024*1024

        self._build()

    def _set_params_from_dict(self, hypers_dict):
        # for name, hyper in self.params.iteritems():
        # doing it the above way is worse-- because if you changed the config
        # to add hyperparameters, they won't be found in the hypers_dict. 
        # this way is more robust
        for name, hyper in hypers_dict.iteritems():
            if name in self.params:
                self.params[name].value = hypers_dict[name]

    def _prepare_cache(self):
        self._cache_list = list()

        # inputs_hash = hash(self.inputs.tostring())
        for i in xrange(self.num_states):
            self.set_state(i)
            chol  = spla.cholesky(self.kernel.cov(self.inputs), lower=True)
            alpha = spla.cho_solve((chol, True), self.values - self.mean.value)
            cache_dict = {
                'chol'  : chol,
                'alpha' : alpha
            }
            self._cache_list.append(cache_dict)


    def jitter_value(self):
        return self.stability_noise_kernel.noise.value

    def noise_value(self):
        if self.noiseless:
            return self.stability_noise_kernel.noise.value
        else:
            return self.params['noise'].value

    def _build(self):
        self.params = dict()

        # these should be in the right order because the json was parsed with an orderedDict
        # could make this more robust by using a list instead...

        # transformer = Transformer(self.num_dims)
        transformer = SimpleTransformer(self.num_dims)

        for trans in self.options['transformations']:
            assert len(trans) == 1 # this is the convention-- a list of length-1 dicts
            trans_class = trans.keys()[0]
            trans_options = trans.values()[0]

            T = getattr(transformations,trans_class)(self.num_dims, **trans_options)
            
            transformer.add_layer(T)
            self.params.update({param.name:param for param in T.hypers})
        # Default is BetaWarp (set in main.py)
        # else: # default uses BetaWarp
            # beta_warp = BetaWarp(self.num_dims)
            # transformer.add_layer(beta_warp)
            # self.params.update({param.name:param} for param in beta_warp.hypers)

        # only used in weird case of PESC/finite basis/"depends on"
        self.transformer = transformer

        # Build the component kernels
        # length_scale_prior = priors.Scale(priors.Beta(1.5, 5.0), 10.0)
        length_scale_prior = priors.Scale(priors.Beta(1.5, 7.0), 5.0) # smaller
        # length_scale_prior = priors.Scale(priors.Beta(0.5, 7.0), 5.0)   # even smaller
        # length_scale_prior = None

        # set initial/default length scale value to be an array. we can't do this in advance
        # because we don't know the size of the GP yet. 
        if self.options['initial_ls'] is not None and isinstance(self.options['initial_ls'], float):
            initial_ls_value = np.ones(self.num_dims) * self.options['initial_ls']
        else:
            initial_ls_value = np.array(self.options['initial_ls'])

        input_kernel             = self.input_kernel_class(self.num_dims, prior=length_scale_prior, value=initial_ls_value)
        self.scaled_input_kernel = Scale(input_kernel, value=self.options['initial_amp2'])
        self.stability_noise_kernel = Noise(self.num_dims, name='stability_jitter', value=self.options['stability_jitter']) # Even if noiseless we use some noise for stability
        sum_kernel               = SumKernel(self.scaled_input_kernel, self.stability_noise_kernel)

        # The final kernel applies the transformation.
        self._kernel = TransformKernel(sum_kernel, transformer)

        # Finally make a noisy version if necessary
        if not self.noiseless:
            noise_kernel = Noise(self.num_dims, value=self.options['initial_noise'])
            self._kernel_with_noise = SumKernel(self._kernel, noise_kernel)

        # Build the mean function (just a constant mean for now)
        self.mean = Hyperparameter(
            initial_value = self.options['initial_mean'],
            prior         = priors.Gaussian(0.0,1.0),
            name          = 'mean'
        )

        self.params['ls'] = input_kernel.hypers

        toSample = self.params.values() # we don't want to copy the params, but we want to copy the list.

        amp2 = self.scaled_input_kernel.hypers

        # PESC examines 'params' so put them here even if not sampled
        self.params['amp2'] = amp2
        self.params['mean'] = self.mean
        if not self.noiseless:
            self.params['noise'] = noise_kernel.noise

        # doesn't really matter if it is in params, what matters is toSample

        if self.options['fit_amp2']:
            toSample.append(amp2)
        if self.options['fit_mean']:
            toSample.append(self.mean)        
        if not self.noiseless and self.options['fit_noise']:
            toSample.append(noise_kernel.noise)

        sampler_class = getattr(spearmint.sampling, self.options['sampler'])
        self._samplers.append(sampler_class(*toSample, compwise=True, thinning=self.options['thinning'], opt=self.options['nlopt_method_no_grad']))
        
        # if len(toSample) > 0:
            # self._samplers.append(sampler_class(*toSample, compwise=False, thinning=self.options['thinning']))
       
    def _burn_samples(self, num_samples):
        if num_samples == 0:
            return

        # logging.debug('GPClassifer: burning %s: ' % ', '.join(self.params.keys()))
        # logging.debug('%05d/%05d' % (0, num_samples))

        logging.debug('  Burning %d samples...' % num_samples)

        for i in xrange(num_samples):
            # if self.options['verbose']:
                # logging.debug('\b'*11+'%05d/%05d' % (i, num_samples))
            
            for sampler in self._samplers:
                sampler.sample(self)

            self.chain_length += 1

        # if self.options['verbose']:
            # logging.debug('\n')


    def _collect_samples(self, num_samples):
        hypers_list = []

        for sampler in self._samplers:
            logging.debug('  Sampling %d samples of %s with %s' % (num_samples, ', '.join(['%s(%d)'%(param.name, param.size()) for param in sampler.params]), sampler.__class__.__name__))
        logging.debug('')

        for i in xrange(num_samples):
            for sampler in self._samplers:
                sampler.sample(self)

            hypers_list.append(self.to_dict()['hypers'])
            self.chain_length += 1

        self._hypers_list = hypers_list

    def _collect_fantasies(self, pending):
        fantasy_values_list = []
        for i in xrange(self.num_states):
            self.set_state(i)
            fantasy_vals = self._fantasize(pending)
            if fantasy_vals.ndim == 1:
                fantasy_vals = fantasy_vals[:,np.newaxis]
            fantasy_values_list.append(fantasy_vals)

        return fantasy_values_list

    def _fantasize(self, pend):
        if self._use_mean_if_single_fantasy and self.options['num_fantasies'] == 1:
            predicted_mean, cov = self.predict(pend)
            return predicted_mean
        else:
            npr.set_state(self._random_state)
            return self.sample_from_posterior_given_hypers_and_data(pend, self.options['num_fantasies'])

    @property
    def inputs(self):
        if self.pending is None or len(self._fantasy_values_list) < self.num_states:
            return self._inputs
        else:
            return np.vstack((self._inputs, self.pending)) # Could perhaps cache this to make it faster.


    @property
    def observed_inputs(self):
        return self._inputs

    @property
    def values(self):
        if self.pending is None or len(self._fantasy_values_list) < self.num_states:
            return self._values
        
        if self.options['num_fantasies'] == 1:
            return np.append(self._values, self._fantasy_values_list[self.state].flatten(), axis=0)
        else:
            return np.append(np.tile(self._values[:,None], (1,self.options['num_fantasies'])), self._fantasy_values_list[self.state], axis=0)

    @property
    def observed_values(self):
        return self._values

    @property
    def kernel(self):
        if self.noiseless:
            return self._kernel
        else:
            return self._kernel_with_noise if self._kernel_with_noise is not None else self._kernel

    @property
    def noiseless_kernel(self):
        return self._kernel

    @property
    def has_data(self):
        return self.observed_inputs is not None and self.observed_inputs.size > 0

    def caching(self):
        if not self.options['caching'] or self.num_states <= 0:
            return False

        # For now this only computes the cost of storing the Cholesky decompositions.
        cache_mem_usage = (self._inputs.shape[0]**2) * self.num_states * 8. # Each double is 8 bytes.

        if cache_mem_usage > self.max_cache_bytes:
            logging.debug('Max memory limit of %d bytes reached. Not caching intermediate computations.' % self.max_cache_bytes)

            return False

        return True

    def set_state(self, state):
        self.state = state
        self._set_params_from_dict(self._hypers_list[state])

    def to_dict(self):
        gp_dict = {'hypers' : {}}
        for name, hyper in self.params.iteritems():
            gp_dict['hypers'][name] = hyper.value

        # I don't understand why this is stored...? as soon as you call fit
        # it gets set to 0 anyway.
        gp_dict['chain length'] = self.chain_length

        return gp_dict

    def from_dict(self, gp_dict):
        self._set_params_from_dict(gp_dict['hypers'])
        self.chain_length = gp_dict['chain length']

    def reset_params(self):
        for param in self.params.values():
            param.reset_value() # set to default

    # if fit_hypers is False, then we do not perform MCMC and use whatever we have
    # in other words, we are just changing setting the data if fit_hypers is False
    def fit(self, inputs, values, pending=None, hypers=None, reburn=False, fit_hypers=True):
        # Set the data for the GP
        self._inputs = inputs
        self._values = values

        if self.options['mcmc_iters'] == 0: # do not do MCMC
            fit_hypers = False

        self._fantasy_values_list = []  # fantasy of pendings

        # Initialize the GP with hypers if provided, or else set them to their default
        if hypers:
            self.from_dict(hypers)
        else:
            self.reset_params()

        if fit_hypers:
            # self._hypers_list = []  # samples hypers
            # self._cache_list  = []  # caching cholesky
            self.chain_length = 0   # chain of hypers

            # Burn samples (if needed)
            num_samples_to_burn = self.options['burnin'] if reburn or self.chain_length < self.options['burnin'] else 0
            self._burn_samples(num_samples_to_burn)

            # Now collect some samples (sets self._hypers_list)
            self._collect_samples(self.options['mcmc_iters'])

            # Now we have more states
            self.num_states = self.options['mcmc_iters']
        else:
            if len(self._hypers_list) == 0:
                # Just use the current hypers as the only state
                self._hypers_list = [self.to_dict()['hypers']]
                self.num_states  = 1

        self._cache_list  = []  # i think you need to do this before collecting fantasies...

        # Set pending data and generate corresponding fantasies
        if pending is not None:
            self.pending              = pending
            self._fantasy_values_list = self._collect_fantasies(pending)

        # Actually compute the cholesky and all that stuff -- this is the "fitting"
        # If there is new data (e.g. pending stuff) but fit_hypers is False
        # we still want to do this... because e.g. new pending stuff does change the cholesky. 
        if self.caching() and self.has_data:
            self._prepare_cache()

        # Set the hypers to the final state of the chain
        self.set_state(len(self._hypers_list)-1)

        return self.to_dict()

    def log_likelihood(self):
        """
        GP Marginal likelihood
        """
        if not self.has_data:
            return 0.0

        # cannot do caching of chol here because we are evaluating different length scales
        # -- nothing to cache yet
        cov   = self.kernel.cov(self.observed_inputs)
        chol  = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), self.observed_values - self.mean.value)

        complexity_penalty = -np.sum(np.log(np.diag(chol)))
        data_fit_term = -0.5*np.dot(self.observed_values - self.mean.value, solve)
        return complexity_penalty + data_fit_term
        # Uses the identity that log det A = log prod diag chol A = sum log diag chol A
        # return -np.sum(np.log(np.diag(chol)))-0.5*np.dot(self.observed_values - self.mean.value, solve)

    # cholK is only used for the Predictive Entropy Search acquisition function
    # Please ignore it otherwise...
    def predict(self, pred, full_cov=False, compute_grad=False):
        inputs = self.inputs
        values = self.values

        # if pred.shape[1] != self.num_dims:
            # raise Exception("Dimensionality of test points is %d but dimensionality given at init time is %d." % (pred.shape[1], self.num_dims))

        # Special case if there is no data yet --> predict from the prior
        if not self.has_data:
            return self.predict_from_prior(pred, full_cov, compute_grad)

        # The primary covariances for prediction.
        cand_cross = self.noiseless_kernel.cross_cov(inputs, pred)
        
        if self.caching() and len(self._cache_list) == self.num_states:
            chol  = self._cache_list[self.state]['chol']
            alpha = self._cache_list[self.state]['alpha']
        else:
            chol  = spla.cholesky(self.kernel.cov(self.inputs), lower=True)
            alpha = spla.cho_solve((chol, True), self.values - self.mean.value)

        # Solve the linear systems.
        # Note: if X = LL^T, cho_solve performs X\b whereas solve_triangular performs L\b
        beta = spla.solve_triangular(chol, cand_cross, lower=True)

        # Predict the marginal means at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean.value

        if full_cov:
            # Return the covariance matrix of the pred inputs, 
            # rather than just the individual variances at each input
            cand_cov = self.noiseless_kernel.cov(pred)
            func_v = cand_cov - np.dot(beta.T, beta)
        else:
            cand_cov = self.noiseless_kernel.diag_cov(pred) # it is slow to generate this diagonal matrix... for stationary kernels you don't need to do this
            func_v = cand_cov - np.sum(beta**2, axis=0)

        if not compute_grad:
            return func_m, func_v

        grad_cross = self.noiseless_kernel.cross_cov_grad_data(inputs, pred)
        grad_xp_m  = np.tensordot(np.transpose(grad_cross, (1,2,0)), alpha, 1)

        # this should be faster than (and equivalent to) spla.cho_solve((chol, True),cand_cross))
        gamma = spla.solve_triangular(chol.T, beta, lower=False)

        # Using sum and multiplication and summing instead of matrix multiplication
        # because I only want the diagonals of the gradient of the covariance matrix, not the whole thing
        grad_xp_v = -2.0*np.sum(gamma[:,:,np.newaxis] * grad_cross, axis=0)

        # Not very important -- just to make sure grad_xp_v.shape = grad_xp_m.shape
        if values.ndim > 1:
            grad_xp_v = grad_xp_v[:,:,np.newaxis]
        
        # In case this is a function over a 1D input,
        # return a numpy array rather than a float
        if np.ndim(grad_xp_m) == 0:
            grad_xp_m = np.array([grad_xp_m])
            grad_xp_v = np.array([grad_xp_v])

        return func_m, func_v, grad_xp_m, grad_xp_v

    def predict_from_prior(self, pred, full_cov=False, compute_grad=False):
        mean = self.mean.value * np.ones(pred.shape[0])
        if full_cov:
            cov = self.noiseless_kernel.cov(pred)
            return mean, cov
        elif compute_grad:
            var = self.noiseless_kernel.diag_cov(pred)
            grad = np.zeros((pred.shape[0], self.num_dims))
            return mean, var, grad, grad
        else:
            var = self.noiseless_kernel.diag_cov(pred)
            return mean, var


    # -------------------------------------------------------- #
    #                                                          #
    # Below are four sampling routines. Each one has the same  #
    # signature. "pred" contains the inputs at which we would  #
    # like to sample. "n_samples" is the number of samples. If #
    # n_samples is 1 we return a squeezed vector. "joint" is a #
    # boolean indicating whether we want to sample jointly.    #
    # joint=True means sample normally. joint=False means      #
    # sample from the conditional distribution at each input,  #
    # and just compute them all together in a vectorized way.  #
    #                                                          #
    # -------------------------------------------------------- #

    # Sample from p(y | theta), where theta is given by the current state
    def sample_from_prior_given_hypers(self, pred, n_samples=1, joint=True):
        N_pred = pred.shape[0]
        if joint:
            mean = self.mean.value
            cov  = self.noiseless_kernel.cov(pred) # Gaussian likelihood happens here
            return npr.multivariate_normal(mean*np.ones(N_pred), cov, size=n_samples).T.squeeze()
        else:
            mean = self.mean.value
            var  = self.noiseless_kernel.diag_cov(pred)
            return np.squeeze(mean + npr.randn(N_pred, n_samples) * np.sqrt(var)[:,None])

    # Sample from p(y)
    # This is achieved by first sampling theta from its hyperprior p(theta), and then
    # sampling y from p(y | theta)
    def sample_from_prior(self, pred, n_samples=1, joint=True):
        fants = np.zeros((pred.shape[0], n_samples))
        for i in xrange(n_samples):
            for param in self.params:
                param.sample_from_prior() # sample from hyperpriors and set value
            fants[:,i] = self.sample_from_prior_given_hypers(pred, joint)
        return fants.squeeze() # squeeze in case n_samples=1

    # Terminology: does "posterior" usually refer to p(theta | data) ?
    # By "posterior" I guess I mean "posterior predictive", p(y | data)

    # Sample from p(y | theta, data), where theta is given by the current state
    def sample_from_posterior_given_hypers_and_data(self, pred, n_samples=1, joint=True):
        if joint:
            predicted_mean, cov = self.predict(pred, full_cov=True) # This part depends on the data
            return npr.multivariate_normal(predicted_mean, cov, size=n_samples).T.squeeze()
        else:
            predicted_mean, var = self.predict(pred, full_cov=False) # This part depends on the data
            return np.squeeze(predicted_mean[:,None] + npr.randn(pred.shape[0], n_samples) * np.sqrt(var)[:,None])

    # Sample from p(y | data), integrating out the hyperparameters (theta)
    # This is achieved by first sampling theta from p(theta | data), and then
    # sampling y from p(y | theta, data)
    def sample_from_posterior_given_data(self, pred, n_samples=1, joint=True):
        fants = np.zeros((pred.shape[0], n_samples))
        for i in xrange(n_samples):
            # Sample theta from p(theta | data)
            self.generate_sample(1)
            # Sample y from p(y | theta, data)
            fants[:,i] = self.sample_from_posterior_given_hypers_and_data(pred, joint)
        return fants.squeeze() # squeeze in case n_samples=1

    # -------------------------------------------------------- #
    #                                                          #
    #               End of sampling functions                  #
    #                                                          #
    # -------------------------------------------------------- #


    # pi = probability that the latent function value is greater than or equal to C
    # This is evaluated separately at each location in pred
    def pi(self, pred, compute_grad=False, C=0):
        if not compute_grad:
            mean, sigma2 = self.predict(pred, compute_grad=False)
        else:
            mean, sigma2, g_m_x, g_v_x = self.predict(pred, compute_grad=True)
        sigma  = np.sqrt(sigma2)

        C_minus_m = C-mean

        # norm.sf = 1 - norm.cdf
        prob = sps.norm.sf(C_minus_m/sigma)  

        if not compute_grad:
            return prob
        else:
            # Gradient of pi w.r.t. GP mean
            g_p_m = sps.norm.pdf( C_minus_m / sigma ) / sigma
            # Gradient of pi w.r.t. GP variance (equals grad w.r.t. sigma / (2*sigma))
            g_p_v = sps.norm.pdf( C_minus_m / sigma ) * C_minus_m / sigma2 / (2*sigma)
            # Total derivative of pi w.r.t. inputs
            grad_p = g_p_m[:,np.newaxis] * g_m_x + g_p_v[:,np.newaxis] * g_v_x
            return prob, grad_p



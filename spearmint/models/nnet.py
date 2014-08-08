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
import copy
import scipy

import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.special
import scipy.stats  as sps

from collections import defaultdict

from spearmint.utils.param            import Param as Hyperparameter
from spearmint.kernels                import matern
from spearmint.sampling.slice_sampler import SliceSampler
from spearmint.utils                  import priors
from spearmint.transformations        import *
from spearmint.models.abstract_model  import AbstractModel

import spearmint.utils.param      as param

#from numba import autojit, jit

def _relu(x, grad=False):
    #z = np.dot(x,w)
    z = x
    y = np.greater(z,0)
    if grad:
        return z*y, y
    return z*y

def _relu_grad(x):
    return np.greater(x,0)

def _sigmoid(x):
    y = 1.0/(1.0+np.exp(-(x)))
    return y

def _sigmoid_grad(x):
    y = x*(1.0-x)
    return y

def _softmax(x, w, grad=False):
    z = np.dot(x,w)
    y = np.exp(-z)/np.sum(np.exp(-z),axis=1)[:,None]

    if grad:
        dy = np.rollaxis(-y.T[:,:,np.newaxis]*y[np.newaxis,:,:],1)
        d = np.diag_indices(y.shape[1],ndim=2)
        dy[:,d[0],d[1]] = y*(1.0-y)
        #dy = np.sum(dy,axis=1)
        return y, dy

    return y

def _relu_norm(x, w, grad=False):
    x = np.dot(x,w)
    y = x*np.greater(x,0)
    z = x
    y = z/(np.sqrt(np.sum(z**2,axis=1)[:,None]))

    if grad:

        dy = np.rollaxis(-(z.T[:,:,np.newaxis]*z[np.newaxis,:,:]),1)
        dydiag = np.sum(z**2,axis=1)[:,None] - z**2
        d = np.diag_indices(dy.shape[1], ndim=2)
        dy[:,d[0],d[1]] = dydiag
        dy = dy/((np.sum(z**2,axis=1)[:,None,None]**(3.0/2.0)))

        return y, dy
    return y

def _rbf(x, w, grad=False, width=0.5):
    z = np.dot(x,w)
    y = np.exp((-z**2)/width)
    if grad:
        dy = -2.0/width*z*np.exp((-z**2)/width)
        return y,dy
    else:
        return y

def rbf(x, w, grad=False, width=0.5):
    y = np.exp(-0.5*np.sum((x[:,:,None]-w)**2, axis=1))
    if grad:
        dy = (x[:,:,None]-w)*y[:,None,:]
        return y, dy.sum(1)#[:,-1,:]
    else:
        return y

def _erf(x, grad=False):
    y = scipy.special.erf(x)
    if grad:
        dy = (1.0/np.sqrt(np.pi))*np.exp(-x**2)
        return y, dy
    else:
        return y

def _probit(x, grad=False):
    y = sps.norm.cdf(x)
    if grad:
        dy = sps.norm.pdf(x)
        return y, dy
    else:
        return y
    
def _sin(x,w, grad=False):    
    z = np.dot(x, w)
    y = np.sin(z)
    if grad:
        dy = np.cos(z)
        return y, dy
    else:
        return y

def _tanh(x, grad=False):
    z = x
    y = np.tanh(z, out=z)
    if grad:
        dy = 1.0 - y**2
        return y, dy
    else:
        return y     

def _softplus(x, w, grad=False):
    z = np.dot(x, w)
    y = np.log(1.0 + np.exp(-z))
    if grad:
        dy = -1.0/(1.0+np.exp(z))
        return y, dy

    return y

def _normalize(w, norm):
    if norm is not None:
        val = np.linalg.norm(w)
        if val > norm:
            w /= val
            w *= norm

def _init_weights(dims, norm):
    w = 0.1*np.random.randn(dims[0], dims[1])
    _normalize(w, norm)
    return w

def _sgd_update(w, dw, w_inc, mom, lrate, norm):
    w_inc *= mom
    w_inc -= lrate*dw
    w     += w_inc
    _normalize(w, norm)

DEFAULT_MCMC_ITERS = 10
DEFAULT_BURNIN = 100

# Jasper Snoek
class ***REMOVED***(AbstractModel):    
    def __init__(self, num_dims, **options):

        dims=options.get("dims", [1, 50, 50, 50, 1])
        l2=options.get("l2", 0.0)
        weight_norm=options.get("weight_norm", 5.0)
        drop=options.get("drop", 0.1)
        self.use_bfgs=options.get("use_bfgs", False)

        self.verbose    = bool(options.get("verbose", False))
        self.mcmc_diagnostics = bool(options.get("mcmc-diagnostics", False))
        self.mcmc_iters = int(options.get("mcmc-iters", DEFAULT_MCMC_ITERS))
        self.burnin     = int(options.get("burn-in", DEFAULT_BURNIN))
        self.thinning   = int(options.get("thinning", 2))
        self.priors = options.get("priors", dict())

        self.numlayers = len(dims) - 2
        dims[0] = num_dims
        self.dims = dims
        self.weight_norm = weight_norm
        self.weights = list()
        self.biases  = list()
        for i in xrange(len(dims)-1):
            self.weights.append(_init_weights([dims[i], dims[i+1]], self.weight_norm))
            self.biases.append(1.0*_init_weights([1, dims[i+1]], None))

        self.drop = drop
        self.droptrain = 0.0
        self.l2 = l2
        self.acts = [_tanh for i in xrange(len(dims)-3)]
        self.acts.append(_tanh)
        self.nesterov = True
        self.ls = 1.0 # Length scales for the last hidden layer
        self.std = 1.0
        self.num_dims = num_dims
        self.noiseless = False
        # if 'likelihood' not in options or options['likelihood'] == "GAUSSIAN":
        #     self.noiseless = False
        # elif options['likelihood'] == "NOISELESS":
        #     self.noiseless = True  
        # else:
        #     pass # this might occur in GP classifier, which inherits. ok for now
        #     # raise Exception("GP only supports Gaussian or noiseless likelihood")

        self.pending_samples = 100  # TODO -- make in config

        self.randomstate     = npr.get_state()

        self.cache_list        = []
        self.fantasy_data_list = []
        self.chol              = None
        self.fantasy_data      = None
        self.state             = None
        self._inputs           = None
        self._values           = None
        self.pending           = None
        self._samplers         = []
        self.num_states        = 0


        # Instantiate transformations
        self.transformations = list()
        if "transformations" in options:
            for trans_class in options["transformations"]:
                T = eval(trans_class)(num_dims, options["transformations"][trans_class])
                self.transformations.append(T)
        else: # default does nothing
            pass
            # self.transformations.append(KumarWarp(input_dims))

        self.build_hypers()
        self.build_components()
        self.build_sampler() # model is the GP object

    def _reset_params(self):
        for param in self.params.values():
            param.value = param.initial_value

    def _reset(self):
        self._cache_list          = []
        self._fantasy_values_list = []
        self._hypers_list         = []
        
        self._reset_params()
        self.chain_length = 0

    @property
    def inputs_and_pend(self):
        if self.pending is None or len(self._fantasy_values_list) < self.num_states:
            return self._inputs
            
        return np.vstack((self._inputs, self.pending)) # Could perhaps cache this to make it faster.

    @property
    def observed_inputs(self):
        return self._inputs

    @property
    def values_and_pend(self):
        if self.pending is None or len(self._fantasy_values_list) < self.num_states:
            return self._values
        
        if self.num_fantasies == 1:
            return np.append(self._values, self._fantasy_values_list[self.state].flatten(), axis=0)
        else:
            return np.append(np.tile(self._values[:,None], (1,self.num_fantasies)), self._fantasy_values_list[self.state], axis=0)

    @property
    def values(self):
        return self.values_and_pend

    @property
    def inputs(self):
        return self.inputs_and_pend

    @property
    def observed_values(self):
        return self._values

    def _set_params_from_dict(self, hypers_dict):
        for name, hyper in self.params.iteritems():
            self.params[name].value = hypers_dict[name]

    def set_state(self, state):
        self.state = state
        self._set_params_from_dict(self._hypers_list[state])

        # if len(self.cache_list) > state:
        #     self.chol  = self.cache_list[state]['chol']
        #     self.alpha = self.cache_list[state]['alpha']
        # else:
        #     self.chol  = None
        #     self.alpha = None

        self.state = state

    def _burn_samples(self, num_samples):
        for i in xrange(num_samples):
            for sampler in self._samplers:
                sampler.sample(self)

            self.chain_length += 1

    def _collect_samples(self, num_samples):
        hypers_list = []
        for i in xrange(num_samples):
            for sampler in self._samplers:
                sampler.sample(self)

            hypers_list.append(self.to_dict()['hypers'])
            self.chain_length += 1

        return hypers_list

    # def num_states(self):
    #     return self.num_states

    def set_random_state(self):
        self.set_state(npr.randint(self.num_states))

    def to_dict(self):
        gp_dict = {'hypers' : {}}
        for name, hyper in self.params.iteritems():
            gp_dict['hypers'][name] = hyper.value

        gp_dict['chain length'] = self.chain_length

        return gp_dict

    def from_dict(self, gp_dict):
        self._set_params_from_dict(gp_dict['hypers'])
        self.chain_length = gp_dict['chain length']


    # def clear_cache(self):
    #     self.fantasy_data_list = []
    #     self.fantasy_data      = None
    #     self.cache_list        = []
    #     self.chol              = None
    #     self.alpha             = None

    def fit(self, inputs, values, pending=None, hypers=None, reburn=False, fit_hypers=True):
        # Set the data for the GP
        self._inputs = inputs
        self._values = values

        # Reset the GP
        self._reset()

        # Initialize the GP with hypers if provided
        if hypers:
            self.from_dict(hypers)

        # if pending is not None:
        #     self.pending              = pending
        #     self._fantasy_values_list = self._collect_fantasies(pending)

        # self.clear_cache()
        if values.ndim == 1:
            values = values[:,None]

        # Don't warp before training the neural net
        for t in self.transformations:
           inputs = t.transform(inputs)

        self.***REMOVED***_train(inputs, values, epochs=2500)

        if fit_hypers:
            # Burn samples (if needed)
            num_samples = self.burnin if reburn or self.chain_length < self.burnin else 0
            self._burn_samples(num_samples)

            # Now collect some samples
            self._hypers_list = self._collect_samples(self.mcmc_iters)

            # Now we have more states
            self.num_states = self.mcmc_iters

        elif not self._hypers_list:
            # Just use the current hypers as the only state
            self._hypers_list = [self.to_dict()['hypers']]
            self.num_states  = 1

        # Set pending data and generate corresponding fantasies
        if pending is not None:
            self._fantasy_values_list = []
            self.pending              = pending
            # If there are pending jobs then we will fantasize their outcomes.
            # Afterward, prediction will behave as it always has, except
            # now the GP will predict values across each fantasized function.
            if 'pending' is not None:
                for i in xrange(self.num_states):
                    self.set_state(i)
                    
                    # Sample fantasies
                    fantasy_vals = self.blinear_sample(
                      inputs, pending,
                      (values-self.mean.value)/self.std, nsamples=self.pending_samples) * self.std + self.mean.value
                if fantasy_vals.ndim == 1:
                    fantasy_vals = fantasy_vals[:,np.newaxis]
                self._fantasy_values_list.append(fantasy_vals)

        # Get caching ready
        # if self.caching:
        #     self._prepare_cache()

        # Set the hypers to the final state of the chain
        self.set_state(len(self._hypers_list)-1)

        return self.to_dict()

    def has_data(self):
        return (self._inputs is not None) and (self._inputs.size > 0)

    # GP marginal likelihood
    def log_likelihood(self, data=None):
        if data is None:                
            data = self._inputs

        # If no data, return 0
        if not self.has_data():
            return 0.0

        alpha = self.noise.value
        beta  = self.amp2.value

        for t in self.transformations:
            data = t.transform(data)

        vals = self._values
        comp = self._inputs

        phi = self.***REMOVED***_last_hid(comp)
        var = np.max([0.1, np.var(vals)])

        (N,M) = comp.shape

        Sinv = beta * np.dot(phi.T,phi) + (alpha+1e-6)*np.eye(phi.shape[1])
        (L, lower) = spla.cho_factor(Sinv, lower=True, overwrite_a=False)
        m = beta * spla.cho_solve((L, lower), np.dot(phi.T, ((vals-self.mean.value)/self.std)))
        yhat = np.dot(m.T, phi.T).T

        lnDetSinv = 2.0*np.sum(np.log(np.diag(L)))
        lnDetS = -lnDetSinv

        lp = (M/2.0)*np.log(alpha) + (N/2.0)*np.log(beta) - ((beta/2.0)*np.sum((((vals-self.mean.value)/self.std)-yhat)**2.0))-(alpha/2.0)*(np.dot(m.T,m)) + 0.5 * lnDetS - (N/2.0)*np.log(2.0*np.pi)

        # Roll in mean Gaussian prior
        #lp -= 0.5*((self.mean.value-np.mean(vals))/(2.0*var))**2

        return lp

    # pred is a dict containing inputs to be predicted.
    def predict(self, pred, full_cov=False, compute_grad=False):
        inputs = self.inputs_and_pend
        values = self.values_and_pend

        if pred.shape[1] != self.num_dims:
            raise Exception("Dimensionality of inputs must match dimensionality given at init time.")

        test = pred
        obsv = inputs
        for t in self.transformations:
            obsv = t.transform(obsv)
            test = t.transform(test)

        test = np.reshape(test, (-1,obsv.shape[1]))

        # Predict in batches
        batchsize = np.min([2048, test.shape[0]])
        means      = []
        variances  = []
        if not compute_grad:
            for i in xrange(0,pred.shape[0],batchsize):
                func_m, func_v = self.blinear_pred(obsv, test[i:i+batchsize,:], values, variance=True, grad=compute_grad)                
                means.append(func_m)
                variances.append(func_v)
            return np.concatenate(means), np.concatenate(variances)

        d_means = []
        d_vars  = []
        for i in xrange(0,pred.shape[0],batchsize):
            func_m, func_v, dm, dv = self.blinear_pred(obsv, test[i:i+batchsize,:], values, variance=True, grad=compute_grad)

            for t in self.transformations:
                dtrans = t.transform_grad(pred[i:i+batchsize,:])
                dm *= dtrans
                dv *= dtrans

            if test.shape[0] == 1:
                return func_m, func_v, dm, dv
            means.append(np.atleast_1d(func_m))
            variances.append(np.atleast_1d(func_v))
            d_means.append(dm)
            d_vars.append(dv)
        return np.concatenate(func_m), np.concatenate(func_v), np.vstack(dm), np.vstack(dv)

    def build_hypers(self):

        default_priors = {
            'ls'   : priors.Tophat(0,10),
            'amp2' : priors.NonNegative(priors.NoPrior()), # Prior is cooked into the ML #Lognormal(1.0),
            'mean' : priors.NoPrior(),
            'noise': priors.NonNegative(priors.Horseshoe(0.1))
        }

        # Overwrite the default priors with any that the user may have specified
        default_priors.update(priors.ParseFromOptions(self.priors))

        self.amp2 = Hyperparameter(
            initial_value = 1.0, # do not initialize as np.var(self.data['targets']) because this might be 0 which has 0 mass under prior--> error
            prior         = default_priors['amp2'],
            name          = 'amp2'
        )

        self.mean = Hyperparameter(
            initial_value    = self.data['targets'].mean() if self.has_data() else 0.0,
            prior            = default_priors['mean'],
            name             = 'mean'
        )

        self.hypers = [self.mean, self.amp2]
        
        if not self.noiseless:
            self.noise = Hyperparameter(
                initial_value = 1e-3,
                prior         = default_priors['noise'],
                name          = 'noise'
            )
            self.hypers.extend([self.noise])
        else:
            self.noise = Hyperparameter(
                initial_value = 1e-6,
                name          = 'unused-noise'
            )

        for transformation in self.transformations:
            self.hypers.extend(transformation.get_hypers())

    def build_components(self):
        pass

    def build_sampler(self):
        # Build the transformer
        kumar_warp  = KumarWarp(self.num_dims)
        transformer = Transformer(self.num_dims)
        transformer.add_layer(kumar_warp)

        # Get the hyperparameters to sample
        kumar_alpha, kumar_beta = kumar_warp.hypers

        self.params = {
            'mean'        : self.mean,
            'amp2'        : self.amp2,
            'noise'       : self.noise,
            'kumar_alpha' : kumar_alpha,
            'kumar_beta'  : kumar_beta
        }

        # Build the samplers
        self._samplers.append(SliceSampler(self.mean, self.amp2, self.noise, compwise=False, thinning=self.thinning))
        self._samplers.append(SliceSampler(kumar_alpha, kumar_beta, compwise=True, thinning=self.thinning))

    # The gradients w.r.t. the mean of the gp
    def blinear_gradient_mean(self, data, targets):
        (N,M) = data.shape
        alpha = self.noise.value
        beta = self.amp2.value
        phi = data.copy()
        (N,M) = phi.shape
        Phi = np.dot(phi.T,phi)
        Sinv = (alpha+1e-6)*np.eye(Phi.shape[0]) + beta * Phi
        (L,lower) = spla.cho_factor(Sinv, lower=True, overwrite_a=True)
        m = beta * spla.cho_solve((L, True), np.dot(phi.T, targets))
        targetout = np.dot(m.T, phi.T).T
        f = (beta/2.0)*np.sum((targets-targetout)**2.0)
        df = beta*np.dot(targetout-targets, m.T)

        return (f, df)

    # The gradients w.r.t. the marginal likelihood
    def blinear_gradient_ml(self, data, targets):
        (N,M) = data.shape
        alpha = self.noise.value
        beta = self.amp2.value
        phi = data.copy()
        (N,M) = phi.shape
        Phi = np.dot(phi.T,phi)
        Sinv = (alpha+1e-6)*np.eye(Phi.shape[0]) + beta * Phi
        L = spla.cholesky(Sinv, lower=True)
        S = spla.cho_solve((L,True), np.eye(L.shape[0]))
        lnDetSinv = 2.0*np.sum(np.log(np.diag(L)))
        lnDetS = -lnDetSinv
        m = beta * spla.cho_solve((L, True), np.dot(phi.T, targets))
        targetout = np.dot(m.T, phi.T).T
        #f = -M/2.0*np.log(alpha) + N/2.0*np.log(beta) - ((2.0/beta)*np.sum((targets-targetout)**2)) + 0.5 * lnDetS + (N/2)*np.log(2.0*np.pi)# +(alpha/2.0)*np.sum(np.dot(m.T,m))
        f = (beta/2.0)*np.sum((targets-targetout)**2.0) + (alpha/2.0)*np.dot(m.T,m) - 0.5*lnDetS
        #f = (2.0/(beta))*np.sum((targets-targetout)**2.0)
        
        dSinv = (np.kron(np.eye(M), phi.T).reshape(M,M,N,M, order='F') +
                np.kron(phi.T, np.eye(M)).reshape(M,M,M,N, order='F').swapaxes(3,2))
        dSinv = beta*dSinv.reshape(M,M,M*N)

        df = np.dot(np.rollaxis(dSinv,2), S)
        df = np.transpose(np.dot(np.transpose(df,(0,2,1)),-S.T),(0,2,1))
        df = beta*np.dot(df, np.dot(phi.T,targets))
        A = np.kron(targets.T,np.eye(M)).T
        df = np.rollaxis(beta*np.dot(A[:,np.newaxis,:],S).T,2) + df
        df = m*alpha*df
        df = df.sum(1).reshape(N,M) 
        df = df + np.dot(beta*phi,S) #- 2*phi
        df = df + ((beta)*np.dot(targetout-targets, m.T))

        return (f, df)

    def blinear_sample(self,data,test,targets,nsamples):
        # Draw some samples
        h3 = self.***REMOVED***_last_hid(data)#/self.ls
        h3test = self.***REMOVED***_last_hid(test)#/self.ls
        phi = h3
        #h3test = h3test/np.sqrt(np.sum(h3test**2,axis=1))[:,np.newaxis]# + 0.1    
        Phi = np.dot(phi.T,phi)
        Sinv = self.amp2.value * Phi + (self.noise.value+1e-6)*np.eye(Phi.shape[0])
        L = spla.cholesky(Sinv, lower=True)
        S = spla.cho_solve((L,True), np.eye(Sinv.shape[0]))
        m = self.amp2.value * spla.cho_solve((L, True), np.dot(phi.T, targets))
        w = np.dot(np.random.randn(nsamples,1,S.shape[0]),S)
        return np.dot(h3test,w.T.squeeze()+m)

    def blinear_pred(self,data,test,targets,variance=False, grad=False):

        test = np.reshape(test, (-1, data.shape[1]))
        phi = self.***REMOVED***_last_hid(data)
        
        h = [test]
        dh = [h[0]]
        for i in xrange(0,len(self.weights)-1):
            hid, dhid = self.acts[i](np.dot(h[i], self.weights[i])+self.biases[i], grad=True)
            h.append(hid)
            dh.append(dhid)

        phi2 = h[-1]

        Sinv = self.amp2.value * np.dot(phi.T,phi) + (self.noise.value+1e-6)*np.eye(phi.shape[1])
        (L, lower) = spla.cho_factor(Sinv, lower=True, overwrite_a=True)
        m = self.amp2.value * spla.cho_solve((L, True), np.dot(phi.T, targets))
        yhat = np.dot(m.T, phi2.T).T

        if variance:
            S = spla.cho_solve((L,True), np.eye(L.shape[0]))
            var = 1.0/self.amp2.value + np.sum(np.dot(phi2[:,np.newaxis,:], S.T).squeeze()*phi2,axis=1)[:,np.newaxis]
            if not grad:
                return yhat,var
        else:
            if not grad:
                return yhat

        # Gradients of test w.r.t. the mean and variance              
        if m.ndim == 1:
            m = m[:,None]
        dm = m.T[:,None,:]

        ix = (dm*dh[-1])[:,:,:]
        for i in np.arange(len(self.weights)-2, 0, -1):
            if dh[i].ndim == 3:
                ix = np.sum((np.dot(ix, self.weights[i].T)[:,:,:,None]*dh[i]), axis=2)
            else:
                ix = (np.dot(ix, self.weights[i].T)*dh[i])
        dyhat = (np.dot(ix,self.weights[0].T))
        if m.shape[1] == 1 and phi2.shape[0] == 1:            
            dyhat = dyhat.reshape(test.shape)

        if not variance:
            return yhat, dyhat
        
        dv = 2.0*np.dot(phi2,S)
        ix = (dv*dh[-1])[:,:]
        for i in np.arange(len(self.weights)-2, 0, -1):
            if dh[i].ndim == 3:
                ix = np.sum((np.dot(ix,self.weights[i].T)[:,:,None]*dh[i]), axis=2)
            else:
                ix = (np.dot(ix,self.weights[i].T)*dh[i])
        dsig = (np.dot(ix,self.weights[0].T))

        if dyhat.ndim == 3:
            dyhat = np.rollaxis(dyhat.squeeze(),1)
        return yhat, var, dyhat, dsig

    # train is the training data NxD
    # Targets is a NxK one hot label for each training case
    def ***REMOVED***_gradient(self, weights, biases, train, targets):
        bayesian = True

        dw      = [w*0 for w in weights]
        dbiases = [b*0 for b in biases]

        # Dropouts
        droptrain = np.nonzero(np.random.rand(train.shape[1]) > self.droptrain)[0]
        dropouts  = []
        dropouts.append(droptrain)
        for i in xrange(len(weights)-1):
            dropouts.append(np.nonzero(np.random.rand(weights[i].shape[1]) > self.drop)[0])

        train = train.take(droptrain,axis=1)

        w = [weights[i].take(dropouts[i],axis=0).take(dropouts[i+1],axis=1) for i in xrange(len(weights)-1)]
        w.append(weights[-1].take(dropouts[-1],axis=0))

        b = [biases[i].take(dropouts[i+1],axis=1) for i in xrange(len(biases)-1)]

        dw_idx = [np.ix_(dropouts[i], dropouts[i+1]) for i in xrange(len(dropouts)-1)]

        numdims  = train.shape[1]
        numclass = targets.shape[1]
        N  = float(train.shape[0])
        h  = [train]
        dh = [train]
        for i in xrange(0,len(w)-1):
            hid, dhid = self.acts[i](np.dot(h[i],w[i])+b[i], grad=True)
            h.append(hid)
            dh.append(dhid)

        targetout = (np.dot(h[-1],w[-1])) + b[-1]
        if bayesian:
            f, df = self.blinear_gradient_mean(h[-1], targets)
            if self.l2 > 0:
                 f += np.sum(np.array([0.5*self.l2*np.sum(w[i]**2) for i in xrange(len(w))]))

            ix = (df*dh[-1])[:,:]            
        else:
            f = 1.0/N * (0.5*np.sum((targets - targetout)**2))
            if self.l2 > 0:
                 f += np.sum(np.array([0.5*self.l2*np.sum(w[i]**2) for i in xrange(len(w))]))

            ix = (targetout-targets)
    
            dw[-1][dropouts[-1],:] = np.dot(h[-1].T,ix)/N + self.l2*w[-1]
            dbiases[-1] = np.sum(ix,axis=0)/N + self.l2*b[-1]


        for i in np.arange(len(dw)-1, 0, -1):
            if dh[i].ndim == 3:
                ix = np.sum((np.dot(ix,w[i].T)[:,:,None]*dh[i])[:,:,:], axis=1)
            else:
                if not bayesian or i < len(dw)-1:
                    ix = ((np.dot(ix,w[i].T))*dh[i])

            dw[i-1][dw_idx[i-1]] = np.dot(h[i-1].T,ix)/N
            dbiases[i-1][:,dropouts[i]] += np.sum(ix,axis=0)/N + self.l2*b[i-1]

            if self.l2 > 0:
                dw[i-1][dw_idx[i-1]] += self.l2*w[i-1]

        return f, dw, dbiases

    def ***REMOVED***_train(self, train, targets, epochs=5000):           
        batchsize = int(np.min([128, train.shape[0]]))
        bintargets = targets
        w = self.weights[:]
        b = self.biases[:]

        if self.use_bfgs: # Use BFGS, otherwise use SGD
            w_vec = w[0].flatten()
            for i in xrange(1,len(w)):
                w_vec = np.concatenate((w_vec, w[i].flatten()))
            for i in xrange(0,len(b)):
                w_vec = np.concatenate((w_vec, b[i].flatten()))

            w_vec = scipy.optimize.fmin_l_bfgs_b(self.bfgs_wrapper, w_vec, args=(self.dims, train, targets), disp=0)[0]
            off = 0
            for i in xrange(len(self.dims)-1):
                w[i] = w_vec[off:off+(self.dims[i])*self.dims[i+1]].reshape((self.dims[i], self.dims[i+1]))
                off  = off+(self.dims[i])*self.dims[i+1]
            for i in xrange(len(b)):
                b[i] = w_vec[off:off+b[i].shape[1]][None,:]
                off += b[i].shape[1]

            self.weights = w[:]
            self.biases  = b[:]
            return

        mom = 0.1
        fs = np.zeros(epochs)
        w_inc = [np.zeros(weight.shape) for weight in w]
        b_inc = [np.zeros(bias.shape) for bias in self.biases]
        accs = []
        lrate = 1.0e-2#
        for i in xrange(epochs):
            randomorder = np.random.permutation(train.shape[0])
            for j in xrange(0,train.shape[0],batchsize):
                batch = train.take(randomorder[j:j+batchsize],axis=0)
                batchtargets = bintargets[randomorder[j:j+batchsize],:]
                if self.nesterov: # Nesterov momentum
                    f, dw, db = self.***REMOVED***_gradient([w[idx] + w_inc[idx] for idx in xrange(len(w))], self.biases, batch, batchtargets)
                else:
                    f, dw, db = self.***REMOVED***_gradient(w, self.biases, batch, batchtargets)
                fs[i] += f

                for jj in xrange(len(w)):
                    _sgd_update(w[jj], dw[jj], w_inc[jj], mom, lrate, self.weight_norm)
                    _sgd_update(b[jj], db[jj], b_inc[jj], mom, lrate, None)

            if i % 100 == 0:
                self.weights = w[:]
                self.biases  = b[:]
                accs.append(self.***REMOVED***_err(train,targets))
                if i > 200:
                    if np.abs(fs[i] - fs[i-2]) < 1e-6:
                        break
                #print 'Epoch %d, Err: %f,%f' %(i,fs[i], accs[-1])

            if i % 1 == 0:
                mom = 0.98

        self.weights = w[:]

        # Adjust for dropout
        for i in xrange(0,len(self.weights)-1):
            drop_adjust = self.droptrain if i == 0 else self.drop
            self.weights[i] *= (1.0 - drop_adjust)

    def bfgs_wrapper(self, weight_vector, dims, train, targets):
        w = []
        b = []
        off = 0
        for i in xrange(len(dims)-1):
            w.append(weight_vector[off:off+(dims[i])*dims[i+1]].reshape((dims[i], dims[i+1])))
            off = off+(dims[i])*dims[i+1]

        for i in xrange(len(dims)-1):
            b.append(weight_vector[off:off+w[i].shape[1]][None,:])
            off += b[-1].shape[1]

        f, dw, db = self.***REMOVED***_gradient(w, b, train, targets)
        dw_vec = np.zeros(weight_vector.shape[0])

        off = 0
        for i in xrange(len(dims)-1):
            dw_vec[off:off+(dims[i])*dims[i+1]] = dw[i].flatten()
            off = off+(dims[i])*dims[i+1]

        for i in xrange(len(b)):
            dw_vec[off:off+b[i].shape[1]] = db[i].flatten()
            off += b[i].shape[1]

        return f, dw_vec

    def check_grad(self,data,targets):
        print '----------------- OUTPUTS ----------------'
        idx = np.zeros(data.shape)
        df1 = data*0.0
        f,df2 = self.blinear_gradient_mean(data, targets)
        for i in xrange(data.shape[0]):
            for j in xrange(data.shape[1]):
                print i,j
                idx[i,j] = 1e-6
                f1,_ = self.blinear_gradient_mean(data-idx, targets)
                f2,_ = self.blinear_gradient_mean(data+idx, targets)
                df1[i,j] = (f2-f1)/(2.0*1e-6)
                idx[i,j] = 0.0
                print '%f %f %f' % (df1[i,j], df2[i,j], df2[i,j]/df1[i,j])

        f, dw, db = self.***REMOVED***_gradient(self.weights, self.biases, data, targets)
        w = self.weights[:]
        b = self.biases[:]
        for ind in xrange(len(b)):
            print '-----------------BIAS ', ind ,'----------------'
            df1 = b[ind]*0.0
            idx = np.zeros(b[ind].shape)
            for i in xrange(b[ind].shape[0]):
                for j in xrange(b[ind].shape[1]):
                    idx[i,j] = 1e-4
                    b[ind] = b[ind] + idx
                    f1, _,_ = self.***REMOVED***_gradient(w, b, data, targets)
                    b[ind] = b[ind] - 2*idx
                    f2, _,_ = self.***REMOVED***_gradient(w, b, data, targets)
                    b[ind] = b[ind] + idx
                    df1[i,j] = (f1-f2)/(2*1e-4)
                    idx[i,j] = 0.0
                    print '%f %f %f' % (df1[i,j], db[ind][i,j], db[ind][i,j]/df1[i,j])

        for ind in xrange(len(w)):
            print '-----------------', ind ,'----------------'
            df1 = w[ind]*0.0
            idx = np.zeros(w[ind].shape)
            for i in xrange(w[ind].shape[0]):
                for j in xrange(w[ind].shape[1]):                
                    idx[i,j] = 1e-4
                    w[ind] = w[ind] + idx
                    f1, _,_ = self.***REMOVED***_gradient(w, self.biases, data, targets)
                    w[ind] = w[ind] - 2*idx
                    f2, _,_ = self.***REMOVED***_gradient(w, self.biases, data, targets)
                    w[ind] = w[ind] + idx
                    df1[i,j] = (f1-f2)/(2*1e-4)
                    idx[i,j] = 0.0
                    print '%f %f %f' % (df1[i,j], dw[ind][i,j], dw[ind][i,j]/df1[i,j])

    def drop_pred(self, data):
        y = np.zeros((data.shape[0],1))
        for i in xrange(1000):
            droptrain = np.nonzero(np.random.rand(data.shape[1]) > self.droptrain)[0]
            dropouts = []
            dropouts.append(droptrain)
            for i in xrange(len(self.weights)-1):
                dropouts.append(np.nonzero(np.random.rand(self.weights[i].shape[1]) > self.drop)[0])

            data = data.take(droptrain,axis=1)

            w = [weights[i].take(dropouts[i],axis=0).take(dropouts[i+1],axis=1) for i in xrange(len(weights)-1)]
            w.append(weights[-1].take(dropouts[-1],axis=0))

            h = [train]
            for i in xrange(0,len(w)-1):
                h.append(self.acts[i](np.dot(h[i],w[i])+self.biases[i]))
            y += (np.dot(h[-1],w[-1]))
        return y/1000.0

    def ***REMOVED***_last_hid(self, data):
        h = [data]
        for i in xrange(0,len(self.weights)-1):
            h.append(self.acts[i](np.dot(h[i],self.weights[i])+self.biases[i]))
        return h[-1]

    def ***REMOVED***_pred(self, data):
        h = self.***REMOVED***_last_hid(data)
        preds = np.dot(h,self.weights[-1]*(1.0-self.drop))+self.biases[-1]
        return preds

    def ***REMOVED***_err(self, data, targets):
        #preds = self.***REMOVED***_pred(data)
        preds = self.blinear_pred(data,data,targets)
        #preds = self.blinear_pred(data,data,targets,variance=False)
        acc = np.mean((preds-targets)**2)
        return acc

if __name__ == '__main__':
    targets = np.random.randn(150, 1)
    train = np.random.randn(150,5)
    train = np.linspace(-10,10,25)[:,np.newaxis]
    #train = np.vstack((train[:50,:], train[80:,:]))
    targets = 10.0+np.sin(train[:,0])[:,np.newaxis]

    datamin = np.min(train)
    datamax = np.max(train-datamin)
    test = np.linspace(-20,20,500)[:,np.newaxis]
    nn = ***REMOVED***(dims=[1, 50, 50, 50, 1], l2=0.0, weight_norm=5.0, use_bfgs=True)
    nn.alpha = 0.01
    nn.beta  = 10.0
    acc = nn.***REMOVED***_err(train, targets)
    print 'MSE before training: ', acc
    #nn.check_grad(train[:25,:],targets[:25])

    nn.***REMOVED***_train(train, targets,epochs=5000)
    acc = nn.***REMOVED***_err(train, targets)
    print 'MSE after training: ', acc
    yhat, sig = nn.blinear_pred(train,test,targets,True)

    test = np.linspace(-120,120,500)[:,np.newaxis]

    yhat, sig = nn.blinear_pred(train,test,targets,True)

    # Draw some samples
    h3 = nn.***REMOVED***_last_hid(train)
    h3test = nn.***REMOVED***_last_hid(test)    
    phi = h3
    phi = phi/np.sqrt(np.sum(phi**2,axis=1))[:,np.newaxis] + 0.1
    h3test = h3test/np.sqrt(np.sum(h3test**2,axis=1))[:,np.newaxis] + 0.1    
    Phi = np.dot(phi.T,phi)
    Sinv = nn.beta * Phi + (nn.alpha+1e-6)*np.eye(Phi.shape[0])
    L = spla.cholesky(Sinv, lower=True)
    S = spla.cho_solve((L,True), np.eye(Sinv.shape[0]))
    m = nn.beta * spla.cho_solve((L, True), np.dot(phi.T, targets))
    w = np.dot(np.random.randn(100,1,S.shape[0]),S)

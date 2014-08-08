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
import os
import ast
import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats  as sps

from spearmint.models                 import GP
from spearmint.utils.param            import Param as Hyperparameter
from spearmint.kernels                import Matern, Noise, Scale, SumKernel, TransformKernel
from spearmint.sampling.slice_sampler import SliceSampler
from spearmint.utils                  import priors
from spearmint.transformations        import BetaWarp, IgnoreDims, Linear, Normalization, Transformer

import spearmint.utils.param      as param_util
import spearmint.sampling.sampler as sampler

import copy
from collections import defaultdict


#TODO: The tests below should be converted into proper nosetests.
class DiagnosticGP(GP):
    # The above functions sample the latent function. This is the observation model
    # (i.i.d. Gaussian noise)
    def observation_model(self, y):
        if self.noiseless:
            return y
        elif type(y) == float or y.size==1:
            return y + npr.randn() * np.sqrt(self.noise.value)
        else:
            return np.squeeze(y + npr.randn(*y.shape) * np.sqrt(self.noise.value))

    # There seem to be 2 tests called the Geweke test. One is the Geweke convergence test
    # and one is the Geweke correctness test. This is the correctness test. It is descibed at
    # https://hips.seas.harvard.edu/blog/2013/06/10/testing-mcmc-code-part-2-integration-tests/
    # This test uses an arbitrary statistic of the data (outputs). Here we use the sum.
    def geweke_correctness_test(self):
        print 'Initiating Geweke Correctness test'
        # Note: the horseshoe prior on the noise will make the line slightly not straight
        # because we don't have the actual log pdf

        import matplotlib.pyplot as plt

        # First, check that all priors and models can be sampled from
        for param in self.hypers:
            if not hasattr(param.prior, 'sample'):
                print 'Prior of param %s cannot be sampled from. Cannot perform the Geweke correctness test.' % param.name
                return

        n = 10000 # number of samples # n = self.mcmc_iters
        statistic_of_interest = np.mean

        true_data = copy.copy(self.data) # reset this at the end

        # Case A: 
            # 1) Draw new hypers from priors
            # 2) Draw new data given hypers (**NOT** given hypers and data !!!!)
        caseA = np.zeros(n)
        for i in xrange(n):
            if i % 1000 == 0:
                print 'Geweke Part A Sample %d/%d' % (i,n)
            for param in self.hypers:
                param.sample_from_prior()
            latent_y = self.sample_from_prior_given_hypers(self.data) # only inputs used
            
            # fants = latent_y
            fants = self.observation_model(latent_y)
            # self.noise.print_diagnostics()
            # print fants

            caseA[i] = statistic_of_interest(fants)

        # Case B:
            # 1) Resample all hypers one step given data
            # 2) Resample data given hypers
            # repeat a bunch of times
        caseB = np.zeros(n)
        for i in xrange(n):
            if i % 1000 == 0:
                print 'Geweke Part B Sample %d/%d' % (i,n)
            # Take MCMC step on theta given data
            self.sampler.generate_sample() # data['inputs'] and data['values'] used

            # Resample data
            latent_y = self.sample_from_prior_given_hypers(self.data) # only data['inputs'] used

            # self.data['values'] = latent_y
            self.data['values'] = self.observation_model(latent_y)  # add noise
            # self.noise.print_diagnostics()
            # print self.data['values']

            caseB[i] = statistic_of_interest(self.data['values'])
        
        print np.mean(caseA)
        print np.std(caseA)
        print np.mean(caseB)
        print np.std(caseB)

        # Then, sort the sets A and B.
        caseA = np.sort(caseA)
        caseB = np.sort(caseB)

        # Then for each a in A, take the fraction of B smaller than it. 
        yAxis = np.zeros(n)
        for i in xrange(n):
            yAxis[i] = np.sum(caseB < caseA[i]) / float(n)

        xAxis = np.arange(n)/float(n)
        # Plot fractional index of a vs this fraction. 
        # Repeat for all a in A so number of points on graph is |A| ( = |B| )

        if not os.path.isdir('diagnostics'):
            os.mkdir('diagnostics')
        if not os.path.isdir('diagnostics/correctness'):
            os.mkdir('diagnostics/correctness')

        plt.figure(1)
        plt.clf()
        plt.plot(xAxis, yAxis, 'b')
        plt.plot(xAxis, xAxis, '--r')
        plt.title('Geweke test P-P plot with %d samples' % n)
        plt.savefig('diagnostics/correctness/GewekeCorrectness_%d_samples.pdf' % n)


        self.data = true_data

def test_gp():

    x     = np.linspace(-5,5,10)[:,None] # 10 data points, 1-D
    xtest = np.linspace(-6,6,200)[:,None]

    y     = np.sin(x.flatten()) + np.sqrt(1e-3)*np.random.randn(x.shape[0])
    ytest = np.sin(xtest.flatten())

    # print 'Inputs'
    # print x
    # print 'Outputs'
    # print y

    data            = {'inputs':x,     'values':y}
    pred            = {'inputs':xtest, 'values':ytest}

    options = {'likelihood':'GAUSSIAN', 
                'mcmc-iters':500, 
                'burn-in':500, 
                'verbose':False, 
                'mcmc-diagnostics':True, 
                'thinning':0, 
                'priors': {'mean':{'distribution':'Gaussian', 'parameters':{'mu':0.0, 'sigma':1.0}},
                         'noise':{'distribution':'Lognormal', 'parameters':{'scale':1.0}},
                         'amp2' :{'distribution':'Lognormal', 'parameters':{'scale':1.0}} 
                         }
            }
    
    gp = GP(x.shape[1], **options)
    gp.fit(data)

    func_m, func_v = gp.predict(pred, full_cov=False, compute_grad=False)

    # func_m, func_v, grad_m, grad_v = gp.predict(pred, full_cov=False, compute_grad=True)

    # print np.hstack((func_m[:,None], ytest[:,None]))

if __name__ == '__main__':
    test_gp()


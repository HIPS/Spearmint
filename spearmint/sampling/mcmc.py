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


import numpy        as np
import numpy.random as npr

def elliptical_slice(xx, chol_Sigma, log_like_fn, *log_like_fn_args):
    D  = xx.size

    # Select a random ellipse.
    nu = np.dot(chol_Sigma, npr.randn(D))

    # Select the slice threshold.
    hh = np.log(npr.rand()) + log_like_fn(xx, *log_like_fn_args)

    # Randomly center the bracket.
    phi     = npr.rand()*2*np.pi
    phi_max = phi
    phi_min = phi_max - 2*np.pi

    # Loop until acceptance.
    while True:

        # Compute the proposal.
        xx_prop = xx*np.cos(phi) + nu*np.sin(phi)

        # If on the slice, return the proposal.
        if log_like_fn(xx_prop, *log_like_fn_args) > hh:
            return xx_prop

        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception("Shrank to zero!")

        phi = npr.rand()*(phi_max - phi_min) + phi_min

def uni_slice_sample(init_x, logprob, lower, upper, *logprob_args):
    llh_s = np.log(npr.rand()) + logprob(init_x, *logprob_args)
    while True:
        new_x = npr.rand()*(upper-lower) + lower
        new_llh = logprob(new_x, *logprob_args)
        if new_llh > llh_s:
            return new_x
        elif new_x < init_x:
            lower = new_x
        elif new_x > init_x:
            upper = new_x
        else:
            raise Exception("Slice sampler shrank to zero!")

def slice_sample(init_x, logprob, *logprob_args, **slice_sample_args):
    """generate a new sample from a probability density using slice sampling
    
    Parameters
    ----------
    init_x : array
    logprob : callable, `lprob = logprob(x, *logprob_args)`
        A functions which returns the log probability at a given
        location
    *logprob_args :
        additional arguments are passed to logprob
        
    TODO: this function has too many levels and is hard to read.  It would be clearer
    as a class or just moving the sub-functions to another location
    
    Returns
    -------
    new_x : float
        the sampled position
    new_llh : float 
        the log likelihood at the new position (I'm not sure about this?)
        
    Notes
    -----
    http://en.wikipedia.org/wiki/Slice_sampling

    """
    sigma         = slice_sample_args.get('sigma', 1.0)
    step_out      = slice_sample_args.get('step_out', True)
    max_steps_out = slice_sample_args.get('max_steps_out', 1000)
    compwise      = slice_sample_args.get('compwise', True)
    doubling_step = slice_sample_args.get('doubling_step', True)
    verbose       = slice_sample_args.get('verbose', False)

    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction*z + init_x, *logprob_args)

        def acceptable(z, llh_s, L, U):
            while (U-L) > 1.1*sigma:
                middle = 0.5*(L+U)
                splits = (middle > 0 and z >= middle) or (middle <= 0 and z < middle)
                if z < middle:
                    U = middle
                else:
                    L = middle
                # Probably these could be cached from the stepping out.
                if splits and llh_s >= dir_logprob(U) and llh_s >= dir_logprob(L):
                    return False
            return True
    
        upper = sigma*npr.rand()
        lower = upper - sigma
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            if doubling_step:
                while (dir_logprob(lower) > llh_s or dir_logprob(upper) > llh_s) and (l_steps_out + u_steps_out) < max_steps_out:
                    if npr.rand() < 0.5:
                        l_steps_out += 1
                        lower       -= (upper-lower)                        
                    else:
                        u_steps_out += 1
                        upper       += (upper-lower)
            else:
                while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                    l_steps_out += 1
                    lower       -= sigma                
                while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                    u_steps_out += 1
                    upper       += sigma

        start_upper = upper
        start_lower = lower

        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*npr.rand() + lower
            new_llh   = dir_logprob(new_z)
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x, *logprob_args)
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s and acceptable(new_z, llh_s, start_lower, start_upper):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in

        return new_z*direction + init_x, new_llh

    if type(init_x) == float or isinstance(init_x, np.number):
        init_x = np.array([init_x])
        scalar = True
    else:
        scalar = False

    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        npr.shuffle(ordering)
        new_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            new_x, new_llh = direction_slice(direction, new_x)

    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))
        new_x, new_llh = direction_slice(direction, init_x)

    if scalar:
        return float(new_x[0]), new_llh
    else:
        return new_x, new_llh
          

def slice_sample_simple(init_x, logprob, *logprob_args, **slice_sample_args):
    sigma         = slice_sample_args.get('sigma', 1.0)
    step_out      = slice_sample_args.get('step_out', True)
    max_steps_out = slice_sample_args.get('max_steps_out', 1000)
    compwise      = slice_sample_args.get('compwise', True)
    verbose       = slice_sample_args.get('verbose', False)

    # Keep track of the number of evaluations of the logprob function
    # funEvals = {'funevals': 0} # sorry, i don't know how to actually do this properly with all these nested function. pls forgive me -MG -- from collections imoprt Counter ?
    
    # this is a 1-d sampling subrountine that only samples along the direction "direction"
    def direction_slice(direction, init_x):
        
        def dir_logprob(z): # logprob of the proposed point (x + dir*z) where z must be the step size
            # funEvals['funevals'] += 1
            try:
                return logprob(direction*z + init_x, *logprob_args)
            except:
                print 'ERROR: Logprob failed at input %s' % str(direction*z + init_x)
                raise
                
    
        # upper and lower are step sizes -- everything is measured relative to init_x
        upper = sigma*npr.rand()  # random thing above 0
        lower = upper - sigma     # random thing below 0
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)  # = log(prob_current * rand) 
        # (above) uniformly sample vertically at init_x
    
    
        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            # increase upper and decrease lower until they overshoot the curve
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower       -= sigma  # make lower smaller by sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper       += sigma
        
        
        # rejection sample along the horizontal line (because we don't know the bounds exactly)
        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*npr.rand() + lower  # uniformly sample between upper and lower
            new_llh   = dir_logprob(new_z)  # new current logprob
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x)
                raise Exception("Slice sampler got a NaN logprob")
            if new_llh > llh_s:  # this is the termination condition
                break       # it says, if you got to a better place than you started, you're done
                
            # the below is only if you've overshot, meaning your uniform sample from the horizontal
            # slice ended up outside the curve because the bounds lower and upper were not tight
            elif new_z < 0:  # new_z is to the left of init_x
                lower = new_z  # shrink lower to it
            elif new_z > 0:
                upper = new_z
            else:  # upper and lower are both 0...
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in, "Final logprob:", new_llh

        # return new the point
        return new_z*direction + init_x, new_llh

    
    # begin main
    
    # # This causes an extra "logprob" function call -- might want to turn off for speed
    initial_llh = logprob(init_x, *logprob_args)
    if verbose:
        sys.stderr.write('Logprob before sampling: %f\n' % initial_llh)
    if np.isneginf(initial_llh):
        sys.stderr.write('Values passed into slice sampler: %s\n' % init_x)
        raise Exception("Initial value passed into slice sampler has logprob = -inf")
    
    if not init_x.shape:  # if there is just one dimension, stick it in a numpy array
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    if compwise:   # if component-wise (independent) sampling
        ordering = range(dims)
        npr.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            cur_x, cur_llh = direction_slice(direction, cur_x)
            
    else:   # if not component-wise sampling
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))  # pick a unit vector in a random direction
        cur_x, cur_llh = direction_slice(direction, init_x)  # attempt to sample in that direction
    
    return cur_x, cur_llh
    # return (cur_x, funEvals['funevals']) if returnFunEvals else cur_x



if __name__ == '__main__':
    npr.seed(1)

    import pylab as pl
    import pymc

    D  = 10
    fn = lambda x: -0.5*np.sum(x**2)

    iters = 1000
    samps = np.zeros((iters,D))
    for ii in xrange(1,iters):
        samps[ii,:] = slice_sample(samps[ii-1,:], fn, sigma=0.1, step_out=False, doubling_step=True, verbose=False)

    ll = -0.5*np.sum(samps**2, axis=1)

    scores = pymc.geweke(ll)
    pymc.Matplot.geweke_plot(scores, 'test')

    pymc.raftery_lewis(ll, q=0.025, r=0.01)

    pymc.Matplot.autocorrelation(ll, 'test')


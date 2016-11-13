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
import numpy as np
from collections import OrderedDict


class InputSpace(object):
    """
    An input space as specified in the Spearmint config file.
    Handles things like squashing the inputs into the unit hypercube.
    """

    def __init__(self, variables_config):
        """
        Converts a dict of variable meta-information from a config-file format into
        a format that can be more easily used by bayesopt routines.
        """
        # Stores the metadata for the dataset that allows a conversion
        # from a config file representation into a matrix representation.
        # The main addition that this variable adds is a mapping between
        # each variable and associated column indices in the matrix
        # representation.
        variables_meta = OrderedDict()
        cardinality    = 0 # The number of distinct variables
        num_dims       = 0 # The number of dimensions in the matrix representation

        for name, variable in variables_config.iteritems():
            cardinality += variable['size']
            vdict = {'type'    : variable['type'].lower(),
                     'indices' : []} # indices stores a mapping from these variable(s) to their matrix column(s)

            if vdict['type'] == 'int' or vdict['type'] == 'float':
                conv_func = int if vdict['type'] == 'int' else float
                
                # here is some hackery for prototyping the unknown "?" bounds
                # basically, if min and max are ? then we start with 0 and 1
                # if only 1 is "?" we start with a region of size 1 for that dimensions
                # we use conv_func so we don't need repeated code for int and float
                if variable['min'] == "?":
                    if variable['max'] == "?":
                        vdict['min'] = 0
                        vdict['max'] = 1
                        vdict['min expandable'] = True
                        vdict['max expandable'] = True
                        print 'Expandable min and max bounds detected for %s' % name
                    else:
                        vdict['max'] = conv_func(variable['max'])
                        vdict['min'] = vdict['max'] - 1
                        vdict['min expandable'] = True
                        vdict['max expandable'] = False
                        print 'Expandable min bound detected for %s' % name
                else:
                    if variable['max'] == "?":
                        vdict['min'] = conv_func(variable['min'])
                        vdict['max'] = vdict['min'] + 1
                        vdict['min expandable'] = False
                        vdict['max expandable'] = True
                        print 'Expandable max bound detected for %s' % name
                    else:
                        vdict['min'] = conv_func(variable['min'])
                        vdict['max'] = conv_func(variable['max'])
                        vdict['min expandable'] = False
                        vdict['max expandable'] = False
            elif vdict['type'] == 'enum':
                vdict['options'] = list(variable['options'])
            else:
                raise Exception("Unknown variable type.")

            for i in xrange(variable['size']):
                if vdict['type'] == 'int':
                    vdict['indices'].append(num_dims)
                    num_dims += 1
                elif vdict['type'] == 'float':
                    vdict['indices'].append(num_dims)
                    num_dims += 1
                elif vdict['type'] == 'enum':
                    vdict['indices'].append(list(np.arange(len(list(variable['options']))) + num_dims))
                    num_dims += len(list(variable['options']))
                else:
                    raise Exception("Unknown variable type.")

            variables_meta[name] = vdict

        self.variables_meta = variables_meta
        self.num_dims = num_dims
        self.cardinality = cardinality

    def print_params(self, params, left_indent=0, indent_top_row=True, print_func=logging.info):
        indentation = ' '*left_indent
        
        top_row = 'NAME          TYPE       VALUE'
        if indent_top_row:
            top_row = indentation + top_row
        print_func(top_row)
        print_func(indentation + '----          ----       -----')

        for param_name, param in params.iteritems():

            if param['type'] == 'float':
                format_str = '%s%-12.12s  %-9.9s  %-12f'
            else:
                format_str = '%s%-12.12s  %-9.9s  %-12d'

            for i in xrange(len(param['values'])):
                if i == 0:
                    print_func(format_str % (indentation, param_name, param['type'], param['values'][i]))
                else:
                    print_func(format_str % (indentation, '', '',                    param['values'][i]))

    def paramify_and_print(self, data_vector, **kwargs):
        params = self.paramify(data_vector)
        self.print_params(params, **kwargs)

    # Converts a vector in input space to the corresponding dict of params
    def paramify(self, data_vector):
        if data_vector.ndim == 0:
            data_vector=data_vector[None]
        if data_vector.ndim != 1:
            raise Exception('Input to paramify must be a 1-D array.')

        params = OrderedDict()
        for name, vdict in self.variables_meta.iteritems():
            indices = vdict['indices']
            params[name] = {}
            params[name]['type'] = vdict['type']

            if vdict['type'] == 'int' or vdict['type'] == 'float':
                params[name]['values'] = data_vector[indices]
            elif vdict['type'] == 'enum':
                params[name]['values'] = []
                for ind in indices:
                    params[name]['values'].append(vdict['options'][data_vector[ind].argmax(0)])
            else:
                raise Exception('Unknown parameter type.')
            
        return params

    # get the list of indices from the names here
    def get_indices(self, param_names):
        inds = np.zeros(0, dtype=int)
        for param_name in param_names:
            inds = np.append(inds, self.variables_meta[param_name]['indices'])
        return inds

    # Converts a dict of params to the corresponding vector in puts space
    def vectorify(self, params):
        v = np.zeros(self.num_dims)
        for name, param in params.iteritems():
            indices = self.variables_meta[name]['indices']

            if param['type'] == 'int' or param['type'] == 'float':
                v[indices] = param['values']
            elif param['type'] == 'enum':
                for i,ind in enumerate(indices):
                    offset           = self.variables_meta[name]['options'].index(param['values'][i])
                    v[ind[0]+offset] = 1
            else:
                raise Exception('Unknown parameter type.')

        return v

    # x is a distance in the original space (a float)
    # we just want to rescale it to be a distance in the new space. 
    # so we only care about changing the size
    # assume it's a float
    def rescale_to_unit(self, x):
        V = x.copy()
        if V.ndim == 1:
            V = V[None]

        for name, variable in self.variables_meta.iteritems():
            indices = variable['indices']
            V[:,indices] /= (variable['max']-variable['min'])
        
        return np.squeeze(V)

    # Consider memoization here if this method becomes a bottleneck
    def to_unit(self, V):
        if V.shape[0] == 0:
            return V

        if V.ndim == 1:
            V = V[None,:]
            squeeze = True
        else:
            squeeze = False

        U = np.zeros(V.shape)
        for name, variable in self.variables_meta.iteritems():
            indices = variable['indices']
            if variable['type'] == 'int':
                vals = V[:,indices]
                U[:,indices] = self.int_to_unit(vals, variable['min'], variable['max'])
            elif variable['type'] == 'float':
                vals = V[:,indices]
                U[:,indices] = self.float_to_unit(vals, variable['min'], variable['max'])
            elif variable['type'] == 'enum':
                for ind in indices:
                    U[:,ind] = V[:,ind] # Assumed to already be stored in a 1-hot encoding
            else:
                raise Exception("Unknown variable type.")

        if squeeze:
            U = np.squeeze(U)

        return U
        
    def from_unit(self, U):
        if U.shape[0] == 0:
            return U

        if U.ndim == 1:
            U = U[None,:]
            squeeze = True
        else:
            squeeze = False

        V = np.zeros(U.shape)
        for name, variable in self.variables_meta.iteritems():
            indices = variable['indices']
            if variable['type'] == 'int':
                vals = U[:,indices]
                V[:,indices] = self.unit_to_int(vals, variable['min'], variable['max'])
            elif variable['type'] == 'float':
                vals = U[:,indices]
                V[:,indices] = self.unit_to_float(vals, variable['min'], variable['max'])
            elif variable['type'] == 'enum':
                for ind in indices:
                    # This is a bit more complicated than to_unit because
                    # the values might come from the unit hypercube, meaning
                    # that U might not have a 1-hot encoding.
                    v = np.zeros(V[:,ind].shape)
                    v[np.arange(v.shape[0]),U[:,ind].argmax(1)] = 1
                    V[:,ind] = v
            else:
                raise Exception("Unknown variable type: %s" % variable['type'])

        if squeeze:
            V = np.squeeze(V)

        return V

    # Convert primitive types to the unit hypercube
    def int_to_unit(self, v, vmin, vmax):
        unit = (np.double(v) - vmin) / (vmax - vmin)
        
        # Make sure we are not over bounds
        try:
            unit[unit > 1] = 1
            unit[unit < 0] = 0
        except:
            if unit > 1:
                unit = 1
            elif unit < 0:
                unit = 0
        return unit

    def float_to_unit(self, v, vmin, vmax):
        unit = (np.double(v) - vmin) / (vmax - vmin)
        
        # Make sure we are not over bounds
        try:
            unit[unit > 1] = 1.0
            unit[unit < 0] = 0.0
        except:
            if unit > 1:
                unit = 1.0
            elif unit < 0:
                unit = 0.0
        return unit

    def enum_to_unit(self, v, options):
        u = np.zeros(len(options))
        u[options.index(v)] = 1
        return u

    # Convert unit hypercube values to primitive types
    def unit_to_int(self, u, vmin, vmax):
        return vmin + np.int32(np.floor((1-np.finfo(float).eps) * u * np.double(vmax-vmin+1)))

    def unit_to_float(self, u, vmin, vmax):
        return vmin + u * (vmax-vmin)

    def unit_to_enum(self, u, options):
        return options[u.argmax()]


# Converts a vector in input space to the corresponding dict of params
def paramify_no_types(input_params):
    params = dict()
    for name, param in input_params.iteritems():
        vals = param['values']

        if param['type'].lower() == 'float':
            params[name] = np.array(vals)
        elif param['type'].lower() == 'int':
            params[name] = np.array(vals, dtype=int)
        elif param['type'].lower() == 'enum':
            params[name] = vals
        else:
            raise Exception("Unknown parameter type.")

    return params
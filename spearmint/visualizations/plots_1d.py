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
# Jasper Snoek and at Harvard University, Kevin Swersky and Richard
# Zemel at the University of Toronto (“Toronto”), and Hugo Larochelle
# at the Université de Sherbrooke (“Sherbrooke”), which assigned its
# rights in the Software to Socpra Sciences et Génie
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



import os
import sys
sys.path.append('..')
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

from choosers.gp_ei_opt_decoupled_constrained_chooser import GPDecoupledConstraintEntropySearchChooser as DecCho
from choosers.gp_ei_opt_chooser import GPEIOptChooser
from models.gp import GP
from choosers.acquisition_functions import compute_ei

if __name__ == '__main__':
	N = 1000

	# create true functions
	obj_fun = lambda x: -np.sin(6*x)*x
	con_fun = lambda x: np.cos(5*x+1.0)*x*x*0.9+0.05
	combined_fun = lambda x: obj_fun(x) + 10*(con_fun(x)<0)

	xgrid = np.linspace(0,1,N)[:,None]
	# obj_observations = np.array([0.2, 0.5, 0.55, 1.0])[:,None]
	# con_observations = np.array([0.1, 0.5,0.72, 0.73])[:,None]
	# total_observations = np.append(obj_observations, con_observations, axis=0)
	obj_observations = np.array([0.1, 0.5,0.72, 0.73])[:,None]
	con_observations = np.array([0.1, 0.5,0.72, 0.73])[:,None]
	total_observations = np.array([0.1, 0.5,0.72, 0.73])[:,None]

	# true min
	comb = obj_fun(xgrid) + 10*(con_fun(xgrid)<0)
	bestx=xgrid[np.argmin(comb)]

	plt.figure()
	plt.clf()

	gs = gridspec.GridSpec(5, 1,height_ratios=[2,2,1.2,2,1.2])

	# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
		               	    # wspace=None, hspace=0)

	### 1st subplot -- objective
	######################################################
	plt.subplot(gs[0])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.plot(xgrid, obj_fun(xgrid), 'r', linewidth=2)
	plt.plot(obj_observations, obj_fun(obj_observations[:,0]), '.b', markersize=12)
	obj_data = {'inputs': obj_observations, 'targets':obj_fun(obj_observations[:,0])}
	obj_gp = GP("test_gp_bv2", obj_data, 1, {"likelihood":"NOISELESS"})
	obj_mean, obj_var = obj_gp.predict({'inputs':xgrid})
	obj_std = np.sqrt(obj_var)
	# plt.plot(xgrid, obj_mean, 'b')
	plt.fill_between(xgrid[:,0], obj_mean-obj_std, obj_mean+obj_std,color='b',alpha=0.25)
	plt.plot(bestx, obj_fun(bestx), 'r*', markersize=14 )
	plt.ylabel('f(x)')
	

	### 2nd subplot -- constraint
	######################################################
	plt.subplot(gs[1])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.axhline(0, color='black')
	plt.plot(xgrid, con_fun(xgrid), 'r', linewidth=2)
	plt.plot(con_observations, con_fun(con_observations[:,0]), '.g', markersize=12)
	con_data = {'inputs': con_observations, 'targets':con_fun(con_observations[:,0])}
	con_gp = GP("test_gp_bvcon", con_data, 1, {"likelihood":"NOISELESS"})
	con_mean, con_var = con_gp.predict({'inputs':xgrid})
	con_std = np.sqrt(con_var)
	plt.plot(xgrid, con_mean, 'g')
	plt.fill_between(xgrid[:,0], con_mean-con_std, con_mean+con_std,color='g',alpha=0.25)
	plt.ylabel('g(x)')
	

	### 3rd subplot -- weighted EI ###
	######################################################
	plt.subplot(gs[2])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	# plt.axhline(0, color='black')
	ei = obj_gp.function_over_hypers(compute_ei, obj_gp, {'inputs':xgrid}, compute_grad=False)
	pi = con_gp.function_over_hypers(con_gp.pi, {'inputs':xgrid}, compute_grad=False)
	weighted_ei = ei*pi
	plt.fill_between(xgrid[:,0], np.zeros(xgrid.shape[0]), weighted_ei, color='k', alpha=0.25)
	plt.plot(xgrid[:,0], weighted_ei, color='k', linewidth=1)
	plt.plot(xgrid[np.argmax(weighted_ei)], np.max(weighted_ei), '.k', markersize=12)
	plt.ylabel('EI')
	plt.ylim([0.0, np.max(weighted_ei)*1.1])
	### 4st subplot -- combined model, naive
	######################################################
	plt.subplot(gs[3])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.plot(xgrid, combined_fun(xgrid), 'r', linewidth=2)
	plt.plot(total_observations, combined_fun(total_observations[:,0]), '.b', markersize=12)
	combined_data = {'inputs': total_observations, 'targets':combined_fun(total_observations[:,0])}
	comb_gp = GP("test_gp_bv5", combined_data, 1, {"likelihood":"NOISELESS"})
	comb_mean, comb_var = comb_gp.predict({'inputs':xgrid})
	comb_std = np.sqrt(comb_var)
	# plt.plot(xgrid, obj_mean, 'b')
	plt.fill_between(xgrid[:,0], comb_mean-comb_std, comb_mean+comb_std,color='b',alpha=0.25)
	bestcombx=xgrid[np.argmin(combined_fun(xgrid))]
	plt.plot(bestcombx, combined_fun(bestcombx), 'r*', markersize=14 )
	plt.ylabel('h(x)')

	### 5th subplot -- naive EI
	######################################################
	plt.subplot(gs[4])
	# Regular EI
	# combined_data = {'inputs': total_observations, 'targets':combined_fun(total_observations[:,0])}
	# comb_gp = GP("test_gp_bv5", combined_data, 1, {"likelihood":"NOISELESS"})
	ei = comb_gp.function_over_hypers(compute_ei, comb_gp, {'inputs':xgrid}, compute_grad=False)
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.axhline(0, color='black')
	plt.fill_between(xgrid[:,0], np.zeros(xgrid.shape[0]), ei, color='k', alpha=0.25)
	plt.plot(xgrid[:,0], ei, color='k', linewidth=1)
	# max of EI
	plt.plot(xgrid[np.argmax(ei)], np.max(ei), '.k', markersize=12)
	plt.ylabel('EI')
	plt.ylim([0.0, np.max(ei)*1.1])

	plt.tight_layout()

	plt.savefig('test.png')
















	plt.figure()
	plt.clf()

	gs = gridspec.GridSpec(3, 1,height_ratios=[1,1,1])

	# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
		               	    # wspace=None, hspace=0)

	### 1st subplot -- objective
	######################################################
	plt.subplot(gs[0])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.plot(xgrid, obj_fun(xgrid), 'r', linewidth=2)
	plt.plot(obj_observations, obj_fun(obj_observations[:,0]), '.b', markersize=12)
	obj_data = {'inputs': obj_observations, 'targets':obj_fun(obj_observations[:,0])}
	obj_gp = GP("test_gp_bv2", obj_data, 1, {"likelihood":"NOISELESS"})
	obj_mean, obj_var = obj_gp.predict({'inputs':xgrid})
	obj_std = np.sqrt(obj_var)
	# plt.plot(xgrid, obj_mean, 'b')
	plt.fill_between(xgrid[:,0], obj_mean-obj_std, obj_mean+obj_std,color='b',alpha=0.25)
	plt.plot(bestx, obj_fun(bestx), 'r*', markersize=14 )
	plt.ylabel('f(x)')
	

	### 2nd subplot -- constraint
	######################################################
	plt.subplot(gs[1])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.axhline(0, color='black')
	plt.plot(xgrid, con_fun(xgrid), 'r', linewidth=2)
	plt.plot(con_observations, con_fun(con_observations[:,0]), '.g', markersize=12)
	con_data = {'inputs': con_observations, 'targets':con_fun(con_observations[:,0])}
	con_gp = GP("test_gp_bvcon", con_data, 1, {"likelihood":"NOISELESS"})
	con_mean, con_var = con_gp.predict({'inputs':xgrid})
	con_std = np.sqrt(con_var)
	plt.plot(xgrid, con_mean, 'g')
	plt.fill_between(xgrid[:,0], con_mean-con_std, con_mean+con_std,color='g',alpha=0.25)
	plt.ylabel('g(x)')
	

	### 4st subplot -- combined model, naive
	######################################################
	plt.subplot(gs[2])
	ax = plt.gca()
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.xticks([])
	plt.yticks([])
	plt.plot(xgrid, combined_fun(xgrid), 'r', linewidth=2)
	plt.plot(total_observations, combined_fun(total_observations[:,0]), '.b', markersize=12)
	combined_data = {'inputs': total_observations, 'targets':combined_fun(total_observations[:,0])}
	comb_gp = GP("test_gp_bv5", combined_data, 1, {"likelihood":"NOISELESS"})
	comb_mean, comb_var = comb_gp.predict({'inputs':xgrid})
	comb_std = np.sqrt(comb_var)
	# plt.plot(xgrid, obj_mean, 'b')
	plt.fill_between(xgrid[:,0], comb_mean-comb_std, comb_mean+comb_std,color='b',alpha=0.25)
	bestcombx=xgrid[np.argmin(combined_fun(xgrid))]
	plt.plot(bestcombx, combined_fun(bestcombx), 'r*', markersize=14 )
	plt.ylabel('h(x)')
	plt.tight_layout()

	plt.savefig('test2.png')


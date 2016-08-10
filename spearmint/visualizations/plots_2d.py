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
import importlib
import numpy             as np
import matplotlib        as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# mpl.rcParams['ps.useafm'] = True
# mpl.rcParams['pdf.use14corefonts'] = True
# mpl.rcParams['axes.unicode_minus']=False
# mpl.rcParams.update({'font.size': 30})
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'

from collections                      import defaultdict 
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.main                   import load_jobs
from spearmint.main                   import print_hypers
from spearmint.models.abstract_model  import function_over_hypers
from spearmint.utils.parsing          import repeat_experiment_name

cmap = mpl.cm.get_cmap('Blues')

x_size = 16
x_width = 4
o_data_size = 12
star_size = 30
star_width = 4

nGridPoints = 100


def get_ready_to_plot(input_space, current_best):

    if len(input_space.variables_meta) == 2:
        xmin = input_space.variables_meta.values()[0]['min']
        xmax = input_space.variables_meta.values()[0]['max']
        ymin = input_space.variables_meta.values()[1]['min']
        ymax = input_space.variables_meta.values()[1]['max']
        bounds = (xmin,xmax,ymin,ymax)
        xlabel = input_space.variables_meta.keys()[0]
        ylabel = input_space.variables_meta.keys()[1]
    elif len(input_space.variables_meta) == 1:
        xmin = input_space.variables_meta.values()[0]['min']
        xmax = input_space.variables_meta.values()[0]['max']
        ymin = input_space.variables_meta.values()[0]['min']
        ymax = input_space.variables_meta.values()[0]['max']
        bounds = (xmin,xmax,ymin,ymax)
        xlabel = input_space.variables_meta.keys()[0] + "_1"
        ylabel = input_space.variables_meta.keys()[0] + "_2"
    else:
        raise Exception("How can num_dims be 2 if the number of variables is not 1 or 2??")

    if current_best is not None:
        current_best = input_space.from_unit(current_best)
        current_best = current_best.flatten()

    x_grid = np.linspace(0.0, 1.0, nGridPoints)
    y_grid = np.linspace(0.0, 1.0, nGridPoints)
    X, Y = np.meshgrid(x_grid, y_grid)
    flat_grid = np.hstack((X.flatten()[:,np.newaxis], Y.flatten()[:,np.newaxis]))

    def setaxes():
        plt.xlim((xmin,xmax))
        plt.ylim((ymin,ymax))
        plt.xticks(np.linspace(xmin, xmax, 6), size=20)
        plt.yticks(np.linspace(ymin, ymax, 6), size=20)

        if xlabel is not None:
            plt.xlabel(xlabel, size=35)
        if ylabel is not None:
            plt.ylabel(ylabel, size=35)

        plt.tight_layout()

    mapped_flat_grid = input_space.from_unit(flat_grid)

    mappedX = np.reshape(mapped_flat_grid[:,0], (nGridPoints,nGridPoints))
    mappedY = np.reshape(mapped_flat_grid[:,1], (nGridPoints,nGridPoints))

    return flat_grid, mappedX, mappedY, current_best, bounds, setaxes


def plot_2d_mean_and_var(model, directory, filename_prefix, input_space, current_best=None):
    
    flat_grid, mappedX, mappedY, mapped_current_best, bounds, setaxes = get_ready_to_plot(input_space, current_best)

    figpath = os.path.join(directory, filename_prefix)

    ############## ---------------------------------------- ############
    ##############                                          ############
    ##############           PLOT MEAN FUNCTION             ############
    ##############                                          ############
    ############## ---------------------------------------- ############
    fig = plt.figure(1)
    plt.clf()

    mean, var = model.function_over_hypers(model.predict, flat_grid)
    mean_on_grid = np.reshape(mean, (nGridPoints,nGridPoints))
    var_on_grid  = np.reshape(var,  (nGridPoints,nGridPoints))

    x_data = model.observed_inputs
    y_data = model.observed_values

    mapped_observed_inputs = input_space.from_unit(x_data)

    CS = plt.contourf(mappedX, mappedY, mean_on_grid, 20, cmap=cmap)
    # plt.clabel(CS, inline=1, fontsize=8)

    plt.colorbar(CS)

    plt.plot(mapped_observed_inputs[:,0], mapped_observed_inputs[:,1], color='k', marker='o', markersize=o_data_size, linestyle='None', markerfacecolor=(0,0,0,0.6), markeredgewidth=2, markeredgecolor='k')


    if mapped_current_best is not None:
        plt.plot(mapped_current_best[0], mapped_current_best[1], marker='*', color='orange', markersize=star_size)

    setaxes()

    plt.savefig('%s_mean.pdf' % figpath)



    ############### ---------------------------------------- ############
    ###############                                          ############
    ###############           PLOT SIGMA FUNCTION            ############
    ###############                                          ############
    ############### ---------------------------------------- ############
    fig = plt.figure(2)
    plt.clf()

    CS = plt.contourf(mappedX, mappedY, var_on_grid, 12, cmap=cmap)
    plt.colorbar(CS)
    # plt.clabel(CS, inline=1, fontsize=8)

    plt.plot(mapped_observed_inputs[:,0], mapped_observed_inputs[:,1], 
        color='k', marker='o', markersize=o_data_size, 
        linestyle='None', markerfacecolor=(0,0,0,0.6), 
        markeredgewidth=2, markeredgecolor='k')


    setaxes()

    plt.savefig('%s_sigma.pdf' % figpath)



def plot_2d_constraints(chooser, directory, input_space, current_best=None):

    overall_probabilistic_constraint = np.ones((nGridPoints, nGridPoints))
    overall_p_valid = np.ones((nGridPoints, nGridPoints))

    for constraint in chooser.constraints:

        model = chooser.models[constraint]
        delta = chooser.tasks[constraint].options['constraint_delta']

        figpath = os.path.join(directory, constraint)

        flat_grid, mappedX, mappedY, mapped_current_best, bounds, setaxes = get_ready_to_plot(input_space, current_best)


        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############        PLOT CONSTRAINT PROBABILITY       ############
        ##############       AND PROBABILISTIC CONSTRAINTS      ############
        ##############                                          ############        
        ############## ---------------------------------------- ############
        fig = plt.figure(3)
        plt.clf()

        pv = model.function_over_hypers(model.pi, flat_grid)
        pv_on_grid = np.reshape(pv, (nGridPoints,nGridPoints))

        # print 'max pv = %f' % np.max(pv)

        CS = plt.contourf(mappedX, mappedY, pv_on_grid, cmap=cmap)#, levels=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        plt.colorbar(CS)#, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])#, drawedges=False)

        # Plot the x's and o's -- even though they come as continuous -- ok fine.
        con_x = model.observed_inputs
        if hasattr(model, 'latent_values'):
            con_y = model.function_over_hypers(lambda: model.latent_values.value)
        else:
            con_y = model.observed_values
        mapped_observed_inputs = input_space.from_unit(con_x)

        if model.options['likelihood'].lower() == 'binomial':
            # Plot the counts (or the labels if total_counts=1)
            plt.plot(mapped_observed_inputs[:,0], mapped_observed_inputs[:,1], '.k', markersize=4)
            c = model.counts
            p_avg = model.function_over_hypers(lambda: model.sigmoid(model.latent_values.value))
            for j in xrange(c.size):
                plt.annotate('%d/%d' % (c[j], model.options['binomial_trials']), (mapped_observed_inputs[j,0], mapped_observed_inputs[j,1]), fontsize=11)
                plt.annotate('%.2f' % p_avg[j], (mapped_observed_inputs[j,0], mapped_observed_inputs[j,1]), 
                    fontsize=8, horizontalalignment='left', verticalalignment='top') # 'top' means below the data point, for some reason

        else:
            plt.plot(mapped_observed_inputs[con_y >= 0, 0], mapped_observed_inputs[con_y >= 0,1], color='k', linestyle='None', marker='o', markersize=x_size+1, markerfacecolor='none', markeredgewidth=2, markeredgecolor='k')
            plt.plot(mapped_observed_inputs[con_y <  0, 0], mapped_observed_inputs[con_y <  0,1], color='k', linestyle='None', marker='x', markersize=x_size, markeredgewidth=x_width)

        setaxes()


        plt.savefig(os.path.join(figpath, 'constraint_prob.pdf'))

        # plot valid region
        fig = plt.figure(4)
        plt.clf()

        validregion = pv_on_grid >= 1-delta
        plt.imshow(validregion, aspect='equal', extent=bounds, origin='lower', vmin=0, vmax=1, cmap=cmap)

        if chooser.numConstraints() == 1:
            plt.plot(mapped_current_best[0], mapped_current_best[1], marker='*', color='orange', markersize=star_size)

        setaxes()

        plt.savefig(os.path.join(figpath, 'probabilistic_constraint.pdf'))
        
        overall_probabilistic_constraint = np.logical_and(overall_probabilistic_constraint, validregion)
        overall_p_valid = overall_p_valid * pv_on_grid

    ############## ---------------------------------------- ############
    ##############                                          ############
    ##############  PLOT OVERALL PROBABILISTIC CONSTRAINT   ############
    ##############                                          ############
    ############## ---------------------------------------- ############
    if chooser.numConstraints() > 1: # only need the overall constraint if there is more than 1 constraint
        fig = plt.figure(5)
        plt.clf()

        plt.imshow(overall_probabilistic_constraint, aspect='equal', extent=bounds, origin='lower', vmin=0, vmax=1, cmap=cmap)

        plt.plot(mapped_current_best[0], mapped_current_best[1], marker='*', color='orange', markersize=star_size)

        setaxes()
        plt.savefig(os.path.join(directory, 'probabilistic_constraint_overall.pdf'))

        fig = plt.figure(6)
        plt.clf()

        CS = plt.contourf(mappedX, mappedY, overall_p_valid, cmap=cmap)#, vmin=0, vmax=1)
        plt.colorbar(CS)#, drawedges=False)

        plt.plot(mapped_current_best[0], mapped_current_best[1], marker='*', color='orange', markersize=star_size)

        # plot all the x's and o's
        for con_model in chooser.constraint_models:
            if con_model.options['likelihood'].lower == 'binomial':
                continue
            con_y = con_model.observed_values
            con_x = con_model.observed_inputs
            mapped_observed_inputs = input_space.from_unit(con_x)
            plt.plot(mapped_observed_inputs[con_y >= 0, 0], mapped_observed_inputs[con_y >= 0,1], color='k', linestyle='None', marker='o', markersize=x_size+1, markerfacecolor='none', markeredgewidth=2, markeredgecolor='k')
            plt.plot(mapped_observed_inputs[con_y <  0, 0], mapped_observed_inputs[con_y <  0,1], color='k', linestyle='None', marker='x', markersize=x_size, markeredgewidth=x_width)

        setaxes()
        plt.savefig(os.path.join(directory, 'constraint_prob_overall.pdf'))


#     # ############## ---------------------------------------- ############
#     # ##############                                          ############
#     # ##############         PLOT OVERALL p(VALID)            ############
#     # ##############                                          ############
#     # ############## ---------------------------------------- ############
#     fig = plt.figure(12)
#     plt.clf()
#     print 'plotting overall constraint prob'

#     pvall = self.confidence_total(flat_grid)
#     pvall_on_grid = np.reshape(pvall, X.shape)

#     CS = plt.contourf(mappedX, mappedY, pvall_on_grid, cmap=blues)#, levels=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
#     plt.colorbar(CS, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])#, drawedges=False)

#     setaxes()

#     plt.savefig('%s%04d_pv_overall.pdf' % (figpath, self.iterCount))


    ############## ---------------------------------------- ############
    ##############                                          ############
    ##############   PLOT OBJECTIVE ACQUISITION FUNCTION    ############
    ##############                                          ############
    ############## ---------------------------------------- ############
def plot_acquisition_function(chooser, directory, input_space, current_best_location, current_best_value):

    # if there is non-competitive decoupling, then the tasks might have different acquisition funcitons
    # so we see if there are multiple acquisition functions in play here...
    # this is a bit hacky and needs to be handled better later
    # acq_funcs = list({chooser.acquisition_functions[task_name]["name"] for task_name in chooser.options["tasks"]})
    # tasks_to_plot = [chooser.tasks.keys()[0]] if len(acq_funcs) == 1 else chooser.tasks.keys()

    # for task_name in tasks_to_plot:

    suggestion = chooser.suggest(chooser.tasks.keys())[0] # must do this before computing acq

    # acq_name = chooser.acquisition_functions[task_name]["name"]
    # acq_fun  = chooser.acquisition_functions[task_name]["class"](chooser.num_dims, DEBUG_input_space=input_space)
    # acq_fun = chooser.acq[task_name] # get the cached one, so that randomness like x* sampling in PES stays the same
    # above is the non-cached one, will sample a different x*
    acq_name = chooser.acquisition_function_name
    acq_fun = chooser.acq


    flat_grid, mappedX, mappedY, mapped_current_best, bounds, setaxes = get_ready_to_plot(input_space, current_best_location)

    fig = plt.figure(5)
    plt.clf()
    acq = function_over_hypers(chooser.models.values(), acq_fun.acquisition, 
                              chooser.objective_model_dict, chooser.constraint_models_dict, 
                              flat_grid, current_best_value, compute_grad=False)
    best_acq_index = np.argmax(acq)
    acq_on_grid = np.reshape(acq, (nGridPoints,nGridPoints))

    CS = plt.contourf(mappedX, mappedY, acq_on_grid, 20, cmap=cmap)
    plt.colorbar(CS)

    # plot the suggestion
    plt.plot(suggestion[0], suggestion[1], color='red', marker='x', markersize=10, markeredgewidth=0.5)
    # best_acq_location = flat_grid[best_acq_index]
    # mapped_best_acq_location = input_space.from_unit(best_acq_location).flatten()
    # plt.plot(mapped_best_acq_location[0], mapped_best_acq_location[1], color='green', marker='x', markersize=10, markeredgewidth=0.5)
    # plt.plot(mapped_best_acq_location[0], mapped_best_acq_location[1], color='red', marker='*', markersize=star_size)#, markeredgecolor='orange') # markeredgewidth=star_width, 
    
    # see the chooser's grid overlay
    # plt.plot(chooser.grid[:,0], chooser.grid[:,1], 'r.')

    setaxes()
    # dire = directory if len(acq_funcs) == 1 else os.path.join(directory, task_name)
    dire = directory
    plt.savefig(os.path.join(dire, '%s_acquisition_function.pdf' % acq_name))


# Plot the hyperparameters of the GP
def plot_hypers(model, directory, filename_prefix):

    plt.figure()
    plt.clf()

    # A function that labels bars in a bar chart with text indicating their value
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            y = rect.get_y()

            if y >= 0:
                textLocation = y+1.05*height
                text = height
            else:
                textLocation = y-0.1*height
                text = -height

            plt.text(rect.get_x()+rect.get_width()/2., textLocation, '%0.3f' % text, ha='center', va='bottom')


    # plot the hyperparameters with bar charts
    # --> plot the hyperparameters that are being sampled, namely the ones in self.hyper_names
    hypers_to_plot = defaultdict(list)
    for hyper_name, hyper in model.params.iteritems():

        for i in xrange(model.num_states):
            model.set_state(i)
            if hyper_name == 'noise':
                hypers_to_plot['noise_sig'].append(np.sqrt(hyper.value))
            elif hyper.size() == 1: 
                hypers_to_plot[hyper_name].append ( hyper.value )
            elif hyper_name == 'ls':
                hypers_to_plot['ls-x'].append( hyper.value[0] )
                hypers_to_plot['ls-y'].append( hyper.value [1] )
            else:
                for j in xrange(hyper.size()):
                    hypers_to_plot['%s%d' % (hyper_name, j)].append(hyper.value[j])
    
    width = 0.35

    # if self.noisy:
    ind = np.arange(len(hypers_to_plot))
    means = [np.mean(hypers_to_plot[h]) for h in hypers_to_plot]
    rects = plt.bar(ind, means, width, color='y')
    plt.boxplot(hypers_to_plot.values(), positions=ind+width/2, widths=width, whis=1e10) # a big value here means the whiskers are the max and min of the data
    plt.xticks(range(len(hypers_to_plot)), hypers_to_plot.keys()) # ind+width/2., 
    
    plt.setp(plt.xticks()[1], rotation=90)

    autolabel(rects)                                

    filename = os.path.join(directory, filename_prefix+"_hypers.pdf")

    plt.savefig(filename)


# Important: this function does not change the state of the experiment in the database...
# we only read from the database here.
def main(expt_dir, repeat=None):

    options = parse_config_file(expt_dir, 'config.json')
    experiment_name = options["experiment_name"]

    if repeat is not None:
        experiment_name = repeat_experiment_name(experiment_name,repeat)

    input_space = InputSpace(options["variables"])

    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    chooser = chooser_module.init(input_space, options)

    db = MongoDB(database_address=options['database']['address'])

    jobs = load_jobs(db, experiment_name)
    hypers = db.load(experiment_name, 'hypers')

    if input_space.num_dims != 2:
        raise Exception("This plotting script is only for 2D optimizations. This problem has %d dimensions." % input_space.num_dims)

    tasks = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)

    hypers = chooser.fit(tasks, hypers)

    print '\nHypers:'
    print_hypers(hypers)

    recommendation = chooser.best()
    current_best_value = recommendation['model_model_value']
    current_best_location = recommendation['model_model_input']

    plots_dir = os.path.join(expt_dir, 'plots')
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    if len(chooser.models) > 1:
        for task_name in chooser.models:
            plots_subdir = os.path.join(plots_dir, task_name)
            if not os.path.isdir(plots_subdir):
                os.mkdir(plots_subdir)

    print 'Plotting...'

    # Plot objective model
    # plot_2d_mean_and_var(chooser.objective_model, plots_dir, 
    #     chooser.objective.name, 
    #     input_space, current_best_location)
    
    # plot_hypers(chooser.objective_model, plots_dir, 'objective_function')
    for task_name, model in chooser.models.iteritems():

        plots_subdir = os.path.join(plots_dir, task_name) if len(chooser.models) > 1 else plots_dir

        plot_hypers(model, plots_subdir, task_name)

        plot_2d_mean_and_var(model, plots_subdir, task_name, input_space, current_best_location)


    if chooser.numConstraints() > 0:
        plot_2d_constraints(chooser, plots_dir, input_space, current_best_location)

    plot_acquisition_function(chooser, plots_dir, input_space, current_best_location, current_best_value)

    print 'Done plotting.'

# usage: python plots_2d.py DIRECTORY
if __name__ == '__main__':
    main(*sys.argv[1:])
  

# -*- coding: utf-8 -*-
# spearmint
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
import imp
import optparse
import numpy             as np
import numpy.random      as npr
import numpy.linalg      as npla
import matplotlib        as mpl
mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


from spearmint.visualizations         import plots_2d
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.parsing          import repeat_experiment_name
from spearmint.utils.parsing          import get_objectives_and_constraints
from spearmint.utils.parsing          import DEFAULTS
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.tasks.input_space      import paramify_no_types
from spearmint.main                   import load_jobs


# Returns the violation value if the constraints are violated, else the objective
def true_func(stored, module, violation_value=np.inf, constraint_tol=0.0, objective=DEFAULTS["task_name"], constraints=[]):
    params = stored['params']

    # if we don't actually know the true objective, just return what we think is the best
    if not hasattr(module, 'true_func'):
        if stored['objective'] is None:
            return np.nan
        else:
            return stored['objective']
        # output = module.main(0, paramify_no_types(params))

    # otherwise, assume we know it
    output = module.true_func(0, paramify_no_types(params))
    
    if not isinstance(output, dict):
        if len(constraints)!=0:
            raise Exception("output is not a dict and yet you said there were constraints..?")
        val = output
    else:
        val = output[objective]

    for c in constraints:
        if output[c] < -constraint_tol:
            print 'Violation value of %f for constraint %s' % (output[c], c)
            # print str(paramify_no_types(params))
            return violation_value

    # if the module defines the true solution value, then use the gap. otherwise don't
    return np.abs(val - module.true_val()) # abs not really needed, it will always be positive

def main(dirs, 
    n_repeat        = -1, 
    n_iter_spec     = None, 
    rec_type        = "model", 
    average         = "mean",
    log_scale       = False,
    violation_value = 1.,
    constraint_tol  = 0.,
    make_dist_plot  = False,
    mainfile        = None,
    stretch_x       = False,
    task_comp_x     = None,
    plot_wall_time  = False,
    bin_size        = 1.0,
    plot_separate   = False,
    labels          = None,
    y_axis_label    = None,
    x_axis_label    = None):

    # Create the figure that plots utility gap
    fig = dict()
    ax  = dict()

    # averaging function
    if average == "mean":
        avg = np.mean
    elif average == "median":
        avg = np.median
    else:
        raise Exception("Unknown average %s" % average)


    fig['err'] = plt.figure()
    ax['err'] = fig['err'].add_subplot(1,1,1)
    if plot_wall_time:
        ax['err'].set_xlabel("wall time (min)", size=25)
    elif x_axis_label:
        ax['err'].set_xlabel(x_axis_label, size=25)
    else:
        ax['err'].set_xlabel('Number of function evaluations', size=25)
    ax['err'].tick_params(axis='both', which='major', labelsize=20)

    # Create the figure that plots L2 distance from solution
    fig['dist'] = plt.figure()
    ax['dist'] = fig['dist'].add_subplot(1,1,1)
    if x_axis_label:
        ax['dist'].set_xlabel(x_axis_label, size=25)
    else:
        ax['dist'].set_xlabel('Number of function evaluations', size=25)
    if y_axis_label:
        ax['dist'].set_ylabel(y_axis_label, size=25)
    elif log_scale:
        ax['dist'].set_ylabel('$\log_{10}\, \ell_2$-distance', size=25)
    else:
        ax['dist'].set_ylabel('$\ell_2$-distance', size=25)
    ax['dist'].tick_params(axis='both', which='major', labelsize=20)

    db_document_name = 'recommendations'

    acq_names = list()
    for expt_dir in dirs:
        options         = parse_config_file(expt_dir, 'config.json')
        experiment_name = options["experiment_name"]
        input_space     = InputSpace(options["variables"])
        chooser_module  = importlib.import_module('spearmint.choosers.' + options['chooser'])
        chooser         = chooser_module.init(input_space, options)
        db              = MongoDB(database_address=options['database']['address'])
        jobs            = load_jobs(db, experiment_name)
        hypers          = db.load(experiment_name, 'hypers')
        tasks           = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)

        if rec_type  == "model":
            if mainfile is None:
                main_file = options['main_file']
            else:
                main_file = mainfile

            sys.path.append(options['main_file_path']) # TODO: make this nicer with proper importing
            module          = importlib.import_module(main_file)
            sys.path.remove(options['main_file_path'])

        obj, con        = get_objectives_and_constraints(options) # get the names
        obj = obj[0] # only one objective
        print 'Found %d constraints' % len(con)

        plot_utility_gap = rec_type=="model" and hasattr(module, 'true_val')

        if plot_utility_gap:
            print 'PLOTTING UTILITY GAP'
            if y_axis_label:
                ax['err'].set_ylabel(y_axis_label, size=25)
            elif log_scale:
                ax['err'].set_ylabel('$\log_{10}$ utility gap', size=25)
            else:
                ax['err'].set_ylabel('utility gap', size=25)
        else:
            if y_axis_label:
                ax['err'].set_ylabel(y_axis_label, size=25)
            elif log_scale:
                ax['err'].set_ylabel('$\log_{10}$ objective value', size=25)
            else:
                ax['err'].set_ylabel('objective value', size=25)

        # Make the directory for the plots
        plots_dir = os.path.join(expt_dir, 'plots')
        if not os.path.isdir(plots_dir):
            os.mkdir(plots_dir)

        # if the module provides the location of the true solution, plot the distance to this solution vs iterations
        if make_dist_plot and not hasattr(module, 'true_sol'):
            raise Exception("make_dist_plot turned on but cannot find true sol in the main_file")

        # If repeat >= 0, then we are averaging a number of experiments
        # We assume the experiments are stored with the original name plus a hyphen plus the number
        n_repeat = int(n_repeat)
        if n_repeat < 0:
            recs = db.load(experiment_name, db_document_name)
            if recs is None:
                raise Exception("Could not find experiment %s in database at %s" % (experiment_name, options['database']['address']))

            # the x axis represents the number of evals of a particular task given by task_comp_x
            # so we only take the data where this number was incrememted, i.e. when this task was evaluated
            if task_comp_x:
                # only include recommendations when you finish a particular task
                new_recs = list()
                last_complete = 0
                for rec in recs:
                    cur_complete = rec['num_complete_tasks'][task_comp_x]
                    if cur_complete > last_complete:
                        last_complete = cur_complete
                        new_recs.append(rec)
                recs = new_recs
                

            n_iter = len(recs) if n_iter_spec is None else n_iter_spec

            iters = range(n_iter)

            if plot_wall_time:
                if task_comp_x:
                    raise Exception("Do not use plot wall_time with task_x")
                iters = [rec['total_elapsed_time']/60.0 for rec in recs]
                iters = iters[:n_iter]
                iters = np.array(iters, dtype=float)

            print 'Found %d iterations' % len(recs)

            if rec_type == "model":
                values = [true_func(rec, module, violation_value, constraint_tol, obj, con) for rec in recs]

                if log_scale:
                    ax['err'].plot(iters, map(np.log10,values))
                else:
                    ax['err'].plot(iters, values)
            else:
                if rec_type == "observations":
                    observations = [x['obj_o'] for x in recs]
                elif rec_type == "mixed":
                    observations = [x['obj_om'] for x in recs]
                else:
                    raise Exception("unknown rec type")

                for i in xrange(len(observations)):
                    if observations[i] is None or np.isnan(observations[i]):
                        observations[i] = violation_value
                # print observations
                # print len(observations)

                if log_scale:
                    ax['err'].plot(iters, np.log10(observations))
                else:
                    ax['err'].plot(iters, observations)

            if make_dist_plot:
                distances = [params_norm(rec['params'], module.true_sol()) for rec in recs]
                if log_scale:
                    ax['dist'].plot(iters, np.log10(distances))
                else:
                    ax['dist'].plot(iters, distances)
        else:
             # MULTIPLE REPEATS
            repeat_recs = [db.load(repeat_experiment_name(experiment_name,j),db_document_name) for j in xrange(n_repeat)]
            if None in repeat_recs:
                for i, repeat_rec in enumerate(repeat_recs):
                    if repeat_rec is None:
                        print 'Could not load experiment %s repeat %d' % (experiment_name, i)
                print 'Exiting...'
                return

            if task_comp_x:
                # only include recommendations when you finish a particular task
                new_repeat_recs = list()
                for recs in repeat_recs:
                    recs = sorted(recs, key=lambda k: k['id']) # sort by id
                    new_recs = list()
                    last_complete = 0
                    for rec in recs:
                        cur_complete = rec['num_complete_tasks'][task_comp_x]
                        if cur_complete == last_complete + 1:
                            last_complete = cur_complete
                            new_recs.append(rec)
                        elif cur_complete == last_complete:
                            pass
                        else: 
                            print('WARNING: cur complete=%d, last_complete=%d' % (cur_complete,  last_complete))
                            break
                    new_repeat_recs.append(new_recs)

                repeat_recs = new_repeat_recs
            
            n_iter_each = map(len, repeat_recs)
            if plot_wall_time:
                """ do everything separately from here if plotting wall time
                here is what we do... we can't have a square array because 
                we don't want to take the minimum number of iterations... 
                we want to take ALL iterations for each repeat, and this number
                may be different for different repeats.
                so we store all times/values in a list of arrays
                then we chop things up into bins
                """
                if rec_type != "model":
                    values      = list()
                    wall_times  = list()
                    for j in xrange(n_repeat):  # loop over repeated experiments

                        wall_times.append( np.array([ repeat_recs[j][i]['total_elapsed_time']/60.0 for i in xrange(n_iter_each[j]) ]))

                        if rec_type == "observations":
                            values.append([ repeat_recs[j][i]['obj_o'] for i in xrange(n_iter_each[j])])
                        elif rec_type == "mixed":
                            values.append([ repeat_recs[j][i]['obj_om'] for i in xrange(n_iter_each[j])])
                        else:
                            raise Exception("unknown rec type")
                            
                        for i in xrange(n_iter_each[j]):
                            if values[-1][i] is None or np.isnan(values[-1][i]):
                                values[-1][i] = violation_value

                        values[-1] = np.array(values[-1])

                    # print values

                else: # if plot wall tiem but using model
                    values     = list()
                    wall_times = list()
                    for j in xrange(n_repeat):  # loop over repeated experiments

                        # for this repeat, get all wall times 
                        wall_times.append( np.array([ repeat_recs[j][i]['total_elapsed_time']/60.0 for i in xrange(n_iter_each[j]) ]))
                        
                        values_j = np.zeros(n_iter_each[j])
                        for i in xrange(n_iter_each[j]):      # loop over iterations
                            val = true_func(repeat_recs[j][i], module, None, constraint_tol, obj, con)
                            if val is None or np.isnan(val): #set to violation value here so we can print out this info...
                                values_j[i] = violation_value
                                print 'Violation with params %s at repeat %d iter %d' % (paramify_no_types(repeat_recs[j][i]['params']), j, i)
                            else:
                                values_j[i] = val
                        values.append(values_j)

                # change the data structure to be time bins and include everything in
                # those time bins across repeats
                end_times = map(max,wall_times)
                for j in xrange(n_repeat):
                    print 'end time for repeat %d: %f' % (j, end_times[j])
                iters = np.arange(0.0,np.round(max(end_times)), bin_size)
                new_values = list()
                for i,timestep in enumerate(iters):
                    # print 'Creating wall time bin from %f to %f. (%d/%d)' % (i, i+bin_size, i, len(iters))
                    new_value = list()
                    for j in xrange(n_repeat):
                        new_value = np.append(new_value, values[j][np.logical_and(wall_times[j] >= timestep, wall_times[j] < timestep+bin_size)].flatten())
                    # if a time bin is empty across all repeats:
                    if len(new_value) == 0:
                        if i == 0:
                            new_value = [violation_value]
                        else:
                            new_value = new_values[-1]
                    new_values.append(new_value)
                values = new_values

                # make the first value equal to the violation value (optional)        
                iters = np.append(iters, max(iters)+bin_size)
                values.insert(0, np.array([violation_value]))

                # Average over the repeated experiments
                average_values = map(avg, values)
                errorbars = bootstrap_errorbars(values, log=log_scale, avg=avg)
                # plt.yscale('log', nonposy='clip')

                if log_scale:
                    ax['err'].errorbar(iters, np.log10(average_values), yerr=errorbars)
                else:
                    ax['err'].errorbar(iters, average_values, yerr=errorbars)

            else:
                # NOT WALL TIME

                n_iter      = reduce(min, n_iter_each, np.inf)
                if n_iter_spec is None:
                    print 'Found %d repeats with at least %d iterations' % (n_repeat, n_iter)
                    print {i:n_iter_each[i] for i in xrange(n_repeat)}
                elif n_iter < n_iter_spec:
                    print 'You specified %d iterations but there are only %d available... so plotting %d' % (n_iter_spec, n_iter, n_iter)
                else:
                    n_iter = n_iter_spec
                    print 'Plotting %d iterations' % n_iter

                iters = range(n_iter)

                if rec_type != "model":
                    values      = np.zeros((n_iter, n_repeat))
                    for j in xrange(n_repeat):  # loop over repeated experiments
                        for i in iters[j]:      # loop over iterations
                            if rec_type == "observations":
                                values[i,j] = repeat_recs[j][i]['obj_o']
                            elif rec_type == "mixed":
                                values[i,j] = repeat_recs[j][i]['obj_om']
                            else:
                                raise Exception("unknown rec type")
                            if values[i,j] is None or np.isnan(values[i,j]):
                                values[i,j] = violation_value

                    print values

                else:
                    values      = np.zeros((n_iter, n_repeat))
                    distances   = np.zeros((n_iter, n_repeat))
                    for j in xrange(n_repeat):  # loop over repeated experiments
                        for i in iters:      # loop over iterations 
                            val = true_func(repeat_recs[j][i], module, None, constraint_tol, obj, con)
                            if val is None: #set to violation value here so we can print out this info...
                                values[i,j] = violation_value
                                print 'Violation with params %s at repeat %d iter %d' % (paramify_no_types(repeat_recs[j][i]['params']), j, i)
                            else:
                                values[i,j] = val

                            if make_dist_plot:
                                distances[i,j] = params_norm(repeat_recs[j][i]['params'], module.true_sol()) 


                if plot_separate:
                    if log_scale:
                        ax['err'].plot(iters, np.log10(values))
                    else:
                        ax['err'].plot(iters, values)

                else:
                    # Average over the repeated experiments
                    average_values = map(avg, values)
                    errorbars = bootstrap_errorbars(values, log=log_scale, avg=avg)
                    # plt.yscale('log', nonposy='clip')

                    if stretch_x:
                        fctr = float(n_iter_spec) / float(n_iter)
                        iters = np.array(iters) * fctr
                        print 'Stretching x axis by a factor of %f' % fctr                

                    if log_scale:
                        ax['err'].errorbar(iters, np.log10(average_values), yerr=errorbars)
                    else:
                        ax['err'].errorbar(iters, average_values, yerr=errorbars)

                    if make_dist_plot:
                        average_dist = map(avg, distances)
                        errorbars_dist = bootstrap_errorbars(distances, log=log_scale, avg=avg)
                        if log_scale:
                            ax['dist'].errorbar(iters, np.log10(average_dist), yerr=errorbars_dist)
                        else:
                            ax['dist'].errorbar(iters, average_dist, yerr=errorbars_dist)

        acq_names.append(options["tasks"].values()[0]["acquisition"])
        if acq_names[-1] == 'PES':
            acq_names[-1] = 'PESC'
        if acq_names[-1] == 'ExpectedImprovement':
            acq_names[-1] = 'EIC'

    if labels:
        ax['err'].legend(labels.split(';'), fontsize=16, loc='lower left')
        ax['dist'].legend(labels.split(';'), fontsize=20)
    elif len(acq_names) > 1:
        ax['err'].legend(acq_names, fontsize=20)
        ax['dist'].legend(acq_names, fontsize=20)

    # save it in the last directory... (if there are multiple directories)
    if not plot_wall_time:
        if n_repeat >= 0:
            print 'Made a plot with %d repeats and %d iterations' % (n_repeat, n_iter)
        else:
            print 'Made a plot with %d iterations' % (n_iter)
    else:
        if n_repeat >= 0:
            print 'Made a plot with %d repeats and %f minutes' % (n_repeat, max(iters))
        else:
            print 'Made a plot with %f minutes' % (max(iters))
    
    file_prefix = '%s_' % average if n_repeat > 0 else ''
    file_postfix = '_wall_time' if plot_wall_time else ''
    fig['err'].tight_layout()
    figname = os.path.join(plots_dir, '%serror%s' % (file_prefix, file_postfix))
    fig['err'].savefig(figname + '.pdf')
    fig['err'].savefig(figname + '.svg')
    print 'Saved to %s' % figname
    if make_dist_plot:
        fig['dist'].tight_layout()
        figname_dist = os.path.join(plots_dir, '%sl2_distance%s.pdf' % (file_prefix, file_postfix))
        fig['dist'].savefig(figname_dist)
        print 'Saved to %s' % figname_dist

# computes the l2 norm between x and y assuming x and y are stores as dicts of params
# and also the first one (x) has this extra 'values' nonsense
def params_norm(x, y):
    if not isinstance(y, list):
        y = [y]
    # if x.keys() != y.keys():
        # raise Exception("x and y must have the same keys")
    
    norm = 0.0
    for key in x:
        min_dist = reduce(min, [np.sum((x[key]['values']-sol[key])**2) for sol in y], np.inf)
        norm += min_dist
    
    return np.sqrt(norm)

# compute error bars with the bootstrap
# if log=True, we compute the standard deviation of the log values
def bootstrap_errorbars(X, M=1000, log=False, avg=np.mean):
    # boots = np.zeros((X.shape[0], M))
    # for k in xrange(M):
    #     # sample n_repeat curves with replacement from the originals
    #     resampled_X = X[:,npr.randint(0, X.shape[1],size=X.shape[1])]
    #     boots[:,k] = map(avg, resampled_X)
    # if log:
    #     boots = np.log10(boots)


    boots = list()
    for i,x in enumerate(X):
        boots_i = np.zeros(M)
        if np.array(x).size>1: # if there's only one or 0 values, errorbar=0
            for m in xrange(M):
                resampled_x = x[npr.randint(0, x.size,size=x.size)]
                boots_i[m] = avg(resampled_x)
            if log:
                boots_i = np.log10(boots_i)
        boots.append(boots_i)
    boots = np.array(boots)
    errorbars = map(np.std, boots)
    return errorbars

# Usage:
# python progress_curve.py dir1 [dir2] ... [dirN] [repeat]
if __name__ == '__main__':
    
    option_names = ['n_repeat', 'average', 'rec_type', 'log_scale', \
    'n_iter_spec',  'violation_value', 'constraint_tol', \
    'make_dist_plot', 'mainfile', 'plot_wall_time', 'plot_separate', 'bin_size',   \
    'stretch_x', 'task_comp_x', 'labels', "y_axis_label", 'x_axis_label']

    parser = optparse.OptionParser(usage="usage: %prog [options] dir1 dir2")

    parser.add_option("--repeat", dest="n_repeat",
                      help="Number of repeated experiments.",
                      type="int", default=-1)
    parser.add_option("--rec-type", dest="rec_type",
                      help="model, observations, or mixed?",
                      default="model")
    parser.add_option("--median", action="store_true",
                      help="Use the median instead of the mean.",
                      dest="average")
    parser.add_option("--logscale", action="store_true", dest="log_scale",
                      help="Whether to plot the y axis on a log scale.")
    parser.add_option("--iter", dest="n_iter_spec",
                      help="Uesd to specify a certain number of iterations to plot.",
                      type="int", default=None)
    parser.add_option("--violation-value", dest="violation_value",
                      help="The penalty value for violating the constraints.",
                      type="float", default=1.0)
    parser.add_option("--constraint-tol", dest="constraint_tol",
                      help="You can violate the constraint by this amount and not be penalized the violation value.",
                      type="float", default=0.0)
    parser.add_option("--make_dist_plot", action="store_true",
                      help="Whether to also make a plot of the L2 distance from the true solution.")
    parser.add_option("--mainfile", dest="mainfile",
                      help="Explicity store the location of the main file.",
                      type="string", default=None)
    parser.add_option("--stretch", action="store_true", dest="stretch_x",
                      help="Only use this if you really know what you are doing.")
    parser.add_option("--task_x", dest="task_comp_x",
                      help="A particular task whose num complete will make up the x-axis of the plot.",
                      type="string", default=None)
    parser.add_option("--wall-time", action="store_true",
                      help="Plot wall time on the x axis.",
                      dest="plot_wall_time")
    parser.add_option("--bin-size", dest="bin_size",
                      help="Only used when plot_wall_time is on and n_repeat > 1.",
                      type="float", default=1.0)
    parser.add_option("--separate", action="store_true",
                      help="Plot repeated experiments separately instead of averaged with error bars.",
                      dest="plot_separate")
    parser.add_option("--labels", dest="labels",
                      help="For non-default legend labels on the curves. If >1, separate with SEMICOLON.",
                      type="string", default=None)
    parser.add_option("--y-axis-label", dest="y_axis_label",
                      help="For non-default y-axis label.",
                      type="string", default=None)
    parser.add_option("--x-axis-label", dest="x_axis_label",
                      help="For non-default x-axis label.",
                      type="string", default=None)
    """ when you add a new options, make sure to add it to the list above"""

    # Stretches the x-axis of one plot to match the other- use to compare coupled and
    # decoupled algs

    (args, dirs) = parser.parse_args()

    # parse this weird args thing into a dict
    options = dict()
    for option_name in option_names:
        if hasattr(args, option_name):
            options[option_name] = getattr(args, option_name)
        else:
            options[option_name] = False

    if options["average"]:
        options["average"] = "median"      
    else:
        options["average"] = "mean"

    main(dirs, **options)


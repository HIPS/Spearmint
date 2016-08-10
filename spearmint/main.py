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
import optparse
import tempfile
import datetime
import subprocess
import importlib
import time
import imp
import os
import re
import pymongo
import hashlib
import logging

import numpy as np

# np.random.seed(1)

try: import simplejson as json
except ImportError: import json

from collections import defaultdict

from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_resources_from_config
from spearmint.utils.parsing          import repeat_experiment_name
from spearmint.utils.parsing          import repeat_output_dir
from spearmint.resources.resource     import print_resources_status
from spearmint.tasks.task             import print_tasks_status

logLevel = logging.INFO
logFormatter = logging.Formatter("%(message)s")
logging.basicConfig(level=logLevel,
                    format="%(message)s")
                    # format="%(asctime)s %(filename)s:%(lineno)s %(message)s", 
                    # datefmt="%H:%M:%S")
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")


def main(expt_dir, config_file="config.json", no_output=False, repeat=-1):
    if not os.path.isdir(expt_dir):
        raise Exception("Cannot find directory %s" % expt_dir)

    options = parse_config_file(expt_dir, config_file)
    experiment_name = options["experiment_name"]

    # Special advanced feature for repeating the same experiment many times
    if repeat >= 0:
        experiment_name = repeat_experiment_name(experiment_name, repeat)

    if not no_output: # if we want output
        if repeat >= 0:
            output_directory = repeat_output_dir(expt_dir, repeat)
        else:
            output_directory = os.path.join(expt_dir, 'output')
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        if repeat < 0:
            rootLogger = logging.getLogger()
            fileHandler = logging.FileHandler(os.path.join(output_directory, 'main.log'))
            fileHandler.setFormatter(logFormatter)
            fileHandler.setLevel(logLevel)
            rootLogger.addHandler(fileHandler)
        # consoleHandler = logging.StreamHandler()
        # consoleHandler.setFormatter(logFormatter)
        # consoleHandler.setLevel(logLevel)
        # rootLogger.addHandler(consoleHandler)
    else:
        output_directory = None

    input_space = InputSpace(options["variables"])

    resources = parse_resources_from_config(options)

    # Load up the chooser.
    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    chooser = chooser_module.init(input_space, options)

    # Connect to the database
    db_address = options['database']['address']
    db         = MongoDB(database_address=db_address)

    overall_start_time = time.time()
    db.save({'start-time':overall_start_time}, experiment_name, 'start-time')


    waiting_for_results = False  # for printing purposes only
    while True:

        for resource_name, resource in resources.iteritems():

            jobs = load_jobs(db, experiment_name)
            # resource.printStatus(jobs)

            # If the resource is currently accepting more jobs
            # TODO: here cost will eventually also be considered: even if the 
            #       resource is not full, we might wait because of cost incurred
            # Note: I could chose to fill up one resource and them move on to the next ("if")
            # You could also do it the other way, by changing "if" to "while" here

            # Remove any broken jobs from pending 
            # note: make sure to do this before the acceptingJobs() condition is checked
            remove_broken_jobs(db, jobs, experiment_name, resources)

            if resource.acceptingJobs(jobs):

                if waiting_for_results:
                    logging.info('\n')
                waiting_for_results = False

                # Load jobs from DB 
                # (move out of one or both loops?) would need to pass into load_tasks
                jobs = load_jobs(db, experiment_name)

                # Print out a list of broken jobs
                print_broken_jobs(jobs)

                # Get a suggestion for the next job
                tasks = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)

                # Special case when coupled and there is a NaN task-- what to do with NaN task when decoupled??
                if 'NaN' in tasks and 'NaN' not in resource.tasks:
                    resource.tasks.append('NaN')

                # Load the model hypers from the database.
                hypers = db.load(experiment_name, 'hypers')

                # "Fit" the chooser - give the chooser data and let it fit the model(s).
                # NOTE: even if we are only suggesting for 1 task, we need to fit all of them
                # because the acquisition function for one task depends on all the tasks
                hypers = chooser.fit(tasks, hypers)
                
                if hypers:
                    logging.debug('GP covariance hyperparameters:')
                print_hypers(hypers, input_space, options)
                # if 'duration hypers' in hypers:
                    # logging.debug('Duration GP covariance hyperparameters:')
                    # print_hypers(hypers['duration hypers'], input_space, options)

                # Save the hyperparameters to the database.
                if hypers:
                    db.save(hypers, experiment_name, 'hypers')

                if options['recommendations'] == "during":
                    # Compute the best value so far, a.k.a. the "recommendation"
                    recommendation = chooser.best()

                    # Save the recommendation in the DB if there are more complete jobs than last time
                    store_recommendation(recommendation, db, experiment_name, tasks, jobs, input_space, time.time()-overall_start_time)

                # Get the decoupling groups
                task_couplings = {task_name : tasks[task_name].options["group"] for task_name in resource.tasks}

                logging.info('\nGetting suggestion for %s...\n' % (', '.join(task_couplings.keys())))

                # Get the next suggested experiment from the chooser.
                suggested_input, suggested_tasks = chooser.suggest(task_couplings)
                suggested_task = suggested_tasks[0] # hack, deal with later

                suggested_job = {
                    'id'          : len(jobs) + 1,
                    'params'      : input_space.paramify(suggested_input),
                    'expt_dir'    : options['main_file_path'],
                    'tasks'       : suggested_tasks,
                    'resource'    : resource_name,
                    'main-file'   : options['tasks'][suggested_task]['main_file'],
                    'language'    : options['tasks'][suggested_task]['language'],
                    'status'      : 'new',
                    'submit time' : time.time(),
                    'start time'  : None,
                    'end time'    : None,
                    'fast update' : chooser.fast_update # just for plotting - not important
                }

                save_job(suggested_job, db, experiment_name)

                # Submit the job to the appropriate resource
                process_id = resource.attemptDispatch(experiment_name, suggested_job, db_address, 
                                                      expt_dir, output_directory)

                # Print the current time
                logging.info('Current time: %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                # Set the status of the job appropriately (successfully submitted or not)
                if process_id is None:
                    suggested_job['status'] = 'broken'
                    logging.info('Job %s failed -- check output file for details.' % job['id'])
                    save_job(suggested_job, db, experiment_name)
                else:
                    suggested_job['status'] = 'pending'
                    suggested_job['proc_id'] = process_id
                    save_job(suggested_job, db, experiment_name)

                jobs = load_jobs(db, experiment_name)

                # Print out the status of the resources
                # resource.printStatus(jobs)
                print_resources_status(resources.values(), jobs)

                if len(set(task_couplings.values())) > 1: # if decoupled
                    print_tasks_status(tasks.values(), jobs)

                # For debug - print pending jobs
                print_pending_jobs(jobs)

        # Terminate the optimization if all resources are finished (run max number of jobs)
        # or ANY task is finished (just my weird convention)
        jobs = load_jobs(db, experiment_name)
        tasks = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)
        terminate_resources = reduce(lambda x,y: x and y, map(lambda x: x.maxCompleteReached(jobs), resources.values()), True)
        terminate_tasks     = reduce(lambda x,y: x or y,  map(lambda x: x.maxCompleteReached(jobs), tasks.values()),     False)
        terminate_maxtime   = (time.time()-overall_start_time) >= (options['max_time_mins']*60.0)
        
        if terminate_resources or terminate_tasks or terminate_maxtime:

            if terminate_resources:
                logging.info('Maximum number of jobs completed on all resources.')
            if terminate_tasks:
                logging.info('Maximum number of jobs reached for at least one task.')
            if terminate_maxtime:
                logging.info('Maximum total experiment time of %f minutes reached.' % options['max_time_mins'])

            # save rec in DB
            if options['recommendations'] in ("during", "end-one"):
                logging.info('Making final recommendation:')
                recommendation = chooser.best()
                store_recommendation(recommendation, db, experiment_name, tasks, jobs, 
                    input_space, time.time()-overall_start_time, final=True)
            elif options['recommendations'] == "end-all":
                logging.info('Making recommendations...')
                all_jobs = jobs
                for i in xrange(len(all_jobs)):
                    logging.info('')
                    logging.info('-------------------------------------------------')
                    logging.info('     Getting recommendations for iter %d/%d      ' % (i, len(all_jobs)) )
                    logging.info('-------------------------------------------------')
                    logging.info('')

                    jobs = all_jobs[:i+1]
                    tasks = parse_tasks_from_jobs(jobs, experiment_name, options, input_space)
                    hypers = chooser.fit(tasks, hypers)
                    print_hypers(hypers, input_space, options)
                    # get the biggest end time of the jobs
                    end_time = max([job['end time'] for job in jobs])
                    elapsed_time = end_time - overall_start_time

                    recommendation = chooser.best()
                    store_recommendation(recommendation, db, experiment_name, tasks, jobs, input_space, elapsed_time)

            logging.info('Have a nice day.')
            return

        # If no resources are accepting jobs, sleep
        if no_free_resources(db, experiment_name, resources):
            # Don't use logging here because it's too much effort to use logging without a newline at the end
            sys.stdout.write('Waiting for results...' if not waiting_for_results else '.')
            sys.stdout.flush()
            # sys.stderr.flush()
            waiting_for_results = True
            time.sleep(options['polling_time'])
        else:
            sys.stdout.write('\n')

# if "final" is true it means we are at the end of optimizing, and we just want to make ONE
# recommendation regardless of the rest of this stuff
def store_recommendation(recommendation, db, experiment_name, tasks, jobs, input_space, elapsed_time, final=False):
    stored_recs = db.load(experiment_name, 'recommendations')
    if stored_recs is None:
        previous_total_num_complete = 0
    elif isinstance(stored_recs, dict):
        previous_total_num_complete = 1
    else:
        previous_total_num_complete = max(map(lambda x: x['num_complete'], stored_recs))
    
    total_num_complete = sum(map(lambda x: x['status'] == 'complete', jobs)) # over all resources
    
    while total_num_complete > previous_total_num_complete or final:
        logging.debug('Storing recommendation %d.' % previous_total_num_complete)
        
        db.save({'num_complete' : total_num_complete, 
             'num_complete_tasks' : {task_name : task.numComplete(jobs) for task_name, task in tasks.iteritems()},
             'params'   : input_space.paramify(recommendation['model_model_input']), 
             'objective': recommendation['model_model_value'],
             'params_o' : None if recommendation['obser_obser_input'] is None else input_space.paramify(recommendation['obser_obser_input']),
             'obj_o'    : recommendation['obser_obser_value'],
             'params_om': None if recommendation['obser_model_input'] is None else input_space.paramify(recommendation['obser_model_input']),
             'obj_om'   : recommendation['obser_model_value'],
             'total_elapsed_time' : elapsed_time}, 
        experiment_name, 'recommendations', {'id' : previous_total_num_complete})
        # We want to save 3 types of recommendations:
        # 1) The best according to the model (model_model)
        # 2) The best objective/constraint observation (obser_obser)
        # 3) The best objective observation at a place where the model thinks the constraint is satisfied (obser_model)
        
        previous_total_num_complete += 1
        # this is a bit complicated. but, basically, we only want to save a recommendation if there are actually new reuslts
        # for example, if we have failed jobs then we don't wan't to keep saving recommendations even though
        # we do want to keep making suggestions. 
        # this while loop has 2 functions. first it serves like an "if" so if there is no new stuff,
        # we do nothing. second, if we have multiple resources/capacity such that we completed more than 1
        # new job since last time, we repeat the same recommendation multiple times. this part is a bit
        # off-- it would be better to repeat the earlier recomendation instead of the later one... but
        # shouldn't be a big deal. 
        # this should also work out so that each recommendation has a unique id and the IDs 
        # have no gaps in between them. good.

        if final: # just do it once
            return

    # return previous_total_num_complete # there is no actual need to return this, it doesn't matter

# Is it the case that no resources are accepting jobs?
def no_free_resources(db, experiment_name, resources):
    jobs = load_jobs(db, experiment_name)
    for resource_name, resource in resources.iteritems():
        if resource.acceptingJobs(jobs):
            return False
    return True

# Look thorugh jobs and for those that are pending but not alive, set
# their status to 'broken'
# there are 3 places a job can be set to broken 
#   1. in launcher.py if the thing throws an error
#   2. above in the main loop, if it fails immediately upon spawning
#   3. here, if it still thinks it is pending but it died
#
def remove_broken_jobs(db, jobs, experiment_name, resources):
    for job in jobs:
        if job['status'] == 'pending':
            if not resources[job['resource']].isJobAlive(job):
                job['status'] = 'broken'
                save_job(job, db, experiment_name)

def print_broken_jobs(jobs):
    broken_jobs = defaultdict(list) 
    for job in jobs:
        if job['status'] == 'broken':
            broken_jobs[', '.join(job['tasks'])].append(str(job['id']))

    for task_names_broken, broken_id_list in broken_jobs.iteritems():
        logging.info('** Failed jobs(s) for %s: %s\n' % (task_names_broken, ', '.join(broken_id_list)))

def print_pending_jobs(jobs):
    pending_jobs = defaultdict(list)
    for job in jobs:
        if job['status'] == 'pending':
            pending_jobs[', '.join(job['tasks'])].append(str(job['id']))

    for task_names_pending, pending_id_list in pending_jobs.iteritems():
        logging.info('ID(s) of pending job(s) for %s: %s' % (task_names_pending, ', '.join(pending_id_list)))


def print_hypers(hypers, input_space, options):
    for task_name, stored_dict in hypers.iteritems():
        if task_name == 'duration hypers':
            continue
        logging.debug(task_name)
        if 'latent values' in stored_dict:
            logging.debug('   Latent values: %s' % ', '.join(map(lambda x: '%.04f'%x, stored_dict['latent values'].values())))
        for hyper_name, hyper_value in stored_dict['hypers'].iteritems():
            # special printing for length scales, to show the dimension names
            if hyper_name == 'ls':
                logging.debug('   length scales:')

                length_scales = input_space.paramify(hyper_value)
                for param_name, length_scale in length_scales.iteritems():
                    logging.debug("      %-15.15s : %s" % (param_name, ', '.join(map(str, length_scale['values']))))
                # logging.debug('')
            else:
                # regular hyperparam
                logging.debug('   %-18.18s : %s' % (hyper_name, hyper_value))
    logging.debug('')

def load_jobs(db, experiment_name):
    jobs = db.load(experiment_name, 'jobs')

    if jobs is None:
        jobs = []
    if isinstance(jobs, dict):
        jobs = [jobs]

    return jobs

def save_job(job, db, experiment_name):
    db.save(job, experiment_name, 'jobs', {'id' : job['id']})

if __name__ == '__main__':
    parser = optparse.OptionParser(usage="usage: %prog [options] directory")

    parser.add_option("--config", dest="config_file",
                      help="Configuration file name.",
                      type="string", default="config.json")
    parser.add_option("--no-output", action="store_true",
                      help="Do not create output files.")
    parser.add_option("--repeat", dest="repeat",
                      help="Used for repeating the same experiment many times.",
                      type="int", default="-1")

    (kwargs, args) = parser.parse_args()

    expt_dir  = os.path.realpath(args[0])

    main(expt_dir, kwargs.config_file, kwargs.no_output, kwargs.repeat)

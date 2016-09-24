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


import re
import numpy        as np
import numpy.random as npr
import math
import json
import os
import copy
import time
from collections import OrderedDict
from collections import defaultdict

import logging
import importlib

from spearmint.tasks.task         import Task
from spearmint.resources.resource import Resource
from spearmint.utils.priors       import *

# For converting a string of args into a dict of args
# (one could then call parse_args on the output)
def unpack_args(str):
    if len(str) > 1:
        eq_re = re.compile("\s*=\s*")
        return dict(map(lambda x: eq_re.split(x),
                        re.compile("\s*,\s*").split(str)))
    else:
        return {}
            
# For parsing the input arguments to a Chooser. 
# "argTypes" is a dict with keys of argument names and
# values of tuples with the (argType, argDefaultValue)
# args is the dict of arguments passd in by the used
def parse_args(argTypes, args):
    opt = dict() # "options"
    for arg in argTypes:
        if arg in args:
            try:
                opt[arg] = argTypes[arg][0](args[arg])
            except:
                # If the argument cannot be parsed into the data type specified by argTypes (e.g., float)
                logging.error("Cannot parse user-specified value %s of argument %s" % (args[arg], arg))
        else:
            opt[arg] = argTypes[arg][1]
 
    return opt


DEFAULTS = dict()
DEFAULTS['resource name']        = 'Main'
DEFAULTS['language']             = 'PYTHON'
DEFAULTS['model']                = "GP"
DEFAULTS['likelihood']           = "gaussian"
DEFAULTS['database address']     = 'localhost'
DEFAULTS['scheduler']            = 'local'
DEFAULTS['max finished jobs']    = np.inf
DEFAULTS['max concurrent']       = 1
DEFAULTS['polling time']         = 2.0
DEFAULTS['database name']        = 'spearmint'
DEFAULTS['experiment name']      = 'unnamed-experiment'
DEFAULTS['chooser']              = 'default_chooser'
DEFAULTS['acquisition']          = 'EI'
DEFAULTS['transformations']      = [{'BetaWarp' : {}}]
DEFAULTS['task name']            = 'Objective'
DEFAULTS['constraint name']      = 'Constraint'
DEFAULTS['cost']                 = 1.0    # only for competitive decoupling, scale by this factor?
DEFAULTS['always sample']        = True   # resample GPs each iteration even if there is no new data
DEFAULTS['group']                = 0      # to specify competitive decoupling 
DEFAULTS['scale duration']       = False  # only for competitive decoupling, do acq per unit time?
DEFAULTS['max time mins']        = np.inf # terminate spearmint after this long
DEFAULTS['fast updates']         = False  # only for PESC, do fast updates sometimes?
DEFAULTS['predict fast updates'] = True   # only for fast updates -- very advanced
DEFAULTS['thoughtfulness']       = 1.0     # only for fast updates -- very advanced
DEFAULTS['normalize outputs']    = True
DEFAULTS['recommendations']      = "during"   # this options has 3 possible values: 
   # "during" means makes recommendations at every iteration, during optimization
   # "end-all" means at the end, make recommendations for ALL iterations
   # "end-one" means at the end, just make a recommendation given all the data
# defaults for the chooser
DEFAULTS['acq_grid_size']        = 10000  # size of grid used to initialize optimization of acquisition function
DEFAULTS['fast_acq_grid_size']   = 1000   # for the fast updates only, if you want to use a different grid size
DEFAULTS['rec_grid_size']        = 10000  # size of grid used to initialize optimization of recommendation
DEFAULTS['pes_x*_grid_size']     = 1000   # size of grid used to initialize optimizaiton of x* in PES/PESC
DEFAULTS['pes_num_rand_features']= 1000   # the number of random features used to approximate the GP in PES/PESC
DEFAULTS['constraint delta']     = 1e-2   # for now we have one for each constraint
DEFAULTS['use multi delta']      = True   # if true, p.c. is p(c_k>0)>1-delta for all k, else p(all c_k>0)>1-delta
DEFAULTS['pes_num_x*_samples']   = 1      # if you want multiple x* samples per GP hyper sample (use with e.g. maximum likelihood)
# DEFAULTS['tolerance']            = None   # optimization tolerance as a fraction of original units
# DEFAULTS['unit tolerance']       = 1e-4   # optimization tolerance as a fraction of the space
DEFAULTS['opt_acq_tol']          = 1e-4   # (fractional) precision to optimize acquisition function
DEFAULTS['fast_opt_acq_tol']     = 1e-3   # for the fast updates only, if you want to use a different tolerance
DEFAULTS['opt_rec_tol']          = 1e-4   # (fractional) precision to optimize recommendations
DEFAULTS['pes_opt_x*_tol']       = 1e-6   # (fractional) precision to optimize x* samples in PESC
DEFAULTS['nlopt constraint tol'] = 1e-6   # let NLOPT violate the PROBABILISTIC constraints a little bit
DEFAULTS['check_grad']           = False
DEFAULTS['parallel_opt']         = False
DEFAULTS['num_spray']            = 10
DEFAULTS['spray_std']            = 1e-4 # set to unit tolerance?
DEFAULTS['optimize_best']        = True
DEFAULTS['optimize_acq']         = True
DEFAULTS['initial design size']  = 1
DEFAULTS['nlopt method has grad']= 'LD_MMA'
DEFAULTS['nlopt method no grad'] = 'LN_BOBYQA'

# defaults for the GP options
GP_OPTION_DEFAULTS = {
    'verbose'           : False,
    'mcmc_diagnostics'  : False,
    'mcmc_iters'        : 10,
    'burnin'            : 20,
    'thinning'          : 0,
    'num_fantasies'     : 1,
    'caching'           : True,
    'max_cache_mb'      : 256,
    'likelihood'        : 'gaussian',
    'kernel'            : 'Matern52',
    'stability_jitter'  : 1e-6,
    'fit_mean'          : True, # can set mcmc_iters to 0 to not sample anything
    'fit_amp2'          : True,
    'fit_ls'            : True,
    'fit_noise'         : True,
    'sampler'           : "SliceSampler",
    # 'transformations'   : [],
    'priors'            : [],
    'initial_ls'        : 0.1,
    'initial_mean'      : 0.0, # initial values of the hypers
    'initial_amp2'      : 1.0,
    'initial_noise'     : 0.0001
}
DEFAULTS.update(GP_OPTION_DEFAULTS)
GP_CLASSIFIER_OPTION_DEFAULTS = {
    'ess_thinning'      : 10,
    'prior_whitening'   : True,
    'binomial_trials'   : 1,
    # 'transformations'   : [],
    # 'priors'            : {'amp2' : {'distribution':'Exponential', 'parameters':[1.0]}},
    # 'verbose'           : False,
    'sigmoid'           : 'probit',
    'epsilon'           : 0.5,
    # 'likelihood'        : 'binomial'
}
DEFAULTS.update(GP_CLASSIFIER_OPTION_DEFAULTS)

def change_to_underscore(d):
    # change all hypens spaces to underscores
    d_copy = d.copy()
    for opt, val in d.iteritems():
        opt_new = opt.replace(" ", "_")
        opt_new = opt_new.replace("-", "_")
        del d_copy[opt]
        d_copy[opt_new] = val
    return d_copy
DEFAULTS = change_to_underscore(DEFAULTS)

def parse_true_false_strings(d):
    for opt in d:
        if isinstance(d[opt], basestring):
            if d[opt].lower() == "false":
                d[opt] = False
            elif d[opt].lower() == "true":
                d[opt] = True
    return d # not actually needed, whatever


"""
Parse the config and set defaults
update_options are for when you have a base config file.
"""
def parse_config_file(config_file_dir, config_file_name, verbose=False, update_options=None):
    
    try:
        with open(os.path.join(config_file_dir, config_file_name), 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict) # important for the order of the transformations!
    except ValueError:
        raise Exception("%s did not load properly. Perhaps a spurious comma?" % config_file_name)

    options = change_to_underscore(options)
    options = parse_true_false_strings(options)

    # allow config files to inherit from each other in a very primitive way
    if "base_config_file" in options:
        base_config_path = options["base_config_file"].replace("$LOCATION_OF_THIS_FILE$", config_file_dir)
        del options['base_config_file']
        logging.debug("Parsing base config file at %s" % base_config_path)
        return parse_config_file(*os.path.split(base_config_path), verbose=verbose, update_options=options)

    options["config"] = config_file_name

    # Look for the main file in expt_dir, unless you explicitly specify otherwise in "main_file_path"
    if "main_file_path" not in options:
        options["main_file_path"] = config_file_dir
    else:
        options["main_file_path"] = options["main_file_path"].replace("$LOCATION_OF_THIS_FILE$", config_file_dir)

    return parse_config(options, verbose, update_options)

def parse_config(options, verbose=False, update_options=None):

    # check for unrecognized options
    warnings_wait = False
    for option in options:
        if option not in DEFAULTS and option not in ["base_config_file", "config", "main_file", "main_file_path", "variables", "tasks", "resources"]:
            logging.info('WARNING: unrecognized option "%s"' % option)
            warnings_wait = True
    if warnings_wait:
        time.sleep(3)

    # print out nondefault options
    # for opt_name, opt_val in options.iteritems():
    #     logging.debug("Got option %s = %s" % (opt_name, opt_val))

    # set things to defaults
    options_with_defaults = DEFAULTS.copy()
    options_with_defaults.update(options)
    options = options_with_defaults

    # Here is where you stick in the options from the specific config file,
    # if it inherited from a base config file.
    if update_options is not None:
        options.update(update_options)

    

    # If tasks field not specified at all, set default
    if 'tasks' not in options:
        options['tasks'] = dict()
        options['tasks'][DEFAULTS['task_name']] = {
            'type'       : 'OBJECTIVE', 
            'likelihood' : options.get('likelihood', DEFAULTS['likelihood']) }

        if 'num_constraints' in options:
            for i in xrange(int(options['num_constraints'])):
                options['tasks']['%s%d' % (DEFAULTS['constraint_name'], i)] = {
                    'type'       : 'CONSTRAINT', 
                    'likelihood' : options.get('likelihood', DEFAULTS['likelihood']) }


    if options["acquisition"] == "EI":
        options["acquisition"] = 'ExpectedImprovement'

    """ stick all top-level options in all the task options if they are not already there """
    for opt_name, opt in options.iteritems():
        for task_name, task_opt in options['tasks'].iteritems():
            if opt_name not in task_opt:
                if opt_name not in ["tasks", "resources", "variables"]:
                    task_opt[opt_name] = copy.copy(opt)
                    # the copying is no longer needed, probably

    # Turn off warping and fit_mean if you are using PES
    for task_name, task_opts in options['tasks'].iteritems():
        # If you are using PES, turn off transformations!!
        if task_opts["acquisition"] == "PES":
            if task_opts['transformations'] != []:
                logging.debug('Warning: turning off transformations for task %s using PESC' % task_name)
                task_opts['transformations'] = []
            if task_opts['fit_mean']:
                logging.debug('Warning: turning off fit_mean for task %s using PESC' % task_name)
                task_opts['fit_mean'] = False

        if options['recommendations'] not in ("during", "end-all", "end-one"):
            logging.debug('Warning: unknown setting for recommendations: %s setting to default' % options['recommendations'])
            options['recommendations'] = DEFAULTS['recommendations']

        if task_opts['acquisition'] == "ExpectedImprovement":
            if options['recommendations'] != "during":
                logging.debug('Warning: turning recommendations to "during" because you are using EI')
                options['recommendations'] = "during"

        if task_opts['sampler'] == "MaximumLikelihood":
            logging.debug('Warning: setting burnin=0, mcmc_iters=1 because you are using MaximumLikelihood')
            task_opts['burnin'] = 0
            task_opts['mcmc_iters'] = 1

    # Make sure there is exactly 1 objective
    numObjectives = len(get_objectives_and_constraints(options)[0])
    if numObjectives != 1:
        raise Exception("You have %d objectives. You must have exactly 1" % numObjectives)

    # Set DB address
    db_address = os.getenv('SPEARMINT_DB_ADDRESS')
    if db_address is not None:
        if 'datebase' not in options:
            options['database'] = dict()
        options['database']['address'] = db_address
        logging.info('Got database address %s from environment variable\n' % db_address)        
    else:
        options['database'] = options.get('database', {'name': DEFAULTS['database_name'], 'address': DEFAULTS['database_address']})
    
    # print out stuff about the tasks    
    if verbose:
        for task_name, task_opts in options['tasks'].iteritems():
            logging.debug('Found Task "%s"' % task_name)
            for task_opt_name, task_opt_val in task_opts.iteritems():
                logging.debug('  %-18s: %s' % (task_opt_name, task_opt_val))
            logging.debug('')

    return options

def get_objectives_and_constraints(config):
    obj = list()
    con = list()
    for task_name, task_opt in config["tasks"].iteritems():
        if task_opt['type'].lower()=='objective':
            obj.append(task_name)
        elif task_opt['type'].lower()=='constraint':
            con.append(task_name)
    return obj, con


# Parses the config dict and returns the actual resource objects.
def parse_resources_from_config(config):

    # If the user did not explicitly specify resources
    # Then use a default name and use the upper level config for resource options
    if "resources" not in config:
        config["resources"] = {DEFAULTS["resource_name"] : config}

    resources = dict()
    for resource_name, resource_opts in config["resources"].iteritems():
        tasks = _parse_tasks_in_resource_from_config(resource_name, config)

        scheduler_class  = resource_opts.get("scheduler", DEFAULTS['scheduler'])
        scheduler_object = importlib.import_module('spearmint.schedulers.' + scheduler_class).init(resource_opts)

        max_concurrent = resource_opts.get('max_concurrent', DEFAULTS['max_concurrent'])
        max_finished_jobs = resource_opts.get('max_finished_jobs', DEFAULTS['max_finished_jobs'])

        resources[resource_name] = Resource(resource_name, tasks, scheduler_object, scheduler_class, max_concurrent, max_finished_jobs)

        logging.debug('Initialized Resource "%s"' % resources[resource_name].name)
        logging.debug('  scheduler         : %s' % resources[resource_name].scheduler_class)
        logging.debug('  max concurrent    : %d' % resources[resource_name].max_concurrent)
        logging.debug('  max finished jobs : %s' % resources[resource_name].max_finished_jobs)
        logging.debug('  tasks             : %s' % ', '.join(resources[resource_name].tasks))
        logging.debug('')

    return resources

# Parses out the names of the tasks associated with a particular resource, from the config dict. 
def _parse_tasks_in_resource_from_config(resource_name, config):

    # If the user did not explicitly specify tasks, then we have to assume
    # the single task runs on all resources


    tasks = list()
    for task_name, task_config in config["tasks"].iteritems():
        # If the user specified tasks but not specific resources for those tasks,
        # We have to assume the tasks run on all resources...
        if "resources" not in task_config:
            tasks.append(task_name)
        else:
            if resource_name in task_config["resources"]:
                tasks.append(task_name)

    return tasks 

def parse_priors_from_config(options):
    parsed = dict()
    for p in options:
        prior_class = eval(options[p]['distribution'])
        args = options[p]['parameters']

        # If they give a list, just stick them in order
        # If they give something else (hopefully a dict of some sort), pass them in as kwargs
        if isinstance(args, list):
            parsed[p] = prior_class(*args)
        elif isinstance(args, dict): # use isinstance() not type() so that defaultdict, etc are allowed
            parsed[p] = prior_class(**args)
        else:
            raise Exception("Prior parameters must be list or dict type")

    return parsed

# A sketch-ball helper class to help with parse_tasks_from_jobs
# Used to get a unique set of inputs across tasks
# these are the inputs for the NaN task
class HashableInput(object):
    def __init__(self, x):
        self.x = x
    def __hash__(self):
        # return int(hashlib.sha1(self.x).hexdigest(),16)
        return hash(self.x.tostring())
    def __eq__(self, other):
        return np.array_equal(self.x, other.x)
def HashableInputsToArray(HIs):
    return np.array(map(lambda x:x.x, HIs))

def parse_tasks_from_jobs(jobs, experiment_name, options, input_space):
    tasks_config = options["tasks"]

    # Create all the tasks
    tasks = dict()
    for task_name, task_options in tasks_config.iteritems():
        if task_options['type'].lower() != 'ignore':
            tasks[task_name] = Task(task_name, task_options, input_space.num_dims)

    if jobs:

        for task_name, task in tasks.iteritems():

            for job in jobs:

                if task_name in job['tasks'] and task_name != 'NaN':

                    if job['status'] == 'pending':
                        task.pending = np.append(task.pending, input_space.vectorify(job['params'])[None],axis=0)

                    elif job['status'] == 'complete':
                        task.inputs = np.append(task.inputs, input_space.vectorify(job['params'])[None],axis=0)
                        val = job['values'][task_name]
                        if isinstance(val, dict):
                            raise Exception("Value returned is a dict containing dicts: %s" % job['values'])
                        task.values = np.append(task.values, val)
                        task.durations = np.append(task.durations, job['end time']-job['start time'])
    # The next step is to add in the "NaN" task if needed
    # This is how it works: if any task was ever NaN, then we add in this new special task
    # That is binary, and "1" if not NaN, 0 if NaN
    # As a hack we make it noiseless iff all the other tasks are noiseless also
    # (This could be done more explicitly/better. For example one better idea
    # is to make it noiseless iff all of the tasks that had failures are noiseless)

    # Also right now we are working on the assumption that all the tasks live in the same
    # input space. I guess this is generally fine, if we do the ignoredims thing.
    # (meaning if they live in difference spaces, concatenate their spaces and just
    # ignore irrelevant dimensions for each task)
    
    # Here is what I actually want: to be able to hash the inputs



    # remove the old NaN task
    if 'NaN' in tasks:
        del tasks['NaN']

    # if any task has NaN values in it, create the special task
    if reduce(lambda x,y:x or y,map(lambda x:np.any(np.isnan(x.values)), tasks.values()), False):

        # First, see if all the tasks currently in this group are noiseless
        # If so, we should make the NaN task noiseless also
        # This is important because if a NaN constraint unnecessarily
        # thinks it's non-deterministic it could take MUCH longer to pass
        # the confidence threshold
        all_noiseless = True
        for task_name in tasks:
            if tasks[task_name].options.get('likelihood', DEFAULTS['likelihood']).lower() not in ['noiseless', 'step']:
                all_noiseless = False
                break

        if options["acquisition"] == "PES":
            # in some cases, for example with PESC, you may not want to use a GP classifier to a NaN task
            # thus use a regular GP here and map {0,1} to {-1,1}
            nan_task_inputs = tasks.values()[0].inputs
            nan_task_values = np.logical_not(reduce(np.logical_or, map(np.isnan, [task.values for task in tasks.values()])))
            nan_task_values = nan_task_values*2.0 - 1.0

            nan_likelihood = "noiseless" if all_noiseless else "gaussian" 
            # (above) could even just check if Objective is noiseless (this is more conservative)

        else:
            nan_likelihood = 'step' if all_noiseless else 'binomial'
            

            # WARNING:
            # There is a problem here. Before I was taking the unique set.  
            # But what if you evaluate at the same place twice? You lose the new data.
            # Then spearmint repeatedly evaluates at this place because it never gets new data
            #  ... this is catastrophic. 
            # So I think I won't do this, at the risk of having repeated entries in here
            # It will make things slower but at least it will be less risky...

            # now get the unique set of results:
            # valids = defaultdict(lambda: True)
            # for task_name, task in tasks.iteritems():
            #     print task.inputs
            #     for inp, val in zip(task.inputs, task.values):
            #         hi = HashableInput(inp)
            #         valids[hi] = valids[hi] and not np.isnan(val)
            # # this should give me a mapping from inputs to valid/invalid
            # # i see. valids is a dict with keys of HashableInputs and values True/False

            # nan_task_inputs = np.zeros((len(valids), input_space.num_dims))
            # nan_task_values = np.zeros(len(valids))
            # for i, inp in enumerate(valids):
            #     nan_task_inputs[i,:] = inp.x
            #     nan_task_values[i] = valids[inp]

            # coupled only---!!!
            nan_task_inputs = tasks.values()[0].inputs
            nan_task_values = np.logical_not(reduce(np.logical_or, map(np.isnan, [task.values for task in tasks.values()])))

        # what a mess...
        nan_task_options = dict()
        nan_task_options['constraint_delta'] = options.get("constraint_delta", DEFAULTS['constraint_delta'])
        nan_task_options['group'] = DEFAULTS['group']
        nan_task_options['nlopt_constraint_tol'] = options.get("nlopt_constraint_tol", DEFAULTS['nlopt_constraint_tol'])
        if options["acquisition"] == "PES":
            nan_task_options['transformations'] = []
        else:    
            nan_task_options['transformations'] = options.get("transformations", DEFAULTS['transformations'])
        nan_task_options['cost']  = DEFAULTS['cost']

        # make sure the options are sorted out, just like the other tasks

        for opt_name, opt in options.iteritems():
            if opt_name not in nan_task_options:
                    if opt_name not in ["tasks", "resources", "variables"]:
                        nan_task_options[opt_name] = opt
        nan_task_options['likelihood'] = nan_likelihood
        nan_task_options['type'] = 'constraint'

        nan_task = Task('NaN', nan_task_options, input_space.num_dims)
        nan_task.inputs  = nan_task_inputs
        nan_task.values  = nan_task_values

        if np.any(np.isnan(nan_task_inputs)) or np.any(np.isnan(nan_task_values)):
            raise Exception("NaN in NaN task inputs/values???")

        options["tasks"]['NaN'] = nan_task_options
        tasks['NaN'] = nan_task

    return tasks

def repeat_experiment_name(name, n):
    return name+"-%d"%int(n)

def repeat_output_dir(path, n):
    return os.path.join(path, 'output' + '_repeat_' + str(n))

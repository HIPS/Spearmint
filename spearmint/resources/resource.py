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


import spearmint

import importlib
from operator import add
import numpy as np
import sys

def parse_resources_from_config(config):
    """Parse the config dict and return a dictionary of resource objects keyed by resource name"""

    # If the user did not explicitly specify resources
    if "resources" not in config:
        default_resource_name = 'Main'
        task_names = parse_tasks_in_resource_from_config(config, default_resource_name)
        return {default_resource_name : resource_factory(default_resource_name, task_names, config)}

    # If resources are specified
    else:
        resources = dict()
        for resource_name, resource_opts in config["resources"].iteritems():
            task_names = parse_tasks_in_resource_from_config(config, resource_name)
            resources[resource_name] = resource_factory(resource_name, task_names, resource_opts)
        return resources

def parse_tasks_in_resource_from_config(config, resource_name):
    """parse the config dict and return a list of task names that use the given resource name"""
    # If the user did not explicitly specify tasks, then we have to assume
    # the single task runs on all resources
    # TODO: THIS IS VERY DANGEROUS, BECAUSE THE TASK MIGHT NOT NAMED MAIN
    # NEED TO HAVE A CONFIG PARSING SECTION OF THE CODE!!!
    if "tasks" not in config:
        return ['main']
    else:
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


def resource_factory(resource_name, task_names, config):
    """return a resource object constructed from the resource name, task names, and config dict"""
    scheduler_class  = config.get("scheduler", "local")
    scheduler_object = importlib.import_module('spearmint.schedulers.' + scheduler_class).init(config)

    max_concurrent = config.get('max-concurrent', 1)
    max_finished_jobs = config.get('max-finished-jobs', np.inf)

    return Resource(resource_name, task_names, scheduler_object, 
                    scheduler_class, max_concurrent, max_finished_jobs)

def print_resources_status(resources, jobs):
    """Print out the status of the resources"""
    if len(resources) == 1:
        sys.stderr.write('Status: %d pending, %d complete.\n\n'
            % (resources[0].numPending(jobs), resources[0].numComplete(jobs)))
    else:
        sys.stderr.write('\nResources:      ')
        left_indent=16
        indentation = ' '*left_indent

        sys.stderr.write('NAME          PENDING    COMPLETE\n')
        sys.stderr.write(indentation)
        sys.stderr.write('----          -------    --------\n')
        totalPending = 0
        totalComplete = 0
        for resource in resources:
            p = resource.numPending(jobs)
            c = resource.numComplete(jobs)
            totalPending += p
            totalComplete += c
            sys.stderr.write("%s%-12.12s  %-9d  %-10d\n" % (indentation, resource.name, p, c))
        sys.stderr.write("%s%-12.12s  %-9d  %-10d\n" % (indentation, '*TOTAL*', totalPending, totalComplete))
        sys.stderr.write('\n')

class Resource(object):
    """class which manages the job resources
    
    Parameters
    ----------
    name : str
        The name of the resource
    tasks : list of strings
        The names of the tasks
    scheduler : scheduler object
        The object which submits and polls jobs
    scheduler_class : class type
        The class type of scheduler.  This is used just for printing
    max_concurrent : int
        The maximum number of jobs that can run concurrantly
    max_finished_jobs : int
        The maximum number of jobs that will be run to completion
    """

    def __init__(self, name, tasks, scheduler, scheduler_class, max_concurrent, max_finished_jobs):
        self.name              = name
        self.scheduler         = scheduler
        self.scheduler_class   = scheduler_class   # stored just for printing
        self.max_concurrent    = max_concurrent
        self.max_finished_jobs = max_finished_jobs
        self.tasks             = tasks

        if len(self.tasks) == 0:
            sys.stderr.write("Warning: resource %s has no tasks assigned to it" % self.name)

    def filterMyJobs(self, jobs):
        """Take a list of jobs and filter only those that are running/run on this resource"""
        if jobs:
            return filter(lambda job: job['resource']==self.name, jobs)
        else:
            return jobs

    def numPending(self, jobs):
        jobs = self.filterMyJobs(jobs)
        if jobs:
            pending_jobs = map(lambda x: x['status'] in ['pending', 'new'], jobs)
            return reduce(add, pending_jobs, 0)
        else:
            return 0

    def numComplete(self, jobs):
        jobs = self.filterMyJobs(jobs)
        if jobs:
            completed_jobs = map(lambda x: x['status'] == 'complete', jobs)
            return reduce(add, completed_jobs, 0)
        else:
            return 0

    def acceptingJobs(self, jobs):
        """Is this resource currently accepting new jobs?"""
        if self.numPending(jobs) >= self.max_concurrent:
            return False
        
        if self.numComplete(jobs) >= self.max_finished_jobs:
            return False

        return True 

    def printStatus(self, jobs):
        sys.stderr.write("%-12s: %5d pending %5d complete\n" %
            (self.name, self.numPending(jobs), self.numComplete(jobs)))

    def isJobAlive(self, job):
        """Is a particular job alive?"""
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        return self.scheduler.alive(job['proc_id'])

    def attemptDispatch(self, experiment_name, job, db_address, expt_dir):
        """submit a new job using the scheduler
        
        Parameters
        ----------
        experiment_name : str
        job : dict
        db_address : str
        expt_dir : str
        
        Returns
        -------
        process_id : str
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        process_id = self.scheduler.submit(job['id'], experiment_name, expt_dir, db_address)

        if process_id is not None:
            sys.stderr.write('Submitted job %d with %s scheduler (process id: %d).\n' % 
                (job['id'], self.scheduler_class, process_id))
        else:
            sys.stderr.write('Failed to submit job %d.\n' % job['id'])

        return process_id


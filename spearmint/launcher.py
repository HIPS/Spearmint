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

import os
import sys
import time
import optparse
import subprocess
import numpy as np

from spearmint.utils.database.mongodb import MongoDB

def main():
    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--experiment-name", dest="experiment_name",
                      help="The name of the experiment in the database.",
                      type="string")

    parser.add_option("--database-address", dest="db_address",
                      help="The address where the database is located.",
                      type="string")

    parser.add_option("--job-id", dest="job_id",
                      help="The id number of the job to launch in the database.",
                      type="int")

    (options, args) = parser.parse_args()

    if not options.experiment_name:
        parser.error('Experiment name must be given.')

    if not options.db_address:
        parser.error('Database address must be given.')

    if not options.job_id:
        parser.error('Job ID not given or an ID of 0 was used.')

    launch(options.db_address, options.experiment_name, options.job_id)

def launch(db_address, experiment_name, job_id):
    """
    Launches a job from on a given id.
    """

    db  = MongoDB(database_address=db_address)
    job = db.load(experiment_name, 'jobs', {'id' : job_id})

    start_time        = time.time()
    job['start time'] = start_time
    db.save(job, experiment_name, 'jobs', {'id' : job_id})

    sys.stderr.write("Job launching after %0.2f seconds in submission.\n" 
                     % (start_time-job['submit time']))

    success = False

    try:
        if job['language'].lower() == 'matlab':
            result = matlab_launcher(job)

        elif job['language'].lower() == 'python':
            result = python_launcher(job)

        elif job['language'].lower() == 'shell':
            result = shell_launcher(job)

        elif job['language'].lower() == 'mcr':
            result = mcr_launcher(job)

        else:
            raise Exception("That language has not been implemented.")

        if not isinstance(result, dict):
            # Returning just NaN means NaN on all tasks
            if np.isnan(result):
                # Apparently this dict generator throws an error for some people??
                # result = {task_name: np.nan for task_name in job['tasks']}
                # So we use the much uglier version below... ????
                result = dict(zip(job['tasks'], [np.nan]*len(job['tasks'])))
            elif len(job['tasks']) == 1: # Only one named job
                result = {job['tasks'][0] : result}
            else:
                result = {'main' : result}
        
        if set(result.keys()) != set(job['tasks']):
            raise Exception("Result task names %s did not match job task names %s." % (result.keys(), job['tasks']))

        success = True
    except:
        import traceback
        traceback.print_exc()
        sys.stderr.write("Problem executing the function\n")
        print sys.exc_info()
        
    end_time = time.time()

    if success:
        sys.stderr.write("Completed successfully in %0.2f seconds. [%s]\n" 
                         % (end_time-start_time, result))
        
        job['values']   = result
        job['status']   = 'complete'
        job['end time'] = end_time

    else:
        sys.stderr.write("Job failed in %0.2f seconds.\n" % (end_time-start_time))
    
        # Update metadata.
        job['status']   = 'broken'
        job['end time'] = end_time

    db.save(job, experiment_name, 'jobs', {'id' : job_id})

def python_launcher(job):
    # Run a Python function
    sys.stderr.write("Running python job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # Convert the JSON object into useful parameters.
    params = {}
    for name, param in job['params'].iteritems():
        vals = param['values']

        if param['type'].lower() == 'float':
            params[name] = np.array(vals)
        elif param['type'].lower() == 'int':
            params[name] = np.array(vals, dtype=int)
        elif param['type'].lower() == 'enum':
            params[name] = vals
        else:
            raise Exception("Unknown parameter type.")

    # Load up this module and run
    main_file = job['main-file']
    if main_file[-3:] == '.py':
        main_file = main_file[:-3]
    sys.stderr.write('Importing %s.py\n' % main_file)
    module  = __import__(main_file)
    sys.stderr.write('Running %s.main()\n' % main_file)
    result = module.main(job['id'], params)

    # Change back out.
    os.chdir('..')

    # TODO: add dict capability

    sys.stderr.write("Got result %s\n" % (result))

    return result

# BROKEN
def matlab_launcher(job):
    # Run it as a Matlab function.

    try:
        import pymatlab
    except:
        raise Exception("Cannot import pymatlab. pymatlab is required for Matlab jobs. It is installable with pip.")

    sys.stderr.write("Booting up Matlab...\n")
    session = pymatlab.session_factory()

    # Add directory to the Matlab path.
    session.run("cd('%s')" % os.path.realpath(job['expt_dir']))

    session.run('params = struct()')
    for name, param in job['params'].iteritems():
        vals = param['values']

        # sys.stderr.write('%s = %s\n' % (param['name'], str(vals)))

        # should have dtype=float explicitly, otherwise
        # if they are ints it will automatically do int64, which
        # matlab will receive, and will tend to break matlab scripts
        # because in matlab things tend to always be double type
        session.putvalue('params_%s' % name, np.array(vals, dtype=float))
        session.run("params.%s = params_%s" % (name,name))
        # pymatlab sucks, so I cannot put the value directly into a struct
        # instead i do this silly workaround to put it in a variable and then
        # copy that over into the struct
        # session.run('params_%s'%param['name'])
        
    sys.stderr.write('Running function %s\n' % job['function-name'])

    # Execute the function
    session.run('result = %s(params)' % job['function-name'])

    # Get the result
    result = session.getvalue('result')

    # TODO: this only works for single-task right now
    result = float(result) 
    sys.stderr.write("Got result %s\n" % (result))

    del session

    return result

# BROKEN
def shell_launcher(job):
    # Change into the directory.
    os.chdir(job['expt_dir'])

    cmd = './%s %s' % (job['function-name'], job_file)
    sys.stderr.write("Executing command '%s'\n" % cmd)

    subprocess.check_call(cmd, shell=True)

    return result

# BROKEN
def mcr_launcher(job):
    # Change into the directory.
    os.chdir(job['expt_dir'])

    if os.environ.has_key('MATLAB'):
        mcr_loc = os.environ['MATLAB']
    else:
        raise Exception("Please set the MATLAB environment variable")

    cmd = './run_%s.sh %s %s' % (job['function-name'], mcr_loc, job_file)
    sys.stderr.write("Executing command '%s'\n" % (cmd))
    subprocess.check_call(cmd, shell=True)

    return result

if __name__ == '__main__':
    main()

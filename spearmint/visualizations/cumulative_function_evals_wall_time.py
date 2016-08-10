import os
import sys
import importlib
import imp
import optparse
import numpy             as np
import numpy.random      as npr
import numpy.linalg      as npla
from collections import defaultdict

import matplotlib        as mpl
mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['axes.unicode_minus'] = False
mpl.rc('font', size=18)
import matplotlib.pyplot as plt

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

def main(expt_dir, n_repeat):
    n_repeat = int(n_repeat)
    # allow n_repeat=-1 to represent a non-repeated (single) experiment
    if n_repeat < 0:
        n_repeat = 1
        repeat_flag = False
    else:
        repeat_flag = True
    options  = parse_config_file(expt_dir, 'config.json')
    tasks    = options['tasks'].keys()

    # Get all the jobs in once place, and store all the wall times
    jobs = dict()
    start_times = np.zeros(n_repeat)
    for j in xrange(n_repeat):
        # get experiment name
        experiment_name = options["experiment_name"]
        print "Loaded experiment %s" % experiment_name
        if repeat_flag:
            experiment_name = repeat_experiment_name(experiment_name, j)
        # get db
        db  = MongoDB(database_address=options['database']['address'])
        # get start time
        loaded_time = db.load(experiment_name, 'start-time')
        if loaded_time is None:
            raise Exception("Could not load start time from DB. This may be because the experiment name is wrong. For a non-repeated experiment, use n_repeat=-1.")
        start_times[j] = loaded_time['start-time']    
        # load jobs
        loaded_jobs = load_jobs(db, experiment_name)
        
        # don't load up broken jobs etc
        jobs[j] = list()
        for loaded_job in loaded_jobs:
            if loaded_job['status'] == 'complete':
                jobs[j].append(loaded_job)
    print(start_times)
    print experiment_name
    wall_times = list()
    for j in xrange(n_repeat):
        # get elapsed wall times
        wall_times.append(np.zeros(len(jobs[j])))
        for i in xrange(len(jobs[j])):
            wall_times[j][i] = (jobs[j][i]['end time'] - start_times[j])/60.0

    # set up the time bins
    bin_size = 1.0
    end_times = map(max, wall_times)
    for j in xrange(n_repeat):
        print 'end time for repeat %d: %f' % (j, end_times[j])
    timesteps = np.arange(0.0,np.round(max(end_times))+bin_size, bin_size)
    print timesteps
    # get the number of evals
    cum_evals = dict()
    for j in xrange(n_repeat):
        cum_evals[j] = dict()

        for task in tasks:
            
            evals = np.zeros(len(timesteps))
            for i in xrange(len(jobs[j])):

                if jobs[j][i]['end time'] is None:
                    raise Exception("Job does not have end time. Maybe it is currently running?")

                # last_num_evals = 0 if i == 0 else cum_evals[j][task][i-1] # cumulative, so get the last number
                if task in jobs[j][i]['tasks']:

                    # find what time bin it goes in
                    timebin = int(np.ceil(wall_times[j][i]/bin_size)) # used to be floor -- conventions!
                    # print timebin
                    if timebin < len(evals): # this is because I have np.round(max(end_times)) above instead of np.ceil... nthis is just cosmetic
                        evals[timebin] += 1

            cum_evals[j][task] = np.cumsum(evals)



    # make new dicts to store the averages of repeats and the error bars
    avg_evals = defaultdict(lambda: np.zeros(len(timesteps)))
    err_evals = defaultdict(lambda: np.zeros(len(timesteps)))
    # average over the j repeats
    for i in xrange(len(timesteps)):
        for task in tasks:
            avg_evals[task][i] = np.mean([cum_evals[j][task][i] for j in xrange(n_repeat)])
            err_evals[task][i] =  np.std([cum_evals[j][task][i] for j in xrange(n_repeat)])

    plt.figure()
    linestyles = ['solid','dashed','dashdot','dotted'] #  ['-' | '--' | '-.' | ':' | 'None' | ' ' | '']
    markers = ['+', 'x', 'o', '*', '.']
    for i,task in enumerate(tasks):
        plt.errorbar(timesteps, avg_evals[task], yerr=err_evals[task], 
            linewidth=2, linestyle=linestyles[i], marker=markers[i], markeredgewidth=2, ms=8)
    plt.legend(tasks, loc='upper left')
    plt.xlabel('Elapsed time (minutes)', size=25)
    plt.ylabel('Cumulative evaluations',size=25)
    plt.xlim(0, max(timesteps))
    ymin,ymax = plt.ylim()
    plt.ylim(-1, ymax) # start it at 0
    plt.xticks(np.arange(0, max(timesteps)+3,3)) # hack -
    plt.tight_layout()

    # Make the directory for the plots
    plots_dir = os.path.join(expt_dir, 'plots')
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    figname = os.path.join(plots_dir, 'cumulative_evals_wall_time')
    print 'Saving figure at %s' % figname
    plt.savefig(figname + '.pdf')
    plt.savefig(figname + '.svg')


if __name__ == '__main__':
    main(*sys.argv[1:])
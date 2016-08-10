
import os
import sys
import importlib
import imp
import optparse
import numpy             as np
import numpy.random      as npr
import numpy.linalg      as npla
import matplotlib        as mpl
from collections import Counter
from collections import defaultdict
mpl.use('Agg')
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['axes.unicode_minus'] = False

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
    options  = parse_config_file(expt_dir, 'config.json')
    tasks    = options['tasks'].keys()

    jobs = dict()
    for j in xrange(n_repeat):
        experiment_name = repeat_experiment_name(options["experiment_name"], j)
        db              = MongoDB(database_address=options['database']['address'])
        jobs[j]         = load_jobs(db, experiment_name)

        
    n_iter_each = map(len, jobs.values())
    print 'Found %s iterations' % n_iter_each
    n_iter = min(n_iter_each)

    cum_evals = defaultdict(lambda: defaultdict(lambda:defaultdict(int)))
    for j in xrange(n_repeat):

        for i in xrange(n_iter):
            for task in tasks:
                if task in jobs[j][i]['tasks']:
                    cum_evals[j][task][i] = cum_evals[j][task][i-1] + 1
                else:
                    cum_evals[j][task][i] = cum_evals[j][task][i-1]

    # average over the j repeats
    for i in xrange(n_iter):
        for task in tasks:
            cum_evals["avg"][task][i] = np.mean([cum_evals[j][task][i] for j in xrange(n_repeat)])
            cum_evals["err"][task][i] =  np.std([cum_evals[j][task][i] for j in xrange(n_repeat)])

    plt.figure()
    iters = range(n_iter)
    for task in tasks:
        plt.errorbar(iters, [cum_evals["avg"][task][i] for i in xrange(n_iter)], 
                       yerr=[cum_evals["err"][task][i] for i in xrange(n_iter)], linewidth=2)
    plt.legend(tasks, loc='upper left')
    plt.xlabel('Iteration number', size=25)
    plt.ylabel('Cumulative evaluations',size=25)

    # Make the directory for the plots
    plots_dir = os.path.join(expt_dir, 'plots')
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    figname = os.path.join(plots_dir, 'cumulative_evals.pdf')
    print 'Saving figure at %s' % figname
    plt.savefig(figname)


if __name__ == '__main__':
    main(*sys.argv[1:])
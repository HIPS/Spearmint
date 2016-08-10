import os
import sys
import numpy        as np
import numpy.random as npr
from collections import defaultdict

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
    start_times = dict()
    for j in xrange(n_repeat):
        experiment_name = repeat_experiment_name(options["experiment_name"], j)
        db              = MongoDB(database_address=options['database']['address'])
        jobs[j]         = load_jobs(db, experiment_name)
        start_times[j]  = db.load(experiment_name, 'start-time')['start-time']


    time_in_evals = defaultdict(lambda: np.zeros(n_repeat))
    time_in_fast_updates = np.zeros(n_repeat)
    time_in_slow_updates = np.zeros(n_repeat)

    for j in xrange(n_repeat):

        last_job_end_time = start_times[j]

        for job in jobs[j]:
            if job['status'] == 'complete':
                time_in_evals[job['tasks'][0]][j] += (job['end time'] - job['start time'])/60.0

                if job['fast update']:
                    time_in_fast_updates[j] += (job['start time'] - last_job_end_time)/60.0
                else:
                    time_in_slow_updates[j] += (job['start time'] - last_job_end_time)/60.0
                last_job_end_time = job['end time']

    for task in tasks:
        print 'Average time on task %s over %d repeats: %f +/- %f minutes (mean +/- std)' % (task, n_repeat, np.mean(time_in_evals[task]), np.std(time_in_evals[task]))
    total_time_in_evals = sum(time_in_evals.values())
    print 'Average time in JOBS over %d repeats: %f +/- %f minutes (mean +/- std)' % (n_repeat, np.mean(total_time_in_evals), np.std(total_time_in_evals))
    print 'Average time in FAST over %d repeats: %f +/- %f minutes (mean +/- std)' % (n_repeat, np.mean(time_in_fast_updates), np.std(time_in_fast_updates))
    print 'Average time in SLOW over %d repeats: %f +/- %f minutes (mean +/- std)' % (n_repeat, np.mean(time_in_slow_updates), np.std(time_in_slow_updates))
    total_optimizer_time = time_in_fast_updates + time_in_slow_updates
    print 'Average time in OPTIMIZER over %d repeats: %f +/- %f minutes (mean +/- std)' % (n_repeat, np.mean(total_optimizer_time), np.std(total_optimizer_time))
    print 'Total average time spent: %f' % np.sum([np.mean(total_time_in_evals),np.mean(time_in_fast_updates),np.mean(time_in_slow_updates)])

if __name__ == '__main__':
    main(*sys.argv[1:])
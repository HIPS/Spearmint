import importlib
import sys
from itertools import izip

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


from spearmint.utils.database.mongodb import MongoDB

from spearmint.main import get_options, parse_resources_from_config, load_jobs, remove_broken_jobs, \
    load_task_group, load_hypers

def print_dict(d, level=1):
    if isinstance(d, dict):
        if level > 1: print ""
        for k, v in d.iteritems():
            print "  " * level, k,
            print_dict(v, level=level+1)
    else:
        print d 

def main():
    """
    Usage: python make_plots.py PATH_TO_DIRECTORY

    TODO: Some aspects of this function are specific to the simple branin example
    We should clean this up so that interpretation of plots are more clear and
    so that it works in more general cases 
    (e.g. if objective likelihood is binomial then values should not be
    unstandardized)
    """
    options, expt_dir = get_options()
    print "options:"
    print_dict(options)
    
    # reduce the grid size
    options["grid_size"] = 400

    resources = parse_resources_from_config(options)

    # Load up the chooser.
    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    chooser = chooser_module.init(options)
    print "chooser", chooser
    experiment_name     = options.get("experiment-name", 'unnamed-experiment')

    # Connect to the database
    db_address = options['database']['address']
    sys.stderr.write('Using database at %s.\n' % db_address)        
    db         = MongoDB(database_address=db_address)
    
    # testing below here
    jobs = load_jobs(db, experiment_name)
    remove_broken_jobs(db, jobs, experiment_name, resources)

    print "resources:", resources
    print_dict(resources)
    resource = resources.itervalues().next()
    
    task_options = { task: options["tasks"][task] for task in resource.tasks }
    print "task_options:"
    print_dict(task_options) # {'main': {'likelihood': u'NOISELESS', 'type': 'OBJECTIVE'}}
    
    task_group = load_task_group(db, options, resource.tasks)
    print "task_group", task_group # TaskGroup
    print "tasks:"
    print_dict(task_group.tasks) # {'main': <spearmint.tasks.task.Task object at 0x10bf63290>}
    
    
    hypers = load_hypers(db, experiment_name)
    print "loaded hypers", hypers # from GP.to_dict()
    
    hypers = chooser.fit(task_group, hypers, task_options)
    print "\nfitted hypers:"
    print_dict(hypers)

    lp, x = chooser.best()
    x = x.flatten()
    print "best", lp, x
    bestp = task_group.paramify(task_group.from_unit(x))
    print "expected best position", bestp
    
    # get the grid of points
    grid = chooser.grid
#     print "chooser objectives:", 
#     print_dict(chooser.objective)
    print "chooser models:", chooser.models
    print_dict(chooser.models)
    obj_model = chooser.models[chooser.objective['name']]
    obj_mean, obj_var = obj_model.function_over_hypers(obj_model.predict, grid)

    # un-normalize the function values and variances
    obj_task = task_group.tasks['main']
    obj_mean = [obj_task.unstandardize_mean(obj_task.unstandardize_variance(v)) for v in obj_mean]
    obj_std = [obj_task.unstandardize_variance(np.sqrt(v)) for v in obj_var]

    
#     for xy, m, v in izip(grid, obj_mean, obj_var):
#         print xy, m, v

    grid = map(task_group.from_unit, grid)
#     return
    
    xymv = [(xy[0], xy[1], m, v) for xy, m, v in izip(grid, obj_mean, obj_std)]# if .2 < xy[0] < .25] 
    
    x = map(lambda x:x[0], xymv)
    y = map(lambda x:x[1], xymv)
    m = map(lambda x:x[2], xymv)
    sig = map(lambda x:x[3], xymv)
#     print y
    
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, m, marker='.', linestyle="None")

    # plot errorbars
    for i in np.arange(0, len(x)):
        ax.plot([x[i], x[i]], [y[i], y[i]], [m[i]+sig[i], m[i]-sig[i]], marker="_", color='k')

    # get the observed points
    task = task_group.tasks['main']
    idata = task.valid_normalized_data_dict
    xy = idata["inputs"]
    xy = map(task_group.from_unit, xy)
    xy = np.array(xy)
    vals = idata["values"]
    vals = [obj_task.unstandardize_mean(obj_task.unstandardize_variance(v)) for v in vals]

    ax.plot(xy[:,0], xy[:,1], vals, marker='o', color="r", linestyle="None")
    
    plt.show()
    

    
if __name__ == "__main__":
    main()

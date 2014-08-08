import math
import numpy as np

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    print 'Evaluating at (%f, %f)' % (x, y)

    if x < 0 or x > 5.0 or y > 5.0:
        return np.nan
    # Feasible region: x in [0,5] and y in [0,5]

    obj = float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)

    con1 = float(y-x)   # y >= x 

    con2 = float(10.0-y)  # y <= 10
    
    return {
        "branin"       : obj, 
        "y_at_least_x" : con1, 
        "y_at_most_10" : con2
    }

    # True minimum is at 2.945, 2.945, with a value of 0.8447

def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in branin_con.py'
        return np.nan

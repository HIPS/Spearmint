# The toy problem in http://arxiv.org/abs/1403.4890
import numpy as np

def main(job_id, params):
  x1 = params['x']
  x2 = params['y']

  f  = x1 + x2
  c1 = 1.5 - x1 - 2.0*x2 - 0.5*np.sin(2*np.pi*(x1**2 - 2.0*x2))
  c2 = x1**2 + x2**2 - 1.5 
  c1 = -c1
  c2 = -c2

  return {'f':f, 'c1':c1, 'c2':c2}


"""
Everything below this point is optional. It is used to specify
the true solution so that one can plot the error curve using 
progress_curve.py in the visualizations/ directory.
"""
def true_val():
    return 0.5998
def true_sol():
    return {'x' : 0.1954, 'y' : 0.4044}
true_func = main    

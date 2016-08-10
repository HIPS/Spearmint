import numpy as np
import numpy.random as npr
import sys
import math
import time

def main(job_id, params):
  x1 = params['x']
  x2 = params['y']

  c1 = 1.5 - x1 - 2.0*x2 - 0.5*np.sin(2*np.pi*(x1**2 - 2.0*x2))
  c1 = -c1

  time.sleep(2)
  
  return {'c1' : c1}

# def true_func(job_id, params):
#   return toy(params['x'], params['y'])
# def true_val():
#     return 0.5998
# def true_sol():
#     return {'x' : 0.1954, 'y' : 0.4044}
    

import numpy as np
import sys
import math
import time

def branin(x, y):

  result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + 
       (5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10

  result = float(result)

  #if np.random.rand > 0.75:
  #  raise Exception('Blah!')

  print 'Result = %f' % result
  time.sleep(np.random.randint(30))
  return {'branin' : result}

# Write a function like this called 'main'
def main(job_id, params):
  print 'Anything printed here will end up in the output directory for job #%d' % job_id
  print params
  return branin(params['x'], params['y'])

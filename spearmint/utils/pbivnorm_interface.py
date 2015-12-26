import numpy as np
import pbivnorm

def pbivnorm_easy_numpy_vectors(x, y, rho = 0.0):

	assert type(x) == np.ndarray and type(y) == np.ndarray and x.shape == y.shape
		
	if type(rho) == float:
		rho = np.ones(x.shape) * rho
	else:
		assert type(rho) == np.ndarray and rho.shape == x.shape
		
	lower = np.zeros(2)
	infin = np.zeros(2)

	original_shape = x.shape

	x = x.reshape(1, np.prod(x.shape)).flatten()
	y = y.reshape(1, np.prod(x.shape)).flatten()
	rho = rho.reshape(1, np.prod(x.shape)).flatten()

	uppera = x
	upperb = y

	lt = int(x.shape[ 0 ])
	prob = np.zeros(lt)

	result = pbivnorm.pbivnorm(lower.tolist(), uppera.tolist(), upperb.tolist(), infin.tolist(), rho.tolist())

	result[ np.where(np.isnan(result)) ] = 0

	return result.reshape(original_shape)

def pbivnorm_easy_numpy_floats(x, y, rho = 0.0):

	assert (type(x) == float and type(y) == float) or (type(x) == int and type(y) == int)

	if type(x) == int:
		x = float(x)

	if type(y) == int:
		y = float(y)
		
	if type(rho) == float:
		rho = np.array([ rho ])
	else:
		assert type(rho) == np.ndarray 
		
	lower = np.zeros(2)
	infin = np.zeros(2)

	original_shape = rho.shape

	x = x * np.ones(original_shape)
	y = y * np.ones(original_shape)

	x = x.reshape(1, np.prod(original_shape)).flatten()
	y = y.reshape(1, np.prod(original_shape)).flatten()
	rho = rho.reshape(1, np.prod(original_shape)).flatten()

	uppera = x
	upperb = y

	lt = int(x.shape[ 0 ])
	prob = np.zeros(lt)

	result = pbivnorm.pbivnorm(lower.tolist(), uppera.tolist(), upperb.tolist(), infin.tolist(), rho.tolist())

	return result.reshape(original_shape)





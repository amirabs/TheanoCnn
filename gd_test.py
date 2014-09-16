import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano import pp

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from theano import pp

def shared_dataset(data_x, borrow=True):
    data_x = data_x
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x

set_x=[1,1,1]
x = shared_dataset(set_x)
y=x[0]*x[1]*x[2]
gy = T.grad(y, x)
updates = []
# for x_i,g_i in zip(x,gy):
# 	updates.append((x_i, x_i - 0.1 * g_i))
# for i in range(3):
#     updates.append((x[i], x[i] - 0.1 * gy[i]))
for x_i in x:
	updates.append((x_i , - 0.1 * x_i))


f = theano.function([], gy, updates=updates)

print f()

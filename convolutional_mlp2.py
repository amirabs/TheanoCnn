"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    [data_x, data_y] = data_xy
    # print data_x[0]
    # print data_y
    # print len(data_x[0])
    # print len(data_x)
    # print len(data_y)

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')



def evaluate_test1(learning_rate=0.1, n_epochs=500,
                    nkerns=[1, 1], batch_size=10):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = numpy.random.RandomState(23455)

    n=10
    d=9
    xs=[[float(i),] * d for i in range(0,n)]
    ys=([0]*int(0.8*n))
    for i in range(0,int(0.2*n)): ys.append(1) 
    ys=[0,0,0,0,0,0,0,0,1,1]
    test_set_x, test_set_y = shared_dataset([xs,ys])
    valid_set_x, valid_set_y = shared_dataset([xs,ys])
    train_set_x, train_set_y = shared_dataset([xs,ys])


    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    

    # compute number of minibatches for training, validation and testing
    batch_size=n
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    # n_train_batches /= batch_size
    # n_valid_batches /= batch_size
    # n_test_batches /= batch_size
    n_train_batches = 1
    n_valid_batches = 1
    n_test_batches = 1

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (3, 3)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,3*3)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 3, 3))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (3-3+1,3-3+1)=(1,1)
    # maxpooling reduces this further to (1/1,1/1) = (1,1)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],1,1)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 3, 3),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(1, 1))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (1,1*1*1) = (20,512)
    layer1_input = layer0.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    # layer1 = HiddenLayer(rng, input=layer1_input, n_in=nkerns[1] * 1 * 1,
    #                      n_out=1, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer1 = LogisticRegression(input=layer1_input, n_in=1, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer1.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer1.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer1.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    bestConvW=layer0.W.get_value();



    while (epoch < n_epochs) and (not done_looping):

        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter

            cost_ij = train_model(minibatch_index)
            print cost_ij


            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    bestConvW=layer0.W.get_value();
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print "bestConvW" + str(bestConvW);
    print sum(sum(sum(sum(bestConvW))))
    print cost




def evaluate_test2(learning_rate=0.1, n_epochs=100,
                    nkerns=[1, 1], batch_size=10):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = numpy.random.RandomState(23455)

    n=10
    d=16
    xs=[[float(i)/float(n+1)] * d for i in range(0,n)]
    ys=([0]*int(0.8*n))
    for i in range(0,int(0.2*n)): ys.append(1) 
    ys=[0,0,0,0,0,0,0,0,1,1]
    test_set_x, test_set_y = shared_dataset([xs,ys])
    valid_set_x, valid_set_y = shared_dataset([xs,ys])
    train_set_x, train_set_y = shared_dataset([xs,ys])


    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    

    # compute number of minibatches for training, validation and testing
    batch_size=n
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    # n_train_batches /= batch_size
    # n_valid_batches /= batch_size
    # n_test_batches /= batch_size
    n_train_batches = 1
    n_valid_batches = 1
    n_test_batches = 1

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (4, 4)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,3*3)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 4, 4))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (3-3+1,3-3+1)=(1,1)
    # maxpooling reduces this further to (1/1,1/1) = (1,1)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],1,1)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 4, 4),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(1, 1))

    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 2, 2),
            filter_shape=(nkerns[1], nkerns[0], 2, 2), poolsize=(1, 1))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (1,1*1*1) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    # layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 1 * 1,
    #                      n_out=10, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer2 = LogisticRegression(input=layer2_input, n_in=1, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer2.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer2.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer2.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})


    # create a list of all model parameters to be fit by gradient descent
    params = layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    bestConvW1=layer0.W.get_value();
    bestConvW2=layer1.W.get_value();



    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            # if iter % 100 == 0:
            print 'training @ iter = ', iter

            cost_ij = train_model(minibatch_index)

            print cost_ij


            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    bestConvW1=layer0.W.get_value();
                    bestConvW2=layer1.W.get_value();
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print "bestConvW1" + str(bestConvW1);
    print "bestConvW2" + str(bestConvW2);


def evaluate_mnist_1(learning_rate=0.1, n_epochs=2000,
                    nkerns=[4, 6], batch_size=2):

    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    rng = numpy.random.RandomState(23455)
    xs=[]
    ys=[]
    f = open('temp_value', 'r+')
    while(1):
        line=f.readline()
        line2=f.readline()
        if not line:
            break
        line=line.replace("\n","")

        values = [float(i) for i in line.split()]
        value= float(line2)

        xs.append(values)
        ys.append(value)
    # print xs
    # xs=[[0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0, 0.00784, 0, 0, 0.0157, 0.00392, 0, 0.0196, 0, 0.0118, 0, 0.00784, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0314, 0, 0.0157, 0, 0, 0, 0.051, 0.00784, 0, 0, 0, 0.0314, 0.0275, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.0275, 0.0196, 0.00392, 0, 0.00784, 0.0118, 0, 0, 0, 0.00784, 0.0118, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0.102, 0, 0, 0.0118, 0.0118, 0.00392, 0, 0.0196, 0.00784, 0.0118, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.0235, 0, 0.0157, 0.0118, 0.0196, 0, 0, 0, 0.0235, 0, 0, 0.0157, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0392, 0.769, 0.969, 1, 1, 1, 0.769, 0.275, 0.0431, 0, 0.051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.0275, 0.00392, 0.216, 0.871, 1, 0.961, 0.949, 0.949, 0.992, 0.898, 0.694, 0.0863, 0, 0, 0.0784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.235, 0.831, 0.945, 1, 0.957, 1, 0.976, 0.996, 1, 0.937, 0.165, 0.0314, 0, 0, 0, 0, 0, 0, 0.00392, 0.00392, 0, 0, 0, 0.00392, 0, 0, 0, 0.161, 0.933, 0.937, 1, 1, 0.937, 1, 1, 0.992, 0.929, 0.51, 0.0275, 0.0157, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.0196, 0.00392, 0.00392, 0.0157, 0.00392, 1, 0.898, 1, 0.957, 1, 0.976, 0.984, 0.969, 0.965, 1, 0.957, 0.38, 0.00784, 0.0235, 0.0314, 0, 0, 0, 0, 0.00392, 0, 0, 0.0157, 0.00392, 0, 0, 0.0157, 0.00392, 0.541, 0.788, 1, 0.945, 1, 0.953, 0.922, 0.651, 0.698, 0.988, 0.898, 1, 0.302, 0, 0, 0, 0, 0, 0, 0.0157, 0.00784, 0, 0, 0, 0, 0, 0.00784, 0.0314, 0.141, 0.714, 0.969, 1, 0.941, 0.733, 0.467, 0.0941, 0.643, 0.937, 1, 0.984, 0.298, 0.0196, 0, 0, 0, 0, 0, 0, 0.00784, 0.00784, 0, 0, 0.0275, 0.0275, 0, 0, 0.847, 1, 0.937, 0.945, 0.859, 0.122, 0, 0.0392, 0.949, 1, 0.91, 0.961, 0, 0.051, 0, 0, 0, 0, 0, 0, 0.00392, 0.0196, 0.00392, 0, 0.00784, 0, 0, 0, 0.357, 0.969, 1, 1, 0.0157, 0.0118, 0.0235, 0, 0.157, 1, 1, 0.949, 0.816, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0196, 0, 0, 0, 0.0706, 0.31, 1, 0.949, 0.98, 0.675, 0, 0, 0.0157, 0.051, 0.498, 0.945, 0.898, 1, 0.435, 0.0353, 0.00392, 0, 0, 0, 0, 0.0275, 0, 0, 0.0275, 0, 0, 0.0902, 0.286, 0.89, 0.949, 1, 0.608, 0.0471, 0.0549, 0.0196, 0, 0, 0.984, 1, 1, 0.976, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.00784, 0.0118, 0.00392, 0.0118, 0.0235, 0.122, 0.886, 0.933, 1, 0.847, 0.125, 0.0314, 0.00392, 0, 0.0667, 0.275, 1, 0.996, 0.949, 0.655, 0, 0.0157, 0, 0, 0, 0, 0.0235, 0.0118, 0.0157, 0, 0, 0, 0, 0.0902, 0.843, 1, 0.906, 0.294, 0.0588, 0, 0, 0.051, 0, 0.416, 0.965, 1, 0.804, 0.62, 0, 0.0118, 0, 0, 0, 0, 0.0157, 0, 0, 0.0471, 0.0235, 0, 0.106, 0.592, 0.929, 0.98, 0.604, 0, 0.0588, 0.0275, 0, 0, 0.00784, 1, 0.976, 0.937, 0.247, 0.0549, 0, 0.00784, 0, 0, 0, 0, 0, 0.0314, 0.0157, 0, 0.0235, 0.0196, 0, 0.176, 0.835, 1, 0.847, 0.239, 0, 0.0392, 0.0118, 0.00784, 0, 1, 0.941, 1, 0.169, 0, 0.0314, 0.0118, 0, 0, 0, 0, 0, 0.0392, 0.00784, 0, 0.0196, 0.00392, 0, 0.0667, 0.839, 0.918, 0.984, 0.843, 0.2, 0, 0.00392, 0.455, 0.78, 0.996, 1, 0.957, 0.22, 0.0863, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0, 0.0353, 0.051, 0, 0.051, 0.431, 0.945, 0.965, 1, 0.973, 0.992, 1, 1, 0.984, 1, 0.957, 0.914, 0.459, 0, 0, 0, 0.0471, 0, 0, 0, 0, 0.00392, 0.00392, 0.00392, 0, 0, 0.0314, 0, 0.0118, 0.506, 0.667, 0.992, 1, 0.973, 0.91, 0.953, 1, 0.914, 1, 0.976, 0.384, 0.0118, 0, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0, 0.0118, 0, 0.0157, 0, 0.0824, 0.608, 0.984, 0.965, 1, 1, 0.957, 1, 0.925, 0.776, 0.133, 0.0196, 0.0392, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.114, 0.643, 0.976, 1, 0.969, 0.525, 0.196, 0.157, 0.098, 0.0392, 0.00392, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.0118, 0, 0.0392, 0.00392, 0, 0.0863, 0, 0.0314, 0.0196, 0, 0, 0, 0, 0.0118, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0431, 0, 0, 0.0549, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0.0118, 0.0196, 0, 0.0196, 0.00784, 0, 0.0118, 0.0157, 0.0196, 0.0196, 0.0118, 0, 0, 0, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0196, 0, 0, 0.0353, 0, 0, 0.0196, 0, 0, 0, 0, 0.0118, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0.0431, 0.0118, 0, 0.0902, 0, 0, 0.0353, 0.0314, 0, 0, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.051, 0, 0.0471, 0.0235, 0, 0, 0, 0.0196, 0, 0.0157, 0, 0, 0.0353, 0, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.0157, 0.0471, 0.0157, 0.0235, 0, 0.0784, 0.00392, 0, 0, 0.0157, 0.0118, 0.00392, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0941, 0, 0, 0, 0, 0, 0.0471, 0, 0, 0.0431, 0.0392, 0, 0, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0.0118, 0, 0.0196, 0.263, 0.945, 1, 1, 0.933, 0.525, 0.0471, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.0353, 0, 0.0314, 0, 0.0157, 0.6, 1, 0.906, 0.996, 1, 0.898, 0.58, 0.235, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0.0353, 0, 0.161, 0.776, 1, 0.929, 1, 0.969, 0.929, 1, 0.98, 0.431, 0, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0.122, 1, 1, 0.953, 0.969, 0.961, 1, 0.929, 1, 0.451, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0.0275, 0, 0.667, 0.898, 0.941, 1, 0.996, 0.467, 0.529, 0.988, 1, 0.396, 0.0118, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0.263, 1, 1, 0.98, 1, 0.463, 0, 0, 0.396, 1, 0.973, 0.439, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.0196, 0.282, 0.973, 0.984, 0.961, 1, 0, 0.051, 0.051, 0.447, 0.984, 0.973, 0.369, 0, 0.0588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0.831, 0.992, 0.937, 1, 0.796, 0.0275, 0, 0, 0.518, 1, 1, 0.392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0902, 0.592, 0.922, 1, 0.988, 0.314, 0, 0, 0.502, 0.945, 0.973, 1, 0.427, 0.0431, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.992, 0.973, 0.957, 0.671, 0.0157, 0.0275, 0.00392, 0.824, 0.984, 0.984, 1, 0.325, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.698, 1, 0.945, 0.882, 0.0157, 0.00392, 0, 0, 0.886, 0.98, 0.996, 0.678, 0.0471, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.259, 0.831, 1, 0.98, 0.859, 0, 0, 0, 0.322, 0.98, 0.961, 0.98, 0.263, 0, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.435, 0.98, 1, 0.976, 0.816, 0.0745, 0, 0.0863, 0.624, 0.933, 1, 0.424, 0, 0.0431, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.396, 1, 0.996, 0.988, 0.537, 0, 0.0157, 0.537, 0.984, 0.992, 1, 0.247, 0, 0.00392, 0.00784, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.431, 0.973, 1, 0.965, 0.188, 0.00784, 0.502, 1, 0.961, 1, 0.639, 0.11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.447, 0.929, 1, 0.98, 0.396, 0.608, 1, 0.984, 1, 0.592, 0.0588, 0, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.416, 1, 1, 0.961, 0.945, 0.984, 0.996, 1, 1, 0.114, 0, 0.0431, 0.0471, 0, 0.0118, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.365, 0.922, 1, 0.922, 1, 0.976, 0.953, 0.843, 0.443, 0, 0.0941, 0, 0, 0.0314, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0.333, 0.984, 1, 0.937, 0.98, 1, 0, 0, 0, 0.0314, 0, 0.00392, 0.0392, 0, 0.0588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.00784, 0.463, 0.965, 0.871, 0.345, 0.192, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.051, 0, 0, 0.0784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0.0196, 0, 0, 0, 0, 0.0706, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0.0235, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.098, 0, 0, 0.0784, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0.0157, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.00784, 0.0275, 0, 0.0392, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0745, 0, 0, 0, 0.0157, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0784, 0, 0.0118, 0.0588, 0.286, 0.729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0.0745, 0, 0.0314, 0.992, 0.953, 0.0941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.051, 0, 0.0314, 0.118, 0.984, 0.965, 0.184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0.0941, 0.945, 1, 0.176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0.00392, 0, 0.118, 1, 0.996, 0.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0.0392, 0, 0.0941, 1, 1, 0.384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.051, 0, 0.0314, 0.89, 1, 0.561, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0.0118, 0, 0, 0.737, 0.992, 0.588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0275, 0, 0, 0.0157, 0.675, 0.937, 0.557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0, 0.0157, 0.702, 0.945, 0.694, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.051, 0, 0, 0.737, 1, 0.914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.0196, 0.00784, 0.00392, 0.0235, 0.478, 0.949, 0.941, 0.0196, 0, 0.0275, 0.0431, 0, 0, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.345, 1, 0.941, 0.118, 0, 0, 0, 0, 0.0353, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0157, 0.0235, 0.00784, 0.0196, 0.216, 1, 0.992, 0.204, 0, 0, 0.0431, 0.00784, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0627, 0.808, 0.976, 0.463, 0.0745, 0, 0.0157, 0.0353, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.549, 0.969, 0.118, 0.00392, 0.0275, 0.00784, 0, 0.0196, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.00392, 0.0235, 0.0118, 0.0235, 0.00392, 0.498, 0.984, 0, 0, 0.0314, 0.0431, 0, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.00784, 0, 0, 0, 0.643, 0.961, 0.365, 0.0471, 0, 0, 0.0118, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0.0471, 0, 0.0118, 0, 0.878, 0.988, 0.125, 0, 0.0118, 0.0118, 0, 0.00784, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0588, 0, 0, 0, 0.0902, 0.565, 0.847, 0.0824, 0.0353, 0, 0, 0, 0.0157, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0431, 0, 0.0275, 0.0275, 0, 0.0392, 0, 0.0392, 0.0118, 0, 0, 0.00392, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.0314, 0, 0, 0.0157, 0, 0.0118, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0275, 0.0157, 0, 0.0431, 0.0392, 0, 0, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0 ],
    #     [0.0196, 0, 0, 0.0275, 0.00784, 0, 0, 0, 0, 0, 0.00392, 0.0157, 0.0235, 0.0235, 0.00784, 0, 0.0157, 0.00392, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0, 0, 0.0118, 0, 0, 0, 0, 0.0235, 0.0588, 0.0196, 0, 0.00392, 0, 0, 0, 0.0196, 0.0118, 0, 0, 0, 0, 0, 0.0118, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0353, 0.00784, 0, 0, 0, 0, 0.0235, 0.0275, 0.0157, 0, 0, 0.00392, 0.00392, 0, 0, 0, 0.00392, 0.00784, 0.0157, 0.0157, 0.00392, 0, 0, 0, 0, 0, 0.0314, 0.00392, 0, 0, 0.0235, 0.00784, 0, 0.0275, 0, 0, 0, 0.00392, 0, 0, 0, 0.0157, 0.00784, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.0118, 0, 0, 0, 0.0118, 0, 0, 0, 0, 0.0627, 0, 0.0118, 0, 0, 0, 0.0157, 0, 0, 0.051, 0.00392, 0, 0, 0, 0, 0, 0, 0.0157, 0.00784, 0, 0, 0, 0.00392, 0.0275, 0.0196, 0.0196, 0, 0.0196, 0.141, 0.38, 0.906, 1, 0.267, 0, 0.0275, 0.0353, 0, 0, 0, 0, 0.0275, 0, 0, 0, 0, 0.0118, 0, 0, 0.0235, 0.00392, 0, 0, 0, 0, 0.0471, 0.631, 0.933, 1, 1, 0.929, 0.918, 0.82, 0.776, 0.8, 0.278, 0.0627, 0.0784, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.0392, 0, 0, 0, 0, 0.0588, 0.345, 0.976, 1, 0.945, 1, 0.965, 1, 0.973, 1, 1, 0.941, 0.765, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0235, 0, 0, 0.0314, 0.0157, 0.0353, 0.153, 0.937, 1, 0.984, 0.957, 0.969, 1, 1, 0.984, 0.925, 0.953, 0.965, 1, 0.451, 0, 0.0667, 0, 0, 0, 0, 0.0235, 0, 0.00392, 0, 0.00784, 0.0431, 0, 0.00784, 0.992, 0.98, 0.898, 1, 0.976, 1, 0.988, 0.651, 0.78, 1, 1, 0.984, 0.992, 0.875, 0.114, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0.0392, 0.00392, 0, 0.204, 0.925, 0.992, 1, 0.941, 1, 1, 0.451, 0.0392, 0.0667, 0.337, 0.894, 1, 1, 0.996, 0.0431, 0.0118, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0.00392, 0, 0.0627, 0.604, 1, 1, 0.976, 0.988, 1, 0.467, 0.0314, 0, 0, 0, 0.263, 0.463, 0.835, 0.988, 0.392, 0, 0, 0, 0, 0, 0.0118, 0, 0, 0.0745, 0.0235, 0, 0.314, 0.957, 1, 0.945, 1, 0.541, 0, 0, 0.0431, 0, 0, 0.0549, 0, 0.0392, 0.69, 0.925, 1, 0.204, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0, 0.0314, 0.576, 1, 1, 0.933, 0.737, 0.106, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0.741, 1, 0.949, 0.655, 0, 0, 0, 0, 0, 0.0314, 0.0196, 0, 0, 0.0667, 0.804, 1, 0.933, 0.722, 0.0431, 0, 0.0314, 0, 0, 0.0392, 0, 0, 0, 0.00784, 0.294, 0.996, 0.973, 0.847, 0, 0, 0, 0, 0.0118, 0, 0, 0.0196, 0, 0, 0.855, 0.98, 1, 0.694, 0.0353, 0.0196, 0, 0.0549, 0.0314, 0, 0.00784, 0.0392, 0, 0.0157, 0, 0.631, 0.996, 0.78, 0, 0, 0, 0, 0.0314, 0, 0, 0.0353, 0, 0, 0.855, 0.957, 0.992, 0.306, 0.00392, 0, 0.0157, 0, 0, 0.0902, 0, 0.0118, 0, 0.00784, 0.0471, 0.424, 0.945, 0.835, 0, 0, 0, 0, 0.00784, 0, 0, 0, 0, 0, 0.863, 0.988, 1, 0.706, 0.0431, 0, 0.0235, 0, 0.0627, 0, 0.00392, 0, 0, 0.0353, 0, 0.647, 0.969, 0.851, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0, 0.0196, 0.784, 1, 0.949, 0.882, 0.0941, 0.0157, 0.0392, 0, 0.0392, 0, 0.0275, 0, 0, 0.0275, 0.0824, 0.953, 1, 0.741, 0, 0, 0, 0, 0, 0.00784, 0, 0.00392, 0.0196, 0, 0.659, 0.988, 1, 0.992, 0.835, 0.125, 0, 0, 0, 0.0157, 0, 0, 0.0431, 0, 0.663, 1, 0.996, 0.682, 0, 0, 0, 0, 0, 0.0314, 0, 0.0196, 0.0314, 0, 0.125, 0.635, 0.973, 0.984, 0.984, 0.851, 0.49, 0.0824, 0, 0.0353, 0, 0.0118, 0.00392, 0.29, 0.949, 0.965, 0.702, 0.0784, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0.051, 0.0157, 0, 0.114, 0.718, 0.961, 1, 1, 0.941, 0.663, 0.231, 0, 0.0235, 0, 0.298, 0.941, 0.973, 0.969, 0.0392, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0, 0.0196, 0, 0, 0.0824, 0.678, 0.992, 0.961, 0.992, 0.973, 0.918, 0.98, 0.988, 0.855, 0.961, 1, 0.894, 0.0549, 0.0706, 0.0275, 0, 0, 0, 0, 0, 0, 0.00392, 0.0157, 0, 0, 0.0118, 0.051, 0, 0.098, 0.255, 0.502, 0.886, 1, 1, 0.961, 0.961, 1, 1, 0.929, 0.384, 0.0118, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0, 0.0431, 0.0314, 0, 0, 0, 0.0353, 0, 0, 0.0314, 0.0784, 0.314, 0.729, 1, 1, 0.886, 0.761, 0.294, 0, 0.0392, 0, 0.0314, 0, 0, 0, 0, 0.0157, 0, 0, 0, 0, 0.00392, 0.00392, 0, 0.0431, 0, 0, 0.0392, 0.0118, 0.0314, 0.0431, 0, 0, 0.0392, 0, 0.0314, 0.0235, 0, 0.0588, 0, 0, 0, 0, 0, 0, 0.0196, 0.0157, 0, 0, 0, 0.0157, 0.0235, 0, 0, 0.0157, 0.0157, 0, 0, 0.0118, 0, 0, 0, 0.0157, 0, 0.0353, 0, 0, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0392, 0.0118, 0, 0, 0.00392, 0.0196, 0, 0, 0.0157, 0.00392, 0, 0.0157, 0, 0.0392, 0, 0, 0, 0.0275, 0, 0.0196, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0.0667, 0, 0, 0.0314, 0, 0, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0.0118, 0, 0.0706, 0.0157, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.0353, 0, 0.0196, 0, 0.0353, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0588, 0, 0, 0.0863, 0, 0.0784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0, 0.0275, 0.0353, 0.992, 0.235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0627, 0.0667, 0, 0.133, 0.973, 0.749, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0, 0.0196, 0.82, 0.867, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0118, 0.0235, 0.0157, 0.812, 1, 0.169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0.0235, 0, 0.0314, 0.671, 1, 0.522, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.0157, 0.0275, 0, 0.302, 0.906, 0.835, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.00784, 0.0431, 0.431, 1, 0.761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0, 0.0157, 0.294, 1, 0.82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0275, 0.00784, 0, 0.231, 0.98, 0.812, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0314, 0.0118, 0.0549, 0.463, 1, 0.741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0, 0, 0, 0.0157, 0.455, 0.992, 0.788, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.0431, 0, 0.0157, 0.678, 0.988, 0.459, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0, 0.0314, 0, 0.812, 1, 0.733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0.0196, 0, 0, 0.859, 1, 0.439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.00392, 0, 0.0941, 0.922, 0.973, 0.161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0.0471, 0.502, 1, 0.89, 0.0745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0, 0.0314, 0.816, 0.996, 0.686, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.871, 0.996, 0.529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0.0353, 0.0118, 0.737, 0.98, 0.427, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0.482, 0.816, 0.224, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.00784, 0, 0, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.00784, 0.0314, 0.0353, 0.0235, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.0275, 0, 0, 0, 0, 0.0667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.118, 0, 0, 0.0706, 0, 0.0627, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.106, 0, 0, 0.051, 0, 0, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0.494, 0.976, 0.553, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0.00784, 0.839, 1, 0.922, 0.439, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.537, 0.976, 0.953, 0.933, 0.129, 0.0157, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0.0275, 0.00392, 0, 0.0471, 0.0275, 0, 0.267, 0.976, 0.961, 1, 0.239, 0.00392, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.051, 0, 0, 0.0353, 0, 0, 0.176, 0.776, 0.976, 1, 0.651, 0.0902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0.00784, 0.00784, 0.0667, 0, 0.796, 0.996, 1, 0.937, 0.263, 0, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0.00784, 0, 0.00784, 0.0392, 1, 0.949, 1, 0.502, 0.0392, 0, 0.0275, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0.0118, 0.00784, 0.522, 0.957, 0.992, 0.682, 0.0353, 0, 0, 0.0275, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0549, 0.00784, 0.533, 0.949, 1, 0.835, 0.153, 0, 0.0471, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.00392, 0, 0.00392, 0.141, 0.957, 0.949, 0.988, 0.4, 0, 0.0235, 0.0784, 0, 0, 0.00392, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0157, 0, 0.624, 1, 1, 0.847, 0.0902, 0.051, 0, 0, 0, 0.0627, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0235, 0.443, 0.953, 1, 0.992, 0.0902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.118, 0.859, 1, 1, 0.545, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.0902, 0.859, 1, 0.933, 0.541, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.282, 1, 0.941, 1, 0.161, 0.0863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.369, 0.965, 0.953, 0.773, 0.114, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.906, 0.98, 1, 0.365, 0, 0.0118, 0.0588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.149, 0.894, 0.988, 0.522, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.408, 0.961, 1, 0.443, 0.00784, 0.00392, 0.0353, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.89, 0.882, 0.247, 0, 0, 0.0118, 0.00392, 0.0314, 0, 0, 0.0118, 0, 0, 0.00392, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0667, 0, 0, 0, 0.0275, 0.0549, 0, 0, 0, 0, 0, 0.0118, 0.0314, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0.051, 0, 0, 0.0275, 0, 0.0275, 0.051, 0.0196, 0, 0, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0.0431, 0, 0, 0, 0.0118, 0.0196, 0, 0, 0, 0.0235, 0.051, 0, 0, 0.0314, 0, 0, 0, 0 ],
    #     [0, 0.00392, 0.00784, 0, 0, 0.00784, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0275, 0, 0, 0, 0.0118, 0.0824, 0.0196, 0, 0.00784, 0.0314, 0.0275, 0.0196, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.0471, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0, 0.0314, 0.00392, 0.0314, 0, 0.00392, 0, 0.00784, 0.0275, 0.0157, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.00392, 0, 0.0392, 0, 0.0157, 0, 0.00392, 0.0196, 0, 0.0392, 0.0902, 0.627, 1, 1, 0.549, 0.00392, 0.00392, 0.051, 0, 0.0118, 0.0118, 0.00784, 0, 0.0118, 0.0157, 0.00392, 0, 0, 0.0196, 0, 0, 0.0824, 0.0745, 0, 0.00392, 0, 0.157, 0.761, 1, 1, 0.941, 0.914, 1, 0.306, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0, 0.0275, 0.0353, 0, 0, 0.0431, 0, 0.576, 0.988, 1, 0.608, 0.157, 0.0745, 0.29, 0.984, 1, 0.522, 0, 0.0627, 0.0275, 0, 0, 0.00784, 0, 0, 0, 0.0431, 0, 0, 0, 0, 0.0353, 0.0706, 0.106, 0.875, 1, 0.741, 0.271, 0.00784, 0, 0, 0, 0.871, 1, 1, 0.918, 0.373, 0, 0.051, 0.00784, 0.0118, 0.00392, 0.00392, 0.0118, 0, 0, 0.0157, 0.0588, 0, 0.0118, 0.0745, 0.984, 0.984, 0.471, 0.0627, 0, 0.0118, 0.0118, 0.00392, 0.0157, 0.596, 0.592, 0.396, 0.984, 1, 0.486, 0.0196, 0, 0, 0, 0.0118, 0.0275, 0.0118, 0.00392, 0.0353, 0, 0, 0.0706, 0.804, 1, 0.639, 0.0902, 0, 0, 0.0235, 0, 0, 0.0235, 0.0706, 0.322, 0, 0.0667, 0.486, 1, 0.882, 0, 0, 0, 0, 0.0118, 0, 0.0667, 0, 0, 0.051, 0.204, 1, 0.663, 0.0353, 0, 0.0471, 0.0314, 0.0471, 0.0392, 0.0118, 0, 0.0157, 0, 0, 0, 0, 0.439, 1, 0.808, 0.0667, 0.0118, 0, 0, 0.0235, 0, 0, 0.0706, 0, 0.914, 1, 0, 0.00784, 0, 0.0549, 0, 0, 0, 0, 0.0392, 0, 0.0471, 0.0196, 0, 0.0392, 0, 0.627, 1, 0.157, 0.0588, 0, 0, 0, 0, 0, 0.00392, 0.00392, 1, 0.847, 0.0275, 0.0471, 0, 0, 0.0549, 0, 0, 0.0314, 0.00392, 0.00784, 0, 0.0392, 0.00392, 0, 0, 0.051, 0.98, 0.639, 0.00784, 0, 0, 0.0118, 0.00392, 0, 0.0235, 0, 0.949, 0.706, 0, 0, 0.0588, 0.0118, 0, 0.00784, 0.0471, 0, 0, 0.0275, 0, 0, 0, 0.051, 0.0235, 0, 0.522, 0.827, 0.0392, 0, 0, 0, 0.00392, 0, 0.0157, 0.0588, 1, 0.886, 0.0431, 0.0118, 0, 0, 0.00784, 0, 0, 0.0275, 0, 0, 0.0275, 0, 0, 0.00784, 0.0196, 0, 0.325, 0.965, 0.0275, 0, 0, 0, 0.0431, 0, 0, 0.00392, 0.792, 0.976, 0, 0.0431, 0, 0.00392, 0.0784, 0, 0, 0.0275, 0, 0, 0.0471, 0.0235, 0, 0, 0, 0.0157, 0.357, 0.988, 0, 0.00784, 0.0196, 0, 0.0235, 0.0392, 0, 0, 0.427, 1, 0.216, 0, 0.0353, 0.0314, 0, 0.0157, 0.00392, 0, 0, 0, 0.00392, 0, 0.0157, 0.0275, 0, 0, 0.337, 1, 0, 0.00392, 0.00784, 0, 0, 0.0235, 0.0235, 0, 0.173, 1, 0.871, 0.0549, 0, 0, 0, 0.0157, 0, 0, 0.00392, 0.00784, 0, 0, 0.0118, 0.0353, 0, 0, 0.533, 0.98, 0.0431, 0, 0, 0.0392, 0, 0.00392, 0, 0, 0, 0.604, 1, 0.569, 0.106, 0, 0.0353, 0, 0, 0.0235, 0.0118, 0.00784, 0, 0.00784, 0, 0, 0, 0.0706, 0.867, 0.757, 0.0235, 0, 0, 0, 0, 0.0353, 0, 0.0431, 0, 0.157, 0.851, 1, 0.58, 0.0863, 0, 0.0157, 0.0235, 0, 0, 0.0118, 0, 0.0314, 0, 0.00392, 0.00784, 0.102, 0.992, 0.482, 0, 0.0118, 0.0157, 0, 0.00784, 0.00392, 0, 0, 0, 0.0392, 0.0745, 0.902, 0.996, 0.549, 0.0275, 0, 0.00392, 0, 0.0471, 0.0471, 0, 0, 0, 0, 0.18, 0.91, 0.867, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0, 0, 0, 0.0157, 0.0353, 0, 0.565, 0.98, 0.969, 0.706, 0.322, 0.0275, 0, 0, 0.00392, 0.114, 0.376, 0.867, 1, 0.89, 0.153, 0, 0, 0, 0, 0, 0, 0.0118, 0.00784, 0, 0, 0, 0, 0, 0.00392, 0.31, 0.757, 0.984, 1, 1, 1, 1, 0.98, 0.973, 1, 1, 0.537, 0.173, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.00784, 0, 0, 0, 0, 0, 0.0157, 0.0392, 0.0235, 0.122, 0.439, 0.667, 0.639, 0.612, 0.741, 0.663, 0.235, 0.0588, 0, 0.0118, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0157, 0, 0, 0, 0.0314, 0.00784, 0, 0.0157, 0.0157, 0, 0.0196, 0.0314, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0157, 0, 0.0784, 0.051, 0, 0, 0.0118, 0, 0.00392, 0.0706, 0, 0, 0, 0, 0.00784, 0.0471, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00392, 0.00392, 0.0275, 0, 0, 0.0157, 0.0667, 0.00784, 0, 0.0235, 0, 0, 0, 0.00784, 0, 0.0157, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0, 0, 0, 0.0471, 0.0157, 0, 0, 0.0314, 0, 0, 0.0588, 0, 0, 0, 0, 0, 0.0392, 0, 0, 0, 0 ],
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0.0667, 0, 0.0196, 0, 0, 0, 0.314, 0.973, 0.973, 0.29, 0, 0.0784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0.0392, 0.00392, 0.0314, 0, 0, 0.0353, 0.376, 0.757, 0.984, 0.988, 0.792, 0.471, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0, 0.0667, 0.0235, 0.0549, 0.353, 0.737, 1, 0.988, 1, 1, 0.957, 0.102, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0706, 0, 0.0314, 0, 0.184, 0.957, 0.933, 0.98, 1, 0.753, 0.816, 0.949, 1, 0.173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0.329, 0.78, 0.992, 0.902, 0.992, 1, 0.98, 0.369, 0.431, 1, 0.937, 0.0824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0118, 0.361, 0.922, 0.996, 1, 1, 1, 0.969, 1, 0.357, 0.298, 1, 0.969, 0.157, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.404, 0.89, 1, 0.937, 0.898, 0.855, 0.992, 1, 0.894, 0.376, 0.376, 0.961, 0.973, 0.176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.322, 0.961, 0.988, 0.996, 0.706, 0, 0.392, 0.976, 0.953, 0.451, 0.0118, 0.345, 1, 0.992, 0.0902, 0.0235, 0, 0, 0, 0, 0, 0.0471, 0, 0.0314, 0, 0, 0.0157, 0.4, 0.941, 0.988, 0.969, 0.745, 0.141, 0.0157, 0.369, 0.812, 0.416, 0.0275, 0.0118, 0, 0.847, 0.957, 0.102, 0.0314, 0, 0, 0, 0, 0, 0, 0.0392, 0.0314, 0, 0.102, 0, 0.847, 0.957, 0.973, 0.502, 0, 0, 0.098, 0.125, 0.0941, 0, 0, 0.0157, 0.0314, 0.757, 1, 0.098, 0, 0, 0, 0, 0, 0.0235, 0, 0.0549, 0, 0, 0, 0.224, 0.918, 1, 0, 0, 0.051, 0, 0, 0.0667, 0, 0.0392, 0.0275, 0, 0, 0.745, 0.988, 0.114, 0.00392, 0, 0, 0, 0, 0.00784, 0.0157, 0, 0.00784, 0.0431, 0.18, 0.867, 0.945, 0.961, 0, 0.0431, 0, 0.0235, 0.0314, 0, 0.0392, 0, 0.0157, 0, 0.102, 0.776, 0.953, 0.122, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0, 0.71, 1, 1, 1, 0, 0.00784, 0.0549, 0, 0, 0.0118, 0, 0.00784, 0.00784, 0.00392, 0.341, 0.976, 1, 0.196, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0, 0.804, 0.898, 0.973, 0.416, 0.0627, 0, 0, 0, 0.0275, 0.051, 0, 0, 0.0588, 0.137, 0.933, 1, 0.89, 0.051, 0, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0.0784, 0.745, 1, 1, 0.337, 0.051, 0, 0.0235, 0.051, 0, 0, 0.0314, 0.00784, 0.169, 0.804, 1, 0.882, 0.157, 0.0275, 0.0314, 0, 0, 0, 0, 0.0431, 0, 0.00392, 0.0392, 0, 0.792, 1, 0.984, 0.996, 0, 0, 0.00784, 0, 0.0431, 0, 0.0471, 0.698, 0.949, 1, 0.894, 0.141, 0.0157, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.051, 0.0196, 0, 0.376, 0.906, 0.976, 0.933, 0.831, 0.737, 0.373, 0.361, 0.686, 0.729, 0.796, 1, 0.886, 0.557, 0.165, 0, 0.0118, 0.0196, 0, 0, 0, 0, 0, 0.0157, 0.0667, 0, 0, 0.0235, 0, 0.22, 0.953, 0.98, 0.945, 0.965, 1, 1, 0.98, 0.953, 0.976, 0.757, 0.349, 0, 0, 0.0157, 0.0196, 0.00784, 0.0353, 0, 0, 0, 0, 0, 0.0549, 0, 0, 0.0588, 0, 0, 0.165, 0.561, 0.949, 1, 0.98, 0.965, 0.996, 0.98, 0.643, 0.0902, 0.00784, 0, 0.0235, 0.0235, 0, 0, 0.00784, 0, 0, 0, 0, 0.0196, 0, 0.00784, 0.0471, 0, 0, 0.0314, 0, 0, 0.416, 0.788, 1, 0.816, 0.306, 0.141, 0.0314, 0, 0.0196, 0.0392, 0, 0, 0.0235, 0.0196, 0, 0, 0, 0, 0, 0.0392, 0, 0, 0.0235, 0, 0, 0.0157, 0.0588, 0, 0.0275, 0, 0, 0, 0.0471, 0.0784, 0, 0.0431, 0.00784, 0, 0, 0, 0.0275, 0.0275, 0.00784, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0.0196, 0.0627, 0.0196, 0, 0.00784, 0, 0.051, 0, 0, 0, 0, 0.00784, 0, 0, 0.0196, 0.0392, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0.0353, 0, 0, 0, 0.0353, 0.0157, 0, 0, 0.0157, 0.0471, 0.0471, 0, 0, 0.0235, 0.0196, 0, 0, 0, 0.0275, 0.0157, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0.0431, 0.0196, 0, 0, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0.0118, 0, 0, 0.0157, 0.0353, 0.00784, 0, 0, 0.0275, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0, 0.00392, 0, 0, 0, 0.0118, 0, 0.0235, 0.0431, 0.0118, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0118, 0.0118, 0.0196, 0.00392, 0.00392, 0.00784, 0.00784, 0, 0, 0, 0, 0.00392, 0.00784, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0.0118, 0, 0, 0, 0, 0, 0.00784, 0.0314, 0.0118, 0, 0, 0, 0.0196, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0, 0, 0, 0.00392, 0.00392, 0, 0.00392, 0, 0.00392, 0.0157, 0.0157, 0.0275, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0627, 0.051, 0.0353, 0, 0, 0.0118, 0.0549, 0, 0, 0.00784, 0.0275, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.051, 0, 0.00784, 0, 0, 0.0941, 0.671, 0.976, 0.412, 0.00784, 0.00784, 0.00392, 0.0118, 0, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.00392, 0, 0.0353, 0.718, 1, 0.824, 0.0824, 0.0314, 0, 0.00784, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.00392, 0.0157, 0.0431, 0.773, 1, 1, 0.0627, 0, 0, 0.0118, 0.00784, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0118, 0, 0.0118, 0.00392, 0.00784, 0.831, 0.969, 0.502, 0.0196, 0.0353, 0, 0, 0.0196, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0.00784, 0.0118, 0.149, 0.953, 1, 0.565, 0.0118, 0, 0, 0, 0.0196, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.00392, 0, 0, 0.271, 0.937, 0.965, 0.643, 0.0235, 0, 0, 0, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0.051, 0.502, 1, 0.984, 0.643, 0.0353, 0, 0.0157, 0, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0, 0.0118, 0.0314, 0.616, 1, 1, 0.494, 0.0118, 0, 0.0275, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0, 0.671, 0.945, 0.973, 0.263, 0, 0, 0.00784, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0196, 0.0314, 0.902, 1, 0.933, 0.0824, 0, 0.0157, 0, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0.00392, 0.0745, 1, 0.949, 0.686, 0.00784, 0.00392, 0.0431, 0, 0.0118, 0.00784, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125, 0.953, 1, 0.714, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.0157, 0.0745, 0.996, 0.949, 0.667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0157, 0.00392, 0.0196, 1, 1, 0.718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0, 0, 0.0745, 1, 1, 0.58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0118, 0.278, 0.937, 0.961, 0.267, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0.0157, 0.502, 0.973, 1, 0.212, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0, 0.596, 1, 1, 0.224, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0235, 0, 0.584, 0.957, 0.78, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0, 0.404, 0.882, 0.576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0235, 0, 0.00392, 0.051, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.0353, 0.00392, 0.0118, 0.0549, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0157, 0, 0, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0, 0.0118, 0.0549, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0157, 0, 0.0235, 0, 0.0196, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0, 0.0275, 0, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0.0275, 0.0471, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0.0275, 0.486, 0.957, 0.973, 0.235, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.392, 0.961, 1, 0.961, 0.298, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.451, 1, 0.992, 0.992, 0.239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0549, 0.0235, 0, 0.0275, 0, 0, 0.275, 0.925, 1, 0.839, 0.0549, 0.0667, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0627, 0, 0, 0.051, 0, 0.051, 0.255, 0.855, 1, 0.933, 0.396, 0.0275, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0, 0.0118, 0, 0.643, 1, 1, 0.733, 0.0118, 0, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0118, 0.0471, 0, 0.0314, 0.149, 0.933, 0.973, 0.855, 0.318, 0, 0, 0, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0157, 0.0196, 0.0157, 0.0627, 0.788, 1, 1, 0.502, 0, 0.00392, 0.0588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0627, 0, 0, 0, 0.443, 1, 0.945, 0.675, 0.11, 0, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0.0588, 0.31, 0.894, 0.937, 0.765, 0.149, 0, 0.0588, 0, 0, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0.133, 0.976, 1, 0.992, 0.639, 0, 0.00784, 0, 0.0353, 0.0118, 0, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.0235, 0, 0.6, 0.945, 1, 0.937, 0.184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.247, 0.863, 0.976, 0.976, 0.69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0.00392, 0.831, 0.973, 1, 0.741, 0.0941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 1, 1, 0.965, 0.349, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.58, 0.969, 0.949, 1, 0.11, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.976, 1, 0.976, 0.518, 0.0118, 0.0784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.165, 0.992, 1, 0.784, 0.0627, 0.0196, 0, 0.0627, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.294, 0.933, 1, 0.882, 0, 0.00392, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0824, 0.769, 1, 0.824, 0.0549, 0.0353, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0, 0.0196, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.00784, 0.0471, 0.0196, 0, 0.0157, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.051, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0, 0.0157, 0.00784, 0.0314, 0, 0, 0.00392, 0.00392, 0.00392, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.0157, 0.0353, 0.0157, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.00392, 0, 0, 0, 0.0196, 0.0353, 0.0392, 0.0118, 0.00784, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.00784, 0.0275, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0.00392, 0.0314, 0, 0.00784, 0.0588, 0, 0.0353, 0, 0.00392, 0, 0.0118, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0667, 0, 0.0196, 0, 0.255, 0.349, 0.808, 1, 0.945, 0.384, 0.0353, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0.051, 0, 0, 0.357, 0.839, 1, 1, 0.929, 0.969, 0.914, 0.639, 0.0667, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.0314, 0.949, 1, 0.937, 0.98, 1, 1, 1, 0.918, 0.392, 0.122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0627, 0.0392, 0, 0.0353, 0.922, 0.98, 1, 0.992, 0.976, 0.984, 1, 0.831, 0.0902, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.678, 1, 0.984, 1, 0.976, 1, 0.988, 0.98, 0.808, 0.0196, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0.0118, 0, 0.361, 0.91, 0.929, 0.984, 0.984, 0.867, 0.996, 0.976, 0.961, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0.914, 0.992, 1, 1, 0.776, 0.169, 0.71, 0.988, 0.98, 0.776, 0, 0.0118, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0.0314, 0, 0.0314, 0.0353, 0, 0, 0, 0.216, 0.608, 0.925, 0.996, 0.757, 0.718, 0.0784, 0.0118, 0.333, 0.965, 1, 0.796, 0.106, 0, 0.0235, 0, 0, 0, 0, 0.00784, 0, 0.00392, 0.0314, 0.0275, 0, 0, 0.0157, 0.00784, 0.875, 0.961, 1, 0.557, 0.0549, 0, 0, 0.0275, 0.0157, 0.69, 1, 1, 0.396, 0, 0, 0, 0, 0, 0, 0.00392, 0.00392, 0, 0.00784, 0, 0.0157, 0.0157, 0, 0.482, 0.918, 0.976, 0.529, 0.0392, 0.0118, 0.0353, 0, 0, 0.0275, 0.169, 1, 0.988, 0.373, 0.0471, 0, 0, 0, 0, 0, 0.0157, 0.0314, 0, 0, 0, 0.0157, 0, 0.137, 0.796, 1, 0.969, 0.0588, 0, 0, 0, 0, 0.0157, 0, 0.216, 0.949, 0.969, 0.431, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.0353, 0.0549, 0.0196, 0.0196, 0.737, 0.996, 0.925, 0.263, 0.0157, 0.0627, 0.0314, 0.051, 0, 0.0275, 0, 0.125, 1, 0.937, 0.431, 0, 0.00392, 0, 0, 0, 0, 0.0235, 0.0196, 0.0353, 0, 0, 0, 0.00784, 0.804, 1, 0.675, 0, 0, 0.0275, 0, 0, 0.0706, 0, 0.0471, 0.18, 1, 0.894, 0.118, 0, 0.0353, 0, 0, 0, 0, 0, 0.0196, 0.0118, 0, 0.0392, 0.102, 0.447, 1, 0.992, 0.141, 0.0588, 0.0353, 0, 0.0431, 0.0392, 0, 0.0196, 0.224, 0.898, 0.902, 0.78, 0, 0, 0.0275, 0, 0, 0, 0, 0.00392, 0.0118, 0, 0.0235, 0, 0, 0.482, 0.961, 0.965, 0.208, 0, 0.00784, 0.0157, 0, 0, 0, 0.176, 0.886, 1, 0.91, 0.224, 0.0118, 0, 0.00392, 0, 0, 0, 0, 0.00392, 0.00392, 0, 0, 0.00784, 0.0471, 0.392, 0.992, 1, 0.098, 0.0549, 0, 0.0392, 0, 0.0157, 0.518, 0.827, 1, 0.82, 0.259, 0.00784, 0, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0.0314, 0.0118, 0.0196, 0, 0.0824, 0.82, 0.992, 0.922, 0.831, 0.302, 0.188, 0.29, 0.639, 1, 1, 0.757, 0.306, 0.0275, 0, 0.0275, 0.00784, 0, 0, 0, 0, 0, 0.0235, 0, 0.0157, 0, 0.0275, 0, 0.0275, 0.745, 0.976, 1, 0.973, 0.98, 1, 0.922, 1, 0.937, 0.804, 0.282, 0, 0, 0, 0.0118, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0, 0.0471, 0.0314, 0, 0.224, 0.702, 0.914, 1, 1, 0.953, 0.965, 1, 0.447, 0.149, 0, 0, 0.051, 0, 0, 0.0353, 0.0118, 0, 0, 0, 0, 0, 0.0157, 0.00392, 0, 0, 0, 0, 0, 0.0471, 0.251, 0.678, 0.953, 0.671, 0.302, 0.278, 0, 0, 0, 0.0118, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0.0196, 0, 0, 0.129, 0.0314, 0, 0.0157, 0.0353, 0.0196, 0.00392, 0, 0.0275, 0.0549, 0.0118, 0.0353, 0, 0, 0.0353, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0588, 0.00392, 0, 0.00784, 0, 0, 0.0235, 0, 0, 0, 0.0392, 0, 0, 0.0157, 0.0118, 0, 0.00392, 0.0431, 0, 0, 0.0196, 0, 0, 0, 0, 0.0275, 0.00392, 0.0196, 0, 0, 0.0431, 0, 0.00392, 0.00392, 0, 0.0275, 0.0588, 0, 0, 0.0392, 0, 0, 0, 0, 0.00784, 0, 0, 0.00392, 0.0157, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0.0235, 0, 0, 0.0392, 0.00392, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0275, 0, 0, 0.0471, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0, 0, 0.0157, 0, 0.0235, 0, 0.0667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0314, 0, 0.0588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0118, 0, 0.00392, 0, 0.286, 0.98, 0.882, 0.302, 0.0471, 0.0275, 0.00784, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0.00392, 0.0196, 0, 0.00392, 0.278, 1, 0.988, 0.922, 0.694, 0.0431, 0, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0.0196, 0, 0, 0.11, 0.631, 0.933, 0.91, 1, 0.953, 0.529, 0, 0.0157, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.00392, 0.271, 0.929, 0.922, 1, 0.973, 1, 0.922, 0.325, 0, 0, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0275, 0, 0.0196, 0.0196, 0.192, 0.886, 1, 0.992, 0.91, 0.914, 1, 0.914, 0.286, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0.541, 1, 0.996, 1, 0.973, 0.957, 1, 1, 0.937, 1, 0.455, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0.00392, 0.00784, 0.557, 0.961, 0.969, 0.984, 0.988, 0.663, 0.38, 0.616, 0.992, 1, 0.745, 0.0941, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0.392, 0.965, 0.961, 0.914, 0.867, 0.627, 0.0784, 0.00784, 0.549, 0.992, 0.976, 0.973, 0.133, 0.0235, 0, 0, 0, 0, 0.0157, 0, 0.0118, 0.00392, 0, 0.0314, 0.0275, 0, 0.157, 0.851, 0.953, 0.984, 0.973, 0.608, 0.00784, 0, 0, 0.149, 1, 1, 0.953, 0.976, 0.118, 0, 0, 0, 0, 0, 0.0314, 0, 0.00784, 0.00784, 0, 0, 0, 0.0667, 0.4, 0.996, 1, 1, 0.933, 0.518, 0, 0.00784, 0.0235, 0.0353, 0.557, 1, 0.953, 1, 0.647, 0, 0, 0, 0, 0, 0, 0.0314, 0.0157, 0, 0.0118, 0.0745, 0, 0, 0.643, 0.929, 0.957, 0.867, 0.098, 0.0941, 0, 0.00392, 0, 0.00392, 0.588, 0.949, 1, 0.788, 0.0392, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.0431, 0, 0.0235, 0.565, 0.992, 1, 0.965, 0.416, 0.00392, 0, 0.0275, 0, 0.0157, 0.173, 1, 0.976, 0.804, 0.169, 0.0588, 0, 0, 0, 0, 0, 0.0157, 0.00784, 0, 0.0275, 0.0157, 0, 0.00392, 0.71, 0.945, 0.996, 0.969, 0.459, 0.114, 0, 0, 0, 0, 0.118, 0.941, 1, 0.204, 0.051, 0, 0, 0, 0, 0, 0, 0, 0.0471, 0, 0, 0, 0.0667, 0, 0.482, 1, 1, 0.992, 0.741, 0.00392, 0.0196, 0.00392, 0.153, 0.435, 0.533, 0.957, 0.694, 0.373, 0, 0.00392, 0.0196, 0, 0, 0, 0, 0.00784, 0, 0, 0.0353, 0, 0, 0.0353, 0.812, 0.973, 0.929, 0.769, 0.227, 0, 0, 0.0392, 0.243, 1, 0.914, 1, 0.247, 0, 0.0824, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0, 0.0392, 0, 0, 0, 0.682, 1, 0.973, 1, 0.549, 0.275, 0.62, 0.973, 1, 0.969, 1, 0.498, 0.0667, 0.00392, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0.0157, 0.0118, 0, 0.576, 0.996, 0.965, 1, 0.961, 1, 0.965, 1, 0.988, 1, 0.812, 0.341, 0, 0, 0.0353, 0.0235, 0, 0, 0, 0, 0, 0.0392, 0.00392, 0.00392, 0, 0, 0.0196, 0.098, 0.667, 0.957, 0.996, 0.996, 1, 0.98, 0.988, 0.953, 0.961, 0.792, 0.22, 0, 0.0196, 0.0275, 0, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0157, 0, 0, 0.141, 0.769, 0.992, 0.965, 0.992, 0.882, 1, 0.702, 0.4, 0, 0.0471, 0.0431, 0.00392, 0.00392, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0275, 0.00784, 0, 0, 0.0784, 0.761, 1, 0.976, 0.396, 0.267, 0.0667, 0.0118, 0, 0.0392, 0, 0, 0.0157, 0.0353, 0, 0.0392, 0, 0, 0, 0, 0, 0.0275, 0.0157, 0.00392, 0, 0, 0.051, 0.0314, 0.0235, 0.00392, 0, 0.00784, 0, 0.0471, 0, 0, 0.11, 0, 0, 0.0275, 0, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0.00392, 0.0235, 0, 0.0118, 0, 0, 0, 0.0235, 0, 0, 0.0196, 0.0118, 0, 0, 0.0314, 0, 0.00392, 0.0627, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0, 0.0392, 0.0196, 0.0196, 0.0275, 0, 0.0196, 0.0235, 0.0118, 0, 0.0196, 0, 0, 0.0275, 0.0353, 0, 0, 0.0431, 0, 0, 0.0784, 0, 0, 0, 0, 0.00784, 0, 0.0471, 0, 0.051, 0, 0, 0.0235, 0.0118, 0, 0, 0.0157, 0, 0.00392, 0, 0.0157, 0, 0, 0.0314, 0.0235, 0, 0.0196, 0.0431, 0, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.0196, 0, 0.0118, 0, 0.00784, 0.0196, 0, 0, 0.0118, 0, 0.0235, 0.00392, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.0157, 0.00784, 0.00784, 0.0471, 0.0275, 0, 0.0392, 0.0235, 0.0314, 0, 0, 0.00392, 0.0196, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.00392, 0, 0.0235, 0, 0, 0, 0, 0, 0.0157, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431, 0.0353, 0.00784, 0.0353, 0, 0, 0, 0.0549, 0.0275, 0.0471, 0, 0, 0.051, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0.0196, 0.0235, 0, 0, 0.0549, 0, 0, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.0863, 0.208, 0.227, 0.431, 0.514, 0.529, 0.388, 0.31, 0.349, 0.149, 0.11, 0, 0, 0.051, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0.00784, 0.0118, 0.0157, 0, 0.0118, 0.00784, 0.0863, 0.6, 0.0157, 0.0314, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0, 0.0706, 0, 0, 0.133, 0.42, 0.318, 0.0314, 0.114, 0.965, 0, 0, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0235, 0, 0.0627, 0, 0, 0.525, 0.984, 0.949, 0.506, 0.153, 0.894, 0, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0, 0, 0.0235, 0.714, 1, 1, 0.376, 0.161, 0.796, 0.0157, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00392, 0, 0, 0.0235, 0.275, 0.949, 1, 0.855, 0.184, 0.114, 0.467, 0.0235, 0, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.0118, 0, 0.0471, 0.627, 1, 0.973, 0.392, 0.051, 0.0392, 0.0863, 0.00784, 0, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0, 0, 0.00392, 0, 0.843, 0.996, 0.98, 0.22, 0.00784, 0.00784, 0, 0, 0.0118, 0, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0118, 0, 0.0784, 0.91, 0.961, 1, 0.318, 0, 0, 0.0196, 0, 0.00784, 0.0157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0196, 0, 0.439, 0.953, 1, 0.933, 0.227, 0, 0.0118, 0.0118, 0, 0, 0.0314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0.808, 0.996, 1, 0.761, 0, 0.0157, 0.0353, 0, 0.0431, 0, 0.0157, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.00784, 0.0157, 0.196, 0.984, 0.992, 0.961, 0.365, 0.00784, 0, 0, 0.122, 0.0235, 0, 0, 0.0235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.333, 0.965, 1, 0.702, 0, 0, 0.0118, 0.141, 0.322, 0.0157, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0.0275, 0.439, 1, 1, 0.627, 0.0196, 0, 0.0235, 0.192, 0.412, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0667, 0, 0.855, 0.957, 0.898, 0.247, 0.00392, 0.00392, 0, 0.0902, 0.243, 0, 0.0157, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.847, 1, 0.404, 0.00392, 0, 0, 0, 0, 0.0235, 0.0275, 0, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0.188, 0.918, 0.973, 0.216, 0, 0.0588, 0, 0.00784, 0.0118, 0, 0.0431, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0.702, 0.984, 0.871, 0.0667, 0.0314, 0, 0, 0.00392, 0.0157, 0, 0, 0, 0, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.867, 0.976, 0.404, 0.0118, 0.0196, 0, 0.0196, 0, 0, 0.0235, 0, 0.0235, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0.855, 0.702, 0, 0, 0.00392, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0, 0.051, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392, 0, 0, 0.00392, 0, 0, 0.0314, 0.0275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0.0863, 0, 0, 0.0235, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0, 0.0745, 0, 0, 0, 0.0471, 0, 0.0275, 0.0118, 0, 0, 0.0275, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0.0118, 0, 0, 0.00784, 0.0353, 0, 0.00784, 0, 0, 0.0353, 0, 0, 0, 0.0118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0196, 0.0196, 0.0745, 0, 0.00784, 0.0235, 0.0549, 0, 0, 0.0275, 0, 0.0275, 0.051, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.0118, 0, 0, 0, 0.00784, 0, 0, 0.00392, 0.0314, 0, 0, 0, 0, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0.0196, 0, 0, 0, 0.0549, 0, 0.0667, 0, 0, 0.0706, 0.0471, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0, 0.0667, 0, 0, 0.161, 0.592, 1, 0.957, 0.502, 0, 0, 0.0235, 0.0353, 0.0431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0549, 0, 0, 0.0824, 0.365, 0.686, 1, 0.91, 1, 0.941, 0.463, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0275, 0, 0, 0.263, 0.804, 0.973, 0.984, 0.98, 1, 0.937, 1, 0.937, 0.459, 0.0627, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0, 0, 0.0392, 0, 0, 0.0392, 0, 0.00392, 0.263, 0.863, 1, 1, 0.878, 1, 0.976, 1, 0.918, 0.957, 1, 0.478, 0, 0.0314, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0.592, 1, 0.973, 0.949, 1, 0.98, 0.984, 0.925, 1, 1, 1, 0.804, 0.0392, 0, 0, 0, 0, 0, 0, 0.00784, 0.00392, 0, 0, 0.0275, 0.0235, 0, 0.0118, 0.502, 0.933, 0.98, 1, 0.961, 0.992, 0.976, 1, 0.984, 1, 0.906, 0.941, 0.933, 0.173, 0.0196, 0, 0, 0, 0, 0, 0, 0.0118, 0, 0.0118, 0.0549, 0.0196, 0, 0.0353, 0.914, 1, 1, 0.969, 1, 1, 0.992, 1, 0.922, 0.98, 1, 1, 0.89, 0.565, 0.00392, 0, 0, 0, 0, 0.00784, 0, 0.0196, 0, 0, 0, 0, 0, 0.0235, 1, 0.961, 0.984, 0.945, 0.922, 0.929, 1, 0.984, 0.584, 0.11, 0.353, 1, 1, 0.89, 0.224, 0, 0, 0, 0, 0.00784, 0, 0.0196, 0, 0, 0, 0, 0.263, 0.878, 1, 1, 0.851, 0.329, 0.4, 1, 0.949, 0.467, 0.051, 0, 0.329, 0.933, 0.969, 0.984, 0.286, 0, 0, 0, 0, 0, 0, 0.0196, 0, 0.0196, 0.00392, 0.161, 0.729, 0.969, 0.937, 0.957, 0.424, 0.0118, 0, 0.302, 0.251, 0.0549, 0, 0, 0.294, 0.98, 0.918, 0.718, 0.0941, 0, 0, 0, 0, 0.00392, 0, 0.0235, 0, 0.0118, 0.0118, 0.282, 1, 0.98, 1, 0.957, 0.133, 0, 0.0275, 0, 0.0275, 0, 0.0471, 0, 0.318, 0.969, 1, 0.976, 0.275, 0, 0, 0, 0, 0, 0.0588, 0, 0.0314, 0, 0.333, 0.91, 0.98, 0.984, 1, 0.984, 0.486, 0, 0.0157, 0.0392, 0, 0, 0.0353, 0, 0.294, 0.965, 0.992, 1, 0.255, 0, 0, 0, 0, 0, 0, 0.0157, 0, 0.271, 0.929, 1, 0.984, 0.98, 0.839, 1, 1, 0.553, 0.0314, 0, 0.0353, 0.00784, 0.0392, 0.125, 0.914, 0.984, 0.992, 0.533, 0.0157, 0, 0, 0, 0, 0.0275, 0, 0.0941, 0, 0.267, 1, 0.941, 0.976, 0.718, 0.145, 0.576, 0.506, 0.561, 0.133, 0.0235, 0, 0, 0.431, 1, 0.98, 1, 0.478, 0.0157, 0.0549, 0, 0, 0, 0, 0, 0, 0, 0, 0.349, 0.996, 0.937, 0.733, 0.0549, 0, 0, 0.0235, 0, 0, 0, 0.0627, 0.153, 0.776, 0.984, 0.976, 0.58, 0.0824, 0, 0, 0, 0, 0, 0, 0.00784, 0.00784, 0, 0.0157, 0.31, 0.945, 0.988, 0.588, 0, 0, 0.0235, 0, 0, 0.0824, 0.00784, 0.671, 1, 0.988, 0.949, 0.0627, 0.0392, 0, 0, 0, 0, 0, 0, 0, 0.0235, 0, 0.051, 0, 0.227, 0.973, 1, 0.769, 0.604, 0, 0.00392, 0.0588, 0, 0, 0.761, 0.949, 0.933, 0.808, 0.443, 0, 0.0157, 0.0745, 0, 0.0667, 0, 0, 0, 0, 0, 0, 0, 0.0353, 0.294, 0.914, 1, 0.937, 0.925, 0.859, 0.847, 0.851, 0.843, 0.882, 0.906, 0.949, 0.765, 0.0824, 0, 0.0392, 0, 0.0471, 0, 0.0353, 0, 0, 0, 0, 0.0157, 0.0471, 0, 0, 0.00392, 0.267, 0.635, 1, 0.988, 1, 1, 0.933, 0.984, 0.992, 0.992, 0.416, 0.00784, 0, 0, 0.00784, 0.00784, 0, 0.0431, 0, 0, 0, 0, 0, 0.0353, 0, 0, 0.00784, 0.0157, 0, 0.204, 0.933, 0.969, 0.961, 1, 1, 0.878, 0.561, 0.275, 0.137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0314, 0.0353, 0, 0.0118, 0.0157, 0, 0.0627, 0.0627, 0.0118, 0, 0.0196, 0.0196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392, 0.00784, 0, 0, 0, 0.0471, 0, 0, 0, 0, 0, 0, 0.0118, 0.00784, 0, 0.00392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784, 0, 0, 0.0314, 0.0118, 0, 0, 0.0314, 0.0235, 0.0314, 0.0275, 0.0118, 0.00784, 0.00784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    #     ]

    # ys=[0,0,0,0,0,0,0,1,1,1,1,1,1,1]

    print(len(xs))
    print(len(xs[0]))
    print(len(ys))
    test_set_x, test_set_y = shared_dataset([xs,ys])
    valid_set_x, valid_set_y = shared_dataset([xs,ys])
    train_set_x, train_set_y = shared_dataset([xs,ys])


    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    

    # compute number of minibatches for training, validation and testing
    batch_size=1202
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    # n_train_batches /= batch_size
    # n_valid_batches /= batch_size
    # n_test_batches /= batch_size
    n_train_batches = 1
    n_valid_batches = 1
    n_test_batches = 1

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (28, 28)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=20, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=20, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    bestConvW=layer0.W.get_value();



    while (epoch < n_epochs) and (not done_looping):

        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            print cost_ij
            # print learning_rate
            print "layer 0 weights"
            print layer0.W.get_value()
            print "layer 1 weights"
            print layer1.W.get_value()
            print "layer 2 weights"
            print layer2.W.get_value()
            print "log reg layer weights"
            print layer3.W.get_value()


            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    bestConvW=layer0.W.get_value();
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # print "bestConvW" + str(bestConvW);
    # print sum(sum(sum(sum(bestConvW))))
    # print cost


if __name__ == '__main__':
    # evaluate_lenet5()
    # evaluate_test1()
    # evaluate_test2()
    evaluate_mnist_1()


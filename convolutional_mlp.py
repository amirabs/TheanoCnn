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
from theano import pp

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
        self.W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        print self.W_bound
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-self.W_bound, high=self.W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        # W_bound = 0.1
        # self.W = theano.shared(numpy.asarray(
        #     rng.uniform(low=W_bound, high=W_bound, size=filter_shape),
        #     dtype=theano.config.floatX),
        #                        borrow=True)
    

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=self.conv_out,
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

def evaluate_mnist_1(learning_rate=0.1, n_epochs=100,
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
    rng = numpy.random.RandomState(3)
    xs=[]
    ys=[]
    # f = open('temp_value', 'r+')
    # f = open('out_10', 'r+')
    f = open('out_10_10', 'r+')


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


    print(len(xs))
    print(len(xs[0]))
    print(len(ys))
    # print(ys)
    # print(xs)

    test_set_x, test_set_y = shared_dataset([xs,ys])
    valid_set_x, valid_set_y = shared_dataset([xs,ys])
    train_set_x, train_set_y = shared_dataset([xs,ys])


    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]
    

    # compute number of minibatches for training, validation and testing
    batch_size=len(ys)    
    # batch_size=1
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    # n_train_batches = 1
    # n_valid_batches = 1
    # n_test_batches = 1

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
    # myprint=theano.function([x],x)
    # myprint([layer2_input])

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=20, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=20, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    prob = layer3.prob_y_given_x(y)

    f1=open('weights', 'w+')
    print "layer 0 weights"
    for w in layer0.W.get_value():
        for r in w:
            for s in r:
                for d in s:
                    f1.write(str(d)+'\n')

    # print layer0.W.get_value()
    # print layer0.b.get_value()
    print "layer 1 weights"
    # print layer1.W.get_value()
    # print layer1.b.get_value()
    for w in layer1.W.get_value():
        for r in w:
            for s in r:
                for d in s:
                    f1.write(str(d)+'\n')

    print "layer 2 weights"
    # print layer2.W.get_value()
    w=layer2.W.get_value();
    # for d in w: 
    #     print d
    for i in range(len(w[0])):
        for j in range(len(w)):
            f1.write(str(w[j][i])+'\n') 
    # print layer2.b.get_value()

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    prob_model = theano.function([index], prob,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    conv_model0 = theano.function([index], layer0.output,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})
    conv_model0_conv = theano.function([index], layer0.conv_out,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})

    conv_model1 = theano.function([index], layer1.output,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})
    conv_model1_conv = theano.function([index], layer1.conv_out,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})
    conv_model2 = theano.function([index], layer2.output,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    # params = layer0.params + layer1.params + layer2.params + layer3.params

    # x_printed = theano.printing.Print('this is a very important value')(x)
    # f_with_print = theano.function([x], x_printed)
    # f_with_print(layer3.params)



    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    val_grads = T.grad(cost, layer3.p_y_given_x)
    # print "AAAA"
    # theano.printing.debugprint(temp_grads)
    # print "AAAA"

    grad_model = theano.function([index], grads,
            givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    val_grad_model = theano.function([index], val_grads,
            givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})
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
            val_grads_ij=val_grad_model(minibatch_index)
            grads_ij=grad_model(minibatch_index)
            conv0_ij=conv_model0(minibatch_index)
            conv1_ij=conv_model1(minibatch_index)
            conv2_ij=conv_model2(minibatch_index)
            conv0_conv_ij=conv_model0_conv(minibatch_index)
            conv1_conv_ij=conv_model1_conv(minibatch_index)

            print 'training @ iter = ', iter
            print "last layer var grads"
            print val_grads_ij[0]

            # print "Layer 0 convolution"
            # for c in conv0_conv_ij[0]:
            #     print c
            #     print ""
            # print ""
            # print "Layer 1 convolution"
            # for c in conv1_conv_ij[0]:
            #     print c
            #     print ""
            # print ""
            probs = prob_model(minibatch_index)
            print "Probs"
            print probs
            # print "layer 0 grads"
            # print grads_ij[6]
            # print grads_ij[7]
            # print "layer 1 grads"
            # print grads_ij[4]
            # print grads_ij[5]
            # print "layer 2 grads"
            # print grads_ij[2]
            # print grads_ij[3]
            print "log reg layer grads"
            print grads_ij[0]
            print grads_ij[1]
            print "Layer 0 output"
            # for c in conv0_ij:
            #     for d in c: 
            #         print d
            # print conv0_ij[0][0]
            print "Layer 1 output"
            # print conv1_ij[0][0]
            # for c in conv1_ij:
            #     for d in c:
            #         print d
            print "Layer 2 output"
            # for c in conv2_ij:
            #     print c
            cost_ij = train_model(minibatch_index)

    

          

           
            # for c in conv0_conv_ij[1]:
            #     print c
            #     print ""

            

            print "learning_rate"
            print learning_rate
            print "layer 0 weights"
            # print layer0.W.get_value()
            # print layer0.b.get_value()
            print "layer 1 weights"
            # print layer1.W.get_value()
            # print layer1.b.get_value()
            print "layer 2 weights"
            w=layer2.W.get_value()
            # print w[0]
            # print w[1]

            # for c in layer2.W.get_value():
            #     print c
            # print layer2.b.get_value()
            print "log reg layer weights"
            print layer3.W.get_value()
            print layer3.b.get_value()
            print "COST"
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
    # print "bestConvW" + str(bestConvW);
    # print sum(sum(sum(sum(bestConvW))))
    # print cost


if __name__ == '__main__':
    evaluate_mnist_1()


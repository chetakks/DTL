"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy
from numpy import * 
import theano
import theano.tensor as T

import pickle


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        print ' y.ndim',  y.ndim
        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
#     def predictions(self, y):
#         if y.ndim != self.y_pred.ndim:
#             raise TypeError('y should have the same shape as self.y_pred',
#                 ('y', target.type, 'y_pred', self.y_pred.type))
#         # check if y is of the correct datatype
#         print ' y.ndim',  y.ndim
# 
#         return T.ceil(self.y_pred)
    
#     def predictions(self,y):
#         return T.mean(T.neq(self.y_pred, y))
    
#     def predictions(self,y):
#         return T.ceil(self.y_pred)
    
#     def predictions(self,y):
#         return [T.ceil(self.y_pred), T.floor(y)]
    
    def predictions(self,y):
        return [T.ceil(self.y_pred), T.floor(y)]
    
    def class_probabilities(self,y):
        # compute vector of class-membership probabilities in symbolic form
        #return [T.ceil(self.p_y_given_x), T.floor(y)]
        return [self.p_y_given_x, T.floor(y)]
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)




def load_data(dataset_reuse):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    #dataset_reuse = 'chars74k_digits'
    
    if dataset_reuse == 'mnist_subset1':
#     # Download the MNIST dataset if it is not present
#     data_dir, data_file = os.path.split(dataset)
#     if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
#         import urllib
#         origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
#         print 'Downloading data from %s' % origin
#         urllib.urlretrieve(origin, dataset)

        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
    
        #if mode == 'reduced_data1':
        print 'Reduced data set'
        num_of_samples1 = 5080 # start for valid set and also end of train set
        num_of_samples2 = 10000 # end for valid set
        valid_set = []
        valid_set = (numpy.array(train_set[0][num_of_samples1+1:num_of_samples2+1]).astype(theano.config.floatX),
              numpy.array(train_set[1][num_of_samples1+1:num_of_samples2+1]).astype(theano.config.floatX))
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))

    elif dataset_reuse == 'mnist_subset2':
        
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        num_of_samples1 = 13208 # start for valid set and also end of train set
        num_of_samples2 = 13208 + 6604 # end for valid set
        valid_set = []
        valid_set = (numpy.array(train_set[0][num_of_samples1+1:num_of_samples2+1]).astype(theano.config.floatX),
              numpy.array(train_set[1][num_of_samples1+1:num_of_samples2+1]).astype(theano.config.floatX))
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        print 'done'
        
#         file_tra = '/home/aditya/repos/Database/mnist/'+'mnist2_tra.pkl'
#         file_val = '/home/aditya/repos/Database/mnist/'+'mnist2_val.pkl'
#         file_tes = '/home/aditya/repos/Database/mnist/'+'mnist2_tes.pkl'
#         pickle.dump(train_set, open(file_tra, 'wb'))
#         pickle.dump(valid_set, open(file_val, 'wb'))
#         pickle.dump(test_set, open(file_tes, 'wb'))
        
        
    elif dataset_reuse == 'mnist_100':
        #dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        num_of_samples1 = 100*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
    elif dataset_reuse == 'mnist_250':
        
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        num_of_samples1 = 250*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
    elif dataset_reuse == 'mnist_500':
        
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        num_of_samples1 = 500*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
    elif dataset_reuse == 'mnist_1000':
        
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        num_of_samples1 = 1000*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
    elif dataset_reuse == 'mnist_2500':
        
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        num_of_samples1 = 2500*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))

    elif dataset_reuse == 'mnist':
        print '... loading data'
        dataset = '/home/aditya/repos/Database/mnist/mnist.pkl.gz'
        # Load the dataset
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        print 'complete data set'
        
#         file_tra = '/home/aditya/repos/Database/mnist/'+'mnist_tra.pkl'
#         file_val = '/home/aditya/repos/Database/mnist/'+'mnist_val.pkl'
#         file_tes = '/home/aditya/repos/Database/mnist/'+'mnist_tes.pkl'
#         pickle.dump(train_set, open(file_tra, 'wb'))
#         pickle.dump(valid_set, open(file_val, 'wb'))
#         pickle.dump(test_set, open(file_tes, 'wb'))
        
        
    
    elif dataset_reuse == 'chars74k_digits28x28':
    
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
     
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        
    elif dataset_reuse == 'chars74k_digits128x128':
    
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits128x128_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits128x128_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits128x128_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
     
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
    elif dataset_reuse == 'chars74k_digits64x64':
    
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits64x64_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits64x64_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_digits64x64_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
     
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()        
        
    elif dataset_reuse == 'chars74k_lowercase28x28':
        
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_lowercase28x28_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_lowercase28x28_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_lowercase28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
     
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
       
        
    elif dataset_reuse == ('lowercase_100' or 'lowercase_250' or 'lowercase_500'):
        
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_lowercase28x28_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_lowercase28x28_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_lowercase28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        
        if dataset_reuse == 'lowercase_100':
            num_of_samples1 = 100*26 # end of train set
            train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
                  numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        elif dataset_reuse == 'lowercase_250':
            num_of_samples1 = 250*26 # end of train set
            train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
                  numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        elif dataset_reuse == 'lowercase_500':
            num_of_samples1 = 500*26 # end of train set
            train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
                  numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
        # To test the proportion of classes
        for t in range(26):
            print 'proportion of target '+str(t)+' in'
            print '    trai set: '+str(mean(train_set[1]==t))                                          
        
    elif dataset_reuse == 'chars74k_uppercase28x28':
        
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_uppercase28x28_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_uppercase28x28_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_uppercase28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
     
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
    
    elif dataset_reuse == ('uppercase_100' or 'uppercase_250' or 'uppercase_500'):
        
        dataset1='/home/aditya/repos/gpu_code/data/chars74k/chars74k_uppercase28x28_tra.pkl'
        dataset2='/home/aditya/repos/gpu_code/data/chars74k/chars74k_uppercase28x28_val.pkl'
        dataset3='/home/aditya/repos/gpu_code/data/chars74k/chars74k_uppercase28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        print 'Reduced data set'
        
        if dataset_reuse == 'uppercase_100':
            num_of_samples1 = 100*26 # end of train set
            train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
                  numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        elif dataset_reuse == 'uppercase_250':
            num_of_samples1 = 250*26 # end of train set
            train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
                  numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        elif dataset_reuse == 'uppercase_500':
            num_of_samples1 = 500*26 # end of train set
            train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
                  numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
        # To test the proportion of classes
        for t in range(26):
            print 'proportion of target '+str(t)+' in'
            print '    trai set: '+str(mean(train_set[1]==t))
        
    elif dataset_reuse == 'MAHDBase':
        
        dataset1='/home/aditya/repos/Database/MAHDBase_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/Database/MAHDBase_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/Database/MAHDBase_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
     
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
    #elif dataset_reuse == ('MAHDBase_100'or'MAHDBase_250'or'MAHDBase_500'or'MAHDBase_1000'or'MAHDBase_2500'):
    elif dataset_reuse == 'MAHDBase_100':
            
        dataset1='/home/aditya/repos/Database/MAHDBase_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/Database/MAHDBase_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/Database/MAHDBase_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()

        num_of_samples1 = 100*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
    elif dataset_reuse == 'MAHDBase_250':
        
        dataset1='/home/aditya/repos/Database/MAHDBase_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/Database/MAHDBase_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/Database/MAHDBase_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        num_of_samples1 = 250*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
    elif dataset_reuse == 'MAHDBase_500':
        
        dataset1='/home/aditya/repos/Database/MAHDBase_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/Database/MAHDBase_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/Database/MAHDBase_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        num_of_samples1 = 500*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
    elif dataset_reuse == 'MAHDBase_1000':
        
        dataset1='/home/aditya/repos/Database/MAHDBase_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/Database/MAHDBase_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/Database/MAHDBase_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        num_of_samples1 = 1000*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
    elif dataset_reuse == 'MAHDBase_2500':
        
        dataset1='/home/aditya/repos/Database/MAHDBase_digits28x28_tra.pkl'
        dataset2='/home/aditya/repos/Database/MAHDBase_digits28x28_val.pkl'
        dataset3='/home/aditya/repos/Database/MAHDBase_digits28x28_tes.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        num_of_samples1 = 2500*10 # end of train set
        train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
              numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
        
        
    elif dataset_reuse == 'shapeset1_1c_2s_3po':
        
        dataset1= '/home/aditya/repos/Database/shapeset/pickled/shapeset1_1c_2s_3po.10000.train.pkl'
        dataset2= '/home/aditya/repos/Database/shapeset/pickled/shapeset1_1c_2s_3po.5000.valid.pkl'
        dataset3= '/home/aditya/repos/Database/shapeset/pickled/shapeset1_1c_2s_3po.5000.test.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        
    elif dataset_reuse == 'shapeset2_1c_2s_3po':
        
        dataset1= '/home/aditya/repos/Database/shapeset/pickled/shapeset2_1c_2s_3po.10000.train.pkl'
        dataset2= '/home/aditya/repos/Database/shapeset/pickled/shapeset2_1c_2s_3po.5000.valid.pkl'
        dataset3= '/home/aditya/repos/Database/shapeset/pickled/shapeset2_1c_2s_3po.5000.test.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
    elif dataset_reuse == 'shapeset1_1csp_2o_3':
        
        dataset1= '/home/aditya/repos/Database/shapeset/pickled/shapeset1_1csp_2o_3.10000.train.pkl'
        dataset2= '/home/aditya/repos/Database/shapeset/pickled/shapeset1_1csp_2o_3.5000.valid.pkl'
        dataset3= '/home/aditya/repos/Database/shapeset/pickled/shapeset1_1csp_2o_3.5000.test.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
        
    elif dataset_reuse == 'shapeset2_1csp_2o_3':
        
        dataset1= '/home/aditya/repos/Database/shapeset/pickled/shapeset2_1csp_2o_3.10000.train.pkl'
        dataset2= '/home/aditya/repos/Database/shapeset/pickled/shapeset2_1csp_2o_3.5000.valid.pkl'
        dataset3= '/home/aditya/repos/Database/shapeset/pickled/shapeset2_1csp_2o_3.5000.test.pkl'
     
        f = open(dataset1, 'rb')
        train_set = cPickle.load(f)
        f.close()
        
        f = open(dataset2, 'rb')
        valid_set = cPickle.load(f)
        f.close()
     
        f = open(dataset3, 'rb')
        test_set = cPickle.load(f)
        f.close()
        
    # To test the proportion of classes
    for t in range(len(list(set(train_set[1])))):
        print 'proportion of target '+str(t)+' in'
        print '    trai set: '+str(mean(train_set[1]==t))
        
        
    
    print 'nr features:       ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
    print 'targets are:    ', list(set(train_set[1])) 
    print 'nr training instances:  ', len(train_set[0])
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr test instances:      ', len(test_set[0])
     
    ############################################################
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset2(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
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

    test_set_x, test_set_y = shared_dataset2(test_set)
    valid_set_x, valid_set_y = shared_dataset2(valid_set)
    train_set_x, train_set_y = shared_dataset2(train_set)
    #train_set_x, train_set_y = shared_dataset2(D)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                            dataset='/home/aditya/repos/deeplearningtutorials/data/mnist.pkl.gz',
                           #dataset='../data/mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    #classifier = LogisticRegression(input=x, n_in=32 * 32, n_out=10)
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
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
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
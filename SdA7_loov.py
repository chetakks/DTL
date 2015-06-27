"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


from logistic_sgd5 import LogisticRegression
from mlp import HiddenLayer
from dA5 import dA5
#from cA5 import cA5


class SdA7(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=None,
                 hidden_layers_sizes=[500, 500], n_outs=None, #n_outs_b=None,
                 corruption_levels=[0.1, 0.1], tau = None):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """
        
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.params_b = []
        self.n_layers = len(hidden_layers_sizes)
        
        self.sigmoid_layers2 = []
        self.dA_layers2 = []
        self.params2 = []
        self.params2_b = []
        hidden_layers_sizes2=[40, 40, 40]
        self.n_layers2 = len(hidden_layers_sizes2)
        self.combine_ins = [] # combine multiple layer inputs

        assert self.n_layers > 0
        print 'code in Sda7'
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            print 'theano_rng', theano_rng.default_instance_seed
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP
        #######################################################################
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
            self.params_b.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA5(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          tau = tau,
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

                
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.params.extend(self.logLayer.params)
        
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        self.predictions = self.logLayer.predictions(self.y)
        
        # for each probability of the classes # compute vector of class-membership probabilities in symbolic form
        self.class_probabilities = self.logLayer.class_probabilities(self.y)
        
        ###################### reuse code ########################################
        
#         # We now need to add a logistic layer on top of the MLP
#         self.logLayer_b = LogisticRegression(
#                          input=self.sigmoid_layers[-1].output,
#                          n_in=hidden_layers_sizes[-1], n_out=n_outs_b)
# 
#         self.params_b.extend(self.logLayer_b.params)
#         # construct a function that implements one step of finetunining
# 
#         # compute the cost for second phase of training,
#         # defined as the negative log likelihood
#         self.finetune_cost_b = self.logLayer_b.negative_log_likelihood(self.y)
#         # compute the gradients with respect to the model parameters
#         # symbolic variable that points to the number of errors made on the
#         # minibatch given by self.x and self.y
#         self.errors_b = self.logLayer_b.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, tau):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA5 in self.dA_layers:
            # get the cost and the updates list
#             cost, updates, y, z, L, h = dA5.get_cost_updates(corruption_level,
#                                                 learning_rate, tau)
            cost, updates = dA5.get_cost_updates(corruption_level,
                                                learning_rate, tau)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=[cost],
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        
#         for dA5 in self.dA_layers2:
#             # get the cost and the updates list
# #             cost, updates, y, z, L, h = dA5.get_cost_updates(corruption_level,
# #                                                 learning_rate, tau)
#             cost, updates = dA5.get_cost_updates(corruption_level,
#                                                 learning_rate, tau)
#             # compile the theano function
#             fn = theano.function(inputs=[index,
#                               theano.Param(corruption_level, default=0.2),
#                               theano.Param(learning_rate, default=0.1)],
#                                  outputs=[cost],
#                                  updates=updates,
#                                  givens={self.x: train_set_x[batch_begin:
#                                                              batch_end]})
#             # append `fn` to the list of functions
#             pretrain_fns.append(fn)
            
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})
        
        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        
        #return train_fn, valid_score, test_score
        #self.y_pred
        test_predictions_i = theano.function([index], self.predictions,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
        
        #, on_unused_input='warn'
        def test_predictions():
            return [test_predictions_i(i) for i in xrange(n_test_batches)]
        
        
        
        test_class_probabilities_i = theano.function([index], self.class_probabilities,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
        
        #, on_unused_input='warn'
        def test_class_probabilities():
            return [test_class_probabilities_i(i) for i in xrange(n_test_batches)]
        
        

        #return train_fn, valid_score, test_score, test_predictions
        return train_fn, valid_score, test_score, test_predictions, test_class_probabilities

    
    def build_per_treatment_test(self, datasets, batch_size):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch
        '''

        (test_set_x, test_set_y) = datasets

        # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        
        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
        
        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        
        test_predictions_i = theano.function([index], self.predictions,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
        
        #return test_score
        
        #, on_unused_input='warn'
        def test_predictions():
            return [test_predictions_i(i) for i in xrange(n_test_batches)]
         
         
         
        test_class_probabilities_i = theano.function([index], self.class_probabilities,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
         
        #, on_unused_input='warn'
        def test_class_probabilities():
            return [test_class_probabilities_i(i) for i in xrange(n_test_batches)]
 
         
        return test_score, test_predictions, test_class_probabilities
    
    def build_finetune_functions_reuse(self, datasets, batch_size, learning_rate, retrain_ft_layers):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        #gparams = T.grad(self.finetune_cost_b, self.params_b)
        gparams = T.grad(self.finetune_cost, self.params)
        
#         ### To show the difference between the two logistic layers
#         diff_LR_w = self.params_b[6].get_value() - self.params[6].get_value()
#         diff_LR_b = self.params_b[7].get_value() - self.params[7].get_value()
#         print 'diffence in the output layer'
#         print 'differnce in weight', diff_LR_w
#         print 'differnce in biases', diff_LR_b

        # compute list of fine-tuning updates
        total_n_layers = range((self.n_layers +1) *2)
        updates = []
        #update_layerwise = [0,0,0,0,1,1,1,1] #example
        #print 'retrain_ft_layers', retrain_ft_layers
        #for param, gparam, update, layer_num in zip(self.params_b, gparams, retrain_ft_layers, total_n_layers):
        for param, gparam, update, layer_num in zip(self.params, gparams, retrain_ft_layers, total_n_layers):
            if update == 0: 
                if layer_num % 2 == 0: # even for weights
                    updates.append((param, param ))
                    print 'locked  layer %d weights' % ((layer_num/2)+1)
                else:
                    updates.append((param, param ))
                    print 'locked  layer %d bias' % ((layer_num/2)+1)
            elif update == 1:
                if layer_num % 2 == 0: # even for weights
                    updates.append((param, param - gparam * learning_rate))
                    print 'retrain layer %d weights' % ((layer_num/2)+1)
                else:
                    updates.append((param, param - gparam * learning_rate))
                    print 'retrain layer %d bias' % ((layer_num/2)+1)

        train_fn = theano.function(inputs=[index],
              #outputs=self.finetune_cost_b,
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        #test_score_i = theano.function([index], self.errors_b,
        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        #valid_score_i = theano.function([index], self.errors_b,
        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        #return train_fn, valid_score, test_score
        
        test_predictions_i = theano.function([index], self.predictions,
                     givens={
                       self.x: test_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                       self.y: test_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
         
        #, on_unused_input='warn'
        def test_predictions():
            return [test_predictions_i(i) for i in xrange(n_test_batches)]
 
    
        test_class_probabilities_i = theano.function([index], self.class_probabilities,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})
        
        #, on_unused_input='warn'
        def test_class_probabilities():
            return [test_class_probabilities_i(i) for i in xrange(n_test_batches)]
        
        
        return train_fn, valid_score, test_score, test_predictions, test_class_probabilities
import cPickle
import gzip
import os
import sys
import time

import numpy
import numpy as np
import theano
import theano.tensor as T
#from sklearn.metrics import confusion_matrix

#from logistic_sgd5 import LogisticRegression



def train_test_mlp(learning_rate=None, training_epochs=None,#1000,
             dataset=None, batch_size=None,
             n_train_batches=None,n_valid_batches=None,n_test_batches=None,
             train_fn=None,validate_model=None,test_model=None, test_predictions=None,
             test_class_probabilities=None,
             mtest_model_pert = None,
             mtest_predictions_pert = None,
             mtest_class_probabilities_pert = None,
             multi_test_labels = None):


    ###############
    # TRAIN MODEL #
    ###############
    print '... finetunning the model'
    
    # early-stopping parameters
    patience = 20000
    #patience = 2000
    #print 'patience', patience
    #patience = 20 * n_train_batches  # look as this many examples regardless
    #patience_increase = 2.
    patience_increase = 6. #2.  # wait this much longer when a new best is
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
    val_epochs = []
    val_epochs_errs = []
    test_epochs = []
    test_epochs_errs = []
    
    
    start_time = time.clock()
    
# #     ft_costs_epochs_array = []
    
    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
# #         ft_costs_epochs = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
  

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                val_epochs.append(epoch)
                val_epochs_errs.append(this_validation_loss * 100.)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss 
                    best_iter = iter
                    
                    # test it on the multiple test sets
                    mtest_predict = []
                    mtest_class_prob = []
                    mtest_scores = []
                    
                    for i in range(len(multi_test_labels)):
                        print '........................ testing :', multi_test_labels[i]
                        mtest_model = mtest_model_pert[i]
                        mtest_predictions = mtest_predictions_pert[i]
                        mtest_class_probabilities = mtest_class_probabilities_pert[i]
                        test_losses = mtest_model()
                        test_predict = mtest_predictions()
                        test_class_prob = mtest_class_probabilities()
                        test_score = numpy.mean(test_losses) * 100.   
                        
                        mtest_predict.append(numpy.array(test_predict))
                        mtest_class_prob.append(test_class_prob)
                        mtest_scores.append(test_score)

                        print(('     epoch %i, minibatch %i/%i, multi test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           (mtest_scores[i]) ))
                    
                    test_epochs.append(epoch)
                    test_epochs_errs.append(test_score)
            
                    
            if patience <= iter:
                done_looping = True
                break
            
    mtest_scores = np.array(mtest_scores)
    print 'mtest_scores', mtest_scores
    mtest_accus = 100 - mtest_scores
    print 'mtest_accus', mtest_accus
    end_time = time.clock()

    print '######################################################'
    print(('Optimization complete with best validation score of %f %%') % (best_validation_loss* 100.))
    for i in range(len(multi_test_labels)):
        print 'for the test set :', multi_test_labels[i]
        print ('with test performance %f %%') % mtest_scores[i]
        
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print '######################################################'
    
    return (best_validation_loss* 100., test_score, val_epochs,val_epochs_errs, 
           test_epochs, test_epochs_errs,
           mtest_predict, mtest_class_prob, mtest_scores, mtest_accus)
    
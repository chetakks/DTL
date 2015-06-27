import cPickle
import pickle
import os
import sys
import time
import csv 

import numpy
from numpy import *




def load_outputs(dir, prefix='outputs_'):
    outputs_list = []
    for filename in os.listdir(dir): 
        if filename.startswith(prefix):
            outputs = pickle.load(open(dir+'/'+filename, 'rb'))
            outputs_list.append(outputs)
    return outputs_list

def reuse_SdA7(finetune_lr=None, pretraining_epochs=None,
             pretrain_lr=None, training_epochs=None,
              n_ins=None,
              hidden_layers_sizes=None,
              dataset_A=None, 
              n_outs=None,
              retrain=None,
              source_reuse_mode=None,
              dataset_B=None,
              n_outs_source=None,
              batch_size=None,
              output_fold = None, 
              rng_seed=None,
              retrain_ft_layers=None,
              tranferred_layers = None,
              sda_reuse_pt_model=None, 
              sda_reuse_ft_model=None, 
              repetition=None,
              tau=None,
              training_data_fraction=None,
              dropout_rate = None,
              gpu_nr = 'gpu0',
              data_path = None,
              fold = None,
              moa_items = None):
    

    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    #theano.sandbox.cuda.use('gpu0')
    theano.sandbox.cuda.use(gpu_nr)
    
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    from load_MFC7_data2 import load_dataset, load_multi_test_dataset
    from mlp import HiddenLayer
    from dA5 import dA5
    from SdA7_loov import SdA7
    #from SdA7_loov_cA5 import SdA7
    from mlp5_train_model2_loov import train_test_mlp
    
    print 'dataset_A', dataset_A
    
    datasets = load_dataset(dataset_A,data_path, fold, reduce=6,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
                   
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_c, test_set_y = datasets[2]



    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_c.get_value(borrow=True).shape[0] / batch_size
    

    #if approach == 'BL':
    if retrain == 0:    
        # numpy random generator
        numpy_rng = numpy.random.RandomState(rng_seed+repetition)
    
        print '... building the model'
        # construct the stacked denoising autoencoder class
        sda = SdA7(numpy_rng=numpy_rng, n_ins=n_ins,
                  hidden_layers_sizes=hidden_layers_sizes,
                  n_outs=n_outs, #n_outs_b=n_outs_b,
                  tau=tau)
        
      
    
        #########################
        # PRETRAINING THE MODEL #
        #########################
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size, tau=tau)
        
        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        corruption_levels = [.2, .3, .3, .3, .3]
        corruption_levels = [.1, .2, .3, .3, .3]
        corruption_levels = [.1, .2, .3, .3, .3, .4, .4,.4, .4]
        corruption_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pt_trai_costs_vs_stage = []
        for i in xrange(sda.n_layers):
            # go through pretraining epochs
            pt_trai_costs = []
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                              corruption=corruption_levels[i],
                              lr=pretrain_lr))
                    
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                pt_trai_costs.append(numpy.mean(c))
                #print 'c', c
            pt_trai_costs_vs_stage.append(pt_trai_costs)
            
        end_time_pt = time.clock()
        pt_time = (end_time_pt - start_time) / 60.
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % (pt_time))
        
#         sda_reuse_pt_model = []
#         for para_copy in sda.params_b:
#             sda_reuse_pt_model.append(para_copy.get_value())

        sda_reuse_pt_model = []
        for para_copy in sda.params:
            sda_reuse_pt_model.append(para_copy.get_value())
 
        
        ########################
        # FINETUNING THE MODEL #
        ########################
    
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'

    
        start_time_ft = time.clock()
                    
        train_fn, validate_model, test_model, test_predictions, \
        test_class_probabilities = sda.build_finetune_functions(
                                            datasets=datasets, batch_size=batch_size,
                                            learning_rate=finetune_lr)
        
        

        multi_test_sets, multi_test_labels = load_multi_test_dataset(dataset_A, data_path, fold)
            
        mtest_model_pert = []
        mtest_predictions_pert = []
        mtest_class_probabilities_pert = []
        
        for i in range(len(multi_test_labels)):
            print '...... build test models :', multi_test_labels[i]   
            
            mtest_model, \
            mtest_predictions, \
            mtest_class_probabilities = sda.build_per_treatment_test(
                                            datasets=multi_test_sets[i], 
                                            batch_size=batch_size)
            
            mtest_model_pert.append(mtest_model)
            mtest_predictions_pert.append(mtest_predictions)
            mtest_class_probabilities_pert.append(mtest_class_probabilities)
        
            
            
        
        print '... finetunning the model'
        
        best_validation_loss, \
        test_score, \
        val_epochs, \
        val_epochs_errs,\
        test_epochs, \
        test_epochs_errs, \
        mtest_predict, mtest_class_probs, mtest_scores, mtest_accus = train_test_mlp(
                              learning_rate=0.01,
                              training_epochs=training_epochs,
                              dataset=dataset_A,
                              batch_size=batch_size,
                              n_train_batches=n_train_batches,
                              n_valid_batches=n_valid_batches,
                              n_test_batches=n_test_batches,
                              train_fn=train_fn,
                              validate_model=validate_model,
                              test_model=test_model,
                              test_predictions=test_predictions,
                              test_class_probabilities=test_class_probabilities,
                              mtest_model_pert = mtest_model_pert,
                              mtest_predictions_pert = mtest_predictions_pert,
                              mtest_class_probabilities_pert = mtest_class_probabilities_pert,
                              multi_test_labels = multi_test_labels)
        
        print 'Before majority voting ...........'
        print 'mtest_scores', mtest_scores
        print 'mtest_accus', mtest_accus
        

        mtest_majvote_accus  = []
        mtest_majvote_errors = []
        my_test_preds = []
        my_tests = []
        mvote_per_class = []
        my_test_pred_maj_votes = []
        
        print 'applying majority voting ...........'
        for i in range(len(multi_test_labels)):
            print '...... testing :', multi_test_labels[i]   
            print 'test error before majority voting:', mtest_scores[i]
            
            import numpy as np
            #MOAs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            #MOAs = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            #MOAs = [0, 1, 2, 3, 4, 5]
            MOAs = range(len(moa_items))
            vote_per_class = []
            my_test_pred = mtest_predict[i][:,0]    
            my_test = mtest_predict[i][:,1]
            my_test_pred = my_test_pred.flatten()
            my_test = my_test.flatten()
            #print 'my_test', my_test
            #print 'my_test_pred', my_test_pred
            print 'True MOA class:', np.unique(my_test)
            for MOA in MOAs:
                a = np.where(my_test_pred == MOA)
                print 'Number of predicted elements %d in MOA  %d' % (len(a[0]), MOA)
                vote_per_class.append(len(np.array(a)[0]))
            
            print 'vote_per_class', vote_per_class
            maj_vote = np.argmax(vote_per_class)
            print 'maj_vote', maj_vote
            
            predicted_MOA = MOAs[np.argmax(vote_per_class)]
            print 'predicted_MOA', predicted_MOA
            print 'True MOA class: %d, predicted MOA: %d' % (np.unique(my_test), predicted_MOA )
            
            my_test_pred_maj_vote = my_test_pred
            my_test_pred_maj_vote.fill(predicted_MOA)
            
            from sklearn.metrics import confusion_matrix, accuracy_score
            print confusion_matrix(my_test, my_test_pred_maj_vote)
            mtest_majvote_accu = accuracy_score(my_test, my_test_pred_maj_vote) * 100.
            
            print ('mtest_majvote_accu %f %%') % (mtest_majvote_accu)
            
            mtest_majvote_error = 100 - mtest_majvote_accu
            print 'mtest_majvote_error', mtest_majvote_error
            
            
            
            my_test_preds.append(my_test_pred)
            my_tests.append(my_test)
            #my_test_class_probs.append(mtest_class_prob)
            #mtest_accus.append(mtest_accu)
            mvote_per_class.append(vote_per_class)  
            my_test_pred_maj_votes.append(my_test_pred_maj_vote)
            
            mtest_majvote_accus.append(mtest_majvote_accu)
            mtest_majvote_errors.append(mtest_majvote_error)
            
        print 'mtest_majvote_accus', mtest_majvote_accus
        print 'mtest_majvote_errors', mtest_majvote_errors

        
        end_time_ft = time.clock()
        ft_time = (end_time_ft - start_time_ft) / 60.
        
        sda_reuse_ft2_model = []
        for para_copy in sda.params:
            #print 'para_copy22.get_value()',para_copy.get_value()
            sda_reuse_ft2_model.append(para_copy.get_value())
        
#         for ids in range(len(sda.params)):
#             a = sda.params[ids].get_value()
#             print 'a',a
#             sda_reuse_ft2_model.append(a)

        sda_reuse_ft_model = sda
        print 'done'
    
    
    ########################
        # RE- FINETUNING THE MODEL #
    ########################
    elif retrain == 1:
        from scipy.stats import bernoulli
        # numpy random generator
        numpy_rng = numpy.random.RandomState(rng_seed+repetition)
        
        print '... building the model'
        # construct the stacked denoising autoencoder class
        sda = SdA7(numpy_rng=numpy_rng, n_ins=n_ins,
                  hidden_layers_sizes=hidden_layers_sizes,
                  n_outs=n_outs, #n_outs_b=n_outs_b,
                  tau=tau)
        
        if n_outs == n_outs_source:
                nr_of_layers = len(sda.params)
        else:
            nr_of_layers = len(sda.params)-2
        for ids in range(nr_of_layers):
            print 'sda before', sda.params[ids].get_value()
            
        
        def trl(tranferred_layers, nr_of_layers):
            for ids in range(nr_of_layers):
                if tranferred_layers[ids] == 1:
                    if ids % 2 == 0: # even for weights
                        print 'transfer layer %d weights' % ((ids/2)+1)
                    else:
                        print 'transfer layer %d bias' % ((ids/2)+1)
                    #print 'transfer layer', ids
                    if dataset_A == 'bbbc+moa' or dataset_A == 'bbbc+comp': 
                        # FOR BBBC data from float64 to float32
                        a = sda_reuse_ft_model.params[ids].get_value() 
                        b = a.astype(dtype='float32') 
                        sda.params[ids].set_value(b)
                    elif dataset_A == 'mnist' or dataset_A == 'chars74k_uppercase28x28' or \
                           dataset_A == 'chars74k_lowercase28x28' or dataset_A == 'chars74k_digits28x28' or \
                           dataset_A == 'lowercase_diff': 
                        a = sda_reuse_ft_model.params[ids].get_value() 
                        print 'a', a
                        b = a.astype(dtype='float32') 
                        sda.params[ids].set_value(b)
                        print 'sda.', sda.params[ids].get_value()
                    else:
                        sda.params[ids].set_value(sda_reuse_ft_model.params[ids].get_value())
                elif tranferred_layers[ids] == 0:
                    if ids % 2 == 0: # even for weights
                        print 'don\'t transfer %d weights' % ((ids/2)+1)
                    else:
                        print 'don\'t transfer %d bias' % ((ids/2)+1)
                        

        if source_reuse_mode == 'R':
            print 'random initialization'
#         elif source_reuse_mode == 'R+D':
#             print 'random initialization with dropout'
#             for ids in range(len(sda.params)):
#                 a = sda.params[ids].get_value()
#                 b = dropout_weights(a, dropout_rate)
#                 sda.params[ids].set_value(b)
        elif source_reuse_mode == 'PT':
            print 'restoring source problem pre-training weights'
            if n_outs == n_outs_source:
                for ids in range(len(sda.params)):
                    if tranferred_layers[ids] == 1:
                        if ids % 2 == 0: # even for weights
                            print 'transfer layer %d weights' % ((ids/2)+1)
                        else:
                            print 'transfer layer %d bias' % ((ids/2)+1)
                        #print 'transfer layer', ids
                        if dataset_A == 'bbbc+moa' or dataset_A == 'bbbc+comp': 
                            # FOR BBBC data from float64 to float32
                            a = sda_reuse_pt_model[ids]
                            b = a.astype(dtype='float32') 
                            sda.params[ids].set_value(b)
                        else:
                            sda.params[ids].set_value(sda_reuse_pt_model[ids])
                    elif tranferred_layers[ids] == 0:
                        if ids % 2 == 0: # even for weights
                            print 'don\'t transfer %d weights' % ((ids/2)+1)
                        else:
                            print 'don\'t transfer %d bias' % ((ids/2)+1)
            else:
                for ids in range(len(sda.params)-2):
                    if tranferred_layers[ids] == 1:
                        if ids % 2 == 0: # even for weights
                            print 'transfer layer %d weights' % ((ids/2)+1)
                        else:
                            print 'transfer layer %d bias' % ((ids/2)+1)
                        #print 'transfer layer', ids
                        if dataset_A == 'bbbc+moa' or dataset_A == 'bbbc+comp': 
                            # FOR BBBC data from float64 to float32
                            a = sda_reuse_pt_model[ids]
                            b = a.astype(dtype='float32') 
                            sda.params[ids].set_value(b)
                        else:
                            sda.params[ids].set_value(sda_reuse_pt_model[ids])
                    elif tranferred_layers[ids] == 0:
                        if ids % 2 == 0: # even for weights
                            print 'don\'t transfer %d weights' % ((ids/2)+1)
                        else:
                            print 'don\'t transfer %d bias' % ((ids/2)+1)
        
#         elif source_reuse_mode == 'PT+D':
#             print 'restoring source problem pre-training weights with dropout'
#             if n_outs == n_outs_source:
#                 for ids in range(len(sda.params)):
#                     a = sda_reuse_pt_model[ids]
#                     b = dropout_weights(a, dropout_rate)
#                     sda.params[ids].set_value(b)                       
#             else:
#                 for ids in range(len(sda.params)-2):
#                     a = sda_reuse_pt_model[ids]
#                     b = dropout_weights(a, dropout_rate)
#                     sda.params[ids].set_value(b)   
                    
#         elif source_reuse_mode == 'PT+FT':
#                 print 'restoring source problem fine-tunned weights'
#                 if n_outs == n_outs_source:
#                     for ids in range(len(sda.params)):
#                         sda.params[ids].set_value(sda_reuse_ft_model.params[ids].get_value())
#                 else:
#                     for ids in range(len(sda.params)-2):
#                         #sda.params[ids].set_value(sda_reuse_ft_model.params[ids].get_value())
#                         
#                         # FOR BBBC data from float64 to float32
#                         a = sda_reuse_ft_model.params[ids].get_value() 
#                         b = a.astype(dtype='float32') 
#                         sda.params[ids].set_value(b)
                    
        elif source_reuse_mode == 'PT+FT':
            print 'restoring source problem fine-tunned weights'
            if n_outs == n_outs_source:
                nr_of_layers = len(sda.params)
                trl(tranferred_layers, nr_of_layers)
            else:
                nr_of_layers = len(sda.params)-2
                trl(tranferred_layers, nr_of_layers)
        
#         elif source_reuse_mode == 'PT+FT+D':
#             print 'restoring source problem fine-tunned weights with dropout'
#             if n_outs == n_outs_source:
#                 for ids in range(len(sda.params)):
#                     a = sda_reuse_ft_model.params[ids].get_value()
#                     b = dropout_weights(a, dropout_rate)
#                     sda.params[ids].set_value(b)
#             else:
#                 for ids in range(len(sda.params)-2):
#                     a = sda_reuse_ft_model.params[ids].get_value()
#                     b = dropout_weights(a, dropout_rate)
# 
#                     sda.params[ids].set_value(b)
   
        print '... getting the finetuning functions'
        start_time_rft = time.clock()
                    
        train_fn, validate_model, test_model, \
        test_predictions, \
        test_class_probabilities = \
        sda.build_finetune_functions_reuse(
                               datasets=datasets,
                               batch_size=batch_size,
                               learning_rate=finetune_lr,
                               retrain_ft_layers= retrain_ft_layers)
        
        

        multi_test_sets, multi_test_labels = load_multi_test_dataset(dataset_A, data_path, fold)
            
        mtest_model_pert = []
        mtest_predictions_pert = []
        mtest_class_probabilities_pert = []
        
        for i in range(len(multi_test_labels)):
            print '...... build test models :', multi_test_labels[i]   
            
            mtest_model, \
            mtest_predictions, \
            mtest_class_probabilities = sda.build_per_treatment_test(
                                            datasets=multi_test_sets[i], 
                                            batch_size=batch_size)
            
            mtest_model_pert.append(mtest_model)
            mtest_predictions_pert.append(mtest_predictions)
            mtest_class_probabilities_pert.append(mtest_class_probabilities)
        
            
            
        
        print '... finetunning the model'
        
        best_validation_loss, \
        test_score, \
        val_epochs, \
        val_epochs_errs,\
        test_epochs, \
        test_epochs_errs, \
        mtest_predict, mtest_class_probs, mtest_scores, mtest_accus = train_test_mlp(
                              learning_rate=0.01,
                              training_epochs=training_epochs,
                              dataset=dataset_A,
                              batch_size=batch_size,
                              n_train_batches=n_train_batches,
                              n_valid_batches=n_valid_batches,
                              n_test_batches=n_test_batches,
                              train_fn=train_fn,
                              validate_model=validate_model,
                              test_model=test_model,
                              test_predictions=test_predictions,
                              test_class_probabilities=test_class_probabilities,
                              mtest_model_pert = mtest_model_pert,
                              mtest_predictions_pert = mtest_predictions_pert,
                              mtest_class_probabilities_pert = mtest_class_probabilities_pert,
                              multi_test_labels = multi_test_labels)
        
        
        print 'Before majority voting ...........'
        print 'mtest_scores', mtest_scores
        print 'mtest_accus', mtest_accus
        

        mtest_majvote_accus  = []
        mtest_majvote_errors = []
        my_test_preds = []
        my_tests = []
        mvote_per_class = []
        my_test_pred_maj_votes = []
        
        print 'applying majority voting ...........'
        for i in range(len(multi_test_labels)):
            print '...... testing :', multi_test_labels[i]   
            print 'test error before majority voting:', mtest_scores[i]
            
            import numpy as np
            #MOAs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            #MOAs = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            #MOAs = [0, 1, 2, 3, 4, 5]
            MOAs = range(len(moa_items))
            vote_per_class = []
            my_test_pred = mtest_predict[i][:,0]    
            my_test = mtest_predict[i][:,1]
            my_test_pred = my_test_pred.flatten()
            my_test = my_test.flatten()
            #print 'my_test', my_test
            #print 'my_test_pred', my_test_pred
            print 'True MOA class:', np.unique(my_test)
            for MOA in MOAs:
                a = np.where(my_test_pred == MOA)
                print 'Number of predicted elements %d in MOA  %d' % (len(a[0]), MOA)
                vote_per_class.append(len(np.array(a)[0]))
            
            print 'vote_per_class', vote_per_class
            maj_vote = np.argmax(vote_per_class)
            print 'maj_vote', maj_vote
            
            predicted_MOA = MOAs[np.argmax(vote_per_class)]
            print 'predicted_MOA', predicted_MOA
            print 'True MOA class: %d, predicted MOA: %d' % (np.unique(my_test), predicted_MOA )
            
            my_test_pred_maj_vote = my_test_pred
            my_test_pred_maj_vote.fill(predicted_MOA)
            
            from sklearn.metrics import confusion_matrix, accuracy_score
            print confusion_matrix(my_test, my_test_pred_maj_vote)
            mtest_majvote_accu = accuracy_score(my_test, my_test_pred_maj_vote) * 100.
            
            print ('mtest_majvote_accu %f %%') % (mtest_majvote_accu)
            
            mtest_majvote_error = 100 - mtest_majvote_accu
            print 'mtest_majvote_error', mtest_majvote_error
            
            
            
            my_test_preds.append(my_test_pred)
            my_tests.append(my_test)
            #my_test_class_probs.append(mtest_class_prob)
            #mtest_accus.append(mtest_accu)
            mvote_per_class.append(vote_per_class)  
            my_test_pred_maj_votes.append(my_test_pred_maj_vote)
            
            mtest_majvote_accus.append(mtest_majvote_accu)
            mtest_majvote_errors.append(mtest_majvote_error)
            
        print 'mtest_majvote_accus', mtest_majvote_accus
        print 'mtest_majvote_errors', mtest_majvote_errors

        
        end_time_rft = time.clock()
        pt_time = 0
        ft_time = (end_time_rft - start_time_rft) / 60.
        
    
        sda_reuse_ft_model = sda
        sda_reuse_ft2_model = []
        for para_copy in sda.params:
            sda_reuse_ft2_model.append(para_copy.get_value())
        sda_reuse_pt_model= None
        pt_trai_costs_vs_stage = None


    
    
    return (sda_reuse_pt_model,sda_reuse_ft2_model, sda_reuse_ft_model,
            best_validation_loss, test_score, 
            pt_time, ft_time, #y_test_pred, 
            #y_test, 
            val_epochs,val_epochs_errs, test_epochs, test_epochs_errs,
            pt_trai_costs_vs_stage, #y_test_class_prob,
            multi_test_labels,
            mtest_scores,
            mtest_accus,
            my_test_preds,
            my_tests,
            mtest_class_probs,
            #my_test_class_probs,
            mvote_per_class,
            my_test_pred_maj_votes,
            mtest_majvote_accus,
            mtest_majvote_errors)
    
    
       

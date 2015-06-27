from source_t_source3_loov_TL import reuse_SdA7
from results_MFC7_cm2 import plot_cm
from sklearn.metrics import confusion_matrix
from numpy import mean,std
import numpy as np
import datetime
import os
import sys
import time
import load_dataset2


def run_n_times(params,nr_reps,target_dataset,source_dataset,training_data_fraction,
                source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                tranferred_layers, gpu_nr,start_rep_nr,data_path,fold,source_fold =None, approach = None, transfer = None):
    
    if not os.path.exists(target_outputs_dir):
            os.makedirs(target_outputs_dir)
             
    if source_outputs_dir is not None:

        retrain_ft_layers_tmp = []
        for x in retrain_ft_layers:
            retrain_ft_layers_tmp.extend([int(x), int(x)])   
        retrain_ft_layers = retrain_ft_layers_tmp   

        
        tranferred_layers_tmp = []
        for x in tranferred_layers:
            tranferred_layers_tmp.extend([int(x), int(x)])   
        tranferred_layers = tranferred_layers_tmp   

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #print timestamp
    tau = None
    sda_reuse_pt_models = []
    sda_reuse_ft2_models = []
    sda_reuse_ft_models = []
    val_epochs_rep = []
    val_epochs_errs_rep = []
    test_epochs_rep = []
    test_epochs_errs_rep = []
    
    
    best_val_errors_rep = []
    mtest_errors_rep = []
    mtest_accus_rep = []
    mtest_majvote_accus_rep = []
    mtest_majvote_errors_rep = []
    pt_times_rep = []
    ft_times_rep = []
    ground_truth_rep = []
    prediction_rep   = []
    cm_rep = []

    start_tune_time = time.clock()
    

    target_outputs_dir_repeat = target_outputs_dir
    for repetition in range(start_rep_nr, nr_reps):
        start_tune_time_cv = time.clock()
        print '------------repetition: ', repetition+1
        
        
        compound_items = load_dataset2.load(data_path+fold+target_dataset+'_compound_items.pkl.gz')
        moa_items = load_dataset2.load(data_path+fold+target_dataset+'_moa_items.pkl.gz')
        
        settings = {}
        settings['target_dataset']  = target_dataset
        settings['compound_items']  = compound_items
        settings['moa_items']       = moa_items

        if source_outputs_dir is not None:
            source_compound_items = load_dataset2.load(data_path+source_fold+source_dataset+'_compound_items.pkl.gz')
            
            settings['source_compound_items']  = source_compound_items
             
            print
        
        sda_reuse_pt_models_cv = []
        sda_reuse_ft2_models_cv = []
        sda_reuse_ft_models_cv = []
        best_val_errors_cv = []
        test_score_errors_cv = []
        pt_times_cv = []
        ft_times_cv = []
        mtest_errors_cv = []
        mtest_accus_cv = []
        mtest_majvote_accus_cv = []
        mtest_majvote_errors_cv = []
        ground_truth    = []
        prediction      = []

        target_outputs_dir = target_outputs_dir_repeat
        print 'target_outputs_dir', target_outputs_dir
        temp_dir  = target_outputs_dir

        
            
        for idx, compound_label in enumerate(compound_items):

            start_tune_time_per_compound = time.clock()
                
            target_outputs_dir = '{0}{1}/'\
            .format(temp_dir,compound_label)
            
            settings['compound_label']      = compound_label
            settings['target_outputs_dir']  = target_outputs_dir
            
            if not os.path.exists(target_outputs_dir):
                os.makedirs(target_outputs_dir)
            

            print target_outputs_dir
            print '-------------------------test set compound:',idx, compound_label
            
            output_file_path = target_outputs_dir+'outputs_'+timestamp+'_%03d.pkl.gz' % (repetition+1)
            print output_file_path
            
                
            if target_dataset == 'MFC7_set1' or target_dataset == 'MFC7_set2':
                print 'Fetching target dataset information ...'
                params['dataset_A']     = target_dataset +'_'+ compound_label
                params['n_ins']         = 453
                params['n_outs']        = 6
                params['dataset_B']     = None
                params['n_outs_source'] = None
                dropout                 = None
                dropout_rate            = None
                print 'target_dataset details', params['dataset_A']
                print '                 n_ins', params['n_ins']
                print '                n_outs', params['n_outs']
                print '               dropout', dropout
                print '          dropout_rate', dropout_rate
            if target_dataset == 'MFC7_set1+2':
                print 'Fetching target dataset information ...'
                params['dataset_A']     = target_dataset +'_'+ compound_label
                params['n_ins']         = 453
                params['n_outs']        = 2
                params['dataset_B']     = None
                params['n_outs_source'] = None
                dropout                 = None
                dropout_rate            = None
                moa_items               = ['set1','set2']
                MOAs                    = [0,1]
                settings['moa_items']   = moa_items
                settings['MOAs']        = MOAs
                print 'target_dataset details', params['dataset_A']
                print '                 n_ins', params['n_ins']
                print '                n_outs', params['n_outs']
                print '               dropout', dropout
                print '          dropout_rate', dropout_rate
            if source_dataset == 'MFC7_set1':
                print 'Fetching source dataset information ...'
                #params['dataset_B']      = source_dataset
                params['dataset_B']      = source_dataset +'_'+ source_compound_items[10] 
                params['n_outs_source']  = 6
                     
                if source_dataset == 'MFC7_set1_comp':
                    params['n_outs_source']  = 20
                elif source_dataset == 'MFC7_set2_comp':
                    params['n_outs_source']  = 18
                    
                dropout                  = None
                
                dropout_rate             = 0.5
                source_outputs_list      = load_dataset2.load_outputs(source_outputs_dir + source_compound_items[10] + '/')
                print 'source_dataset details', params['dataset_B']
                print '          n_ins_source', 453
                print '         n_outs_source', params['n_outs_source']
                print 'source_outputs_dir'    , source_outputs_dir + source_compound_items[10] + '/'
                
            if source_dataset == 'MFC7_set2':
                print 'Fetching source dataset information ...'

                params['dataset_B']      = source_dataset +'_'+ source_compound_items[5] 
                params['n_outs_source']  = 6
                     
                if source_dataset == 'MFC7_set1_comp':
                    params['n_outs_source']  = 20
                elif source_dataset == 'MFC7_set2_comp':
                    params['n_outs_source']  = 18
                    
                dropout                  = None
                
                dropout_rate             = 0.5

                source_outputs_list      = load_dataset2.load_outputs(source_outputs_dir + source_compound_items[5] + '/')
                print 'source_dataset details', params['dataset_B']
                print '          n_ins_source', 453
                print '         n_outs_source', params['n_outs_source']

                print 'source_outputs_dir'    , source_outputs_dir + source_compound_items[5] + '/'
                
            if source_outputs_dir is not None:
                print 'Restoration of weights of source dataset' 
                task = 'TL: '+ params['dataset_A'] + ' reusing ' + str(params['dataset_B']) +  ' mode ' + str(source_reuse_mode) +' layers '+ str(retrain_ft_layers)
                print task
                 
                source_outputs = source_outputs_list[repetition]
                if   source_reuse_mode == 'PT' or 'PT+D':
                    sda_reuse_pt_model = source_outputs['sda_reuse_pt_model']
                    sda_reuse_ft_model = source_outputs['sda_reuse_ft_model']
                elif source_reuse_mode == 'PT+FT' or 'PT+FT+D':
                    sda_reuse_pt_model = None
                    sda_reuse_ft_model = source_outputs['sda_reuse_ft_model']
     
            else:
                task = 'Experiment with ' + params['dataset_A'] + ' and ' + str(params['dataset_B']) +  ' mode ' + str(source_reuse_mode) 
                print task
                sda_reuse_ft_model = None
                sda_reuse_pt_model = None
              
                                   
            sda_reuse_pt_model,sda_reuse_ft2_model,sda_reuse_ft_model,best_val_error, \
            test_score, pt_time, ft_time, val_epochs_rep, \
            val_epochs_errs_rep, test_epochs_rep, test_epochs_errs_rep, \
            pt_trai_costs_vs_stage, \
            multi_test_labels, \
            mtest_errors, \
            mtest_accus, \
            my_test_preds, \
            my_tests, \
            my_test_class_probs,\
            mvote_per_class, \
            my_test_pred_maj_votes,\
            mtest_majvote_accus,\
            mtest_majvote_errors = reuse_SdA7(
                 params['finetune_lr'],
                 params['pretraining_epochs'],
                 params['pretrain_lr'],
                 params['training_epochs'],
                 params['n_ins'],
                 params['hidden_layers_sizes'],
                 params['dataset_A'],
                 params['n_outs'],
                 params['retrain'],
                 source_reuse_mode,
                 params['dataset_B'],
                 params['n_outs_source'],
                 params['batch_size'],
                 params['output_fold'],
                 params['rng_seed'],
                 retrain_ft_layers = retrain_ft_layers,
                 tranferred_layers = tranferred_layers,
                 sda_reuse_pt_model=sda_reuse_pt_model,
                 sda_reuse_ft_model=sda_reuse_ft_model,
                 repetition=repetition,
                 tau=tau,
                 training_data_fraction=training_data_fraction,
                 dropout_rate = dropout_rate,
                 gpu_nr = gpu_nr,
                 data_path = data_path,
                 fold = fold,
                 moa_items = moa_items)  
             
             
            sda_reuse_pt_models_cv.append(sda_reuse_pt_model)
            sda_reuse_ft2_models_cv.append(sda_reuse_ft2_model)
            sda_reuse_ft_models_cv.append(sda_reuse_ft_model)
            best_val_errors_cv.append(best_val_error)
            test_score_errors_cv.append(test_score)
            pt_times_cv.append(pt_time)
            ft_times_cv.append(ft_time)

            mtest_errors_cv.append(mtest_errors)
            mtest_accus_cv.append(mtest_accus)
            mtest_majvote_accus_cv.append(mtest_majvote_accus)
            mtest_majvote_errors_cv.append(mtest_majvote_errors)
 
            outputs = {}
          
            # parameters
            outputs['hidden_sizes']     = params['hidden_layers_sizes']
            outputs['pt_learning_rate'] = params['pretrain_lr']
            outputs['ft_learning_rate'] = params['finetune_lr']
            outputs['ft_look_ahead']    = params['training_epochs']

            outputs['ft_vali_err']     = best_val_error
            outputs['ft_test_err']     = test_score

                  
            # the pre-trained model
            outputs['sda_reuse_pt_model'] = sda_reuse_pt_model
            # the fine-tuned model
            outputs['sda_reuse_ft_model'] = sda_reuse_ft_model
              
            # pre-training time for each hidden layer
            outputs['pt_trai_times'] = pt_time
            # fine-tuning time
            outputs['ft_trai_time']  = ft_time
      
            outputs['val_epochs_rep']       = val_epochs_rep
            outputs['val_epochs_errs_rep']  = val_epochs_errs_rep
            outputs['test_epochs_rep']      = test_epochs_rep
            outputs['test_epochs_errs_rep'] = test_epochs_errs_rep
            outputs['pt_trai_costs_vs_stage']   = pt_trai_costs_vs_stage
             
            outputs['data_path']              = data_path
            outputs['fold']                   = fold
             
            outputs['multi_test_labels']      = multi_test_labels
            outputs['mtest_errors']           = mtest_errors
            outputs['mtest_accus']            = mtest_accus
            outputs['my_test_preds']          = my_test_preds
            outputs['my_tests']               = my_tests
            outputs['my_test_class_probs']    = my_test_class_probs
            outputs['mvote_per_class']        = mvote_per_class
            outputs['my_test_pred_maj_votes'] = my_test_pred_maj_votes
            outputs['mtest_majvote_accus']    = mtest_majvote_accus
            outputs['mtest_majvote_errors']   = mtest_majvote_errors
             
                      
              
            load_dataset2.save(outputs,output_file_path)
            load_dataset2.save(settings,output_file_path)
              
            save_training_info = 0
            if save_training_info:
                import scipy.io as sio
                print 'saving info PT weights'
                sio.savemat(target_dataset+'_PT_WB.mat', {'WB':sda_reuse_pt_model})
                print 'saving info FT weights'
                sio.savemat(target_dataset+'_FT_WB.mat', {'WB':sda_reuse_ft2_model})   
  
                    
            print '===================================================================='
            print 'Approach' + approach + ' using GPU ' + gpu_nr
            if transfer == None:
                print 'approach', approach  
            elif transfer % 2 == 0: # even for STS
                print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
            elif transfer % 2 == 1: # odd for TL
                print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
            print 'data used:' + data_path + fold 
            print 'training data fraction ='+ str(training_data_fraction)
            print 'target_dataset details: '+ str(params['dataset_A'])
            print '            features  = '+ str(params['n_ins']) 
            print '             targets  = '+ str(params['n_outs'])
            print 'source_dataset details: '+ str(params['dataset_B'])
            print '             features = '+ str(453)
            print '              targets = '+ str(params['n_outs_source'])
            print 'Architecture details:   '
            print 'hidden_layers_sizes= '+ str(params['hidden_layers_sizes'])
            print 'Max Nr. PT epochs  = '+ str(params['pretraining_epochs'])
            print 'Max Nr. FT epochs  = '+ str(params['training_epochs'])
            print 'PT learning_rate   = '+ str(params['pretrain_lr'])
            print 'FT learning_rate   = '+ str(params['finetune_lr'])
            print 'batch_size         = '+ str(params['batch_size'])
            print 'dropout            = '+ str(dropout)
            print 'dropout_rate       = '+ str(dropout_rate)
            print 'source_reuse_mode  = '+ str(source_reuse_mode)
            print
            print '====  results for MFC7 compound # ' + str(idx+1) + ' test '+ compound_label + ' at repetition ' + str(repetition + 1)
            print 'training data fraction =' +str(training_data_fraction)
            print 'best_validation error =' + str(best_val_error)
            print 'mtest_errors', mtest_errors
            print 'mtest_accus', mtest_accus
            print 'mtest_majvote_accus', mtest_majvote_accus
            print 'mtest_majvote_errors', mtest_majvote_errors
            print 'mvote_per_class', mvote_per_class
            print
            MOAs = range(len(moa_items))

            print 'MOAs', MOAs
            for i in range(len(multi_test_labels)):
                print 'Test set results for ......', multi_test_labels[i]
                predicted_MOA = MOAs[np.argmax(mvote_per_class[i])]
                print 'Test accuracy before majority voting:', mtest_accus[i]
                print 'mvote_per_class=', mvote_per_class[i]
                print 'Test accuracy after majority voting:', mtest_majvote_accus[i]
                print 'True MOA class: %d, predicted MOA: %d' % (np.unique(my_tests[i]), predicted_MOA )
                prediction.append(predicted_MOA)
                ground_truth.append(np.unique(my_tests[i]))
                print
            print confusion_matrix(np.hstack(my_tests), np.hstack(my_test_pred_maj_votes))
            print confusion_matrix(np.hstack(ground_truth), np.hstack(prediction))
            if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
                print 'layers retrained by dataset_A : ', retrain_ft_layers
                print 'layers transfered from source: ', tranferred_layers
                print 'Time take for train ft layers in min = ', ft_time
                print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_time), std(ft_time))
            elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
                print 'Time take for train pt layers in min = ', pt_time
                print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(pt_time), std(pt_time))
                print 'Time take for train ft layers in min = ', ft_time
                print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_time), std(ft_time))
            print
             
            print '###########Cross Validation results at compound id %d ######################' % (idx+1)
            print 'compound # ' + str(idx+1) + ' out of '+ str(len(compound_items))
            print 'best_validation error cv =' + str(best_val_errors_cv)
            print 'mean best validation error cv = %.2f(%.2f)' % (mean(best_val_errors_cv), std(best_val_errors_cv))
            print 'results before majority voting'
            print 'mean test error = %.2f(%.2f)' % (mean(np.hstack(mtest_errors_cv)), std(np.hstack(mtest_errors_cv)))
            print 'mean test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_accus_cv)), std(np.hstack(mtest_accus_cv)))
            print 'Test error cv =' + str(mtest_errors_cv)
            print 'Test accu cv  =' + str(mtest_accus_cv)
            print 'results after majority voting'
            print 'mean majority test error = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_errors_cv)), std(np.hstack(mtest_majvote_errors_cv)))
            print 'mean majority test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_accus_cv)), std(np.hstack(mtest_majvote_accus_cv)))
            print 'Test majority error cv =' + str(mtest_majvote_errors_cv)
            print 'Test majority accu cv  =' + str(mtest_majvote_accus_cv)
            print 'Time taken'
            if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
                print 'layers retrained by dataset_A : ', retrain_ft_layers
                print 'layers transfered from source: ', tranferred_layers
                print 'Time take for train ft layers in min = ', ft_times_cv
                print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times_cv), std(ft_times_cv))
            elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
                print 'Time take for train pt layers in min = ', pt_times_cv
                print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(pt_times_cv), std(pt_times_cv))
                print 'Time take for train ft layers in min = ', ft_times_cv
                print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
                print 'Total time to train = mean %.2f(%.2f)m' % ((mean(pt_times_cv) +mean(ft_times_cv)), (std(pt_times_cv) +std(ft_times_cv)))
            print '###########Cross Validation results ######################'
            print
            print
             
            fcv = open(temp_dir+'Results_crossvalid'+'_rep'+str(repetition+1)+'_'+ gpu_nr+'.txt','a')
            old_stdout = sys.stdout   
            sys.stdout = fcv 
            print '===================================================================='
            print 'Approach' + approach + ' using GPU ' + gpu_nr
            if transfer == None:
                print 'approach', approach  
            elif transfer % 2 == 0: # even for STS
                print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
            elif transfer % 2 == 1: # odd for TL
                print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
            print 'data used:' + data_path + fold  
            print 'training data fraction ='+ str(training_data_fraction)
            print 'target_dataset details: '+ str(params['dataset_A'])
            print '            features  = '+ str(params['n_ins']) 
            print '             targets  = '+ str(params['n_outs'])
            print 'source_dataset details: '+ str(params['dataset_B'])
            print '             features = '+ str(453)
            print '              targets = '+ str(params['n_outs_source'])
            print 'Architecture details:   '
            print 'hidden_layers_sizes= '+ str(params['hidden_layers_sizes'])
            print 'Max Nr. PT epochs  = '+ str(params['pretraining_epochs'])
            print 'Max Nr. FT epochs  = '+ str(params['training_epochs'])
            print 'PT learning_rate   = '+ str(params['pretrain_lr'])
            print 'FT learning_rate   = '+ str(params['finetune_lr'])
            print 'batch_size         = '+ str(params['batch_size'])
            print 'dropout            = '+ str(dropout)
            print 'dropout_rate       = '+ str(dropout_rate)
            print 'source_reuse_mode  = '+ str(source_reuse_mode)
            print
            print '====  results for MFC7 compound # ' + str(idx+1) + ' test '+ compound_label + ' at repetation ' + str(repetition + 1)
            print 'training data fraction =' +str(training_data_fraction)
            print 'best_validation error =' + str(best_val_error)
            print 'mtest_errors', mtest_errors
            print 'mtest_accus', mtest_accus
            print 'mtest_majvote_accus', mtest_majvote_accus
            print 'mtest_majvote_errors', mtest_majvote_errors
            print 'mvote_per_class', mvote_per_class
            print 'MOAs', MOAs
            print
            for i in range(len(multi_test_labels)):
                print 'Test set results for ......', multi_test_labels[i]
                predicted_MOA = MOAs[np.argmax(mvote_per_class[i])]
                print 'Test accuracy before majority voting:', mtest_accus[i]
                print 'mvote_per_class=', mvote_per_class[i]
                print 'Test accuracy after majority voting:', mtest_majvote_accus[i]
                print 'True MOA class: %d, predicted MOA: %d' % (np.unique(my_tests[i]), predicted_MOA )
                print
            print confusion_matrix(np.hstack(my_tests), np.hstack(my_test_pred_maj_votes))
            print confusion_matrix(np.hstack(ground_truth), np.hstack(prediction))
            if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
                print 'layers retrained by dataset_A : ', retrain_ft_layers
                print 'layers transfered from source: ', tranferred_layers
                print 'Time take for train ft layers in min = ', ft_time
                print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_time), std(ft_time))
            elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
                print 'Time take for train pt layers in min = ', pt_time
                print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(pt_time), std(pt_time))
                print 'Time take for train ft layers in min = ', ft_time
                print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_time), std(ft_time))
            print
             
            print '###########Cross Validation results at compound id %d ######################' % (idx+1)
            print 'compound # ' + str(idx+1) + ' out of '+ str(len(compound_items))
            print 'best_validation error cv =' + str(best_val_errors_cv)
            print 'mean best validation error cv = %.2f(%.2f)' % (mean(best_val_errors_cv), std(best_val_errors_cv))
            print 'results before majority voting'
            print 'mean test error = %.2f(%.2f)' % (mean(np.hstack(mtest_errors_cv)), std(np.hstack(mtest_errors_cv)))
            print 'mean test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_accus_cv)), std(np.hstack(mtest_accus_cv)))
            print 'Test error cv =' + str(mtest_errors_cv)
            print 'Test accu cv  =' + str(mtest_accus_cv)
            print 'results after majority voting'
            print 'mean majority test error = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_errors_cv)), std(np.hstack(mtest_majvote_errors_cv)))
            print 'mean majority test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_accus_cv)), std(np.hstack(mtest_majvote_accus_cv)))
            print 'Test majority error cv =' + str(mtest_majvote_errors_cv)
            print 'Test majority accu cv  =' + str(mtest_majvote_accus_cv)
            print 'Time taken'
            if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
                print 'layers retrained by dataset_A : ', retrain_ft_layers
                print 'layers transfered from source: ', tranferred_layers
                print 'Time take for train ft layers in min = ', ft_times_cv
                print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
            elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
                print 'Time take for train pt layers in min = ', pt_times_cv
                print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(pt_times_cv), std(pt_times_cv))
                print 'Time take for train ft layers in min = ', ft_times_cv
                print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
                print 'Total time to train = mean %.2f(%.2f)m' % ((mean(pt_times_cv) +mean(ft_times_cv)), (std(pt_times_cv) +std(ft_times_cv)))
            print '###########Cross Validation results ######################'
            print
            sys.stdout=old_stdout 
            fcv.close()
         
             
        best_val_errors_rep.append(best_val_errors_cv)
        mtest_errors_rep.append(mtest_errors_cv)
        mtest_accus_rep.append(mtest_accus_cv)
        mtest_majvote_accus_rep.append(mtest_majvote_accus_cv)
        mtest_majvote_errors_rep.append(mtest_majvote_errors_cv)
        pt_times_rep.append(pt_times_cv)
        ft_times_rep.append(ft_times_cv)
        ground_truth_rep.append(np.hstack(ground_truth))
        prediction_rep.append(np.hstack(prediction))
         
        sda_reuse_pt_models.append(sda_reuse_pt_models_cv)
        sda_reuse_ft2_models.append(sda_reuse_ft2_models_cv)
        sda_reuse_ft_models.append(sda_reuse_ft_models_cv)
         
        print 'np.hstack(ground_truth)', np.hstack(ground_truth)
        print 'np.hstack(prediction)', np.hstack(prediction)
        cm = confusion_matrix(np.hstack(ground_truth), np.hstack(prediction))
        cm_rep.append(cm)
         
        plot_cm(cm, mean(np.hstack(mtest_majvote_accus_cv)), std(np.hstack(mtest_majvote_accus_cv)), target_outputs_dir_repeat, repetition + 1, type_of_cm = 'per_repeatation', labels= moa_items)
        print '0000000000000000000000000000000000000000000000000000000000000000000000000'
        print '====  cross validation results for # ' + str(idx+1) + ' out of '+ str(len(compound_items))
        print 'mean best validation error cv = %.2f(%.2f)' % (mean(best_val_errors_cv), std(best_val_errors_cv))
        print 'best_validation error cv =' + str(best_val_errors_cv)
        print 'results before majority voting'
        print 'mean test error = %.2f(%.2f)' % (mean(np.hstack(mtest_errors_cv)), std(np.hstack(mtest_errors_cv)))
        print 'mean test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_accus_cv)), std(np.hstack(mtest_accus_cv)))
        print 'Test error cv =' + str(mtest_errors_cv)
        print 'Test accu cv  =' + str(mtest_accus_cv)
        print 'results after majority voting'
        print 'mean majority test error = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_errors_cv)), std(np.hstack(mtest_majvote_errors_cv)))
        print 'mean majority test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_accus_cv)), std(np.hstack(mtest_majvote_accus_cv)))
        print 'Test majority error cv =' + str(mtest_majvote_errors_cv)
        print 'Test majority accu cv  =' + str(mtest_majvote_accus_cv)
        print 'Time taken'
         
        if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
            print 'layers retrained by dataset_A : ', retrain_ft_layers
            print 'layers transfered from source: ', tranferred_layers
            print 'Time take for train ft layers in min = ', ft_times_cv
            print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
        elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
            print 'Time take for train pt layers in min = ', pt_times_cv
            print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(pt_times_cv), std(pt_times_cv))
            print 'Time take for train ft layers in min = ', ft_times_cv
            print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
            print 'Total time to train = mean %.2f(%.2f)m' % ((mean(pt_times_cv) +mean(ft_times_cv)), (std(pt_times_cv) +std(ft_times_cv)))
        print '0000000000000000000000000000000000000000000000000000000000000000000000000'  
         
        end_tune_time_cv = time.clock()    
        fcv = open(temp_dir+'Results_crossvalid'+'_rep'+str(repetition+1)+'_'+ gpu_nr+'.txt','a')
        old_stdout = sys.stdout   
        sys.stdout = fcv 
        print 'np.hstack(ground_truth)', np.hstack(ground_truth)
        print 'np.hstack(prediction)', np.hstack(prediction)
        print 'cm', cm
         
        print '0000000000000000000000000000000000000000000000000000000000000000000000000'
        print '====  cross validation results for # ' + str(idx+1) + ' out of '+ str(len(compound_items))
        print 'mean best validation error cv = %.2f(%.2f)' % (mean(best_val_errors_cv), std(best_val_errors_cv))
        print 'best_validation error cv =' + str(best_val_errors_cv)
        print 'results before majority voting'
        print 'mean test error = %.2f(%.2f)' % (mean(np.hstack(mtest_errors_cv)), std(np.hstack(mtest_errors_cv)))
        print 'mean test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_accus_cv)), std(np.hstack(mtest_accus_cv)))
        print 'Test error cv =' + str(mtest_errors_cv)
        print 'Test accu cv  =' + str(mtest_accus_cv)
        print 'results after majority voting'
        print 'mean majority test error = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_errors_cv)), std(np.hstack(mtest_majvote_errors_cv)))
        print 'mean majority test accu  = %.2f(%.2f)' % (mean(np.hstack(mtest_majvote_accus_cv)), std(np.hstack(mtest_majvote_accus_cv)))
        print 'Test majority error cv =' + str(mtest_majvote_errors_cv)
        print 'Test majority accu cv  =' + str(mtest_majvote_accus_cv)
        print 'Time taken'
        if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
            print 'layers retrained by dataset_A : ', retrain_ft_layers
            print 'layers transfered from source: ', tranferred_layers
            print 'Time take for train ft layers in min = ', ft_times_cv
            print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
        elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
            print 'Time take for train pt layers in min = ', pt_times_cv
            print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(pt_times_cv), std(pt_times_cv))
            print 'Time take for train ft layers in min = ', ft_times_cv
            print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(ft_times_cv), std(ft_times_cv))
            print 'Total time to train = mean %.2f(%.2f)m' % ((mean(pt_times_cv) +mean(ft_times_cv)), (std(pt_times_cv) +std(ft_times_cv)))
            print 'The code ran for %.2fm' % ((end_tune_time_cv - start_tune_time) / 60.)
        print '0000000000000000000000000000000000000000000000000000000000000000000000000'  
        sys.stdout=old_stdout 
        fcv.close()          
                 
             
    def cal_rep_scores(dat_rep):
        a = []
        b = []
        for i in range(len(dat_rep)):
            a.append(np.mean(np.hstack(dat_rep[i])))
            b.append(np.std(np.hstack(dat_rep[i])))
        print 'mean of each repititon', a
        print 'std  of each repititon', b
        return np.mean(a), np.std(b)
 
         
    avg_cms = np.mean(cm_rep,axis=0)
    print 'avg_cms', avg_cms
    mean_accu, std_accu = cal_rep_scores(mtest_majvote_accus_rep)
    plot_cm(avg_cms, mean_accu, std_accu, target_outputs_dir_repeat, None, type_of_cm = 'average', labels= moa_items)
     
    print '===================================================================='
    print 'Approach' + approach + ' using GPU ' + gpu_nr
    if transfer == None:
        print 'approach', approach  
    elif transfer % 2 == 0: # even for STS
        print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
    elif transfer % 2 == 1: # odd for TL
        print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
      
    print 'training data fraction ='+ str(training_data_fraction)
    print 'target_dataset details: '+ str(params['dataset_A'])
    print '            features  = '+ str(params['n_ins']) 
    print '             targets  = '+ str(params['n_outs'])
    print 'source_dataset details: '+ str(params['dataset_B'])
    print '             features = '+ str(453)
    print '              targets = '+ str(params['n_outs_source'])
    print 'Architecture details:   '
    print 'hidden_layers_sizes= '+ str(params['hidden_layers_sizes'])
    print 'Max Nr. PT epochs  = '+ str(params['pretraining_epochs'])
    print 'Max Nr. FT epochs  = '+ str(params['training_epochs'])
    print 'PT learning_rate   = '+ str(params['pretrain_lr'])
    print 'FT learning_rate   = '+ str(params['finetune_lr'])
    print 'batch_size         = '+ str(params['batch_size'])
    print 'dropout            = '+ str(dropout)
    print 'dropout_rate       = '+ str(dropout_rate)
    print 'source_reuse_mode  = '+ str(source_reuse_mode)
    print
    print 'Results for the number of repetition'
    print 'best_validation error rep =' + str(best_val_errors_rep)
    print 'results before majority voting'
    print 'Test error rep =' + str(mtest_errors_rep)
    print 'Test accu rep  =' + str(mtest_accus_rep)
    print 'results after majority voting'
    print 'Test majority error rep =' + str(mtest_majvote_errors_rep)
    print 'Test majority accu rep  =' + str(mtest_majvote_accus_rep)
     
 
     
    print 'mean best validation error rep = %.2f(%.2f)' % (mean(mean(best_val_errors_rep, axis=1)), std(std(best_val_errors_rep, axis=1)))    
    print 'results before majority voting'
    print 'mean test error = %.2f(%.2f)' % cal_rep_scores(mtest_errors_rep)  # (mean(np.hstack(mtest_errors_rep)), std(np.hstack(mtest_errors_rep)))
    print 'mean test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_accus_rep)  #(mean(np.hstack(mtest_accus_rep)), std(np.hstack(mtest_accus_rep)))
    print 'results after majority voting'
    print 'mean majority test error = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_errors_rep) #(mean(np.hstack(mtest_majvote_errors_rep)), std(np.hstack(mtest_majvote_errors_rep)))
    print 'mean majority test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_accus_rep)  #(mean(np.hstack(mtest_majvote_accus_rep)), std(np.hstack(mtest_majvote_accus_rep)))
    print 'Time taken'
    if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
        print 'layers retrained by dataset_A : ', retrain_ft_layers
        print 'layers transfered from source: ', tranferred_layers
        print 'Time take for train ft layers in min = ', ft_times_rep
        print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(mean(ft_times_rep, axis=1)), std(std(ft_times_rep, axis=1)))
    elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
        print 'Time take for train pt layers in min = ', pt_times_rep
        print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(mean(pt_times_rep, axis=1)), std(std(pt_times_rep, axis=1)))
        print 'Time take for train ft layers in min = ', ft_times_rep
        print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(mean(ft_times_rep, axis=1)), std(std(ft_times_rep, axis=1)))
        print 'Total time to train = mean %.2f(%.2f)m' % ((mean(mean(pt_times_rep, axis=1)) + mean(mean(ft_times_rep, axis=1))),
                                                           (std(std(pt_times_rep, axis=1)) + std(std(ft_times_rep, axis=1))))
    print
    print '===================================================================='
     
     
    end_tune_time = time.clock()
     
    fm = open(temp_dir+'Results_crossvalid'+'_'+ gpu_nr+'.txt','a')
    old_stdout = sys.stdout   
    sys.stdout = fm
    print '===================================================================='
    print 'Approach' + approach + ' using GPU ' + gpu_nr
    if transfer == None:
        print 'approach', approach  
    elif transfer % 2 == 0: # even for STS
        print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
    elif transfer % 2 == 1: # odd for TL
        print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
      
    print 'training data fraction ='+ str(training_data_fraction)
    print 'target_dataset details: '+ str(params['dataset_A'])
    print '            features  = '+ str(params['n_ins']) 
    print '             targets  = '+ str(params['n_outs'])
    print 'source_dataset details: '+ str(params['dataset_B'])
    print '             features = '+ str(453)
    print '              targets = '+ str(params['n_outs_source'])
    print 'Architecture details:   '
    print 'hidden_layers_sizes= '+ str(params['hidden_layers_sizes'])
    print 'Max Nr. PT epochs  = '+ str(params['pretraining_epochs'])
    print 'Max Nr. FT epochs  = '+ str(params['training_epochs'])
    print 'PT learning_rate   = '+ str(params['pretrain_lr'])
    print 'FT learning_rate   = '+ str(params['finetune_lr'])
    print 'batch_size         = '+ str(params['batch_size'])
    print 'dropout            = '+ str(dropout)
    print 'dropout_rate       = '+ str(dropout_rate)
    print 'source_reuse_mode  = '+ str(source_reuse_mode)
    print
    print 'Results for the number of repetition #######################'
    print 'best_validation error rep =' + str(best_val_errors_rep)
    print 'results before majority voting'
    print 'Test error rep =' + str(mtest_errors_rep)
    print 'Test accu rep  =' + str(mtest_accus_rep)
    print 'results after majority voting'
    print 'Test majority error rep =' + str(mtest_majvote_errors_rep)
    print 'Test majority accu rep  =' + str(mtest_majvote_accus_rep)
    print
    print
    print 'mean best validation error rep = %.2f(%.2f)' % (mean(mean(best_val_errors_rep, axis=1)), std(std(best_val_errors_rep, axis=1)))    
    print 'results before majority voting'
    print 'mean test error = %.2f(%.2f)' % cal_rep_scores(mtest_errors_rep)  # (mean(np.hstack(mtest_errors_rep)), std(np.hstack(mtest_errors_rep)))
    print 'mean test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_accus_rep)  #(mean(np.hstack(mtest_accus_rep)), std(np.hstack(mtest_accus_rep)))
    print 'results after majority voting'
    print 'mean majority test error = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_errors_rep) #(mean(np.hstack(mtest_majvote_errors_rep)), std(np.hstack(mtest_majvote_errors_rep)))
    print 'mean majority test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_accus_rep)  #(mean(np.hstack(mtest_majvote_accus_rep)), std(np.hstack(mtest_majvote_accus_rep)))
    print 'Time taken'
    if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
        print 'layers retrained by dataset_A : ', retrain_ft_layers
        print 'layers transfered from source: ', tranferred_layers
        print 'Time take for train ft layers in min = ', ft_times_rep
        print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(mean(ft_times_rep, axis=1)), std(std(ft_times_rep, axis=1)))
    elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
        print 'Time take for train pt layers in min = ', pt_times_rep
        print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(mean(pt_times_rep, axis=1)), std(std(pt_times_rep, axis=1)))
        print 'Time take for train ft layers in min = ', ft_times_rep
        print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(mean(ft_times_rep, axis=1)), std(std(ft_times_rep, axis=1)))
        print 'Total time to train = mean %.2f(%.2f)m' % ((mean(mean(pt_times_rep, axis=1)) + mean(mean(ft_times_rep, axis=1))),
                                                           (std(std(pt_times_rep, axis=1)) + std(std(ft_times_rep, axis=1))))
    print
    print 'avg_cms', avg_cms
    print
    print 'The code ran for %.2fm' % ((end_tune_time - start_tune_time) / 60.)
    print '===================================================================='        
    sys.stdout=old_stdout
    fm.close()
     
     
    print 'The code ran for %.2fm' % ((end_tune_time - start_tune_time) / 60.)
     
    print >> sys.stderr, ('Testing the Reusability SDA code for file ' +
                       os.path.split(__file__)[1] +
                       ' ran for %.2fm' % ((end_tune_time - start_tune_time) / 60.))
import os, sys
from repetitions_LOOCV import run_n_times

def bl_or_tl(experiment,training_data_fraction,results_dir,params,nr_reps, gpu_nr, start_rep_nr, data_path, fold, source_fold):
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers, tranferred_layers] = experiment
       
    if approach == 'BL':
        params['retrain'] = 0
        params['reset_pt'] = 0

        
        if source_reuse_mode == None:
        
            source_outputs_dir = None
            target_outputs_dir = '{0}{1}_{2:4.2f}/'\
            .format(results_dir,
                    target_dataset,
                    training_data_fraction)
            
        elif source_reuse_mode == 'Join':
            
            source_outputs_dir = source_dataset
            target_outputs_dir = '{0}{1}_{2}_{3}_{4:4.2f}/'\
            .format(results_dir,
                    target_dataset,
                    source_reuse_mode,
                    source_dataset,
                    training_data_fraction)
        
        
        
        
        run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,tranferred_layers,
                    gpu_nr,start_rep_nr,data_path,fold,source_fold, approach)
        
    
    elif approach == 'Tune':
        print 'BL with different architectures approach for parameter search'
        params['retrain'] = 0
        params['reset_pt'] = 0
        nn = params['hidden_layers_sizes']
        pt = params['pretraining_epochs']
        print "params['hidden_layers_sizes']", nn
        print "params['pretraining_epochs']",  pt

        if source_reuse_mode == None:
            source_outputs_dir = None
            target_outputs_dir = '{0}{1}{2}{3}_{4:5.2f}/'\
            .format(results_dir,
                    target_dataset,
                    nn,
                    pt,
                    training_data_fraction)   

            
        print target_outputs_dir
        run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,source_outputs_dir,target_outputs_dir,
                    source_reuse_mode, retrain_ft_layers,tranferred_layers, gpu_nr,start_rep_nr,data_path, fold,source_fold, approach)
         
    elif approach == 'TL':
            params['retrain'] = 1
            if source_reuse_mode == 'PT+FT':
                params['reset_pt']= 0
            elif source_reuse_mode == 'PT+FT+D':
                params['reset_pt']= 0

            elif source_reuse_mode == 'PT':
                params['reset_pt']= 1
            elif source_reuse_mode == 'PT+D':
                params['reset_pt']= 1
            elif source_reuse_mode == 'R':
                params['reset_random']= 1
            

            source_outputs_dir = '{0}{1}'\
                    .format(results_dir,
                            source_dataset)
            source_outputs_dir = source_outputs_dir+'_1.00/'
            target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5}_{6:4.2f}/'\
                 .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         str(tranferred_layers),
                         training_data_fraction)
                 
            run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,tranferred_layers, gpu_nr,start_rep_nr,
                    data_path, fold,source_fold,approach)
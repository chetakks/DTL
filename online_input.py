import sys

sys.argv.pop(0)
approach = sys.argv.pop(0)

if approach == 'BL':
    print 'approach is BL'
    target_dataset = sys.argv.pop(0)
    data_path = sys.argv.pop(0)
    fold = sys.argv.pop(0)
    source_fold = None
    results_dir = sys.argv.pop(0)
    
    nr_reps = sys.argv.pop(0)
    start_rep_nr = 0
    gpu_nr = sys.argv.pop(0)
    
    params = {
    'finetune_lr': sys.argv.pop(0),
    'training_epochs': sys.argv.pop(0),
    'pretrain_lr': sys.argv.pop(0),
    'pretraining_epochs': sys.argv.pop(0),
    'hidden_layers_sizes': sys.argv.pop(0),
    'batch_size':sys.argv.pop(0),         
    'rng_seed': sys.argv.pop(0)
    }

    training_data_fraction = sys.argv.pop(0)
    
    experiment = [approach,target_dataset , None, None, None, None]
    from pprint import pprint
    print experiment
    print 'params='
    pprint(params, width=1)
    
    
elif approach == 'TL':
    print 'approach is TL'
    
    
    data_path = sys.argv.pop(0)
    
    target_dataset = sys.argv.pop(0)
    fold = sys.argv.pop(0)
    source_dataset = sys.argv.pop(0)
    source_fold = sys.argv.pop(0)
    
    source_reuse_mode = sys.argv.pop(0)
    retrain_ft_layers = sys.argv.pop(0)
    tranferred_layers = sys.argv.pop(0)
    
    results_dir = sys.argv.pop(0)
    
    nr_reps = sys.argv.pop(0)
    start_rep_nr = 0
    gpu_nr = sys.argv.pop(0)
    
    params = {
    'finetune_lr': sys.argv.pop(0),
    'training_epochs': sys.argv.pop(0),
    'pretrain_lr': sys.argv.pop(0),
    'pretraining_epochs': sys.argv.pop(0),
    'hidden_layers_sizes': sys.argv.pop(0),
    'batch_size':sys.argv.pop(0),         
    'rng_seed': sys.argv.pop(0)
    }

    training_data_fraction = sys.argv.pop(0)
    
    experiment = [approach,target_dataset,source_dataset, source_reuse_mode, retrain_ft_layers,tranferred_layers]
    
    from pprint import pprint
    print experiment
    print 'params='
    pprint(params, width=1)
    


from approaches import bl_or_tl
params = {
'finetune_lr':0.1,
'pretraining_epochs':1,#60, #30,
'pretrain_lr':0.001,
'training_epochs': 1, #000, #
'hidden_layers_sizes': [5,5,5,5,5,5,5],#[500,500,500,500, 500, 500, 500  ],
'batch_size':100,         
#'output_fold':results_dir, #output_fold,
'rng_seed': 1234
}


import os
host_path = os.getenv('HOME')
data_path=host_path+'/store/Datasets/pickled/'
from pprint import pprint


gpu_nr = 'gpu1'
start_rep_nr = 0
nr_reps = 2
training_data_fractions = [1.00] 

############## results_dir       # fold               # source_fold  
settings = [['results/BL_set1a22/', 'MFC7/set1_loov3a/', None               ],
            ['results/BL_set1a22/', 'MFC7/set2_loov3a/', 'MFC7/set1_loov3a/'],
            ['results/BL_set1a22/', 'MFC7/set2_loov3a/', 'MFC7/set1_loov3a/'],
            ['results/BL_set1a22/', 'MFC7/set2_loov3a/', 'MFC7/set1_loov3a/'],
            ['results/BL_set1a22/', 'MFC7/set2_loov3a/', 'MFC7/set1_loov3a/'],
            #['results/BL_set1_compound/', 'MFC7/set1_loov3a/', None],
            #['results/BL_set1_compound/', 'MFC7/set2_loov3a/', 'MFC7/set1_loov3a_compound/'],
            ]

####[approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers,tranferred_layers]
experiments = [['BL'   ,'MFC7_set1' , None        , None   , None              , None               ],
               ['TL'   ,'MFC7_set2' , 'MFC7_set1' , 'PT+FT', [1,1,1,1,1,1,1,1] , [1,1,1,1,1,1,1,1]  ],
               ['TL'   ,'MFC7_set2' , 'MFC7_set1' , 'PT+FT', [1,1,1,1,1,1,1,1] , [0,0,1,1,1,1,1,1]  ],
               ['TL'   ,'MFC7_set2' , 'MFC7_set1' , 'PT+FT', [1,1,1,1,1,1,1,1] , [0,0,0,0,1,1,1,1]  ],
               ['TL'   ,'MFC7_set2' , 'MFC7_set1' , 'PT+FT', [1,1,1,1,1,1,1,1] , [0,0,0,0,0,0,1,1]  ],
              #['BL'   ,'MFC7_set1_comp' , None , None, None , None  ],
              #['TL'   ,'MFC7_set2' , 'MFC7_set1_comp' , 'PT+FT', [1,1,1,1,1,1,1,1] , [1,1,1,1,1,1,1,1] ],
               ]
                    
for idx, experiment in enumerate(experiments):
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers,tranferred_layers] = experiment
    metadata = {}
    metadata['approach']            = approach
    metadata['target_dataset']      = target_dataset
    metadata['source_dataset']      = source_dataset
    metadata['source_reuse_mode']   = source_reuse_mode
    
    results_dir, fold, source_fold = settings[idx]
    
    params['output_fold'] = results_dir
    experiment = [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers,tranferred_layers]
    for training_data_fraction in training_data_fractions:
        print '='*80
        print 'experiment=',experiment
        print 'settings', settings
        print 'params='
        pprint(params, width=1)
        print 'training_data_fraction=',training_data_fraction
        print '\tretrain_ft_layers', retrain_ft_layers
        print '\ttranferred_layers', tranferred_layers
        bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps, gpu_nr, start_rep_nr, data_path, fold, source_fold)

            
# NOTE: If openblas is installed (libopenblas-base and libopenblas-dev in
# Ubuntu) and multi-processing is needed (enabled by setting nr_processes>0),
# then the OMP_NUM_THREADS environment variable should be set to 1.
    
#from data_handling import *
#from execution import *
from plotting import *

import numpy
import string

if __name__ == "__main__":
    
#     results_dir = 'results/BL_bbbc_feat2/'    
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['BL' , 'bbbc+feat3',     None,    None    , None       ],]
#     file_name ='pt5.png'  
#     #plot_train_info(results_dir,experiments,nr_reps,file_name)
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name)

    
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['BL' , 'bbbc+comp5',     None,   None    , None       ],]
#     file_name ='pt5.png'  
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'compound')
#     
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['BL' , 'bbbc+comp',     None,   None    , None       ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'compound')
# 
#     
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['BL' , 'bbbc+moa',     None,   None    , None       ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'moa')
# 
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['TL' , 'bbbc+comp',    'bbbc+moa',   'PT+FT' , [1,1,1,1]  ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'compound')
#  
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['TL' , 'bbbc+moa',    'bbbc+comp',   'PT+FT' , [1,1,1,1]  ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'moa')
#     
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['TL' , 'bbbc+moa',    'bbbc+comp',   'PT+FT' , [0,1,1,1]  ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'moa')
#     
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['TL' , 'bbbc+moa',    'bbbc+comp',   'PT+FT' , [0,0,1,1]  ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'moa')
#     
#     results_dir = 'results/BL_bbbc_feat4/'
#     nr_reps = 1
#     training_data_fractions = [1.00]
#     experiments = [['TL' , 'bbbc+moa',    'bbbc+comp',   'PT+FT' , [0,0,0,1]  ],]
#     file_name = None 
#     plot_train_batch_info(results_dir,experiments,nr_reps,file_name, 'moa')
    
    results_dir = '/home/aditya/store/Theano/Clean_code/results/BL_bbbc_20rep/' 
    #results_dir ='results/BL_bbbc_20rep/'
    nr_reps = 20#3# 10
    training_data_fractions = [1.00]
    experiments = [['BL' , 'bbbc+comp',     None,   None    , None       ],]
    file_name = None 
    #plot_train_batch_info2(results_dir,experiments,nr_reps,file_name, 'compound')
    
    nr_reps = 1
    training_data_fractions = [1.00]
    plot_layerwise_features(results_dir,experiments,training_data_fractions,nr_reps)
    
#     results_dir = '/home/aditya/store/Theano/Clean_code/results/BL_bbbc_20rep/' 
#     #results_dir ='results/BL_bbbc_20rep/'
#     nr_reps = 20#3# 10
#     training_data_fractions = [1.00]
#     experiments = [['BL' , 'bbbc+moa',     None,   None    , None       ],]
#     file_name = None 
#     plot_train_batch_info2(results_dir,experiments,nr_reps,file_name, 'moa')
#      
#     results_dir = '/home/aditya/store/Theano/Clean_code/results/BL_bbbc_20rep/' 
#     #results_dir ='results/BL_bbbc_20rep/'
#     nr_reps = 20#3# 10
#     training_data_fractions = [1.00]
#     experiments = [['TL',  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [1,1,1,1] ],]
#     file_name = None 
#     plot_train_batch_info2(results_dir,experiments,nr_reps,file_name, 'moa')
#     
#     results_dir = '/home/aditya/store/Theano/Clean_code/results/BL_bbbc_20rep/' 
#     #results_dir ='results/BL_bbbc_20rep/'
#     nr_reps = 20#3# 10
#     training_data_fractions = [1.00]
#     experiments = [['TL',  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,1,1,1] ],]
#     file_name = None 
#     plot_train_batch_info2(results_dir,experiments,nr_reps,file_name, 'moa')
    
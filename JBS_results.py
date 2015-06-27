from results import results_per_rep
''' BL Experiments '''
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set1_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = None
source_fold = None
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer=None)

##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set2_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = None
source_fold = None
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer=None)
# ##################################
# ##################################
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
''' TL Experiments '''
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set2_reusing_MFC7_set1_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[1, 1, 1, 1, 1, 1, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = 'PT+FT'
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = 'MFC7_set1'
source_fold ='MFC7/set1_loov3a/'
transfer = '[11111111]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set2_reusing_MFC7_set1_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 1, 1, 1, 1, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = 'PT+FT'
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = 'MFC7_set1'
source_fold ='MFC7/set1_loov3a/'
transfer = '[00111111]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set2_reusing_MFC7_set1_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 1, 1, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = 'PT+FT'
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = 'MFC7_set1'
source_fold ='MFC7/set1_loov3a/'
transfer = '[00001111]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set2_reusing_MFC7_set1_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 0, 0, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = 'PT+FT'
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = 'MFC7_set1'
source_fold ='MFC7/set1_loov3a/'
transfer = '[00000011]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
##################################
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set1_reusing_MFC7_set2_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[1, 1, 1, 1, 1, 1, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = 'MFC7_set2'
source_fold ='MFC7/set2_loov3a/'
transfer = '[11111111]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps, transfer)
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set1_reusing_MFC7_set2_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 1, 1, 1, 1, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = 'MFC7_set2'
source_fold ='MFC7/set2_loov3a/'
transfer = '[00111111]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set1_reusing_MFC7_set2_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 1, 1, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = 'MFC7_set2'
source_fold ='MFC7/set2_loov3a/'
transfer = '[00001111]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set1_reusing_MFC7_set2_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 0, 0, 1, 1]_1.00/'
approach = 'TL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = 'MFC7_set2'
source_fold ='MFC7/set2_loov3a/'
transfer = '[00000011]'
nr_reps = 10
results_per_rep(result_dir,approach, source_reuse_mode, target_dataset, fold, source_dataset, source_fold, nr_reps,transfer)
##################################
 


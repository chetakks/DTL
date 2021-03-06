from results_MFC7_cm2 import plot_cm
from sklearn.metrics import confusion_matrix
from numpy import mean,std
import numpy as np
import os
from load_MFC7_data3 import load, load_outputs, save
from select_true_set import select

host_path = os.getenv('HOME')
data_path=host_path+'/store/Datasets/pickled/'




##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set1_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = None
source_fold = None
nr_reps = 4

##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set2_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = None
source_fold = None
nr_reps = 4
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set1a/MFC7_set2_reusing_MFC7_set1_PT+FT_[1 1 1 1 1 1 1 1]_[1 1 1 1 1 1 1 1]_1.00/'
approach = 'TL'
source_reuse_mode = 'PT+FT'
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = 'MFC7_set1'
source_fold ='MFC7/set1_loov3a/'
nr_reps = 4
# ##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_set2a/MFC7_set1_reusing_MFC7_set2_None_[1 1 1 1 1 1 1 1]_[1 1 1 1 1 1 1 1]_1.00/'
approach = 'TL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = 'MFC7_set2'
source_fold ='MFC7/set2_loov3a/'
nr_reps = 2
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/BL_set1+2a/MFC7_set1+2_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set1+2'
fold = 'MFC7/loov3a2/'
source_dataset = None
source_fold =None
nr_reps = 8
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/only_true_BL_set1/MFC7_set1_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set1'
fold = 'MFC7/set1_loov3a/'
source_dataset = None
source_fold =None
nr_reps = 8
##################################
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/only_true_BL_set2/MFC7_set2_1.00/'
approach = 'BL'
source_reuse_mode = None
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = None
source_fold =None
nr_reps = 8
##################################
result_dir ='/home/aditya/store/Theano/DTL_HPC2/results2/only_true_BL_set1/MFC7_set2_reusing_MFC7_set1_PT+FT_[1 1 1 1 1 1 1 1]_[0 0 0 0 0 0 1 1]_1.00/'
approach = 'TL'
source_reuse_mode = 'PT+FT'
target_dataset = 'MFC7_set2'
fold = 'MFC7/set2_loov3a/'
source_dataset = 'MFC7_set1'
source_fold ='MFC7/set1_loov3a/'
nr_reps = 6
##################################

if approach == 'BL':
    treatment_items = load(data_path+fold+target_dataset+'_treatment_items.pkl.gz')
    compound_items = load(data_path+fold+target_dataset+'_compound_items.pkl.gz')
    moa_items = load(data_path+fold+target_dataset+'_moa_items.pkl.gz')
elif approach == 'TL':
    treatment_items = load(data_path+fold+target_dataset+'_treatment_items.pkl.gz')
    compound_items = load(data_path+fold+target_dataset+'_compound_items.pkl.gz')
    moa_items = load(data_path+fold+target_dataset+'_moa_items.pkl.gz')
    source_compound_items = load(data_path+source_fold+source_dataset+'_compound_items.pkl.gz')
    source_moa_items = load(data_path+source_fold+source_dataset+'_moa_items.pkl.gz')
    




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
avg_errs = []
avg_accs = []
avg_cms = []

for ri in range(nr_reps):
    print '---------------------------------------------------------repetition: ', ri+1
    best_val_errors_cv = []
    mtest_errors_cv = []
    mtest_accus_cv = []
    mtest_majvote_accus_cv = []
    mtest_majvote_errors_cv = []
    ground_truth    = []
    prediction      = []
    pt_times_cv = []
    ft_times_cv = []     
    
    if target_dataset == 'MFC7_set1' or  target_dataset == 'MFC7_set2':
        two_cl_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/BL_set1+2a/MFC7_set1+2_1.00/'
        true_compound_items, true_treatment_items = select(two_cl_dir,target_dataset,fold, ri)
        looping = true_compound_items 
        MOAs = [0, 1, 2, 3, 4, 5]
        print 'true_compound_items'
    else:
        #MOAs = settings['MOAs']
        MOAs = [0, 1]
        looping = compound_items
    
          
    for idx, compound_label in enumerate(looping):
        print '------------------000000000000000-------------test set compound:',idx, compound_label
        target_outputs_dir = result_dir + compound_label +'/'
        for filename in os.listdir(target_outputs_dir): 
            if ri+1 == int(filename[23:-7]):
                outputs = load(target_outputs_dir+filename)
                print 'filename', filename

        #outputs = outputs_list[ri]
        best_val_error          = outputs['ft_vali_err']
        multi_test_labels       = outputs['multi_test_labels']
        mtest_errors            = outputs['mtest_errors']
        mtest_accus             = outputs['mtest_accus']
        my_test_preds           = outputs['my_test_preds']
        my_tests                = outputs['my_tests']
        my_test_class_probs     = outputs['my_test_class_probs']
        mvote_per_class         = outputs['mvote_per_class']
        my_test_pred_maj_votes  = outputs['my_test_pred_maj_votes']
        mtest_majvote_errors    = outputs['mtest_majvote_errors']
        mtest_majvote_accus     = outputs['mtest_majvote_accus']
        pt_time                 = outputs['pt_trai_times']
        ft_time                 = outputs['ft_trai_time']
        settings                = outputs['settings']
        moa_items               = settings['moa_items']
        idx                     = settings['compound_idx'] 
        compound_label          = settings['compound_label'] 
        
         
        print 'best_validation error =' + str(best_val_error)
        print 'mtest_accus', mtest_accus
        print 'mtest_majvote_accus', mtest_majvote_accus
             
         
 
        for i in range(len(multi_test_labels)):
            print 'mvote_per_class[i]', mvote_per_class[i]
            print 'np.argmax(mvote_per_class[i])', np.argmax(mvote_per_class[i])
            predicted_MOA = MOAs[np.argmax(mvote_per_class[i])]
            print 'True MOA class: %d, predicted MOA: %d' % (np.unique(my_tests[i]), predicted_MOA )
            prediction.append(predicted_MOA)
            ground_truth.append(np.unique(my_tests[i]))
 
        print
        best_val_errors_cv.append(best_val_error)
        mtest_errors_cv.append(mtest_errors)
        mtest_accus_cv.append(mtest_accus)
        mtest_majvote_accus_cv.append(mtest_majvote_accus)
        mtest_majvote_errors_cv.append(mtest_majvote_errors)
        pt_times_cv.append(pt_time)
        ft_times_cv.append(ft_time)
         
 
             
    best_val_errors_rep.append(best_val_errors_cv)
    mtest_errors_rep.append(mtest_errors_cv)
    mtest_accus_rep.append(mtest_accus_cv)
    mtest_majvote_accus_rep.append(mtest_majvote_accus_cv)
    mtest_majvote_errors_rep.append(mtest_majvote_errors_cv)
    pt_times_rep.append(pt_times_cv)
    ft_times_rep.append(ft_times_cv)
    ground_truth_rep.append(np.hstack(ground_truth))
    prediction_rep.append(np.hstack(prediction))         
        
    print 'np.hstack(ground_truth)', np.hstack(ground_truth)
    print 'np.hstack(prediction)', np.hstack(prediction)
     
     
     
    cm = confusion_matrix(np.hstack(ground_truth), np.hstack(prediction))
    print cm
    cm_rep.append(cm)
    plot_cm(cm, mean(np.hstack(mtest_majvote_accus_cv)), std(np.hstack(mtest_majvote_accus_cv)), result_dir, ri + 1, type_of_cm = 'per_repeatation', labels= moa_items,
            approach=approach,target_dataset=target_dataset,source_dataset=source_dataset)
     
    outputs_cv = {}
    outputs_cv['ground_truth']              = np.hstack(ground_truth)
    outputs_cv['prediction']                = np.hstack(prediction)
    outputs_cv['best_val_errors_cv']        = best_val_errors_cv
    outputs_cv['mtest_accus_cv']            = mtest_accus_cv
    outputs_cv['mtest_majvote_accus_cv']    = mtest_majvote_accus_cv
    outputs_cv['pt_times_cv']               = pt_times_cv
    outputs_cv['ft_times_cv']               = ft_times_cv
    outputs_cv['cm']                        = cm
    outputs_cv['treatment_items']           = treatment_items
    outputs_cv['compound_items']            = compound_items
    if target_dataset == 'MFC7_set1' or  target_dataset == 'MFC7_set2':
        outputs_cv['true_treatment_items']      = true_treatment_items
        outputs_cv['true_compound_items']       = true_compound_items
    outputs_cv['moa_items']                 = moa_items
     
 
     
    output_file_path = result_dir+'outputs_cv'+'_%03d.pkl.gz' % (ri+1)
    save(outputs_cv,output_file_path)
      
 
     
 
def cal_rep_scores(dat_rep):
    a = []
    b = []
    for i in range(len(dat_rep)):
        a.append(np.mean(np.hstack(dat_rep[i])))
        b.append(np.std(np.hstack(dat_rep[i])))
    print 'mean of each repititon', a
    print 'std  of each repititon', b
    return np.mean(a), np.std(b)
     
     
print 'mean best validation error rep = %.2f(%.2f)' % cal_rep_scores(best_val_errors_rep)
print 'results before majority voting'
print 'mean test error = %.2f(%.2f)' % cal_rep_scores(mtest_errors_rep)  
print 'mean test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_accus_rep)  
print 'results after majority voting'
print 'mean majority test error = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_errors_rep)
print 'mean majority test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_accus_rep) 
print 'Time taken'
if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
    #print 'layers retrained by dataset_A : ', retrain_ft_layers
    #print 'layers transfered from source: ', tranferred_layers
    print 'Time take for train ft layers in min = ', ft_times_rep
    print 'Time to train ft layers = mean %.2f(%.2f)m' % cal_rep_scores(ft_times_rep)  
elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
    print 'Time take for train pt layers in min = ', pt_times_rep
    print 'Time to train pt layers = mean %.2f(%.2f)m' % cal_rep_scores(pt_times_rep) 
    print 'Time take for train ft layers in min = ', ft_times_rep
    print 'Time to train ft layers = mean %.2f(%.2f)m' % cal_rep_scores(ft_times_rep) 
    print 'Total time to train = mean %.2f(%.2f)m' % cal_rep_scores(np.hstack((pt_times_rep,ft_times_rep))) 
 
 
avg_cms = np.mean(cm_rep,axis=0)
print 'avg_cms', avg_cms
mean_accu, std_accu = cal_rep_scores(mtest_majvote_accus_rep)
plot_cm(avg_cms, mean_accu, std_accu, result_dir, None, type_of_cm = 'average', labels= moa_items,
        approach=approach,target_dataset=target_dataset,source_dataset=source_dataset)
      
output_file_path2 = result_dir+'outputs_avg_conf_mat.pkl.gz'
save(avg_cms,output_file_path2)
    
         
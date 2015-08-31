import csv
import numpy as np
from numpy import mean,std
import os
import pickle
import theano
import cPickle
import gzip



def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
         
def load(filename):
    """Loads a compressed object from disk.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
     
    return object

def nc_data():
    print 'loading the data ... '
    data_path = '/home/aditya/store/Datasets/pickled/MFC7/ljosa_data/'
    cell_id = load(data_path+'cell_id.pkl')
    #features = load(data_path+'features.pkl')
    Image_file = load(data_path+'Image_file.pkl')
    treatment_items = load(data_path+'treatment_items.pkl')
    treatment_id = load(data_path+'treatment_id.pkl')
    compound_items = load(data_path+'compound_items.pkl')
    compound_id = load(data_path+'compound_id.pkl')
    moa_items = load(data_path+'moa_items.pkl')
    moa_id = load(data_path+'moa_id.pkl')
    
    
    nr_samples = 454793
    cell_metadata = np.zeros([nr_samples,6    ])
    cell_metadata[0:,0:3] = cell_id
    cell_metadata[0:,3] = moa_id[0:,0]
    cell_metadata[0:,4] = compound_id[0:,0]
    cell_metadata[0:,5] = treatment_id[0:,0]
    cell_metadata       = cell_metadata.astype(int)
    
    comp_ids            = cell_metadata[0:,4]
    
    # ## Test Cell Metadata
    #print np.unique(cell_metadata[0:,3])
    #print np.unique(cell_metadata[0:,4])
    #print np.unique(cell_metadata[0:,5])
     
    
    # Split the data into control and non-control cells
    ## The compound id of DMSO = 8 ('cutting DMSO from metadata list')
    non_control_cell_metadata = cell_metadata[np.where(comp_ids != 8)[0]]
    for ind in range(len(non_control_cell_metadata)):
        if non_control_cell_metadata[0:,4][ind] > 8:
            non_control_cell_metadata[0:,4][ind] = non_control_cell_metadata[0:,4][ind] -1
            non_control_cell_metadata[0:,5][ind] = non_control_cell_metadata[0:,5][ind] -1
    
    non_control_cell_metadata[0:,5] = non_control_cell_metadata[0:,5].astype(float)
    
    return moa_items, non_control_cell_metadata


def plot_cm(m,mean_accus_cv, std_accus_cv, target_outputs_dir, rep, type_of_cm):
    print 'avg cms'
    #print target_outputs_dir
    #print 'rep', rep
    print target_outputs_dir+type_of_cm+str(rep)+'cm2.png'
    print "Overall average accuracy = %.2f(%.2f)%%" % (mean_accus_cv, std_accus_cv)
    
    cm = m
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #m = np.matrix.round(np.mean(cm,axis=0)).astype(int)
    
    print m
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle('Baseline DNN', fontsize=14, fontweight='bold')
    plt.imshow(cm_normalized, interpolation='nearest',cmap=plt.cm.Blues)

    #plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
    for x_val, y_val in zip(np.arange(np.shape(m)[0]), np.arange(np.shape(m)[1])):
        if np.sum(m[x_val,:]) == 0:
            value = 0
        else: 
            value = int(round((float(m[x_val,y_val]) / np.sum(m[x_val,:])) *100))
        print m[x_val,y_val], np.sum(m[x_val,:]), value
        for idy, val in enumerate(m[x_val]):
            if val != 0:
                plt.text(idy, x_val, m[x_val,idy], fontsize = '15',  va='center', ha='center')
        #plt.text(13, y_val, str(value)+'%', fontsize = '15',  va='center', ha='center')
        plt.text(np.shape(m)[0]+1, y_val, str(value)+'%', fontsize = '15',  va='center', ha='center')
    labels = ['Act', 'Aur', 'Ch', 'DD', 'DR', 'Eg5', 'Epi', 'KI', 'MD', 'MS', 'PD', 'PS']
    #labels = ['MD', 'PS']
    #print "Confusion matrix (Collective): Acc = (%.2f)" % numpy.mean(avg_accs)*100
    #plt.matshow(m)
    #plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
    #plt.title("Overall average accuracy = (%.2f %%)" % (np.mean(mtest_accus_cv)))
    #plt.title("Overall average accuracy = (%.2f %%)" % (accus_cv))
    plt.title("Overall average accuracy = %.2f(%.2f)%%" % (mean_accus_cv, std_accus_cv))
    
    #np.mean(mtest_accus_cv)
    #plt.colorbar()
    plt.ylabel('True mechanistic class')
    plt.yticks(np.arange(np.shape(m)[1]), labels)
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(np.shape(m)[0]), labels, rotation='vertical')
    if type_of_cm == 'per_repeatation':
        #print 'rep', rep
        print target_outputs_dir+type_of_cm+str(rep)+'cm2.png'
        #plt.savefig(target_outputs_dir+type_of_cm+str(rep)+'cm2.png')
        plt.savefig(target_outputs_dir+type_of_cm+str(rep)+'cm2.png',bbox_inches='tight',dpi=200)
    elif type_of_cm == 'average':
        #plt.savefig(target_outputs_dir+type_of_cm+'_cm2.png')
        plt.savefig(target_outputs_dir+type_of_cm+'_cm2.png',bbox_inches='tight',dpi=200)
    plt.close(fig)

from load_MFC7_data2 import load_outputs
##################################
nr_reps = 6
source_reuse_mode = None

path = '/home/aditya/store/Theano/DTL/results/BL_MFC7_loov_per_comp/MFC7_1.00/'
path = '/home/aditya/store/Theano/DTL/results/BL_MFC7_loov_per_comp_gpu0/MFC7_1.00/'
path = '/home/aditya/store/Theano/DTL/results/BL4_MFC7_loov_per_comp_gpu0/MFC7_1.00/'
path = '/home/aditya/store/Theano/DTL/results/BL4_MFC7_loov_per_comp/MFC7_1.00/'

path = '/home/aditya/store/Theano/DTL/results/BL8_MFC7_loov_per_comp/MFC7_1.00/'
#path = '/home/aditya/store/Theano/DTL/results/BL8_MFC7_loov_per_comp_gpu0/MFC7_1.00/'
#path = '/home/aditya/store/Theano/DTL/results/BL9a_gpu0/MFC7_1.00/'
#path = '/home/aditya/store/Theano/DTL/results/BL9a_gpu1/MFC7_1.00/'

path = '/home/aditya/store/Theano/DTL_HPC2/results/BL_12MOA_loov3/MFC7_1.00/'
path = '/home/aditya/store/Theano/DTL_HPC2/results/BL_12MOA_loov3_new/MFC7_1.00/'

##################################

data_path = '/home/aditya/store/Datasets/pickled/MFC7/ljosa_data/'
compound_items = load(data_path+'compound_items.pkl')
nc_compound_items         = np.delete(compound_items, 8)

moa_items, non_control_cell_metadata = nc_data()


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
    
    data_path = '/home/aditya/store/Datasets/pickled/MFC7/ljosa_data/'
    compound_items = load(data_path+'compound_items.pkl')
    nc_compound_items         = np.delete(compound_items, 8)  
    
    
    
    best_val_errors_cv = []
    mtest_errors_cv = []
    mtest_accus_cv = []
    mtest_majvote_accus_cv = []
    mtest_majvote_errors_cv = []
    ground_truth    = []
    prediction      = []
    pt_times_cv = []
    ft_times_cv = []     
          
    for idx, compound_label in enumerate(nc_compound_items):
        if compound_label == 'mevinolin/lovastatin':
            compound_label = 'mevinolin'
        print compound_label
        
        #Find the MOA of a particular compound#
        MOA = np.unique(non_control_cell_metadata[np.where(non_control_cell_metadata[0:,4]==idx)][0:,3])
        print 'The compound belongs to MOA:', moa_items[MOA]

        target_outputs_dir = path + compound_label +'/'
        outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')  
        

        outputs = outputs_list[ri]
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
         
        print multi_test_labels
        print mtest_errors
        
        #mvote_per_class = outputs['mvote_per_class']
        print 'mtest_scores', mtest_errors
        print 'mtest_accus', mtest_accus
        mtest_accus_cv.append(mtest_accus)
        mtest_errors_cv.append(mtest_errors)
        
        for i in range(len(multi_test_labels)):
            #print '...... testing :', multi_test_labels[i]   
            #print 'test error before majority voting:', mtest_scores[i]

            MOAs = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            vote_per_class = []
            #print 'True MOA class:', np.unique(my_tests[i])
            for MOA in MOAs:
                a = np.where(my_test_preds[i] == MOA)
                #print 'Number of predicted elements %d in MOA  %d' % (len(a[0]), MOA)
                vote_per_class.append(len(np.array(a)[0]))
             
            predicted_MOA = MOAs[np.argmax(vote_per_class)]
            print 'True MOA class: %d, predicted MOA: %d' % (np.unique(my_tests[i]), predicted_MOA )
            
            #print 'Test accuracy before majority voting:', 100 - mtest_scores[i]
            #print 'vote_per_class=', mvote_per_class[i]
            #print 'Predicted Test Accuracy =', mtest_accus[i]
            prediction.append(predicted_MOA)
            ground_truth.append(np.unique(my_tests[i]))
        
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
       
        
    
     
#     #print type(mtest_scores_cv)
#     print 'mtest_scores_cv', mtest_errors_cv
#     mtest_errors_cv = np.hstack(mtest_errors_cv)
#     mtest_accus_cv  = np.hstack(mtest_accus_cv)
#     print type(mtest_errors_cv)
#     print 'mtest_errors_cv', mtest_errors_cv
#     print np.mean(mtest_errors_cv)
#     print 
    ground_truth = np.hstack(ground_truth)
    print ground_truth
    prediction = np.hstack(prediction)
    print prediction
     
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ground_truth, prediction)
#     if ri > 0:
#         accus_cv = 100 - np.mean(mtest_accus_cv)
#     else:
#         accus_cv = np.mean(mtest_accus_cv)

    #accus_cv = np.mean(mtest_accus_cv)
    
    mean_accus_cv = mean(np.hstack(mtest_majvote_accus_cv))
    std_accus_cv  = std(np.hstack(mtest_majvote_accus_cv))
    #avg_accs.append(mean_accus_cv)
    #avg_accs.append(std_accus_cv)
    avg_cms.append(cm)
    #print cm
    plot_cm(cm,mean_accus_cv, std_accus_cv, path, ri, 'per_repeatation')
    

def cal_rep_scores(dat_rep):
    a = []
    b = []
    for i in range(len(dat_rep)):
        a.append(np.mean(np.hstack(dat_rep[i])))
        b.append(np.std(np.hstack(dat_rep[i])))
    print 'mean of each repititon', a
    print 'std  of each repititon', b
    return np.mean(a), np.std(b)
    
    
print 'mean best validation error rep = %.2f(%.2f)' % (mean(mean(best_val_errors_rep, axis=1)), std(std(best_val_errors_rep, axis=1)))    
print 'results before majority voting'
print 'mean test error = %.2f(%.2f)' % cal_rep_scores(mtest_errors_rep)  # (mean(np.hstack(mtest_errors_rep)), std(np.hstack(mtest_errors_rep)))
print 'mean test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_accus_rep)  #(mean(np.hstack(mtest_accus_rep)), std(np.hstack(mtest_accus_rep)))
print 'results after majority voting'
print 'mean majority test error = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_errors_rep) #(mean(np.hstack(mtest_majvote_errors_rep)), std(np.hstack(mtest_majvote_errors_rep)))
print 'mean majority test accu  = %.2f(%.2f)' % cal_rep_scores(mtest_majvote_accus_rep)  #(mean(np.hstack(mtest_majvote_accus_rep)), std(np.hstack(mtest_majvote_accus_rep)))
print 'Time taken'
if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
    #print 'layers retrained by dataset_A : ', retrain_ft_layers
    #print 'layers transfered from source: ', tranferred_layers
    print 'Time take for train ft layers in min = ', ft_times_rep
    print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(mean(ft_times_rep, axis=1)), std(std(ft_times_rep, axis=1)))
elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D' or source_reuse_mode == None:
    print 'Time take for train pt layers in min = ', pt_times_rep
    print 'Time to train pt layers = mean %.2f(%.2f)m' % (mean(mean(pt_times_rep, axis=1)), std(std(pt_times_rep, axis=1)))
    print 'Time take for train ft layers in min = ', ft_times_rep
    print 'Time to train ft layers = mean %.2f(%.2f)m' % (mean(mean(ft_times_rep, axis=1)), std(std(ft_times_rep, axis=1)))
    print 'Total time to train = mean %.2f(%.2f)m' % ((mean(mean(pt_times_rep, axis=1)) + mean(mean(ft_times_rep, axis=1))),
                                                       (std(std(pt_times_rep, axis=1)) + std(std(ft_times_rep, axis=1))))
    

     
print 'avg cms'
#m = np.matrix.round(np.mean(avg_cms,axis=0)).astype(int)
m = np.mean(avg_cms,axis=0)
print 'm', m
m = m.astype(float)
m = format(m, '.1f')
print 'm', m

# print 'avg_accs', avg_accs
# avg = np.mean(avg_accs)
# print 'avg_accs', avg

mean_avg_accus_cv, std_avg_accus_cv = cal_rep_scores(mtest_majvote_accus_rep)
plot_cm(m, mean_avg_accus_cv, std_avg_accus_cv, path, None, 'average')
    
        
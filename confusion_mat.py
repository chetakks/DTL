from load_MFC7_data3 import load, load_outputs, save
import numpy as np


two_cl_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/BL_set1+2a/MFC7_set1+2_1.00/'
two_cl_avg_cms = load(two_cl_dir+'outputs_avg_conf_mat.pkl.gz')
print 'two_cl_avg_cms ', two_cl_avg_cms 

bl1_result_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/only_true_BL_set1/MFC7_set1_1.00/'
bl1_avg_cms = load(bl1_result_dir+'outputs_avg_conf_mat.pkl.gz')
print 'bl1_avg_cms', bl1_avg_cms
moa_items1       = load('/home/aditya/store/Datasets/pickled/MFC7/set1_loov3a/MFC7_set1_moa_items.pkl.gz')


#bl2_result_dir = '/home/aditya/store/Theano/DTL_HPC2/results2/only_true_BL_set2/MFC7_set2_1.00/'
#avg_cms2 = load(bl2_result_dir+'outputs_avg_conf_mat.pkl.gz')
#print 'bl2_avg_cms', bl2_avg_cms
moa_items2       = load('/home/aditya/store/Datasets/pickled/MFC7/set2_loov3a/MFC7_set2_moa_items.pkl.gz')

#print np.shape(avg_cms)

avg_cms1 = bl1_avg_cms
#avg_cms2 = bl1_avg_cms

print two_cl_avg_cms

 
#b1 = np.empty((6,6))
b1 = np.zeros((6,6))
b1.fill(two_cl_avg_cms[0,1]/6)
print b1
#b2 = np.empty((6,6))

b2 = np.zeros((6,6))
b2.fill(two_cl_avg_cms[1,0]/6)
print b2
 
total_cm1 = np.concatenate((bl1_avg_cms, b1), axis=1)
total_cm2 = np.concatenate((b2, bl1_avg_cms), axis=1)
total_cm  = np.concatenate((total_cm1, total_cm2), axis=0)
print total_cm
print np.shape(total_cm)

cm1 = np.concatenate((bl1_avg_cms,np.zeros((6,6))), axis=1)
cm2 = np.concatenate((np.zeros((6,6)), bl1_avg_cms), axis=1)
zero_cm  = np.concatenate((cm1, cm2), axis=0)
print zero_cm
print np.shape(zero_cm)


a1 = np.zeros((6, 6))
np.fill_diagonal(a1, two_cl_avg_cms[0,1]/6)
print a1
a2 = np.zeros((6, 6))
np.fill_diagonal(a2, two_cl_avg_cms[1,0]/6)
print a2
diag_cm1 = np.concatenate((bl1_avg_cms, a1), axis=1)
diag_cm2 = np.concatenate((a2, bl1_avg_cms), axis=1)
diag_cm  = np.concatenate((diag_cm1, diag_cm2), axis=0)
print diag_cm
print np.shape(diag_cm)


cm1 = np.concatenate((bl1_avg_cms,np.zeros((6,6))), axis=1)
cm2 = np.concatenate((np.zeros((6,6)), bl1_avg_cms), axis=1)
zero_cm  = np.concatenate((cm1, cm2), axis=0)
print zero_cm
print np.shape(zero_cm)





m = total_cm
approach = 'BL'
mean_accu = 100
std_accu = 1
type_of_cm = 'average'
labels = np.concatenate((moa_items1, moa_items2), axis=1)
rep = None

print labels
import numpy as np
import matplotlib.pyplot as plt


#plot_cm(m,mean_accu, std_accu, target_outputs_dir, rep, type_of_cm,labels, approach=None,target_dataset=None,source_dataset=None ):
print 'avg cms'
#print 
target_outputs_dir = two_cl_dir

cm = m

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#m = np.matrix.round(np.mean(cm,axis=0)).astype(int)

#print m
m = np.around(diag_cm, decimals=1)
#print m

fig = plt.figure()

if approach == 'BL':
    fig.suptitle('Baseline DNN', fontsize=14, fontweight='bold')
if approach == 'TL':
    print
    #fig.suptitle('Reusing '+ source_dataset +' to classify '+ target_dataset, fontsize=14, fontweight='bold')



plt.imshow(cm_normalized, interpolation='nearest',cmap=plt.cm.Blues)

#plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
for x_val, y_val in zip(np.arange(np.shape(m)[0]), np.arange(np.shape(m)[1])):
#for x_val, y_val in zip(np.arange(6), np.arange(6)):
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
#labels = ['Act', 'Aur', 'Ch', 'DD', 'DR', 'Eg5', 'Epi', 'KI', 'MD', 'MS', 'PD', 'PS']
#labels = ['1', '2', '3', '4', '5', '6']
#labels = ['MD', 'PS']
#print "Confusion matrix (Collective): Acc = (%.2f)" % numpy.mean(avg_accs)*100
#plt.matshow(m)
#plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
#plt.title("Overall average accuracy = (%.2f %%)" % (np.mean(mtest_accus_cv)))


plt.title("Overall average accuracy = %.2f(%.2f)%%" % (mean_accu, std_accu))


#plt.colorbar()
plt.ylabel('True mechanistic class')
plt.yticks(np.arange(np.shape(m)[1]), labels)
plt.xlabel('Predicted label')
plt.xticks(np.arange(np.shape(m)[0]), labels, rotation='vertical')
if type_of_cm == 'per_repeatation':
    print target_outputs_dir+type_of_cm+str(rep)+'cm.png'
    plt.savefig(target_outputs_dir+type_of_cm+str(rep)+'cm_full.png',bbox_inches='tight',dpi=200)
elif type_of_cm == 'average':
    plt.savefig(target_outputs_dir+type_of_cm+'_cm_full.png',bbox_inches='tight',dpi=200)
plt.close(fig)
    
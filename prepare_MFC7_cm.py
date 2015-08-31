import csv
import numpy as np
import os
import pickle
import theano
import cPickle
import gzip


def load(filename):
    """Loads a compressed object from disk.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
     
    return object

def MFC7_cm():
    
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
print np.unique(cell_metadata[0:,3])
print np.unique(cell_metadata[0:,4])
print np.unique(cell_metadata[0:,5])
 

# Split the data into control and non-control cells
## The compound id of DMSO = 8 ('cutting DMSO from metadata list')
non_control_cell_metadata = cell_metadata[np.where(comp_ids != 8)[0]]
for ind in range(len(non_control_cell_metadata)):
    if non_control_cell_metadata[0:,4][ind] > 8:
        non_control_cell_metadata[0:,4][ind] = non_control_cell_metadata[0:,4][ind] -1
        non_control_cell_metadata[0:,5][ind] = non_control_cell_metadata[0:,5][ind] -1

non_control_cell_metadata[0:,5] = non_control_cell_metadata[0:,5].astype(float)



print non_control_cell_metadata[0]




nc_compound_items         = np.delete(compound_items, 8)

#nc_compound_items = np.unique(non_control_cell_metadata[0:,4])

# ## Test Cell Metadata
print np.unique(non_control_cell_metadata[0:,3])
print np.unique(non_control_cell_metadata[0:,4])
print np.unique(non_control_cell_metadata[0:,5])

for idx, compound_label in enumerate(nc_compound_items):
    if compound_label == 'mevinolin/lovastatin':
        non_control_cell_metadata[0:,4] = [s.replace('mevinolin/lovastatin', 'mevinolin') for s in non_control_cell_metadata[0:,4]]
        compound_label = 'mevinolin'
    treatments = np.where(np.unique(non_control_cell_metadata[0:,4]) == compound_label )
    print 'compound', compound_label
    
    


nc_treatment_items        = treatment_items[np.where(treatment_items[0:,0] != 'DMSO')]
nc_treatment_items[0:,1]  = nc_treatment_items[0:,1].astype(float)
nc_compound_items         = np.delete(compound_items, 8)

for idx, compound_label in enumerate(nc_compound_items):
    if compound_label == 'mevinolin/lovastatin':
        nc_treatment_items[0:,0] = [s.replace('mevinolin/lovastatin', 'mevinolin') for s in nc_treatment_items[0:,0]]
        compound_label = 'mevinolin'
    treatments = np.where(nc_treatment_items[0:,0] == compound_label )
    print 'compound', compound_label
    #print 'treatments for the compound:'
    #print nc_treatment_items[treatments]
    
    host_path = os.getenv('HOME')
    path=host_path+'/store/Datasets/pickled/MFC7/loov_with_per_treatment/'

    multi_test_labels = []
    for ids in  range(len(np.array(treatments)[0])):
        idst = treatments[0][ids]
        #print idst
        f1  = 'MFC7_'+nc_treatment_items[idst][0]+'_'+nc_treatment_items[idst][1]+'_test.pkl.gz'
        if not os.path.exists(path+f1):
            print f1
            print 'data path is wrong for the file'
        multi_test_labels.append(f1)       
        print '     treatments', f1       
#             for i in range(len(multi_test_labels)):
#                 print 'Test set results for ......', multi_test_labels[i]
#                 print 'True MOA class:', np.unique(my_tests[i])
#                 print 'Test accuracy before majority voting:', 100 - mtest_scores[i]
#                 print 'vote_per_class=', mvote_per_class[i]
#                 print 'maj_vote', np.argmax(mvote_per_class[i])  
#                 print 'Predicted Test Accuracy =', mtest_accus[i]
    
    
    
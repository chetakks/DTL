import numpy as np
import cPickle
import gzip
from prepare_data import relable_data, prepare_metadata, cv_dat, prepare_data, normalize
from load_MFC7_data3 import load, save

         






############################### load data ########################
print 'loading the data ... '
data_path = '/home/aditya/store/Datasets/pickled/MFC7/DMSO_data/'
cell_metadata = load(data_path+'cell_metadata.pkl.gz')
features = load(data_path+'features.pkl.gz')
nr_samples = load(data_path+'nr_samples.pkl.gz')
cell_metadata = load(data_path+'cell_metadata.pkl.gz')
meta_items = load(data_path+'meta_items.pkl.gz')
feature_names = load(data_path+'feature_names.pkl.gz')
############################### load data ########################

print 'Set classification task as classifying MOA labels'
dat_target = cell_metadata[0:,3].astype(int)
MOA_ids = np.unique(dat_target)
print 'dat_target', MOA_ids

cv_by_hand = cv_dat(cell_metadata)
# Normalize the features between [0 and 1]
dat_inp    = normalize(features)
print 'normalized features', dat_inp

print ' Non-control compound labels'
comp_labels = cell_metadata[0:,4].astype(int)


output_folder = '/home/aditya/store/Datasets/pickled/MFC7/norm/'
name = 'norm'

moa_items        = meta_items['nc_moa_items'] 
compound_items   = meta_items['nc_compound_items'] 
treatment_items  = meta_items['nc_treatment_items'] 
    
print moa_items
print compound_items
print treatment_items

metadat = prepare_metadata(moa_items, MOA_ids, cell_metadata, treatment_items)


    
print metadat.keys()

prepare_data(output_folder,cv_by_hand,
             comp_labels, 
             moa_items,
             MOA_ids, 
             compound_items,treatment_items, 
             cell_metadata , 
             dat_inp,
             dat_target, metadat, name )




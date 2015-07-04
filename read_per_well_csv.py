# !/usr/bin/python      
import os
import csv
import numpy as np
from load_per_well_csv import load_per_well_csv
from load_MFC7_data3 import load, save


print 'loading the data ... '
#data_path = '/home/aditya_1t/BBBC_data/Data_S3_reproduce/output/'
data_path = '/home/aditya/store/Datasets/pickled/MFC7/per_well_MFC7_output/'
output_folder = '/home/aditya/store/Datasets/pickled/MFC7/DMSO_data/'

treatment_items = load(data_path+'treatment_items.pkl')
compound_items = load(data_path+'compound_items.pkl')
moa_items = load(data_path+'moa_items.pkl')

compound_items[31] = 'mevinolin'
treatment_items[78][0] = 'mevinolin'
treatment_items[79][0] = 'mevinolin'
treatment_items[80][0] = 'mevinolin'

# print 'treatment_items', treatment_items
# print 'compound_items', compound_items
# print 'moa_items', moa_items

nc_moa_items              = np.delete(moa_items, 3)
nc_compound_items         = np.delete(compound_items, 8)
nc_treatment_items        = treatment_items[np.where(treatment_items[0:,0] != 'DMSO')]
nc_treatment_items[0:,1]  = nc_treatment_items[0:,1].astype(float)

# print 'nc_moa_items', nc_moa_items
# print 'nc_compound_items', nc_compound_items
# print 'nc_treatment_items', nc_treatmen

meta_items = {}
meta_items['nc_moa_items']            = nc_moa_items
meta_items['nc_compound_items']       = nc_compound_items
meta_items['nc_treatment_items']      = nc_treatment_items

count = 0
for filename in os.listdir(data_path):
    if filename.startswith('percell') and not filename.endswith('DMSO.csv'):
        non_DMSO_filename = filename
        print 'non_DMSO_filename', count, non_DMSO_filename
        if count == 0:
            nr_samples, cell_metadata, features = load_per_well_csv(data_path,non_DMSO_filename, meta_items)
        elif count > 0:
            nr_samples_new, cell_metadata_new, features_new = load_per_well_csv(data_path,non_DMSO_filename, meta_items)
            nr_samples= nr_samples + nr_samples_new
            cell_metadata = np.vstack((cell_metadata, cell_metadata_new))
            features      = np.vstack((features, features_new))
        count = count + 1
        #if count == 3:
        #    break
print cell_metadata
print features


save(cell_metadata,output_folder+'cell_metadata.pkl.gz')
save(features,output_folder+'features.pkl.gz')
save(nr_samples,output_folder+'nr_samples.pkl.gz')
save(cell_metadata,output_folder+'cell_metadata.pkl.gz')
save(meta_items,output_folder+'meta_items.pkl.gz')

ifile  = open(data_path + non_DMSO_filename, "rb")
print 'loading the data ...'
reader = csv.reader(ifile, delimiter=',')
for idx, row in enumerate(reader):
#     a=raw_input()
#     print row
    if idx == 0:
        read_row = csv.reader(row, delimiter='\t')
        feature_names = []
        for  idy, col in enumerate(read_row):
            if idy > 5:
                print 'features', idy, col[0]
                feature_names.append(col[0])
#print feature_names
save(feature_names,output_folder+'feature_names.pkl.gz')
print 'nr_samples', nr_samples
# nr_samples = 454793



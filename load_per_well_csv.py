def load_per_well_csv(data_path,non_DMSO_filename, meta_items): 
    import csv
    import numpy as np
    nc_moa_items        = meta_items['nc_moa_items'] 
    nc_compound_items   = meta_items['nc_compound_items'] 
    nc_treatment_items  = meta_items['nc_treatment_items'] 
    
    ifile1  = open(data_path + non_DMSO_filename, "rb")
    reader = csv.reader(ifile1, delimiter=',')
    nr_samples = sum(1 for row in reader) -1
    print 'nr_samples', nr_samples
    ifile1.close()
    
    cell_metadata = np.zeros([nr_samples,6   ])
    features      = np.zeros([nr_samples,453 ])
    
    ifile2  = open(data_path + non_DMSO_filename, "rb")
    print 'loading the data ...'
    reader = csv.reader(ifile2, delimiter=',')
    for idx, row in enumerate(reader):
        if idx > 0:
            if row[2] == 'mevinolin/lovastatin':
                row[2] = 'mevinolin'
# #             a=raw_input()
# #             print 'row[0].split("_")[0][4:]', row[0].split("_")[0][4:]
# #             print 'row[0].split("_")[1]', row[0].split("_")[1]
# #             print 'row[1]', row[1]
# #             print 'row[2]', row[2]
# #             print 'row[3]', str(float(row[3]))
# #             print 'row[4]', row[4]
# #             print 'row[5]', row[5]
# #             print 'row[6:453+6]', row[6:453+6]
# #             print 'MOA_id', np.where(nc_moa_items == row[4])[0][0]
# #             print 'compound_id', np.where(nc_compound_items== row[2])[0][0]
# #             print 'treatment_id', np.where( (nc_treatment_items[:,0] == row[2]) & \
# #                                             (nc_treatment_items[:,1] == str(float(row[3]))) )[0][0]
            
            ### batch_id 
            cell_metadata[idx-1][0] = row[0].split("_")[0][4:]
            ### well_id
            cell_metadata[idx-1][1] = row[0].split("_")[1]
            ### cell_id
            cell_metadata[idx-1][2] = row[5]
            ### MOA_id
            cell_metadata[idx-1][3] = np.where(nc_moa_items == row[4])[0][0]
            ###  compound_id
            cell_metadata[idx-1][4] = np.where(nc_compound_items== row[2])[0][0]
            ### treatment_id
            cell_metadata[idx-1][5] = np.where( (nc_treatment_items[:,0] == row[2]) & \
                                                (nc_treatment_items[:,1] == str(float(row[3]))) )[0][0]
            ### 453 features
            features[idx-1][:]       = row[6:453+6]
            #print 'features', features
            #if idx == 6000:
            #    break
    ifile2.close()
     
    return nr_samples, cell_metadata.astype(int), features
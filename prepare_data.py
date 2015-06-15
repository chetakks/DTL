import numpy as np
import os
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

def normalize(dat_inp):
    """Normalizes the input array between range [0,1].
       using Sklearn preprocessing module.
    """
    #normalize the features between [-1,1]
    from sklearn import preprocessing
    
    #min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    min_max_scaler = preprocessing.MinMaxScaler((0,1))
    dat_inp = min_max_scaler.fit_transform(dat_inp)
    
    #dat_inp = preprocessing.scale(dat_inp)
    #dat_inp_normalized = preprocessing.normalize(dat_inp, norm='l2')
    #print np.amin(dat_inp,axis=1)
    #print np.amax(dat_inp,axis=1)
    
    return dat_inp
    
def relable_data(dat):
    for idx, val in enumerate(np.unique(dat)):
        dat[dat == val] = idx 
    return dat

def relable_two_cl_data(dat): 
    for idx, val in enumerate(np.unique(dat)):
        if  val == 0 or val == 4 or val == 6 or  val == 7 or  val == 9 or  val == 10:  
            dat[dat == val] = 0
        elif val == 1 or val == 2 or val == 3 or  val == 5 or  val == 8 or  val == 11:
            dat[dat == val] = 1
    return dat
    
    
def prepare_metadata(set_moa_items, set_MOA_ids, set_cell_org_metadata, set_treatment_items):
    metadata = {}
    for idx, moa_item in enumerate(set_moa_items):
        print 'moa_item', moa_item
        comps = np.unique(set_cell_org_metadata[np.where(set_cell_org_metadata[0:,3]==set_MOA_ids[idx])][0:,5])
        metadata[set_moa_items[idx]] = set_treatment_items[comps]
        print 'compounds', set_treatment_items[comps]
    
    return metadata

def cv_dat(set_cell_org_metadata):
    # Non-control compound labels
    set_comp_labels = set_cell_org_metadata[0:,4]
    # create train and test folds from our labels for actual data:
    cv_by_hand = [(np.where(set_comp_labels != label)[0], np.where(set_comp_labels == label)[0])
                   for label in np.unique(set_comp_labels)]
    return cv_by_hand

def dat_info(dat_set):
    print 'nr instances: ', len(dat_set[0])
    print 'nr features:  ',len(dat_set[0][0])
    print 'nr targets:   ', len(list(set(dat_set[1])))
    print 'targets are:  ', list(set(dat_set[1])) 
    
def convert_theano_format(dat_inp,dat_tar):
    dat_set = []
    dat_set = (np.array(dat_inp).astype(theano.config.floatX),
            np.array(dat_tar.flatten()).astype(theano.config.floatX))
    return dat_set

def prepare_compound_data(output_folder,target_data, set_dat_inp,set_dat_tar, set_cell_org_metadata):
    from sklearn.cross_validation import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(set_dat_inp, set_dat_tar, test_size=0.15, random_state=42)
    #X_train_met, X_test_met, dummy1, dummy1 = train_test_split(set_cell_org_metadata, set_dat_tar, test_size=0.15, random_state=42)
    
    X_train_only, X_valid, y_train_only, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    #X_train_only_met, X_valid_met, dummy3, dummy4 = train_test_split(X_train_met, X_test_met, test_size=0.15, random_state=42)
    
    
    tra_inp  = X_train_only
    tra_tar  = y_train_only
    #tra_met  = X_train_only_met
    val_inp  = X_valid
    val_tar  = y_valid
    #val_met  = X_valid_met
    tes_inp  = X_test
    tes_tar  = y_test
    #tes_met  = X_test_met

    train_set = convert_theano_format(tra_inp,tra_tar)
    print 'training data details'
    dat_info(train_set)
    
    valid_set = convert_theano_format(val_inp,val_tar)
    print 'validation data details'
    dat_info(valid_set)
    
    test_set = convert_theano_format(tes_inp,tes_tar)
    print 'test data details'
    dat_info(test_set) 

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
     
    os.chdir(output_folder)
    
    
    print target_data+'_train.pkl.gz'
    print target_data+'_valid.pkl.gz'
    print target_data+'_test.pkl.gz'
    
    save(train_set,target_data+'_train.pkl.gz')
    save(valid_set,target_data+'_valid.pkl.gz')
    save(test_set, target_data+'_test.pkl.gz' )
    #save(tra_met,target_data+'met_train.pkl.gz')
    #save(val_met,target_data+'met_valid.pkl.gz')
    #save(tes_met,target_data+'met_test.pkl.gz' )
    
    print "done writing"

         
def prepare_data(output_folder,cv_by_hand,set_comp_labels, set_moa_items,set_MOA_ids, set_compound_items,set_treatment_items, set_cell_org_metadata, dat_inp,dat_target, set_metadat,name ):
    
    ground_truth = []
    for count, t in enumerate(cv_by_hand):
            
        print 'count', count
        print 'Saving data for the compound .................................', set_compound_items[count]
        
        train_index, test_index = t
        
        #Find the MOA of a particular compound#
        MOA = np.unique(set_cell_org_metadata[np.where(set_cell_org_metadata[0:,4]==count)][0:,3])
        #print 'The compound belongs to MOA:', set_moa_items[MOA]
        #print 'The compound belongs to MOA:', nc_moa_items[MOA]
        
        #Find the list of compounds for a particular MOA
        comps = np.unique(set_cell_org_metadata[np.where(set_cell_org_metadata[0:,3]==MOA)][0:,4])
         
        print 'All compound that belongs to this MOA are:', set_compound_items[comps]
        tes_id = count
        val_id = comps[comps != count][0]
        tra_idx = np.where((set_comp_labels != val_id) & (set_comp_labels != tes_id))[0]
        tmp_idx = np.where(set_comp_labels == val_id)[0]
        stp = np.floor(len(tmp_idx)/2).astype(int)
        val_idx = tmp_idx[0: stp]
        
        tra_idx = np.concatenate([tra_idx, tmp_idx[stp:]])

        tra_inp  = dat_inp[tra_idx]
        tra_tar  = dat_target[tra_idx]
        tra_met  = set_cell_org_metadata[tra_idx]
        val_inp  = dat_inp[val_idx]
        val_tar  = dat_target[val_idx]
        val_met  = set_cell_org_metadata[val_idx]
        print 'Consistency check:               ',len(train_index), (len(tra_inp)+len(val_inp)),(len(tra_met)+len(val_met))
        print 'Compounds used in training set   ', set_compound_items[np.unique(tra_met[0:,4])]
        print 'Compounds used in validation set ', set_compound_items[np.unique(val_met[0:,4])]
        print 'Compounds used in test set       ', set_compound_items[count]
        
        train_set = convert_theano_format(tra_inp,tra_tar)
        print 'training data details'
        dat_info(train_set)
        
        valid_set = convert_theano_format(val_inp,val_tar)
        print 'validation data details'
        dat_info(valid_set)
                    
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        
        os.chdir(output_folder)   
        print 'MFC7_'+name+'_'+set_compound_items[count]+'_train.pkl.gz'
        print 'MFC7_'+name+'_'+set_compound_items[count]+'_valid.pkl.gz'
        save(train_set,'MFC7_'+name+'_'+set_compound_items[count]+'_train.pkl.gz')
        save(valid_set,'MFC7_'+name+'_'+set_compound_items[count]+'_valid.pkl.gz')
        save(tra_met,'MFC7_'+name+'_'+'met_'+set_compound_items[count]+'_train.pkl.gz')
        save(val_met,'MFC7_'+name+'_'+'met_'+set_compound_items[count]+'_valid.pkl.gz')
          
        print 'Treatment TEST:', np.unique(set_cell_org_metadata[test_index][0:,5])
        
        n_test_sets = np.unique(set_cell_org_metadata[test_index][0:,5])
        print 'n_test_sets   :', n_test_sets
        #print 'nr. of test_set concentrations   :', nc_treatment_items[n_test_sets]
        
            ##################### Preparing multiple test sets with per treatment for the compound x  ##########################
        for i, test_set_id in enumerate(n_test_sets):
            #print i, test_set_id 
            test_set_index = np.where(set_cell_org_metadata[0:,5] == test_set_id)
            #print np.unique(non_control_cell_metadata[test_set_index][0:,5]), len(non_control_cell_metadata[test_set_index][0:,5])
            
            tes_inp = dat_inp[test_set_index]
            tes_tar = dat_target[test_set_index]
            tes_met = set_cell_org_metadata[test_set_index]
            
            ground_truth.append(np.unique(tes_tar))
            
            test_set = convert_theano_format(tes_inp,tes_tar)
            print 'test data details'
            dat_info(test_set) 
            
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            
            os.chdir(output_folder)
            
            print 'MFC7_'+name+'_'+set_treatment_items[test_set_id][0]+'_'+str(set_treatment_items[test_set_id][1])+'_test.pkl.gz'
            save(test_set,'MFC7_'+name+'_'+set_treatment_items[test_set_id][0]+'_'+str(set_treatment_items[test_set_id][1])+'_test.pkl.gz')
            save(tes_met,'MFC7_'+name+'_'+'met_'+set_treatment_items[test_set_id][0]+'_'+str(set_treatment_items[test_set_id][1])+'_test.pkl.gz') 
    ##################### Preparing multiple test sets with per treatment for the compound x  ##########################
    
    print "completed writing all compounds with per treatment test sets" 
    
    ground_truth = np.hstack(ground_truth)
    print len(ground_truth)
    print np.unique(ground_truth)
    print ground_truth
    
    save(ground_truth,'MFC7_'+name+'_ground_truth.pkl.gz')
    save(set_cell_org_metadata, 'MFC7_'+name+'_cell_org_metadata.pkl.gz')
    save(set_metadat, 'MFC7_'+name+'_metadat.pkl.gz')
    save(set_moa_items,'MFC7_'+name+'_moa_items.pkl.gz')
    save(set_MOA_ids,'MFC7_'+name+'_moa_ids.pkl.gz' )
    save(set_compound_items, 'MFC7_'+name+'_compound_items.pkl.gz')
    save(set_treatment_items, 'MFC7_'+name+'_treatment_items.pkl.gz')
    
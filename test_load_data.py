import cPickle
import gzip
import numpy as np
import csv
import os

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

def normalize(dat_inp, min_max_scaler_train=None):
    """Normalizes the input array between range [0,1].
       using Sklearn preprocessing module.
    """
    #normalize the features between [-1,1]
    from sklearn import preprocessing
    
    #min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    #min_max_scaler = preprocessing.MinMaxScaler((0,1))
    #dat_inp = min_max_scaler.fit_transform(dat_inp)
    
    #dat_inp = preprocessing.scale(dat_inp)
    #dat_inp_normalized = preprocessing.normalize(dat_inp, norm='l2')
    #print np.amin(dat_inp,axis=1)
    #print np.amax(dat_inp,axis=1)
    
#     min_max_scaler = preprocessing.MinMaxScaler((-1,1)).fit(dat_inp)
#     X_train_std = min_max_scaler.transform(dat_inp)
#     X_test_std  = min_max_scaler.transform(dat_inp)
    
    if min_max_scaler_train == None:
        min_max_scaler = preprocessing.MinMaxScaler((-1,1)).fit(dat_inp)
        dat_inp = min_max_scaler.transform(dat_inp)
        print 'min_max_scaler', min_max_scaler
        return dat_inp, min_max_scaler
    elif min_max_scaler_train is not None:
        dat_inp  = min_max_scaler_train.transform(dat_inp)
        return dat_inp
        



def shuffel(data,rng_seed,frac):

    nr_examples = len(data[0])
    num_of_samples = int(np.floor(nr_examples*frac))
    
    np.random.seed(rng_seed)
    rand_inds = np.random.permutation(num_of_samples)   
    data = (np.array(data[0][rand_inds]),
                 np.array(data[1][rand_inds]))
    return data



def load_theano_data(path, filename, frac, rng_seed, min_max_scaler_train=None):
    dat_set = load(path+filename+'.pkl.gz')
    print 'nr instances:   ', len(dat_set[0])
    print 'nr features:    ', len(dat_set[0][0])
    print 'nr targets:     ', len(list(set(dat_set[1])))
    print 'shape of dat set', np.shape(dat_set[0])
    print 'shape of dat set', np.shape(dat_set[1])

    dat_set = shuffel(dat_set,rng_seed,1)
    nr_samples = int(len(dat_set[0])*frac)
    print 'nr_samples', nr_samples
    data = np.zeros([nr_samples,453+1    ])
    data[:,0] = dat_set[1][:nr_samples]
    data[:,0] = data[:,0] + 1
    
#     n_components = 20
#     pca_data = np.zeros([nr_samples,n_components+1    ])
#     # Set target
#     pca_data[:,0] = dat_set[1][:nr_samples]
#     pca_data[:,0] = pca_data[:,0] + 1
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components)
#     X = pca.fit_transform(dat_set[0][:nr_samples])
#     print 'np.shape(X).....', np.shape(X)
# 
#     if min_max_scaler_train == None:
#         pca_data[:,1:], min_max_scaler_train = normalize(X)
#         return pca_data, min_max_scaler_train
#     elif min_max_scaler_train is not None:
#         pca_data[:,1:] = normalize(X,min_max_scaler_train)
#         return pca_data
    
    if min_max_scaler_train == None:
        data[:,1:], min_max_scaler_train = normalize(dat_set[0][:nr_samples])
        return data, min_max_scaler_train
    elif min_max_scaler_train is not None:
        data[:,1:] = normalize(dat_set[0][:nr_samples],min_max_scaler_train)
        return data
    
    #data[:,1:] = normalize(dat_set[0][:nr_samples])
    
#     np.random.seed(rng_seed)
#     rand_inds = np.random.permutation(nr_samples)
#     data = data[rand_inds]




def save_csv(path, filename, data):
    filename = filename+'.csv'
    print 'writing to csv:', filename
    writer = csv.writer(open(path+filename, 'w'))
    for row in data:
        writer.writerow(row)

def save_libsvm(path,filename, target_path):
    print 'converting to libsvm:', filename
    from csv2libsvm_ver2 import convert_csv2libsvm
    convert_csv2libsvm(path+filename+'.csv', target_path+filename+'.dat')
   
def write(csv_path,libsvm_path, filename, data):
    save_csv(csv_path, filename, data)
    save_libsvm(csv_path,filename, libsvm_path)
    print 'saved'

def write_each_compound(path, target_dataset, compound_label, csv_path,libsvm_path,treatment_items, frac):
    print '='*20
    print compound_label
      
    #train_set = load(path+target_dataset+'_'+compound_label+'_train.pkl.gz')
    #frac = 1.0
    rng_seed    = 1234
    #train_set = load_theano_data(path, target_dataset+'_'+compound_label+'_train', frac, rng_seed)
    
    train_set, min_max_scaler_train = load_theano_data(path, target_dataset+'_'+compound_label+'_train', frac, rng_seed)
      
    #train_name = target_dataset+'_'+compound_label+'_train'
    train_name = target_dataset+'_'+compound_label+'_train_X'+str(frac)
    write(csv_path,libsvm_path, train_name, train_set)
    
    
    treatments = np.where(treatment_items[0:,0] == compound_label )
    print 'treatments for the compound:', treatment_items[treatments]
    for ids in  range(len(np.array(treatments)[0])):
        idst = treatments[0][ids]
        #print idst
        f1  = target_dataset+'_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test'
        if not os.path.exists(path+f1+'.pkl.gz'):
            print 'data path is wrong for the file',f1
            print path
            
        frac = 1.0   
        test_set = load_theano_data(path, f1, frac, rng_seed, min_max_scaler_train)
        #test_set = load_theano_data(path, f1, frac, rng_seed)
        #load(path+f1)
        #test_name = f1
        test_name = f1+'_X'+str(frac)
        write(csv_path,libsvm_path, test_name, test_set)
        


# --- 
target_datasets = ['MFC7_set1', 'MFC7_set2']
#target_datasets = ['MFC7_set1']
fractions = [0.01]
for target_dataset in target_datasets:
    if target_dataset == 'MFC7_set1':
        fold        = 'set1_loov3a/'
        path        = '/home/aditya/store/Datasets/pickled/MFC7/' + fold
        compound_items  = load(path+target_dataset+'_compound_items.pkl.gz')
        #moa_items       = load(path+target_dataset+'_moa_items.pkl.gz')
        treatment_items = load(path+target_dataset+'_treatment_items.pkl.gz')
        for idx, compound_label in enumerate(compound_items):
            #if idx == 8:
                #fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
                #fractions = [0.03, 0.04, 0.06, 0.07, 0.08, 0.09]
                #fractions = [0.1]
                #fractions = [0.05]
                print fractions
                for frac in fractions: 
                    csv_path    = '/home/aditya/store/Datasets/csv_data_1/'+target_dataset+'/'
                    libsvm_path = '/home/aditya/store/Datasets/libsvm_data_1/'+target_dataset+'/'
                    print idx, compound_label
                    #write_each_compound(path, target_dataset, compound_label, csv_path,libsvm_path,treatment_items)
                    write_each_compound(path, target_dataset, compound_label, csv_path,libsvm_path,treatment_items,frac)
                
    elif target_dataset == 'MFC7_set2':
        fold        = 'set2_loov3a/'
        path        = '/home/aditya/store/Datasets/pickled/MFC7/' + fold
        compound_items  = load(path+target_dataset+'_compound_items.pkl.gz')
        #moa_items       = load(path+target_dataset+'_moa_items.pkl.gz')
        treatment_items = load(path+target_dataset+'_treatment_items.pkl.gz')
        for idx, compound_label in enumerate(compound_items):
            #if target_dataset == 'MFC7_set2': #and idx >= 6:  
            csv_path    = '/home/aditya/store/Datasets/csv_data_1/'+target_dataset+'/'
            libsvm_path = '/home/aditya/store/Datasets/libsvm_data_1/'+target_dataset+'/'
            print idx, compound_label
            #write_each_compound(path, target_dataset, compound_label, csv_path,libsvm_path,treatment_items)
            write_each_compound(path, target_dataset, compound_label, csv_path,libsvm_path,treatment_items,frac)
        

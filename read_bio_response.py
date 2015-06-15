from sklearn import svm
#import csv_io
import math
import numpy, csv, os
import pickle
import theano
        
def read_csv_train_target(path, filename):
    ifile1  = open(path+filename, "rb")
    ifile2  = open(path+filename, "rb")
    nr_samples = len(list(csv.reader(ifile1)))-1
    reader = csv.reader(ifile2)
    data = numpy.zeros([nr_samples,1777  ])
    for idx, row in enumerate(reader):
        if idx > 0:
            data[idx-1,:] = row
            
    ifile1.close()
    ifile2.close()
    print 'shape of the labels ', data.shape
    print 'is their a NAN ', numpy.any(numpy.isnan(data))
    return data

def main():
    
    host = 'PC'
    host = 'HPC'
    if host == 'PC':
        path = '/media/aditya/nimi1/repos/BioResponse-master/Data/'
        output_folder = '/media/aditya/nimi1/repos/BioResponse-master/Data/' 
    elif host == 'HPC':
        path = '/home/aditya/store/Theano/BioResponse-master/Data/'
        output_folder = '/home/aditya/store/Datasets/pickled/' 


    data = read_csv_train_target(path,'train.csv')
    train_full  = data[0:,1:]
    target_full = data[0:,0]
    print train_full.shape
    print target_full.shape
    
#     from sklearn import preprocessing   
#     min_max_scaler = preprocessing.MinMaxScaler((-1,1))
#     dat_inp_minmax = min_max_scaler.fit_transform(train_full)
#     print numpy.amin(dat_inp_minmax,axis=1)
#     print numpy.amax(dat_inp_minmax,axis=1)
#     from sklearn.cross_validation import train_test_split
#     tra_inp_full, val_inp, tra_tar_full, val_tar = train_test_split(dat_inp_minmax, target_full, test_size=0.3333, random_state=42)
    

    from sklearn.cross_validation import train_test_split
    tra_inp_full, val_inp, tra_tar_full, val_tar = train_test_split(train_full, target_full, test_size=0.3333, random_state=42)
    
    #print tra_inp_full.shape
    #print val_inp.shape
    
    tra_inp, tes_inp, tra_tar, tes_tar = train_test_split(tra_inp_full, tra_tar_full, test_size=0.5, random_state=42)
    
    print tra_inp.shape
    print val_inp.shape
    print tes_inp.shape
    
    targets = set(target_full.flatten())
    #targets = set(tra_tar.flatten())
    print targets
    
    #monitor target balance
    for t in targets:
        print 'proportion of target '+str(t)+' in'
        print '    trai set: '+str(numpy.mean(tra_tar==t))
        print '    vali set: '+str(numpy.mean(val_tar==t))
        print '    test set: '+str(numpy.mean(tes_tar==t))
         
    print 'tra_tar type ---', type(tra_tar)
          
    train_set = []
    train_set = (numpy.array(tra_inp).astype(theano.config.floatX),
             numpy.array(tra_tar.flatten()).astype(theano.config.floatX))
    print 'nr training instances:  ', len(train_set[0])
    print 'nr features:       ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
                         
    valid_set = []
    valid_set = (numpy.array(val_inp).astype(theano.config.floatX),
                 numpy.array(val_tar.flatten()).astype(theano.config.floatX))
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr features:       ',len(valid_set[0][0])
    print 'nr targets:       ', len(list(set(valid_set[1])))
                          
    test_set = []
    test_set = (numpy.array(tes_inp).astype(theano.config.floatX),
                 numpy.array(tes_tar.flatten()).astype(theano.config.floatX))
    print 'nr test instances:      ', len(test_set[0])
    print 'nr features:       ',len(test_set[0][0])
    print 'nr targets:       ', len(list(set(test_set[1])))
                  
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder) 
          
         
    pickle.dump(train_set, open('bio_train.pkl', 'wb'))
    pickle.dump(valid_set, open('bio_valid.pkl', 'wb'))
    pickle.dump(test_set,  open('bio_test.pkl', 'wb'))
    print "done writing"

    
if __name__=="__main__":
    main()

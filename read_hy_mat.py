import numpy as np
import theano
def load_mat(dataset, path):
    import scipy.io
    path = path + dataset + '/'
    mat = scipy.io.loadmat(path+dataset+'.mat')
    #print mat
    img = mat[dataset]
    if dataset == 'pavia' or dataset == 'paviaU' or dataset == 'KSC' or dataset == 'Botswana':
        imgc = img
    else:
        mat = scipy.io.loadmat(path+dataset+'_corrected.mat')
        imgc = mat[dataset+'_corrected']
    
    mat = scipy.io.loadmat(path+dataset+'_gt.mat')
    img_gt = mat[dataset+'_gt']
    return img, imgc, img_gt

def dataset_info(img_gt):
    gt_data =img_gt.flatten()
    #print 'classes = ', set(gt_data)
    print 'Total number of samples = ', len(gt_data)
    #for t in range(len(list(set(gt_data)))):
    for t in set(gt_data):
        print 'Number of samples in class '+str(t)+' = '+str(np.sum(gt_data==t))
        #print 'proportion of target '+str(t)+' in'
        #print '    trai set: '+str(np.mean(gt_data==t))
        

def split_data_2(dat,split):
# to do: this function could be used in other functions in this file
    """
    Split data into training, validation, and test sets.
    """
    length_all = dat.shape[0]
    length_tra = round(split[0]*length_all)
    length_val = round(split[1]*length_all)
    length_tes = length_all - length_tra - length_val
    tra = dat[0                    :length_tra           ,:] 
    val = dat[length_tra           :length_tra+length_val,:] 
    tes = dat[length_tra+length_val:length_all           ,:]
    print tra.shape, val.shape, tes.shape
    return tra,val,tes

def min_max_scaler(dat_inp):
    from sklearn import preprocessing
    #print dat_inp
    #dat_inp_normalized = (dat_inp-np.min(dat_inp))/(np.max(dat_inp)-np.min(dat_inp))
    #dat_inp_normalized = preprocessing.normalize(dat_inp, norm='l2')
    #min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    min_max_scaler = preprocessing.MinMaxScaler((0,1))
    #min_max_scaler = preprocessing.MinMaxScaler()
    dat_inp_minmax = min_max_scaler.fit_transform(dat_inp)
    #dat_inp_minmax = min_max_scaler.fit_transform(dat_inp_normalized)
    #print np.amin(dat_inp_normalized,axis=1)
    #print np.amax(dat_inp_normalized,axis=1)
    print np.amin(dat_inp_minmax,axis=1)
    print np.amax(dat_inp_minmax,axis=1)
    return dat_inp_minmax
    #return dat_inp_normalized


    

path = '/media/883E0F323E0F1938/Chetak/dataset/'
path = '/media/aditya_1t/1892AA8692AA67C8/D_drive/Datasets/hy/'
path = '/home/aditya/store/Datasets/hy/'
dataset = 'indian_pines'
#dataset = 'salinas'
#dataset = 'salinasA'
#dataset = 'pavia'
# dataset = 'paviaU'
# dataset = 'KSC'
# dataset = 'Botswana'
img, imgc, img_gt  = load_mat(dataset, path)
print dataset + ' img.shape' + str(img.shape)
print dataset + ' corrected'+ ' img.shape' + str(imgc.shape)
pixel = imgc[50,50]
print 'pixel.shape', pixel.shape
band6 = imgc[:,:,5]
print 'band6.shape', band6.shape
#dataset_info(img_gt)

nr_samples = len(img_gt.flatten())
print 'nr_samples', nr_samples
nr_features = pixel.shape[0]
print 'nr_features', nr_features
# For the original samples
#dat_inp = np.reshape(imgc, [nr_samples,nr_features])
# For the corrected samples
dat_inp = np.reshape(imgc, [nr_samples,nr_features])
dat_tar = np.reshape(img_gt, [nr_samples,1])

print np.shape(dat_inp)
print np.shape(dat_tar)

print '-------------'
print dat_inp.dtype
dat_inp = np.asarray(dat_inp, dtype=np.float32)
print dat_inp.dtype

# # scale inputs
dat_inp = min_max_scaler(dat_inp)
print dat_inp
print np.any(np.isnan(dat_inp))

# shuffle inputs and targets
np.random.seed(1234)
rand_inds = np.random.permutation(nr_samples)
dat_inp = dat_inp[rand_inds,:]
dat_tar = dat_tar[rand_inds]
 
split = [0.5,0.25]
#split = [0.5,0.1]
# split inputs and targets
tra_inp,val_inp,tes_inp = split_data_2(dat_inp,split)
tra_tar,val_tar,tes_tar = split_data_2(dat_tar,split)

targets = set(dat_tar.flatten())
print targets       

# monitor target balance
for t in targets:
    print 'proportion of target '+str(t)+' in'
    print '    trai set: '+str(np.mean(tra_tar==t))
    print '    vali set: '+str(np.mean(val_tar==t))
    print '    test set: '+str(np.mean(tes_tar==t))
 
#nr_cols = 6
#nr_rows = 1
#show_samples(dat_inp,tra_tar,nr_cols,nr_rows)
print 'done'
print 'tra_tar type ---', type(tra_tar)

train_set = []
train_set = (np.array(tra_inp).astype(theano.config.floatX),
         np.array(tra_tar.flatten()).astype(theano.config.floatX))
print 'nr training instances:  ', len(train_set[0])
print 'nr features:       ',len(train_set[0][0])
print 'nr targets:       ', len(list(set(train_set[1])))
                
valid_set = []
valid_set = (np.array(val_inp).astype(theano.config.floatX),
             np.array(val_tar.flatten()).astype(theano.config.floatX))
print 'nr validation instances: ',  len(valid_set[0])
print 'nr features:       ',len(valid_set[0][0])
print 'nr targets:       ', len(list(set(valid_set[1])))
                 
test_set = []
test_set = (np.array(tes_inp).astype(theano.config.floatX),
             np.array(tes_tar.flatten()).astype(theano.config.floatX))
print 'nr test instances:      ', len(test_set[0])
print 'nr features:       ',len(test_set[0][0])
print 'nr targets:       ', len(list(set(test_set[1])))
      
import os
output_folder = '/home/aditya/store/Datasets/pickled/'
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder) 
  
import cPickle as pickle
pickle.dump(train_set, open(dataset+'_train.pkl', 'wb'))
pickle.dump(valid_set, open(dataset+'_valid.pkl', 'wb'))
pickle.dump(test_set,  open(dataset+'_test.pkl', 'wb'))
print "done writing"



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
    #elif dataset == 'indian_pines' or dataset == 'salinas':
    #    imgc = img
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
dataset = 'indian_pines_land'
#dataset = 'salinas'
#dataset = 'salinasA'
#dataset = 'pavia'
#dataset = 'paviaU'
#dataset = 'KSC'
#dataset = 'Botswana'
# # # if dataset == 'indian_pines_land':
# # #     path2 = '/media/aditya_1t/1892AA8692AA67C8/D_drive/Datasets/hy/AVIRIS/'
# # #     from spectral import *
# # #     img    = open_image(path2 +'92AV3C.lan').load()
# # #     #print 'type', type(img)
# # #     imgc   = img
# # #     img_gt = open_image(path2 +'92AV3GT.GIS').read_band(0)
# # #    
# # # else:
# # #     img, imgc, img_gt  = load_mat(dataset, path)
# # #     print type(img)
# # #     
# # # print dataset + ' img.shape' + str(img.shape)
# # # pixel = img[50,50]
# # # print 'pixel.shape', pixel.shape
# # # band6 = img[:,:,5]
# # # print 'band6.shape', band6.shape
# # # #imgc = img
# # # 
# # # # # MODIFY (REMOVE) THE SELECTED BANDS
# # # if dataset == 'salinas':
# # #     remove_bands = [107,108,109,110]
# # #     print 'len(remove_bands)', len(remove_bands)
# # #     imgc = np.delete(img,remove_bands, axis=2)
# # #     
# # # # remove_bands2 = [153,154,155,156,157,158,159,160,161,162,163,164,165,166]
# # # # remove_bands3 = [223]
# # # # remove_bands = [107,108,109,110,111,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,223]
# # # # print 'len(remove_bands)', len(remove_bands)
# # # # imgc = np.delete(img,remove_bands, axis=2)
# # # # print 'corrected.shape', imgc.shape
# # # # imgc = np.delete(img,remove_bands1, axis=2)
# # # # print 'corrected.shape', imgc.shape
# # # # imgc = np.delete(img,remove_bands2, axis=2)
# # # # print 'corrected.shape', imgc.shape
# # # # imgc = np.delete(img,remove_bands3, axis=2)
# # # # print 'corrected.shape', imgc.shape
# # # # print
# # # 
# # # 
# # # print dataset + ' corrected'+ ' img.shape' + str(imgc.shape)
# # # pixel = imgc[50,50]
# # # print 'pixel.shape', pixel.shape
# # # band6 = imgc[:,:,5]
# # # print 'band6.shape', band6.shape
# # # #dataset_info(img_gt)
# # # 
# # # nr_samples = len(img_gt.flatten())
# # # print 'nr_samples', nr_samples
# # # nr_features = pixel.shape[0]
# # # print 'nr_features', nr_features
# # # dat_inp = np.reshape(imgc, [nr_samples,nr_features])
# # # dat_tar = np.reshape(img_gt, [nr_samples,1])
# # # 
# # # print np.shape(dat_inp)
# # # print np.shape(dat_tar)
# # # 
# # # print '-------------'
# # # print dat_inp.dtype
# # # dat_inp = np.asarray(dat_inp, dtype=np.float32)
# # # print dat_inp.dtype
# # # 
# # # # # scale inputs
# # # dat_inp = min_max_scaler(dat_inp)
# # # print dat_inp
# # # print 'Number of bk ground samples', len(dat_tar[np.where(dat_tar==0)[0]])
# # # print 'Number of Data samples ', len(dat_tar[np.where(dat_tar!=0)[0]]) 
# # # dataset_info(dat_tar)
# # # 
# # # # balance the dataset by removing background samples
# # # def cut_DMSO(data, dat_target, remove_class, item = None):
# # #  
# # #     
# # #     data_tmp1 = data[np.where(dat_target < remove_class)[0]]
# # #     data_tmp2 = data[np.where(dat_target > remove_class)[0]]
# # #     
# # #     
# # #     # to reset the label series from zero
# # #     if item == 'label':
# # #         data_tmp2 = data_tmp2 -1
# # #     new_data = np.vstack((data_tmp1,data_tmp2))
# # #         
# # #  
# # #     return new_data
# # # 
# # # 
# # # remove_class = 0
# # # dat_inp = cut_DMSO(dat_inp, dat_tar, remove_class)
# # # dat_tar = cut_DMSO(dat_tar, dat_tar, remove_class)
# # # nr_samples = len(dat_tar.flatten())
# # # print 'nr_samples', nr_samples
# # # 
# # # print 'Number of Data samples ', len(dat_tar[np.where(dat_tar!=0)[0]])
# # # dataset_info(dat_tar)
# # # 
# # # print np.any(np.isnan(dat_inp))
# # # 
# # # # shuffle inputs and targets
# # # np.random.seed(1234)
# # # rand_inds = np.random.permutation(nr_samples)
# # # dat_inp = dat_inp[rand_inds,:]
# # # dat_tar = dat_tar[rand_inds]

# import os
# output_folder = path2
# if not os.path.isdir(output_folder):
#     os.makedirs(output_folder)
# os.chdir(output_folder)
# import cPickle as pickle
# pickle.dump(dat_inp, open(dataset+'_dat_inp.pkl', 'wb'))
# pickle.dump(dat_tar, open(dataset+'_dat_tar.pkl', 'wb')) 

data_path = '/home/aditya/store/Datasets/pickled/'
import cPickle
f1 = open(data_path+'indian_pines_land_dat_inp.pkl', 'rb')
dat_inp = cPickle.load(f1)
f2 = open(data_path+'indian_pines_land_dat_tar.pkl', 'rb')
dat_tar = cPickle.load(f2)


 
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
pickle.dump(train_set, open(dataset+'_nbg_train.pkl', 'wb'))
pickle.dump(valid_set, open(dataset+'_nbg_valid.pkl', 'wb'))
pickle.dump(test_set,  open(dataset+'_nbg_test.pkl', 'wb'))
if dataset == 'salinas':
    pickle.dump(train_set, open(dataset+'_nbg_TLmod_train.pkl', 'wb'))
    pickle.dump(valid_set, open(dataset+'_nbg_TLmod_valid.pkl', 'wb'))
    pickle.dump(test_set,  open(dataset+'_nbg_TLmod_test.pkl', 'wb'))
print "done writing"
  


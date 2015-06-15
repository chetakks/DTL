def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def convert(dict):
    label = dict['labels']
    for idx, val in enumerate(label):
        if val == 0 or val == 1 or val == 8 or val == 9:
            label[idx] = 0
        else:
            label[idx] = 1
    dict['labels'] = label
    return dict

def save(dict,file):
    import cPickle
    f = open(file, 'wb')
    cPickle.dump(dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def normalize(dict):
    from sklearn import preprocessing
    dat_inp = dict['data']
    dat_inp_normalized = preprocessing.normalize(dat_inp, norm='l2')
    dict['data'] = dat_inp_normalized
    return dict
    
def min_max_scaler(dict):
    from sklearn import preprocessing
    import numpy
    dat_inp = dict['data']
    print dat_inp
    #min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    min_max_scaler = preprocessing.MinMaxScaler((0,1))
    dat_inp_minmax = min_max_scaler.fit_transform(dat_inp)
    print numpy.amin(dat_inp_minmax,axis=1)
    print numpy.amax(dat_inp_minmax,axis=1)
    dict['data'] = dat_inp_minmax
    return dict
    
def greayscale(dict):
    # Y = 0.3R + 0.59G + 0.11B from 
    # http://gimp-savvy.com/BOOK/index.html?node54.html  
    import numpy as np 
    #import theano 
    r = np.asarray(.3, dtype='float32') 
    g = np.asarray(.59, dtype='float32') 
    b = np.asarray(.11, dtype='float32') 
    
    dat_inp = dict['data']
    print dat_inp
    print np.shape(dat_inp)
    # to convert to color to 'grey'
    dat_inp = r * dat_inp[:,:1024] + g * dat_inp[:,1024:2048] + b * dat_inp[:,2048:]
    print dat_inp
    print np.shape(dat_inp)
    dict['data'] = dat_inp
    print 
    return dict
#                 #if dtype=='uint8': 
#                     #x = theano.tensor.cast(x, 'uint8') 
#             elif color=='rgb': 
#                 # the strides aren't what you'd expect between channels, 
#                 # but theano is all about weird strides 
#                x = x.reshape((N, 3,32*32)).dimshuffle(0, 2, 1) 
#             else: 
#                 raise NotImplemented('color', color) 
#         else: 
#             if color=='grey': 
#                 x = r * x[:,:1024] + g * x[:,1024:2048] + b * x[:,2048:] 
#                 if dtype=='uint8': 
#                     x = theano.tensor.cast(x, 'uint8') 
#                 x.reshape((N, 32, 32)) 
#             elif color=='rgb': 
#                 # the strides aren't what you'd expect between channels, 
#                # but theano is all about weird strides 
#                 x = x.reshape((N,3,32,32)).dimshuffle(0, 2, 3, 1) 
#             else: 
#                 raise NotImplemented('color', color) 
#     if x.ndim == 1: 
#         if rasterized: 
#             if color=='grey': 
#                 x = r * x[:1024] + g * x[1024:2048] + b * x[2048:] 
#                 if dtype=='uint8': 
#                     x = theano.tensor.cast(x, 'uint8') 
#             elif color=='rgb': 
#                 # the strides aren't what you'd expect between channels, 
#                 # but theano is all about weird strides 
#                 x = x.reshape((3,32*32)).T 
#             else: 
#                  raise NotImplemented('color', color) 




if __name__ == "__main__":
    
    import gzip
    #path = '/home/aditya_1t/Downloads/'
    path = '/home/aditya/store/Datasets/pickled/'
    fold1 = 'cifar10/'
    fold2 = 'cifar2/'
    fold3 = 'cifar10n/'
    tenclasses = ['cifar10_train_bat0.pkl', 'cifar10_train_bat1.pkl', 'cifar10_train_bat2.pkl',
            'cifar10_train_bat3.pkl', 'cifar10_valid_bat0.pkl', 'cifar10_test_bat0.pkl']
    tenclassesn = ['cifar10n_train_bat0.pkl', 'cifar10n_train_bat1.pkl', 'cifar10n_train_bat2.pkl',
            'cifar10n_train_bat3.pkl', 'cifar10n_valid_bat0.pkl', 'cifar10n_test_bat0.pkl']
    twoclasses = ['cifar2_train_bat0.pkl', 'cifar2_train_bat1.pkl', 'cifar2_train_bat2.pkl',
            'cifar2_train_bat3.pkl', 'cifar2_valid_bat0.pkl', 'cifar2_test_bat0.pkl']
    
    
    # create normalized cifar10n
    normalize_inp = 2
    for tenclass,tenclassn in zip(tenclasses,tenclassesn):
        file_name = path + fold1 + tenclass
        dict = unpickle(file_name)
        if normalize_inp == 1:
            dict = normalize(dict)
        elif normalize_inp == 2:
            dict = greayscale(dict)
            dict = min_max_scaler(dict)
            dat_inp = dict['data']
            print dat_inp
            import numpy as np
            print np.shape(dat_inp)
        else:
            dict = min_max_scaler(dict)
        new_file_name = path + fold3 + tenclassn
        save(dict,new_file_name)
        

        
        
#     normalize_inp = 1
#     for tenclass,twoclass in zip(tenclasses,twoclasses):
#         file_name = path + fold1 + tenclass
#         dict = unpickle(file_name)
#         #dict = convert(dict)
#         if normalize_inp == 1:
#             dict = normalize(dict)
#         else:
#             dict = min_max_scaler(dict)
#         new_file_name = path + fold3 + tenclass
#         save(dict,new_file_name)
#         
#         #new_file_name = path + fold2 + twoclass
#         #save(dict,new_file_name)
    
    


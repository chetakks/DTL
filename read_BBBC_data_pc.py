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

def show_samples(tra_inp,tra_tar,nr_cols,nr_rows):
    """
    Show nr_cols samples from each set in dat_set (training, validation, test).
    """
#     print type(tra_inp), tra_inp.dtype
#     scipy.misc.imsave("something.png",numpy.reshape(tra_inp[10,:],[48,64]))
    
    
    print inspect.stack()[0][3]
    figure = plt.figure(figsize=(16.,6.))
    si = 0
    nr_cols = 7
    nr_rows = 7
    sub_x = tra_inp
    sub_y = tra_tar 
    counter = 0
    for i in range(1,nr_rows+1):
        for ci in range(1,nr_cols+1):
            #print ci+((i-1)*nr_cols)
            #print ci
            counter = counter +1
            x = sub_x[counter]
#             x = sub_x[(ci+((i-1)*nr_cols))]
#             y = sub_y[(ci+((i-1)*nr_cols))] 
            #print 'y', y
            x = numpy.reshape(x,[28,28])
            #x = numpy.reshape(x,[48,64])
            plt.subplot(nr_rows,nr_cols,si+1)
            #plt.imshow(x)
            plt.imshow(x,cmap=plt.cm.gray)
            
            #plt.imshow(x,cmap=plt.cm.gray,interpolation='nearest')
            #plt.imshow(x,cmap=plt.cm.winter,interpolation='nearest')
             
            #plt.imshow(x, interpolation='bilinear')
            #plt.annotate('%d'%y, xy=(3, 2),xytext=(21.5, 26), size=20, 
                 #bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),)
#             plt.annotate('%d'%y, xy=(3, 2),xytext=(53, 40), size=10, 
#                          bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            si = si+1
    plt.show()
#To be able to read csv formated files, we will first have to import the
#csv module.
# import csv
# with open('/home/aditya/Downloads/week1.csv', 'rb') as f:
#     reader = csv.reader(f)
    #print reader[0]
#     for row in reader:
#         print row
#         print row[0]
#         print row[1]
#         print row[2]
#         print row[3]
#         print row[4]


def save_batch(dat_set, dat_name):
    #import numpy
    batch = 1000
    nr_samples = len(dat_set[0])
    nr_batch = int(numpy.floor(nr_samples / batch))
    
    for bat in range(nr_batch):
        tmp = [dat_set[0][bat*batch:(bat+1)*batch],dat_set[1][bat*batch:(bat+1)*batch]]
        pickle.dump(tmp, open(dat_name+'_bat'+str(bat)+'.pkl', 'wb'))
        #pickle.dump(tmp, open('mnist_64x80_'+str(dat_name)+'_bat'+str(bat)+'.pkl', 'wb'))
        print 'saving batch num ', bat
    print 'done'


if __name__ == "__main__":
    import csv
    #from upsample_dataset import save_batch
    
    ifile  = open('/home/aditya/Downloads/week1.csv', "rb")
    #ifile  = open('/home/aditya/store/Datasets/bbbc/week1.csv', "rb")
    reader = csv.reader(ifile)
    
    rownum = 0
    DNA_filename = []
    tubulin_filename = []
    actin_filename = []
    
    for idx, row in enumerate(reader):
        # Save header row.
        if rownum == 0:
            header = row
        else:
            colnum = 0
            
            for col in row:
                #print '%-8s: %s' % (header[colnum], col)
                if colnum == 2:
                    DNA_filename.append(col)
                    #print '    DNA_filename', DNA_filename
                elif colnum == 4:
                    tubulin_filename.append(col)
                    #print '    tubulin_filename', tubulin_filename
                elif colnum == 6:
                    actin_filename.append(col)
                    #print '    actin_filename', actin_filename
     
                colnum += 1
                
        rownum += 1
        #print rownum
        #if rownum == 5:
        #    break
    
    ifile.close()
    #print
    
    import os
    import numpy
    import scipy
    import scipy.ndimage
    import matplotlib.pyplot as plt
    import random
    import theano
    import theano.tensor as T
    #import pickle
    from sys import getsizeof
    import cPickle as pickle
    #import marshal as pickle
    #import cPickle
    host_path = os.getenv('HOME')
    output_folder=host_path+'/store/Datasets/pickled/bbbc/'
    #output_folder=host_path+'/store/Datasets/pickled/bbbc_100/'
    #output_folder= '/media/aditya/nimi1/repos/data/BBBC/' 
     
    dir = '/home/aditya/Downloads/Week1_22123/'
    #dir = '/home/aditya/store/Datasets/bbbc/Week1_22123/'
    sample_ind = 0
    nr_samples = len(os.listdir(dir))
    print 'total nr_samples', nr_samples
    nr_samples = 500
    print 'nr_samples', nr_samples
    #nr_samples = nr_samples * 64
    nr_samples = nr_samples * 128
    #nr_samples = 25
    
    #dat_inp = numpy.zeros([nr_samples,1024,1280])
    #dat_inp = numpy.zeros([nr_samples,256,320])
    #dat_inp = numpy.zeros([nr_samples,128,160])
    dat_inp = numpy.zeros([nr_samples,64,80])
    dat_tar = numpy.zeros([nr_samples,1    ])
    print 'Loading the data ...' 
    for filename in os.listdir(dir):
        #print filename
        
        for item in DNA_filename:
            if item == filename:
                target = 0  # DNA
                #print 'DNA_file_found'
        for item in tubulin_filename:
            if item == filename:
                target = 1  # tubulin
                #print 'tubulin_file_found'
        for item in actin_filename:
            if item == filename:
                target = 2  # actin
                #print 'actin_file_found'

        #dat_tar[sample_ind,0] = target
        #print target
        filepath = dir+'/'+filename
        #import scipy
        #from scipy import misc
        #im = misc.imread(filepath,flatten=True)
        #import Image
        #im = Image.open(filepath).convert('L')
        #im = numpy.array(im)      
        #im = scipy.ndimage.imread(filepath)
        #im = scipy.ndimage.imread(filepath, mode='L')
        im = scipy.ndimage.imread(filepath,flatten=True)
        
        # scale inputs
        #print numpy.max(numpy.max(im, axis=1))
        im = im / 2**16.
        #plt.imshow(im,cmap=plt.cm.gray)
        #plt.show()
        #print im.shape
        #print im.size
        im = im.astype(float)
        
        # Define the window size
        #windowsize_r = 128
        #windowsize_c = 160
        windowsize_r = 64
        windowsize_c = 80
        test_image = im
        if target == 0:
            #print 'target', target
            # Crop out images into windows
            count = 0
            for r in range(0,test_image.shape[0]+1 - windowsize_r, windowsize_r):
                for c in range(0,test_image.shape[1]+1 - windowsize_c, windowsize_c):
                    window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                    #print 'count', count
                    if not (window.mean() > 0.006 and window.mean() < 0.009):
                        print 'count', count
                        plt.imshow(window,cmap=plt.cm.gray)
                        plt.show() 
                        dat_inp[sample_ind,:,:] = window
                        dat_tar[sample_ind,0] = target
                        sample_ind += 1
                    #print 'mean', window.mean()
                        
                    #dat_inp[sample_ind,:,:] = window
                    #dat_tar[sample_ind,0] = target
                    
                    #print target
                    #plt.imshow(window,cmap=plt.cm.gray)
                    #plt.show()
                    #print window
                    #print numpy.max(numpy.max(window, axis=1))
                    #sample_ind += 1
                    count += 1
            #print count
        
        
        if target == 1:
            #print 'target', target
            # Crop out images into windows
            count = 0
            for r in range(0,test_image.shape[0]+1 - windowsize_r, windowsize_r):
                for c in range(0,test_image.shape[1]+1 - windowsize_c, windowsize_c):
                    window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                    
                    if not (window.mean() > 0.014 and window.mean() < 0.02):
                        #print 'count', count
                        #plt.imshow(window,cmap=plt.cm.gray)
                        #plt.show() 
                        dat_inp[sample_ind,:,:] = window
                        dat_tar[sample_ind,0] = target
                        sample_ind += 1
                    #print 'mean', window.mean()
                        
                    
                    #dat_inp[sample_ind,:,:] = window
                    #dat_tar[sample_ind,0] = target
                    
                    #print target
                    #plt.imshow(window,cmap=plt.cm.gray)
                    #plt.show()
                    #print window
                    #print numpy.max(numpy.max(window, axis=1))
                    #sample_ind += 1
                    count += 1
            #print count
        if target == 2:
            #print 'target', target
            # Crop out images into windows
            count = 0
            for r in range(0,test_image.shape[0]+1 - windowsize_r, windowsize_r):
                for c in range(0,test_image.shape[1]+1 - windowsize_c, windowsize_c):
                    window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                    
                    if not (window.mean() > 0.008 and window.mean() < 0.010):
                        #print 'count', count
                        #plt.imshow(window,cmap=plt.cm.gray)
                        #plt.show() 
                        dat_inp[sample_ind,:,:] = window
                        dat_tar[sample_ind,0] = target
                        sample_ind += 1
                    #print 'mean', window.mean()
                        
                    #dat_inp[sample_ind,:,:] = window
                    #dat_tar[sample_ind,0] = target
                    
                    #print target
                    #plt.imshow(window,cmap=plt.cm.gray)
                    #plt.show()
                    #print window
                    #print numpy.max(numpy.max(window, axis=1))
                    #sample_ind += 1
                    count += 1
            #print count
        if sample_ind == 500 * 128:  #100*64:
            break
    print 'sample_ind', sample_ind   
        
        # subsample
        #im = scipy.misc.imresize(im, [256,320], interp='bicubic' )
        #im = scipy.misc.imresize(im, [64,80], interp='bicubic' )
        #plt.imshow(im,cmap=plt.cm.gray)
        #plt.show()
        
        #im = scipy.ndimage.filters.gaussian_filter(im,1.14)
        #dat_inp[sample_ind,:,:] = im

        #sample_ind += 1
        #if sample_ind == 100 * 128:  #100*64:
            #break
    #print dat_tar
    
    # reshape each 2d image as one row
    print dat_inp.shape
    #dat_inp = numpy.reshape(dat_inp, [nr_samples,1024*1280])
    #dat_inp = numpy.reshape(dat_inp, [nr_samples,256*320])
    #dat_inp = numpy.reshape(dat_inp, [nr_samples,128*160])
    dat_inp = numpy.reshape(dat_inp, [nr_samples,64*80])
    
    # scale inputs
    #dat_inp = dat_inp / 255.
    
    print numpy.any(numpy.isnan(dat_inp))
    
    # shuffle inputs and targets
    numpy.random.seed(1234)
    rand_inds = numpy.random.permutation(nr_samples)
    dat_inp = dat_inp[rand_inds,:]
    dat_tar = dat_tar[rand_inds,:]
    
    split = [0.5,0.25]
    #split = [0.5,0.1]
    # split inputs and targets
    tra_inp,val_inp,tes_inp = split_data_2(dat_inp,split)
    tra_tar,val_tar,tes_tar = split_data_2(dat_tar,split)
    
    targets = set(dat_tar.flatten())
    print targets       
    
#     batch = 1000 
#     tra_frac = numpy.floor(len(tra_inp) / batch)
#     val_frac = numpy.floor(len(val_inp) / batch)
#     tes_frac = numpy.floor(len(tes_inp) / batch)
#     
#     tra_inp = tra_inp[0:tra_frac*batch]
#     tra_tar = tra_tar[0:tra_frac*batch]
#     
#     val_inp = val_inp[0:val_frac*batch]
#     val_tar = val_tar[0:val_frac*batch]
#     
#     tes_inp = tes_inp[0:tes_frac*batch]
#     tes_tar = tes_tar[0:tes_frac*batch]
    
    
    # monitor target balance
    for t in targets:
        print 'proportion of target '+str(t)+' in'
        print '    trai set: '+str(numpy.mean(tra_tar==t))
        print '    vali set: '+str(numpy.mean(val_tar==t))
        print '    test set: '+str(numpy.mean(tes_tar==t))

    #nr_cols = 6
    #nr_rows = 1
    #show_samples(dat_inp,tra_tar,nr_cols,nr_rows)
    print 'done'
    
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
    
    print getsizeof(train_set)
    #fileObject = open('bbbc_train.pkl','wb') 
    # this writes the object a to the file named 'testfile'
    #pickle.dump(train_set,fileObject) 
    #pickle.dump(train_set, open('bbbc_train.pkl', 'wb'))
    # here we close the fileObject
    #fileObject.close()
#     import cPickle
#     f = file('bbbc_train.pkl', 'wb')
#     cPickle.dump(train_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
#     f.close()
    
    
    save_batch(train_set,'bbbc_train')
    save_batch(valid_set,'bbbc_valid')
    save_batch(test_set,'bbbc_test')
   
    #pickle.dump(train_set, open('bbbc_train.pkl', 'wb'))
    #pickle.dump(valid_set, open('bbbc_valid.pkl', 'wb'))
    #pickle.dump(test_set,  open('bbbc_test.pkl', 'wb'))

    print "done writing"
    
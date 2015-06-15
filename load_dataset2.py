import cPickle
import gzip
import pickle
import numpy
import theano
import theano.tensor as T
import os
from scipy.stats import bernoulli

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

def load_outputs(dir,prefix='outputs_'):
    #print inspect.stack()[0][3]
    print 'loading',dir
    outputs_list = list()
    for filename in os.listdir(dir): 
        if filename.startswith(prefix):
            print '.',
            if   filename.endswith('.pkl'):
                outputs = cPickle.load(open(dir+filename,'rb'))
            elif filename.endswith('.gz' ):
                outputs = load(dir+filename)
            outputs_list.append(outputs)
    print '\n',
    return outputs_list

def unpack_gzip_data(data_path):
    # Load the dataset
    f = gzip.open(data_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set
  
def unpack_data(data_path,data_name):
    # Load the dataset
    f = open(data_path+data_name+'_train.pkl', 'rb')
    train_set = cPickle.load(f)
    f.close()
    
    f = open(data_path+data_name+'_valid.pkl', 'rb')
    valid_set = cPickle.load(f)
    f.close()
    
    f = open(data_path+data_name+'_test.pkl', 'rb')
    test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set    

def load_batch(data_path,data_name,dat_name, nr_batch):
        #nr_batch = 50
        if data_name == 'bbbc+feat':
            batch = 50000
        elif data_name == 'cifar10' or data_name == 'cifar10n' or data_name == 'cifar2':
            batch = 10000
        else:
            batch = 1000
        
        dat_set = ([],[])
        tmp_inp = []
        tmp_tar = []
        for bat in range(nr_batch):
            
            #f = open(data_path+data_name+'_'+str(dat_name)+'_bat'+str(bat)+'.pkl', 'rb')
            f = open(data_path+data_name+'/'+data_name+'_'+str(dat_name)+'_bat'+str(bat)+'.pkl', 'rb')
            if data_name == 'cifar10' or data_name == 'cifar10n' or data_name == 'cifar2':
                dict = cPickle.load(f)
                x = dict['data']
                x = theano.tensor.cast(x, 'uint8')
                d = (x,dict['labels'])
                (dat_set[0][bat*batch:(bat+1)*batch],dat_set[1][bat*batch:(bat+1)*batch]) = d
            else: 
                (dat_set[0][bat*batch:(bat+1)*batch],dat_set[1][bat*batch:(bat+1)*batch]) = cPickle.load(f)
            
            tmp_inp = numpy.asarray(dat_set[0])
            tmp_tar = numpy.asarray(dat_set[1])
            
#             tmp_inp[bat*batch:(bat+1)*batch] = numpy.array(dat_set[0][bat*batch:(bat+1)*batch])
#             tmp_tar[bat*batch:(bat+1)*batch] = numpy.array(dat_set[1][bat*batch:(bat+1)*batch])
            
#             tmp_inp = numpy.asarray(tmp_inp)
#             tmp_tar = numpy.asarray(tmp_tar)
            
            print 'Loading ' + str(dat_name)+ ' batch num ' + str(bat)
            
        dat_set = (numpy.array(tmp_inp).astype(theano.config.floatX),
                 numpy.array(tmp_tar.flatten()).astype(theano.config.floatX))
        print 'done'
        return dat_set

def unpack_batch_data(data_path,data_name):
    # Load the dataset
    
    if data_name == 'mnist_64x80':
        train_set = load_batch(data_path,data_name,'train', 50)
        valid_set = load_batch(data_path,data_name,'valid', 10)
        test_set = load_batch(data_path,data_name,'test', 10)
    elif data_name == 'bbbc':
        train_set = load_batch(data_path,data_name,'train', 31)
        valid_set = load_batch(data_path,data_name,'valid', 15)
        test_set = load_batch(data_path,data_name,'test', 15)
    elif data_name == 'bbbc+feat':
        filename1 =  'bbbc+feat/bbbc+feat_gzip_train.pkl'
        train_set = load(data_path+filename1)
        filename2 =  'bbbc+feat/bbbc+feat_gzip_valid.pkl'
        valid_set = load(data_path+filename2)
        filename3 =  'bbbc+feat/bbbc+feat_gzip_test.pkl'
        test_set = load(data_path+filename3)
        
        #train_set = load_batch(data_path,data_name,'train', 4)
        #valid_set = load_batch(data_path,data_name,'valid', 2)
        #test_set = load_batch(data_path,data_name,'test', 2)
        #train_set = load_batch(data_path,data_name,'train', 1)
        #f1 = open(data_path+data_name+'_valid.pkl', 'rb')
        #valid_set = cPickle.load(f1)
        #f2 = open(data_path+data_name+'_test.pkl', 'rb')
        #test_set = cPickle.load(f2)
        #valid_set = load_batch(data_path,data_name,'valid', 1)
        #test_set = load_batch(data_path,data_name,'test', 1)
    elif data_name == 'bbbc+feat2':
        filename1 =  'bbbc+feat2/bbbc+feat2_gzip_train.pkl'
        train_set = load(data_path+filename1)
        filename2 =  'bbbc+feat2/bbbc+feat2_gzip_valid.pkl'
        valid_set = load(data_path+filename2)
        filename3 =  'bbbc+feat2/bbbc+feat2_gzip_test.pkl'
        test_set = load(data_path+filename3)
    elif data_name == 'bbbc+feat3':
        filename1 =  'bbbc+feat3/bbbc+feat3_gzip_train.pkl'
        train_set = load(data_path+filename1)
        filename2 =  'bbbc+feat3/bbbc+feat3_gzip_valid.pkl'
        valid_set = load(data_path+filename2)
        filename3 =  'bbbc+feat3/bbbc+feat3_gzip_test.pkl'
        test_set = load(data_path+filename3)
    elif data_name == 'bbbc+moa':
        filename1 =  'bbbc+feat4/bbbc+moa_gzip_train.pkl'
        train_set = load(data_path+filename1)
        filename2 =  'bbbc+feat4/bbbc+moa_gzip_valid.pkl'
        valid_set = load(data_path+filename2)
        filename3 =  'bbbc+feat4/bbbc+moa_gzip_test.pkl'
        test_set = load(data_path+filename3)
    elif data_name == 'bbbc+comp':
        filename1 =  'bbbc+feat4/bbbc+comp_gzip_train.pkl'
        train_set = load(data_path+filename1)
        filename2 =  'bbbc+feat4/bbbc+comp_gzip_valid.pkl'
        valid_set = load(data_path+filename2)
        filename3 =  'bbbc+feat4/bbbc+comp_gzip_test.pkl'
        test_set = load(data_path+filename3)
    elif data_name == '20news_4':
        train_set = load_batch(data_path,data_name,'train', 3)
        valid_set = load_batch(data_path,data_name,'valid', 1)
        test_set = load_batch(data_path,data_name,'test', 2)
    elif data_name == 'cifar10' or data_name == 'cifar10n' or data_name == 'cifar2':
        train_set = load_batch(data_path,data_name,'train', 4)
        valid_set = load_batch(data_path,data_name,'valid', 1)
        test_set = load_batch(data_path,data_name,'test', 1)
    elif data_name[0:9] == 'bbbc+moa_':
        train_set = load(data_path+'/bbbc_loov/'+data_name+'_train.pkl')
        valid_set = load(data_path+'/bbbc_loov/'+data_name+'_valid.pkl')
        test_set  = load(data_path+'/bbbc_loov/'+data_name+'_test.pkl')
    
    return train_set, valid_set, test_set    


def unpack_data_target(data_path,data_name):
    # Load the dataset
    f = open(data_path+data_name+'_train[0].pkl', 'rb')
    train_set_0 = cPickle.load(f)
    f.close()
    f = open(data_path+data_name+'_train[1].pkl', 'rb')
    train_set_1 = cPickle.load(f)
    f.close()
    f = open(data_path+data_name+'_valid[0].pkl', 'rb')
    valid_set_0 = cPickle.load(f)
    f.close()
    f.close()
    f = open(data_path+data_name+'_valid[1].pkl', 'rb')
    valid_set_1 = cPickle.load(f)
    f.close()
    f = open(data_path+data_name+'_test[0].pkl', 'rb')
    test_set_0 = cPickle.load(f)
    f.close()
    f = open(data_path+data_name+'_test[1].pkl', 'rb')
    test_set_1 = cPickle.load(f)
    f.close()
    
    train_set = []
    train_set = (train_set_0,train_set_1)
                  
    valid_set = []
    valid_set = (valid_set_0, valid_set_1)
                   
    test_set = []
    test_set = (test_set_0,test_set_1)

    
    print 'nr training instances:  ', len(train_set[0])
    print 'nr features:       ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr features:       ',len(valid_set[0][0])
    print 'nr targets:       ', len(list(set(valid_set[1])))
    print 'nr test instances:      ', len(test_set[0])
    print 'nr features:       ',len(test_set[0][0])
    print 'nr targets:       ', len(list(set(test_set[1])))
    print 'done'
    proportion_of_classes(train_set, valid_set, test_set)
    
    return train_set, valid_set, test_set    


def reduce_data(data_path,data_name,num_of_samples1,num_of_samples2):
    # Load the dataset
    print data_name
    f = open(data_path+data_name+'_train.pkl', 'rb')
    train_set = cPickle.load(f)
    f.close()
    
    f = open(data_path+data_name+'_valid.pkl', 'rb')
    valid_set = cPickle.load(f)
    f.close()
    
    f = open(data_path+data_name+'_test.pkl', 'rb')
    test_set = cPickle.load(f)
    f.close()
    
    print 'Reduced data set'
    #num_of_samples1 = 13208 # start for valid set and also end of train set
    #num_of_samples2 = 13208 + 6604 # end for valid set
    valid_set = []
    valid_set = (numpy.array(train_set[0][num_of_samples1+1:num_of_samples2+1]).astype(theano.config.floatX),
          numpy.array(train_set[1][num_of_samples1+1:num_of_samples2+1]).astype(theano.config.floatX))
    train_set = (numpy.array(train_set[0][0:num_of_samples1]).astype(theano.config.floatX),
          numpy.array(train_set[1][0:num_of_samples1]).astype(theano.config.floatX))
    print 'done'
    
    return train_set, valid_set, test_set

def reduce_frac_data(file,frac, rng_seed):
    # Load the dataset
    f = open(file, 'rb')
    data = cPickle.load(f)
    f.close()
    
    nr_examples = len(data[0])
    num_of_samples = int(numpy.floor(nr_examples*frac))
    
    numpy.random.seed(rng_seed)
    rand_inds = numpy.random.permutation(num_of_samples)   
    data = (numpy.array(data[0][rand_inds]).astype(theano.config.floatX),
                 numpy.array(data[1][rand_inds]).astype(theano.config.floatX))

    data = (numpy.array(data[0][0:num_of_samples]).astype(theano.config.floatX),
                 numpy.array(data[1][0:num_of_samples]).astype(theano.config.floatX))
    return data



def reduce_times_data(data_path,data_name,samples):
    # Load the dataset
    f = open(data_path+data_name+'_train.pkl', 'rb')
    train_set = cPickle.load(f)
    f.close()
    
    f = open(data_path+data_name+'_valid.pkl', 'rb')
    valid_set = cPickle.load(f)
    f.close()
    
    f = open(data_path+data_name+'_test.pkl', 'rb')
    test_set = cPickle.load(f)
    f.close()
    
    print 'Reduced data set'
    num_of_samples = samples*len(list(set(train_set[1]))) # end of train set
    train_set = (numpy.array(train_set[0][0:num_of_samples]).astype(theano.config.floatX),
                 numpy.array(train_set[1][0:num_of_samples]).astype(theano.config.floatX))
    print 'done'
    
    return train_set, valid_set, test_set
  
def proportion_of_classes(train_set, valid_set, test_set):
    # To test the proportion of classes
    for t in range(len(list(set(train_set[1])))):
        print 'proportion of target '+str(t)+' in'
        print '    trai set: '+str(numpy.mean(train_set[1]==t))
    
    print 'nr features:       ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
    print 'targets are:    ', list(set(train_set[1])) 
    print 'nr training instances:  ', len(train_set[0])
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr test instances:      ', len(test_set[0])
    
def dataset_details(data_name, reduce=None):
    host_path = os.getenv('HOME')
    data_path=host_path+'/store/Datasets/pickled/'
    if reduce == None:
        train_set, valid_set, test_set = unpack_data(data_path,data_name)
    elif reduce == 4:
        train_set, valid_set, test_set = unpack_data_target(data_path,data_name)
    elif reduce == 5:
        train_set, valid_set, test_set = unpack_batch_data(data_path,data_name)
    return len(train_set[0][0]), len(list(set(train_set[1])))

def shared_dataset2(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def combine(data_a, data_b,frac,rng_seed):
        
    data = [[],[]]

    data[0] = numpy.concatenate((data_a[0], data_b[0]), axis = 0)
    data[1] = numpy.concatenate((data_a[1], data_b[1]), axis = 0)
    
    nr_examples = len(data[0])   
    num_of_samples = int(numpy.floor(nr_examples*frac))     
    rand_inds = numpy.random.permutation(num_of_samples) 
    
    data = (numpy.array(data[0][rand_inds]).astype(theano.config.floatX),
                numpy.array(data[1][rand_inds]).astype(theano.config.floatX))
    
    data = (numpy.array(data[0][0:num_of_samples]).astype(theano.config.floatX),
                 numpy.array(data[1][0:num_of_samples]).astype(theano.config.floatX))
    
    return data
  

def combine_two_dataset(data_name_a, data_name_b,frac,rng_seed):
    print 'Combining datasets ....'
    host_path = os.getenv('HOME')
    data_path=host_path+'/store/Datasets/pickled/'
    #data_path= '/media/aditya/nimi1/repos/store/Datasets/pickled/'
    train_set_a, valid_set_a, test_set_a = unpack_data(data_path,data_name_a)
    train_set_b, valid_set_b, test_set_b = unpack_data(data_path,data_name_b)
        
    train_set = combine(train_set_a, train_set_b,frac,rng_seed)
    valid_set = combine(valid_set_a, valid_set_b,frac,rng_seed)
    test_set  = combine(test_set_a,  test_set_b, frac,rng_seed)
    
    proportion_of_classes(train_set, valid_set, test_set)
    
    test_set_x, test_set_y = shared_dataset2(test_set)
    valid_set_x, valid_set_y = shared_dataset2(valid_set)
    train_set_x, train_set_y = shared_dataset2(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    

def dropout_weights(a, dropout_rate):
    R = bernoulli.rvs(dropout_rate, size=a.shape)
    R = numpy.reshape(R, a.shape)
    return (a*R).astype(theano.config.floatX)
      
def load_dataset(data_name,reduce=None,samples=None, frac=None, rng_seed=None, fold=None):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############
    host_path = os.getenv('HOME')
    data_path=host_path+'/store/Datasets/pickled/'
    #data_path=host_path+'/repos/Database/pickled/'
    #data_path='/home/aditya/repos/Database/pickled/'
    #data_path='/homes/up201109823/repos/Database/pickled/'
    
    data_path=data_path+fold
    
    print '... loading data'
    print data_name
    if reduce == None:
        train_set, valid_set, test_set = unpack_data(data_path,data_name)
        proportion_of_classes(train_set, valid_set, test_set)
    elif reduce == 1:
        train_set, valid_set, test_set = reduce_times_data(data_path,data_name,samples)
        proportion_of_classes(train_set, valid_set, test_set)
    elif reduce == 2:
        if data_name == 'mnist_subset2':
            num_of_samples1 = 13208 # start for valid set and also end of train set
            num_of_samples2 = 13208 + 6604 # end for valid set
            data_name_reduce = 'mnist'
            train_set, valid_set, test_set = reduce_data(data_path,data_name_reduce,num_of_samples1,num_of_samples2)
            proportion_of_classes(train_set, valid_set, test_set)
        elif data_name == 'mnist_subset1':
            num_of_samples1 = 5080 # start for valid set and also end of train set
            num_of_samples2 = 10000 # end for valid set
            data_name_reduce = 'mnist'
            train_set, valid_set, test_set = reduce_data(data_path,data_name_reduce,num_of_samples1,num_of_samples2)
            proportion_of_classes(train_set, valid_set, test_set)
    elif reduce == 3:
        file_train = data_path+data_name+'_train.pkl'  
        train_set = reduce_frac_data(file_train,frac, rng_seed)
        file_valid = data_path+data_name+'_valid.pkl'
        valid_set = reduce_frac_data(file_valid,frac, rng_seed)
        f = open(data_path+data_name+'_test.pkl', 'rb')
        test_set = cPickle.load(f)
        f.close() 
        print 'train_set[1]', train_set[1][0:5]
        proportion_of_classes(train_set, valid_set, test_set)
    elif reduce == 4:
        train_set, valid_set, test_set = unpack_data_target(data_path,data_name)
        proportion_of_classes(train_set, valid_set, test_set)
    elif reduce == 5:
        train_set, valid_set, test_set = unpack_batch_data(data_path,data_name)
        print 'test_set[1]', test_set[1]
        proportion_of_classes(train_set, valid_set, test_set)
    elif reduce == 10:  
        train_set = load(data_path+data_name+'_train.pkl.gz')
        valid_set = load(data_path+data_name+'_valid.pkl.gz')
        test_set = load(data_path+data_name+'_test.pkl.gz')        
        proportion_of_classes(train_set, valid_set, test_set)
        

        
        
 
        
           
    
        
    
    ############################################################
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
#     def shared_dataset2(data_xy, borrow=True):
#         """ Function that loads the dataset into shared variables
#     
#         The reason we store our dataset in shared variables is to allow
#         Theano to copy it into the GPU memory (when code is run on GPU).
#         Since copying data into the GPU is slow, copying a minibatch everytime
#         is needed (the default behaviour if the data is not in a shared
#         variable) would lead to a large decrease in performance.
#         """
#         data_x, data_y = data_xy
#         shared_x = theano.shared(numpy.asarray(data_x,
#                                                dtype=theano.config.floatX),
#                                  borrow=borrow)
#         shared_y = theano.shared(numpy.asarray(data_y,
#                                                dtype=theano.config.floatX),
#                                  borrow=borrow)
#         # When storing data on the GPU it has to be stored as floats
#         # therefore we will store the labels as ``floatX`` as well
#         # (``shared_y`` does exactly that). But during our computations
#         # we need them as ints (we use labels as index, and if they are
#         # floats it doesn't make sense) therefore instead of returning
#         # ``shared_y`` we will have to cast it to int. This little hack
#         # lets ous get around this issue
#         return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset2(test_set)
    valid_set_x, valid_set_y = shared_dataset2(valid_set)
    train_set_x, train_set_y = shared_dataset2(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
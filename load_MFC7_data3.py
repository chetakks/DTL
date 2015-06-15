import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
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

def load_outputs(dir,prefix='outputs_'):
    #print inspect.stack()[0][3]
    print 'loading',dir
    outputs_list = list()
    
    #for filename in sorted(os.listdir(dir)): 
    for filename in os.listdir(dir): 
        if filename.startswith(prefix):
            print '.',
            if   filename.endswith('.pkl'):
                outputs = cPickle.load(open(dir+filename,'rb'))
            elif filename.endswith('.gz' ):
                outputs = load(dir+filename)
            print 'filename', filename
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

def shuffel(data,rng_seed,frac):

    nr_examples = len(data[0])
    num_of_samples = int(np.floor(nr_examples*frac))
    
    np.random.seed(rng_seed)
    rand_inds = np.random.permutation(num_of_samples)   
    data = (np.array(data[0][rand_inds]).astype(theano.config.floatX),
                 np.array(data[1][rand_inds]).astype(theano.config.floatX))
    return data
    
# def unpack_MFC7_gzip_data(settings, data_name, rng_seed):
#     
#     compound_label= settings['compound_label']
#     data_path = settings['data_path'] 
#     fold = settings['fold']     
#     print 'compound_label', compound_label
#     
#     data_path = data_path + fold
#     print 'data_path', data_path
#     
#    
#     train_set = load(data_path+data_name+'_train.pkl.gz')
#     valid_set = load(data_path+data_name+'_valid.pkl.gz')
#     
#     train_set = shuffel(train_set,rng_seed,frac= settings['training_data_fraction'])
#     valid_set = shuffel(valid_set,rng_seed,frac= settings['training_data_fraction'])
#     
#     treatment_items =load(data_path+settings['target_dataset']+'_treatment_items.pkl.gz')
#     treatments = np.where(treatment_items[0:,0] == compound_label )
#     #print treatments
#     print 'treatments for the compound:'
#     print treatment_items[treatments]
# 
# 
#     idst = treatments[0][0]
#     print idst
#     f1  = settings['target_dataset']+'_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
#     if not os.path.exists(data_path+f1):
#         print 'data path is wrong for the file', f1
#     
#     test_set  = load(data_path+f1)    
#     print 'done loading'
#     
#     return train_set, valid_set, test_set 





def unpack_MFC7_gzip_data(settings, data_name, rng_seed):
    
    compound_label  = settings['compound_label']
    data_path       = settings['data_path'] 
    fold            = settings['fold']     
    print 'compound_label', compound_label
    
    data_path = data_path + fold
    print 'data_path', data_path
    
    toggle = 0
    if not os.path.exists(data_path+data_name+'_train.pkl.gz'):
        if settings['target_dataset'] == 'MFC7_set1':
            print 'Using randomly selected model '
            data_name = 'MFC7_set1_ALLN'
            toggle = 1
        elif settings['target_dataset'] == 'MFC7_set2':
            data_name = 'MFC7_set2_AZ-A'
            print 'Using randomly selected model '
            toggle = 1

    train_set = load(data_path+data_name+'_train.pkl.gz')
    valid_set = load(data_path+data_name+'_valid.pkl.gz')
    
    train_set = shuffel(train_set,rng_seed,frac= settings['training_data_fraction'])
    valid_set = shuffel(valid_set,rng_seed,frac= settings['training_data_fraction'])
    
    
    if toggle == 1:
        if settings['target_dataset'] == 'MFC7_set1':
            print 'Switching the test sets '
            data_path = '/home/aditya/store/Datasets/pickled/MFC7/set2_loov3/'
            treatment_items =load(data_path+'MFC7_set2_treatment_items.pkl.gz')
            treatments = np.where(treatment_items[0:,0] == compound_label )
            print 'treatments for the compound:', treatment_items[treatments]
            idst = treatments[0][0]
            f1  = 'MFC7_set2_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
        elif settings['target_dataset'] == 'MFC7_set2':
            print 'Switching the test sets '
            data_path = '/home/aditya/store/Datasets/pickled/MFC7/set1_loov3/'
            treatment_items =load(data_path+'MFC7_set1_treatment_items.pkl.gz')
            treatments = np.where(treatment_items[0:,0] == compound_label )
            print 'treatments for the compound:', treatment_items[treatments]
            idst = treatments[0][0]
            f1  = 'MFC7_set1_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
        
        if not os.path.exists(data_path+f1):
            print 'data path is wrong for the file', f1
        
        test_set  = load(data_path+f1)    
        print 'done loading'
    
    elif toggle == 0:
        treatment_items =load(data_path+settings['target_dataset']+'_treatment_items.pkl.gz')
        treatments = np.where(treatment_items[0:,0] == compound_label )
        print 'treatments for the compound:', treatment_items[treatments]
        idst = treatments[0][0]
        print idst
        f1  = settings['target_dataset']+'_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
        if not os.path.exists(data_path+f1):
            print 'data path is wrong for the file', f1
        
        test_set  = load(data_path+f1)    
        print 'done loading'
    
    return train_set, valid_set, test_set 



  
def proportion_of_classes(train_set, valid_set, test_set):
    # To test the proportion of classes
    for t in range(len(list(set(train_set[1])))):
        print 'proportion of target '+str(t)+' in'
        print '    trai set: '+str(np.mean(train_set[1]==t))
    
    print 'nr features:       ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
    print 'targets are:    ', list(set(train_set[1])) 
    print 'nr training instances:  ', len(train_set[0])
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr test instances:      ', len(test_set[0])
    


def shared_dataset2(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
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

def load_test_dataset(f1):
    ''' Loads test dataset
    :type dataset: string
    '''
    print '... loading test data per_treatment'
    print f1
    test_set  = load(f1)
    test_set_x, test_set_y = shared_dataset2(test_set)
    rval = (test_set_x, test_set_y)

    return rval
    
    

    
def load_multi_test_dataset(settings,data_name):
    
    compound_label= settings['compound_label']
    data_path = settings['data_path'] 
    fold = settings['fold']     
    print 'compound_label', compound_label
    
    data_path = data_path + fold
    print 'data_path', data_path
    print '... loading test data per treatement for compound', compound_label
    
    toggle = 0
    if not os.path.exists(data_path+data_name+'_train.pkl.gz'):
        toggle = 1
    
    
    if toggle == 1:
        if settings['target_dataset'] == 'MFC7_set1':
            print 'Switching the test sets '
            data_path = '/home/aditya/store/Datasets/pickled/MFC7/set2_loov3/'
            treatment_items =load(data_path+'MFC7_set2_treatment_items.pkl.gz')
            treatments = np.where(treatment_items[0:,0] == compound_label )
            print 'treatments for the compound:', treatment_items[treatments]
            multi_test_sets = []
            multi_test_labels = []
            for ids in  range(len(np.array(treatments)[0])):
                idst = treatments[0][ids]
                #print idst
                f1  = 'MFC7_set2_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
                if not os.path.exists(data_path+f1):
                    print 'data path is wrong for the file',f1
                    
                multi_test_set = load_test_dataset(data_path+f1)
                multi_test_sets.append(multi_test_set)
                multi_test_labels.append(f1)
        
        elif settings['target_dataset'] == 'MFC7_set2':
            print 'Switching the test sets '
            data_path = '/home/aditya/store/Datasets/pickled/MFC7/set1_loov3/'
            treatment_items =load(data_path+'MFC7_set1_treatment_items.pkl.gz')
            treatments = np.where(treatment_items[0:,0] == compound_label )
            print 'treatments for the compound:', treatment_items[treatments]
            multi_test_sets = []
            multi_test_labels = []
            for ids in  range(len(np.array(treatments)[0])):
                idst = treatments[0][ids]
                #print idst
                f1  = 'MFC7_set1_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
                if not os.path.exists(data_path+f1):
                    print 'data path is wrong for the file',f1
                    
                multi_test_set = load_test_dataset(data_path+f1)
                multi_test_sets.append(multi_test_set)
                multi_test_labels.append(f1)
    
    elif toggle == 0:
        treatment_items =load(data_path+settings['target_dataset']+'_treatment_items.pkl.gz')
        treatments = np.where(treatment_items[0:,0] == compound_label )
        print 'treatments for the compound:', treatment_items[treatments]
        multi_test_sets = []
        multi_test_labels = []
        for ids in  range(len(np.array(treatments)[0])):
            idst = treatments[0][ids]
            #print idst
            f1  = settings['target_dataset']+'_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.pkl.gz'
            if not os.path.exists(data_path+f1):
                print 'data path is wrong for the file',f1
                
            multi_test_set = load_test_dataset(data_path+f1)
            multi_test_sets.append(multi_test_set)
            multi_test_labels.append(f1)
        

            
    return multi_test_sets, multi_test_labels
    

def load_dataset(settings,data_name, rng_seed):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############
    
    print '... loading data'
    train_set, valid_set, test_set = unpack_MFC7_gzip_data(settings,data_name, rng_seed)
    proportion_of_classes(train_set, valid_set, test_set)

        
    ############################################################
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    

    
    test_set_x, test_set_y = shared_dataset2(test_set)
    valid_set_x, valid_set_y = shared_dataset2(valid_set)
    train_set_x, train_set_y = shared_dataset2(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval
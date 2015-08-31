import os
import sys
import numpy as np
import theano
import theano.tensor as T
import pickle
sys.argv.pop(0)
input_folder = sys.argv.pop(0)
file_name = sys.argv.pop(0)
output_folder = sys.argv.pop(0)
split = int(sys.argv.pop(0))


print 'file name:', file_name
#file_name = 'mnist_train'
file_path = '/home/aditya/store/Datasets/mnist_variations/'+input_folder
fd = open(file_path + '.amat','r') 
 
datas = []
labels = []
for line in fd:
    line_split = line.split()
    #print len(line_split)
    datas.append(line_split[0:(28*28)])
    label = line_split[(28*28):]
    labels.append(label[0])
fd.close()

dat_set = []
dat_set = (np.array(datas).astype(theano.config.floatX),
             np.array(labels).astype(theano.config.floatX))

#print output_folder
#output_folder= '/home/aditya/store/Datasets/mnist_var/pickled'
#output_folder= '/home/aditya/repos/Database/shapeset/pickled'
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)  

#split = 1
if split == 1:
    #dat_set_train= ([],[])
    #(dat_set_train[0][0:10000],dat_set_train[1][0:10000]) = dat_set
    dat_set_train = (dat_set[0][0:10000],dat_set[1][0:10000])
    print 'nr train instances:', len(dat_set_train[0])
    print 'nr features:       ', len(dat_set_train[0][0])
    print 'nr targets:        ', len(list(set(dat_set_train[1])))
    dat_set_valid = (dat_set[0][10000:],dat_set[1][10000:])
    print 'nr valid instances:', len(dat_set_valid[0])
    print 'nr features:       ', len(dat_set_valid[0][0])
    print 'nr targets:        ', len(list(set(dat_set_valid[1])))
    
    pickle.dump(dat_set_train, open(file_name+'_train.pkl', 'wb'))
    pickle.dump(dat_set_valid, open(file_name+'_valid.pkl', 'wb'))

else:
    print 'nr test instances:   ', len(dat_set[0])
    print 'nr features:         ', len(dat_set[0][0])
    print 'nr targets:          ', len(list(set(dat_set[1]))) 
    pickle.dump(dat_set, open(file_name+'_test.pkl', 'wb'))

print "done writing"
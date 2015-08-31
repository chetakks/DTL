#!/usr/bin/env python

import subprocess
import time
import datetime
import sys
import cPickle
import gzip
import numpy as np
import os
import csv 

def load(filename):
    """Loads a compressed object from disk.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()

    return object

def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def read_results(path, case):
    
    Rfile = path + "check_results.txt"
    i = open( Rfile, 'rb' )
    reader = csv.reader( i )
    
    for line in reader:
        dat = line.pop(0).split()
        if dat[0] == 'error':
            print 'error is', dat[1]
            error = float(dat[1])
        elif dat[0] == 'correct':
            print 'correct',  dat[1]
            correct = dat[1]
        elif dat[0] == 'incorrect':
            print 'incorrect', dat[1]
            incorrect = dat[1]
        elif dat[0] == 'total':
            print 'total', dat[1]
            total = dat[1]
        elif dat[0] == 'Test_time':
            print 'Runtime (without IO) in cpu-seconds:',  dat[1]
            test_time = dat[1]
    if case == 'Train':
        return error, correct, incorrect, total
    elif case == 'Test':
        return error, correct, incorrect, total, test_time

def read_training(path):
    Rfile = path + "training.txt"
    i = open( Rfile, 'rb' )
    reader = csv.reader( i )
    
    for line in reader:
        dat = line.pop(0).split()
        if dat[0] == 'time':
            print 'Runtime in cpu-seconds:', dat[1]
            train_time = float(dat[1])
        elif dat[0] == 'Num_SV':
            print 'Number of SV', dat[1]
            Num_SV = float(dat[1])
        elif dat[0] == 'Num_i':
            print 'Number of iterations', dat[1]
            Num_i = float(dat[1])
        
    return train_time, Num_SV, Num_i
            
            
def exe(target_dataset,compound_label, dir, cost, kernel,g, frac, repetition):
    print "Running libsvm for the bbbc data ...";
    """
    the folder libsvm_data1 has normalized data all compounds with 100% data.
    
    the folder libsvm_data2 has test sets normalized with min and max from train set.
    all compounds with 100% data.
    
    the folder libsvm_data3 has test sets normalized with min and max from train set.
    the data are only 10%
    
    the folder libsvm_data4 has test sets normalized with min and max from train set 
    and PCA with 20 components. has only 10% data of campothein
    
    the folder libsvm_data_10 has test sets normalized with min and max from train set.
    all compounds with 10% data.
    
    the folder libsvm_data_1 has test sets normalized with min and max from train set.
    all compounds with 1% data.
    
    """
    
    if target_dataset == 'MFC7_set1':
        path = "/home/aditya/store/Datasets/libsvm_data_10/MFC7_set1/"
        
    elif target_dataset == 'MFC7_set2':
        path = "/home/aditya/store/Datasets/libsvm_data_10/MFC7_set2/"
    #path = "/home/aditya/store/Datasets/libsvm_data_only1/"
    code = "/home/aditya/store/Theano/SVM/multiclass_svm_light/svm_multiclass/"
    

    
    
    #target_dataset = 'MFC7_set1'
    #compound_label = 'ALLN'
    
    
    
    MODEL_path = code+dir+'/models/'+target_dataset+"/"
    
    if not os.path.exists(MODEL_path):
        os.makedirs(MODEL_path)
    
#     FNAME = target_dataset +'_'+ compound_label +'_train.dat'    
#     MODEL             = MODEL_path+target_dataset +'_'+ compound_label +'_train.model'
#     ALL_RESULTS       = dir +"/results/"+target_dataset+"/"+target_dataset +"_results.txt"
#     CHART             = dir +"/results/"+target_dataset+"/"+target_dataset +"_chart.txt"
#     CONSOLE           = dir +"/results/"+target_dataset+"/"+target_dataset +'_'+ compound_label +"_console.txt"
#     train_PREDICTIONS = dir +"/results/"+target_dataset+"/"+target_dataset +'_'+ compound_label +"_train_pred.txt"
#     test_PREDICTIONS  = dir +"/results/"+target_dataset+"/"+target_dataset +'_'+ compound_label +"_test_pred.txt"
    
    ## Experimetal setup for fractional model
    FNAME = target_dataset +'_'+ compound_label +'_train_X'+str(frac)+'.dat'
    MODEL             = MODEL_path+target_dataset +'_'+ compound_label +'_train_X'+str(frac)+'model_rep'+str(repetition)
    ALL_RESULTS       = dir +"/results/"+target_dataset+"/"+target_dataset +"_results2_rep"+str(repetition)+".txt"
    CHART             = dir +"/results/"+target_dataset+"/"+target_dataset +"_chart2_rep"+str(repetition)+".txt"
    CONSOLE           = dir +"/results/"+target_dataset+"/"+target_dataset +'_'+ compound_label +"_console.txt"
    train_PREDICTIONS = dir +"/results/"+target_dataset+"/"+target_dataset +'_'+ compound_label +"_train_X"+str(frac)+"pred_rep"+str(repetition)+".txt"
    test_PREDICTIONS  = dir +"/results/"+target_dataset+"/"+target_dataset +'_'+ compound_label +"_test_X"+str(frac)+"pred_rep"+str(repetition)+".txt"
    
   
   
    # Training
    #execute1 = "./svm_multiclass_learn", "-c", cost, "-t", kernel, path+FNAME, MODEL
    execute1 = "./svm_multiclass_learn", "-c", cost, "-t", kernel, "-g", g, path+FNAME, MODEL
    
    #./svm_multiclass_learn -c 1.0, -t 1 -d 3 /home/aditya/store/Datasets/libsvm_data/MFC7_set1/MFC7_set1_ALLN_train.dat, tra.model
    
    
    # Training Accuracy
    execute2 = "./svm_multiclass_classify", path+FNAME, MODEL, train_PREDICTIONS
    
    #taskset -c 1 ./svm_multiclass_classify /home/aditya/store/Datasets/libsvm_data/MFC7_set1/MFC7_set1_ALLN_train.dat tra.model tra.pred
    
    # Test Accuracy
    execute3 = "./svm_multiclass_classify"
    
#     # Training
#     execute1 = "./svm_multiclass_learn", "-c", cost, path+FNAME, MODEL, ">", CONSOLE, "2",">","&","1", "&"
#     # Training Accuracy
#     execute2 = "./svm_multiclass_classify", path+FNAME, MODEL, train_PREDICTIONS, ">>", "foo.err", "1",">>", CONSOLE
#     # Test Accuracy
#     execute3 = "./svm_multiclass_classify"
    

    
    
    options = [execute1,execute2,execute3]
    options = [execute3]
    for idx, option in enumerate(options):
#         print idx, option
#         if idx == 0:
#             f0 = open(ALL_RESULTS,'a')
#             old_stdout = sys.stdout   
#             sys.stdout = f0
#             print '='*30 
#             print "Training ... ", FNAME
#             process = subprocess.call(option)
#             train_time, Num_SV, Num_i = read_training(code)
#             print 'Total training time %.2fs' % train_time
#             sys.stdout=old_stdout 
#             f0.close()
#              
#  
#         elif idx == 1:
#             f1 = open(ALL_RESULTS,'a')
#             old_stdout = sys.stdout   
#             sys.stdout = f1 
#             print '.'*30
#             print 'Training examples'
#             process = subprocess.call(option)
#             train_error, correct, incorrect, total  =  read_results(code,'Train')
#             sys.stdout=old_stdout 
#             f1.close()
#              
#             f11 = open(CHART,'a')
#             old_stdout = sys.stdout   
#             sys.stdout = f11
#             print 'Train', compound_label, 'None', total, train_time, train_error, Num_SV, Num_i
#             sys.stdout=old_stdout 
#             f11.close()
#              
#         elif idx == 2: 
            treatments = np.where(treatment_items[0:,0] == compound_label )
            print 'treatments for the compound:', treatment_items[treatments]
            mtest_errors = []
            for ids in  range(len(np.array(treatments)[0])):
                f2 = open(ALL_RESULTS,'a')
                old_stdout = sys.stdout   
                sys.stdout = f2 
                idst = treatments[0][ids]
                #TNAME  = target_dataset+'_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test.dat'
                
                ## Experimetal setup for fractional model
                frac = 1.0
                TNAME  = target_dataset+'_'+treatment_items[idst][0]+'_'+str(treatment_items[idst][1])+'_test_X'+str(frac)+'.dat'
                execute3 = "./svm_multiclass_classify", path+TNAME, MODEL, test_PREDICTIONS, ">>", "foo.err", "1",">>", CONSOLE
                print '.'*30
                print 'Testing examples', TNAME
                process = subprocess.call(execute3)
                mtest_error, correct, incorrect, total, test_time =  read_results(code, 'Test')
                sys.stdout=old_stdout 
                f2.close()
                
                f12 = open(CHART,'a')
                old_stdout = sys.stdout   
                sys.stdout = f12
                print 'Test', compound_label, str(treatment_items[idst][1]), total, test_time, mtest_error
                sys.stdout=old_stdout 
                f12.close()
                
                mtest_error, correct, incorrect, total, test_time =  read_results(code, 'Test')
                mtest_errors.append(mtest_error)
    train_time = 0
    train_error = 0            
    return mtest_errors, train_time, train_error







# ---      


sys.argv.pop(0)
dir = sys.argv.pop(0)
cost = sys.argv.pop(0)
kernel = sys.argv.pop(0)
g = sys.argv.pop(0)


#dir = 'experiments/default'
#cost = "1.0"

# dir = 'experiments/cost_50000'
# cost = "50000"
# 
# dir = 'experiments/cost_100000'
# cost = "100000"

results_dir = "/home/aditya/store/Theano/SVM/multiclass_svm_light/svm_multiclass/"+dir+"/results/"                
target_datasets = ['MFC7_set1', 'MFC7_set2']
target_datasets = ['MFC7_set1']
repetition = 1
frac = 0.01
frac = 0.1
mtest_errors_cv = []
train_time_cv = []
train_errors_cv = []
for target_dataset in target_datasets:
    if target_dataset == 'MFC7_set1':
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        print 'target_dataset', target_dataset
        fold        = 'set1_loov3a/'
        path        = '/home/aditya/store/Datasets/pickled/MFC7/' + fold
        compound_items  = load(path+target_dataset+'_compound_items.pkl.gz')
        #moa_items       = load(path+target_dataset+'_moa_items.pkl.gz')
        treatment_items = load(path+target_dataset+'_treatment_items.pkl.gz')
        if not os.path.exists(results_dir +target_dataset+"/"):
            os.makedirs(results_dir +target_dataset+"/")

        for idx, compound_label in enumerate(compound_items):
            #if idx <= 3:
            #if compound_label == 'camptothecin':
#                 ## Experimetal setup for fractional model
#                 fractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5]
#                 for frac in fractions: 
            print 'compound_label', compound_label
            mtest_errors, train_time, train_error = exe(target_dataset,compound_label, dir, cost, kernel,g, frac, repetition)
            #mtest_errors, train_time, train_error = exe(target_dataset,compound_label, dir, cost, kernel,g, None)
            mtest_errors_cv.append(mtest_errors)
            train_time_cv.append(train_time)
            train_errors_cv.append(train_error)
    elif target_dataset == 'MFC7_set2':
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        fold        = 'set2_loov3a/'
        path        = '/home/aditya/store/Datasets/pickled/MFC7/' + fold
        compound_items  = load(path+target_dataset+'_compound_items.pkl.gz')
        #moa_items       = load(path+target_dataset+'_moa_items.pkl.gz')
        treatment_items = load(path+target_dataset+'_treatment_items.pkl.gz')

        if not os.path.exists(results_dir +target_dataset+"/"):
            os.makedirs(results_dir +target_dataset+"/")
        
        for idx, compound_label in enumerate(compound_items):
            #if compound_label == 'emetine':
            #if compound_label == 'AZ-A':
            #if compound_label == 'vincristine':
            
            
            print 'compound_label', compound_label
            mtest_errors, train_time, train_error = exe(target_dataset,compound_label, dir, cost, kernel,g, frac, repetition)
            #mtest_errors, train_time, train_error = exe(target_dataset,compound_label, dir, cost, kernel,g,None)
            mtest_errors_cv.append(mtest_errors)
            train_time_cv.append(train_time)
            train_errors_cv.append(train_error)
        
    outputs = {}
    
    outputs['mtest_errors_cv']        = mtest_errors_cv
    outputs['train_time_cv']          = train_time_cv
    outputs['train_errors_cv']        = train_errors_cv
    

    output_file_path = results_dir +target_dataset+"/"+'outputs_'+timestamp+'_%03d.pkl.gz' % (repetition)
    save(outputs,output_file_path)
    
    print 'mean test error = %.2f(%.2f)' % (np.mean(np.hstack(mtest_errors_cv)), np.std(np.hstack(mtest_errors_cv)))
    print 'Test error =' + str(mtest_errors_cv)
    print 'mean Time to train = mean %.2f(%.2f)s' % (np.mean(train_time_cv), np.std(train_time_cv))
    print 'mean Train error = mean %.2f(%.2f)s' % (np.mean(train_errors_cv), np.std(train_errors_cv))
    
    f3 = open(results_dir+target_dataset+"/"+target_dataset +"_results2_rep"+str(repetition)+".txt",'a')
    old_stdout = sys.stdout   
    sys.stdout = f3
    print '*'*30 
    print '*'*30 
    print 'target_dataset', target_dataset
    print 'repetition', repetition
    print 'Experiment', dir
    print 'cost', cost
    print 'kernel', kernel
    print 'gamma', g
    print 'fraction', frac
    print 'mean test error = %.2f(%.2f)' % (np.mean(np.hstack(mtest_errors_cv)), np.std(np.hstack(mtest_errors_cv)))
    print 'Test error =' + str(mtest_errors_cv)
    print 'Mean time to train = mean %.2f(%.2f)s' % (np.mean(train_time_cv), np.std(train_time_cv))
    print 'mean train error = mean %.2f(%.2f)' % (np.mean(train_errors_cv), np.std(train_errors_cv))
    print 'Train error =' + str(train_errors_cv)
    print '*'*30 
    print '*'*30
    sys.stdout=old_stdout 
    f3.close()
    
  
    
    
        
                
     


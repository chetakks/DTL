def load_majvote_accus_cv(result_dir,nr_reps):
    dat = []
    for ri in range(nr_reps):
        print '---------------------------------repetition: ', ri+1
    
        output_file_path = result_dir+'outputs_cv'+'_%03d.pkl.gz' % (ri+1)
        print 'output_file_path', output_file_path
        outputs_cv = load(output_file_path)  
        
        mtest_majvote_accus_cv = outputs_cv['mtest_majvote_accus_cv']   
        
        dat.append(np.mean(np.hstack(mtest_majvote_accus_cv)))
        #print np.mean(np.hstack(mtest_majvote_accus_cv))
    
    return dat

def load_classification_report_cv(result_dir,nr_reps):
    dat = []
    nr_class = 6
    precisions = np.zeros([nr_class,nr_reps])
    recalls = np.zeros([nr_class,nr_reps])
    f1s = np.zeros([nr_class,nr_reps])
    supports = np.zeros([nr_class,nr_reps])
    
    for ri in range(nr_reps):
        print '---------------------------------repetition: ', ri+1
    
        output_file_path = result_dir+'outputs_cv'+'_%03d.pkl.gz' % (ri+1)
        print 'output_file_path', output_file_path
        outputs_cv = load(output_file_path)  
        ground_truth = outputs_cv['ground_truth']
        prediction   = outputs_cv['prediction']
        labels       = outputs_cv['labels']
        
        #print ground_truth, prediction, labels
        from sklearn.metrics import classification_report
        import sys
        
        report = classification_report(ground_truth, prediction, target_names=labels)
        
        fc = open(result_dir+'Report.txt','w+')
        old_stdout = sys.stdout   
        sys.stdout = fc
        print report
        sys.stdout=old_stdout 
        fc.close()
        precisions[:,ri], recalls[:,ri], f1s[:,ri], supports[:,ri] = read_results(result_dir)

    fcv = open(result_dir+'classification_report.txt','w+')
    old_stdout = sys.stdout   
    sys.stdout = fcv 
    
    print "MOA precision recall f1 support"
    for i in range(nr_class):    
        print '%s %.2f(%.2f) %.2f(%.2f) %.2f(%.2f) %d' % (labels[i],
                                                               np.mean(precisions, axis=1)[i], np.std(precisions, axis=1)[i],
                                                               np.mean(recalls,    axis=1)[i], np.std(recalls,    axis=1)[i],
                                                               np.mean(f1s,        axis=1)[i], np.std(f1s,        axis=1)[i],
                                                               np.mean(supports,   axis=1)[i])
    print 'Avg %.2f(%.2f) %.2f(%.2f) %.2f(%.2f) %d' % (np.mean(precisions),np.std(precisions),
                                                     np.mean(recalls), np.std(recalls),
                                                     np.mean(f1s),np.std(f1s),
                                                     np.sum(np.mean(supports,   axis=1)))
    sys.stdout=old_stdout 
    fcv.close()
        
    return dat


def read_results(path):
    import csv
    Rfile = path + 'Report.txt'
    i = open( Rfile, 'rb' )
    reader = csv.reader( i )
    print reader
    import numpy as np

    precision = np.zeros([6])
    recall = np.zeros([6])
    f1 = np.zeros([6])
    support = np.zeros([6])
    count = 0
    for line_nr, line in enumerate(reader):
        #print line
        #print line_nr
        if (line_nr != 0 and line_nr != 1 and line_nr != 8 and line_nr != 9 and line_nr != 10):
            dat = line.pop(0).split()
            #print dat
            precision[count]= float(dat[1])
            recall[count]=float(dat[2])
            f1[count]=float(dat[3])
            support[count]=float(dat[4])
            count = count + 1
    #print np.mean(precision), np.mean(recall), np.mean(f1), np.sum(support)
    return precision, recall, f1, support
            

import cPickle
import gzip
def load(filename):
    """Loads a compressed object from disk.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()

    return object

import numpy as np
def cal_rep_scores(dat_rep):
    a = []
    b = []
    for i in range(len(dat_rep)):
        a.append(np.mean(np.hstack(dat_rep[i])))
        b.append(np.std(np.hstack(dat_rep[i])))
    #print 'mean of each repititon', a
    #print 'std  of each repititon', b
    return np.mean(a), np.std(b)

def box_plot_performance(target_outputs_dir, target_name, data):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title("Performance of "+target_name+ " with DNNs")
    plt.ylabel('Accuracy (higher the better)')
    #plt.xlabel('Baseline and DTL approaches')
    
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    
    ## add patch_artist=True option to ax.boxplot() 
    ## to get fill color
    bp = ax.boxplot(data, patch_artist=True)
    
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    
    ## Custom x-axis labels
    ax.set_xticklabels(['baseline', 'DTL_1', 'DTL_2', 'DTL_3', 'DTL_4'])

    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    fig.savefig(target_outputs_dir+target_name+'_dnn_performance.png',bbox_inches='tight',dpi=200)


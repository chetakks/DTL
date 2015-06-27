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
    ax.set_xticklabels(['baseline', 'DTL_all', 'DTL_generic', 'DTL_both', 'DTL_specific'])
    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    fig.savefig(target_outputs_dir+target_name+'_dnn_performance.png',bbox_inches='tight',dpi=200)


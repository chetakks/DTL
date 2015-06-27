import numpy as np
import matplotlib.pyplot as plt


def plot_cm(m,mean_accu, std_accu, target_outputs_dir, rep, type_of_cm,labels, approach=None,target_dataset=None,source_dataset=None,transfer=None ):
    print 'avg cms'
    print target_outputs_dir
    
    cm = m
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #m = np.matrix.round(np.mean(cm,axis=0)).astype(int)
    print m
    m = np.around(m, decimals=1)
    print m
    
    fig = plt.figure()
    
    if approach == 'BL':
        fig.suptitle('Baseline DNN', fontsize=14, fontweight='bold')
    if approach == 'TL':
        if target_dataset == 'MFC7_set1':
            dat_name = 'Set1'
        elif target_dataset == 'MFC7_set2':
            dat_name = 'Set2'
        
        #fig.suptitle('Reusing Set1 to classify Set2', fontsize=14, fontweight='bold')
        fig.suptitle('Reusing '+ source_dataset +' to classify '+ target_dataset, fontsize=14, fontweight='bold')
        
        fig.suptitle(dat_name + ' DTL '+transfer, fontsize=14, fontweight='bold')
    

    
    plt.imshow(cm_normalized, interpolation='nearest',cmap=plt.cm.Blues)

    #plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
    for x_val, y_val in zip(np.arange(np.shape(m)[0]), np.arange(np.shape(m)[1])):
        if np.sum(m[x_val,:]) == 0:
            value = 0
        else: 
            value = int(round((float(m[x_val,y_val]) / np.sum(m[x_val,:])) *100))
        print m[x_val,y_val], np.sum(m[x_val,:]), value
        for idy, val in enumerate(m[x_val]):
            if val != 0:
                plt.text(idy, x_val, m[x_val,idy], fontsize = '15',  va='center', ha='center')

        #plt.text(13, y_val, str(value)+'%', fontsize = '15',  va='center', ha='center')
        plt.text(np.shape(m)[0], y_val, str(value)+'%', fontsize = '15',  va='center', ha='center')
    #labels = ['Act', 'Aur', 'Ch', 'DD', 'DR', 'Eg5', 'Epi', 'KI', 'MD', 'MS', 'PD', 'PS']
    #labels = ['1', '2', '3', '4', '5', '6']
    #labels = ['MD', 'PS']
    #print "Confusion matrix (Collective): Acc = (%.2f)" % numpy.mean(avg_accs)*100
    #plt.matshow(m)
    #plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
    #plt.title("Overall average accuracy = (%.2f %%)" % (np.mean(mtest_accus_cv)))
    

    
    
    plt.title("Overall average accuracy = %.2f(%.2f)%%" % (mean_accu, std_accu))

    
    #plt.colorbar()
    plt.ylabel('True mechanistic class')
    plt.yticks(np.arange(np.shape(m)[1]), labels)
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(np.shape(m)[0]), labels, rotation='vertical')
    if type_of_cm == 'per_repeatation':
        if approach == 'BL':
            print target_outputs_dir+approach+type_of_cm+str(rep)+'cm.png'
            plt.savefig(target_outputs_dir+approach+type_of_cm+str(rep)+'cm.png',bbox_inches='tight',dpi=200)
        if approach == 'TL':
            print target_outputs_dir+approach+transfer+type_of_cm+str(rep)+'cm.png'
            plt.savefig(target_outputs_dir+approach+transfer+type_of_cm+str(rep)+'cm.png',bbox_inches='tight',dpi=200)
    elif type_of_cm == 'average':
        if approach == 'BL':
            print target_outputs_dir+approach+type_of_cm+'_cm.png'
            plt.savefig(target_outputs_dir+approach+type_of_cm+'_cm.png',bbox_inches='tight',dpi=200)
        if approach == 'TL':
            print target_outputs_dir+approach+transfer+type_of_cm+'_cm.png'
            plt.savefig(target_outputs_dir+approach+transfer+type_of_cm+'_cm.png',bbox_inches='tight',dpi=200)
    plt.close(fig)
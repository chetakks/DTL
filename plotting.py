#from data_handling import *
from load_dataset2 import *
from scipy import stats
#from pymatbridge import Matlab

import numpy
import matplotlib.pyplot as plt
import sys

def print_row_errors(e_mea,e_std,p=None):
    """
    Print results that can be pasted into a table row in LyX with Control+Shift+v.
    """
    
    nr_columns = numpy.shape(e_mea)[0]
    for ci in range(nr_columns):
        if p == None:
            print '{:04.1f} ({:3.1f})\t'.format(e_mea[ci]*100,e_std[ci]*100),
        else:
            if p[ci] >= 0.005:
                print '{:04.1f} ({:3.1f}) [={:4.2f}]\t'.format(e_mea[ci]*100,e_std[ci]*100,p[ci]),
            else:
                print '{:04.1f} ({:3.1f}) [<0.01]\t'.format(e_mea[ci]*100,e_std[ci]*100,p[ci]),
    print '\n'
    pass

def print_row_errors2(e_mea,e_std,p=None):
    """
    Print results that can be pasted into a table row in LyX with Control+Shift+v.
    """
    
    nr_columns = numpy.shape(e_mea)[0]
    for ci in range(nr_columns):
        if p == None:
            print '{:04.1f} ({:3.1f})\t'.format(e_mea[ci],e_std[ci]),
        else:
            if p[ci] >= 0.005:
                print '{:04.1f} ({:3.1f}) [={:4.2f}]\t'.format(e_mea[ci],e_std[ci],p[ci]),
            else:
                print '{:04.1f} ({:3.1f}) [<0.01]\t'.format(e_mea[ci],e_std[ci],p[ci]),
    print '\n'
    pass

def print_row_errors3(e_mea,e_std,p=None):
    """
    Print results that can be pasted into a table row in LyX with Control+Shift+v.
    """
    
    nr_columns = numpy.shape(e_mea)[0]
    for ci in range(nr_columns):
        if p == None:
            print '{:04.2f} ({:3.2f})\n'.format(e_mea[ci]*100,e_std[ci]*100),
        else:
            if p[ci] >= 0.005:
                print '{:04.2f} ({:3.2f}) [={:4.2f}]\n'.format(e_mea[ci]*100,e_std[ci]*100,p[ci]),
            else:
                print '{:04.2f} ({:3.2f}) [<0.01]\n'.format(e_mea[ci]*100,e_std[ci]*100,p[ci]),
    print '\n'
    pass

def print_row_times(e_mea,e_std,p=None):
    """
    Print results that can be pasted into a table row in LyX with Control+Shift+v.
    """
    
    nr_columns = numpy.shape(e_mea)[0]
#   for ci in range(nr_columns):
    for ci in [nr_columns-1]:
        print '{:.0f} ({:.0f})\t'.format(e_mea[ci],e_std[ci]),
    print '\n'
    pass

def plot_e(results_dir,experiments,training_data_fractions,n_ds_max,c,nr_reps,file_name,ylim=None,loc='upper right'):
    """
    Error plots.
    """
#     mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# #     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
# #     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
#     mlab.stop() # in case the server is already running 
#     mlab.start()
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset_name,source_dataset_name,source_reuse_mode,nr_rotations,legend_ei,linestyle,linewidth,color] = experiment
        
        legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            if source_dataset_name is None:
                # baseline or source experiment
                target_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,training_data_fraction,nr_rotations)
            else:
                # reusability experiment
                target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                    
            outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
            
            for ri in range(nr_reps):
                outputs = outputs_list[ri]
                test_error = outputs['ft_test_err']
                test_errors[ri,fi] = test_error
            
        e_mea = numpy.mean(test_errors,0)
        e_std = numpy.std (test_errors,0)
        p = None
        if ei == 0:
            # store baseline stats
            bl_e_mea = e_mea
            bl_e_std = e_std
        else:
            # run t-test
            nr_columns = numpy.shape(e_mea)[0]
            p = -numpy.ones((nr_columns))
#             for ci in range(5):
#                 res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
#                 p[ci] = res['result']
    
        print_row_errors(e_mea,e_std,p)
        
#         nrs_design_samples = numpy.array(training_data_fractions)*n_ds_max/c
#         X = nrs_design_samples + plot_shift
#         Y     = e_mea
#         Y_err = e_std
#         plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#         # this helps to avoid overlapping error bars
#         plot_shift += nrs_design_samples[-1]/300
#         pass
#         
#     alfs=15
#     plt.legend(legend,prop={'size':alfs},loc=loc)
#     plt.xlabel('$n_{ds.ori}/c$')
#     plt.ylabel('$\overline{\\varepsilon}$',rotation=0,fontsize=alfs)
#     plt.xticks(nrs_design_samples,rotation=90)
#     yticks,locs = plt.yticks()
#     ll = ['%.2f' % a for a in yticks]
#     plt.yticks(yticks,ll)
# #   plt.title(title)
# #   plt.xlim(n_ds_max/40,n_ds_max+n_ds_max/40)
#     if ylim is not None:
#         plt.ylim(ylim)
#     else:
#         ax = plt.gca()
#         ax.margins(None,0.03)
#         print 'ylim=',plt.ylim()
#     plt.tight_layout()
#     plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
#     plt.show()
#     plt.close()
#      
#     mlab.stop()

def plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=None,loc='upper right'):
    """
    Error plots.
    """
#     mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# #     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
# #     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
#     mlab.stop() # in case the server is already running 
#     mlab.start()
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers,legend_ei,linestyle,linewidth,color] = experiment
        
        legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            if source_dataset is None:
                # baseline or source experiment
#                 target_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,target_dataset_name,training_data_fraction,nr_rotations)
                target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
            else:
                # reusability experiment
#                 target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                
                target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                 .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
                 
            outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
            #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
            
            for ri in range(nr_reps):
                outputs = outputs_list[ri]
                test_error = outputs['ft_test_err']
                test_errors[ri,fi] = test_error
            
        e_mea = numpy.mean(test_errors,0)
        e_std = numpy.std (test_errors,0)
        #print e_mea
        #print e_std
        p = None
#         if ei == 0:
#             # store baseline stats
#             bl_e_mea = e_mea
#             bl_e_std = e_std
#         else:
#             # run t-test
#             nr_columns = numpy.shape(e_mea)[0]
#             p = -numpy.ones((nr_columns))
#             for ci in range(5):
#                 res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
#                 p[ci] = res['result']
        
        print_row_errors2(e_mea,e_std,p)
        #zzz
        
        nrs_induction_samples = numpy.array(training_data_fractions)*xmax
        X = nrs_induction_samples + plot_shift
        Y     = numpy.mean(test_errors,0)
        Y_err = numpy.std (test_errors,0)
        print_row_errors2(Y,Y_err,p)
        #print_row(Y,Y_err)
        #plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        plot_shift += xmax/200 # helps to distinguish overlapping error bars
        
    alfs=15
    plt.legend(legend,prop={'size':alfs})
    plt.xlabel('$N$')
    plt.ylabel('$\overline{\\varepsilon}$',rotation=0,fontsize=alfs)
    print nrs_induction_samples
    plt.xticks(nrs_induction_samples,rotation=90)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll)
#   plt.title(title)
    plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight')
    plt.show()
    plt.close()
    
    
#         nrs_design_samples = numpy.array(training_data_fractions)*xmax #n_ds_max/c
#         X = nrs_design_samples #+ plot_shift
#         Y     = e_mea
#         Y_err = e_std
#         plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#         #print test_errors
#         #plt.boxplot(test_errors[0],widths=0.1)
#         # this helps to avoid overlapping error bars
#         #plot_shift += nrs_design_samples[-1]/300
#         pass
#         
#     alfs=15
#     plt.legend(legend,prop={'size':alfs},loc=loc)
#     plt.xlabel('$n_{ds.ori}/c$')
#     plt.ylabel('$\overline{\\varepsilon}$',rotation=0,fontsize=alfs)
#     plt.xticks(nrs_design_samples,rotation=90)
#     yticks,locs = plt.yticks()
#     ll = ['%.2f' % a for a in yticks]
#     plt.yticks(yticks,ll)
#     #plt.title(title)
#     plt.xlim(n_ds_max/40,n_ds_max+n_ds_max/40)
#     if ylim is not None:
#         plt.ylim(ylim)
#     else:
#         ax = plt.gca()
#         ax.margins(None,0.03)
#         print 'ylim=',plt.ylim()
#     plt.tight_layout()
#     plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
#     plt.boxplot(test_errors, widths=0.01)
#     plt.show()
#     plt.close()
     
   # mlab.stop()
   


#def plot_e2_sts(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=None,loc='upper right'):
def plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=None, approach=None):
    """
    Error plots.
    """
    loc='upper right'
#     mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# #     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
# #     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
#     mlab.stop() # in case the server is already running 
#     mlab.start()
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    #legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    
    linestyles = ['--','-',':','--',':','--','-',':','--',':','--','-',':','--',':','--','-',':','--',':']
    linewidths = [4,2,2,4,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1]
    colors=['black','red','green','blue','cyan','red','green','blue','cyan','red','green','blue','cyan','red','green','red','green','blue','cyan']
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        #legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        if approach == 'MSTS':  # Cyclic STS
            num_of_STS = 3
            total_transfers = num_of_STS *2
            for transfer in range(1,total_transfers+1):
                
                for fi in range(nr_td_fractions):
                    training_data_fraction = training_data_fractions[fi]
                    
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                                 .format(results_dir,
                                         target_dataset,
                                         source_dataset,
                                         str(transfer))
                    outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                    #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                    
                    for ri in range(nr_reps):
                        outputs = outputs_list[ri]
                        test_error = outputs['ft_test_err']
                        test_errors[ri,fi] = test_error
                
                e_mea = numpy.mean(test_errors,0)
                e_std = numpy.std (test_errors,0)
                print_row_errors2(e_mea,e_std,p)
                nrs_induction_samples = numpy.array(training_data_fractions)*xmax
                X = nrs_induction_samples + plot_shift
                Y     = numpy.mean(test_errors,0)
                Y_err = numpy.std (test_errors,0)
                print_row_errors2(Y,Y_err,p)
                plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
                plot_shift += xmax/200 # helps to distinguish overlapping error bars
                
                             
    
                
        else:            
            for fi in range(nr_td_fractions):
                
                training_data_fraction = training_data_fractions[fi]
                      
                if source_reuse_mode == None:
                
                    target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
                    
                elif source_reuse_mode == 'Join':
                    
                    target_outputs_dir = '{0}{1}_{2}_{3}_{4:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_reuse_mode,
                            source_dataset,
                            training_data_fraction)
                    
                elif source_reuse_mode is not None and source_reuse_mode is not 'Join':
        
                    # reusability experiment
                    target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_dataset,
                            source_reuse_mode,
                            str(retrain_ft_layers),
                            training_data_fraction)
                     
                outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                
                for ri in range(nr_reps):
                    outputs = outputs_list[ri]
                    test_error = outputs['ft_test_err']
                    test_errors[ri,fi] = test_error
            
            e_mea = numpy.mean(test_errors,0)
            e_std = numpy.std (test_errors,0)
            print 'test_errors'
            print test_errors
            #print e_mea
            #print e_std
            p = None
    #         if ei == 0:
    #             # store baseline stats
    #             bl_e_mea = e_mea
    #             bl_e_std = e_std
    #         else:
    #             # run t-test
    #             nr_columns = numpy.shape(e_mea)[0]
    #             p = -numpy.ones((nr_columns))
    #             for ci in range(5):
    #                 res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
    #                 p[ci] = res['result']
            
            print_row_errors2(e_mea,e_std,p)
            #zzz
            
            nrs_induction_samples = numpy.array(training_data_fractions)*xmax
            X = nrs_induction_samples + plot_shift
            Y     = numpy.mean(test_errors,0)
            Y_err = numpy.std (test_errors,0)
            print_row_errors2(Y,Y_err,p)
            #print_row(Y,Y_err)
            plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
            #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
            plot_shift += xmax/700 # helps to distinguish overlapping error bars
        
    alfs=20
    #plt.legend(legend,prop={'size':alfs},loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(legend,prop={'size':alfs}) #,loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(2.1, 2.05))
    plt.xlabel('Number of training samples',fontsize=20)
    plt.ylabel('(lower the better) Avg. Error',rotation=90,fontsize=20)
    print nrs_induction_samples
    plt.xticks(nrs_induction_samples,rotation=90,fontsize=20)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll,fontsize=20)
#   plt.title(title)
    plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
   # mlab.stop()
   
   
def plot_r2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=None, approach=None):
    """
    Relative Error plots for STS ibPRIYA 2015.
    """
    loc='upper right'
#     mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# #     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
# #     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
#     mlab.stop() # in case the server is already running 
#     mlab.start()
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    #legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    
    linestyles = ['--','-',':','--',':','--','-',':','--',':','--','-',':','--',':','--','-',':','--',':']
    linewidths = [4,2,2,4,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1]
    colors=['black','red','green','blue','cyan','red','green','blue','cyan','red','green','blue','cyan','red','green','red','green','blue','cyan']
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        #legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        if approach == 'MSTS':  # Cyclic STS
            num_of_STS = 3
            total_transfers = num_of_STS *2
            for transfer in range(1,total_transfers+1):
                
                for fi in range(nr_td_fractions):
                    training_data_fraction = training_data_fractions[fi]
                    
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                                 .format(results_dir,
                                         target_dataset,
                                         source_dataset,
                                         str(transfer))
                    outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                    #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                    
                    for ri in range(nr_reps):
                        outputs = outputs_list[ri]
                        test_error = outputs['ft_test_err']
                        test_errors[ri,fi] = test_error
                
                e_mea = numpy.mean(test_errors,0)
                e_std = numpy.std (test_errors,0)
                print_row_errors2(e_mea,e_std,p)
                nrs_induction_samples = numpy.array(training_data_fractions)*xmax
                X = nrs_induction_samples + plot_shift
                Y     = numpy.mean(test_errors,0)
                Y_err = numpy.std (test_errors,0)
                print_row_errors2(Y,Y_err,p)
                plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
                plot_shift += xmax/200 # helps to distinguish overlapping error bars
                
                             
    
                
        else:            
            for fi in range(nr_td_fractions):
                
                training_data_fraction = training_data_fractions[fi]
                      
                if source_reuse_mode == None:
                
                    target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
                    
                elif source_reuse_mode == 'Join':
                    
                    target_outputs_dir = '{0}{1}_{2}_{3}_{4:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_reuse_mode,
                            source_dataset,
                            training_data_fraction)
                    
                elif source_reuse_mode is not None and source_reuse_mode is not 'Join':
        
                    # reusability experiment
                    target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_dataset,
                            source_reuse_mode,
                            str(retrain_ft_layers),
                            training_data_fraction)
                     
                outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                
                for ri in range(nr_reps):
                    outputs = outputs_list[ri]
                    test_error = outputs['ft_test_err']
                    test_errors[ri,fi] = test_error
            
            e_mea = numpy.mean(test_errors,0)
            e_std = numpy.std (test_errors,0)
            #print 'test_errors'
            #print test_errors
            #print e_mea
            #print e_std
            nrs_induction_samples = numpy.array(training_data_fractions)*xmax
            X = nrs_induction_samples + plot_shift
            plot_shift += xmax/300
            p = None
            if approach == 'BL' and source_reuse_mode == None:
            #if ei == 0:
                # store baseline stats
                print experiment
                print 'baseline'
                bl_e_mea = e_mea
                bl_e_std = e_std
                print 'BL:', bl_e_mea
                print 'ei:', e_mea
            else:
                r_mea = (bl_e_mea - e_mea) / bl_e_mea
                print experiment
                print 'BL:', bl_e_mea
                print 'ei:', e_mea
                print 'ri:', r_mea
                print '-----------------------------------'
                Y = r_mea
                plt.plot(X, Y, linewidth=2.0)
                
            
                  #numpy.mean(test_errors,0)
                #Y     = numpy.mean(test_errors,0)
                #Y_err = numpy.std (test_errors,0)
                #print_row_errors2(Y,Y_err,p)
                #print_row(Y,Y_err)
                #plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
                #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
             # helps to distinguish overlapping error bars
            
                #plt.plot(X, Y, linewidth=2.0, linestyle=linestyles[ei], color=colors[ei])
                
            
#             else:
#                 # run t-test
#                 nr_columns = numpy.shape(e_mea)[0]
#                 p = -numpy.ones((nr_columns))
#                 for ci in range(5):
#                     res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
#                     p[ci] = res['result']
            
#             print_row_errors2(e_mea,e_std,p)
#             #zzz
#             
#             nrs_induction_samples = numpy.array(training_data_fractions)*xmax
#             X = nrs_induction_samples + plot_shift
#             #Y     = numpy.mean(test_errors,0)
#             #Y_err = numpy.std (test_errors,0)
#             print_row_errors2(Y,Y_err,p)
#             #print_row(Y,Y_err)
#             plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
#             #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#             plot_shift += xmax/700 # helps to distinguish overlapping error bars
#         
    alfs=20
    #plt.legend(legend,prop={'size':alfs},loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(legend,prop={'size':alfs-10}) #,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(legend,prop={'size':alfs-5},loc='upper right') #, bbox_to_anchor=(1, 0.5))
    #plt.legend(legend,prop={'size':alfs-5},loc='lower left') #, bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(2.1, 2.05))
    plt.xlabel('Number of training samples',fontsize=20)
    #plt.ylabel('(lower the better) Avg. Error',rotation=90,fontsize=20)
    plt.ylabel('relative improvement',rotation=90,fontsize=20)
    Y = numpy.zeros(len(e_mea))
    plt.plot(X, Y, linewidth=1.0, color='black', linestyle= '-.' )
    print nrs_induction_samples
    plt.xticks(nrs_induction_samples,rotation=90,fontsize=20)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll,fontsize=20)
#   plt.title(title)
    plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

   
def plot_r3_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=None, approach=None):
    """
    Relative Error plots for STS ibPRIYA 2015.
    """
    loc='upper right'
#     mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# #     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
# #     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
#     mlab.stop() # in case the server is already running 
#     mlab.start()
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    #legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    TLu_r_mea = []
    TLs_r_mea = []
    STS1_r_mea = []
    
    linestyles = ['--','-',':','--',':','--','-',':','--',':','--','-',':','--',':','--','-',':','--',':']
    linewidths = [4,2,2,4,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1]
    colors=['black','red','green','blue','cyan','red','green','blue','cyan','red','green','blue','cyan','red','green','red','green','blue','cyan']
   
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        #legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        if approach == 'MSTS':  # Cyclic STS
            num_of_STS = 3
            total_transfers = num_of_STS *2
            for transfer in range(1,total_transfers+1):
                
                for fi in range(nr_td_fractions):
                    training_data_fraction = training_data_fractions[fi]
                    
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                                 .format(results_dir,
                                         target_dataset,
                                         source_dataset,
                                         str(transfer))
                    outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                    #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                    
                    for ri in range(nr_reps):
                        outputs = outputs_list[ri]
                        test_error = outputs['ft_test_err']
                        test_errors[ri,fi] = test_error
                
                e_mea = numpy.mean(test_errors,0)
                e_std = numpy.std (test_errors,0)
                print_row_errors2(e_mea,e_std,p)
                nrs_induction_samples = numpy.array(training_data_fractions)*xmax
                X = nrs_induction_samples + plot_shift
                Y     = numpy.mean(test_errors,0)
                Y_err = numpy.std (test_errors,0)
                print_row_errors2(Y,Y_err,p)
                plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
                plot_shift += xmax/200 # helps to distinguish overlapping error bars
                
                             
    
                
        else:            
            for fi in range(nr_td_fractions):
                
                training_data_fraction = training_data_fractions[fi]
                      
                if source_reuse_mode == None:
                
                    target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
                    
                elif source_reuse_mode == 'Join':
                    
                    target_outputs_dir = '{0}{1}_{2}_{3}_{4:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_reuse_mode,
                            source_dataset,
                            training_data_fraction)
                    
                elif source_reuse_mode is not None and source_reuse_mode is not 'Join':
        
                    # reusability experiment
                    target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_dataset,
                            source_reuse_mode,
                            str(retrain_ft_layers),
                            training_data_fraction)
                     
                outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                
                for ri in range(nr_reps):
                    outputs = outputs_list[ri]
                    test_error = outputs['ft_test_err']
                    test_errors[ri,fi] = test_error
            
            e_mea = numpy.mean(test_errors,0)
            e_std = numpy.std (test_errors,0)
            #print 'test_errors'
            #print test_errors
            #print e_mea
            #print e_std
            nrs_induction_samples = numpy.array(training_data_fractions)*xmax
            X = nrs_induction_samples #+ plot_shift
            #plot_shift += xmax/300
            
            p = None
            if approach == 'BL' and source_reuse_mode == None:
            #if ei == 0:
                # store baseline stats
                print experiment
                print 'baseline'
                bl_e_mea = e_mea
                bl_e_std = e_std
                print 'BL:', bl_e_mea
                print 'ei:', e_mea
                
            elif approach == 'cBL':
            #if ei == 0:
                # store combined baseline stats
                print experiment
                print 'combined baseline'
                cbl_e_mea = e_mea
                cbl_e_std = e_std
                print 'cBL:', cbl_e_mea
                print 'ei:', e_mea
            elif approach == 'TLu':
                r_mea = (bl_e_mea - e_mea) / bl_e_mea
                print experiment
                print 'BL:', bl_e_mea
                print 'ei:', e_mea
                print 'ri:', r_mea
                print '-----------------------------------'
                TLu_r_mea = numpy.append(TLu_r_mea, r_mea)
                #Y = r_mea
                #plt.plot(X, Y, linewidth=2.0)
            elif approach == 'TLs':
                r_mea = (bl_e_mea - e_mea) / bl_e_mea
                print experiment
                print 'BL:', bl_e_mea
                print 'ei:', e_mea
                print 'ri:', r_mea
                print '-----------------------------------'
                TLs_r_mea = numpy.append(TLs_r_mea, r_mea)
                #Y = r_mea
                #plt.plot(X, Y, linewidth=2.0)
            elif approach == 'STS1':
                r_mea = (bl_e_mea - e_mea) / bl_e_mea
                print experiment
                print 'BL:', bl_e_mea
                print 'ei:', e_mea
                print 'ri:', r_mea
                #print numpy.shape(STS1_r_mea)
                #print numpy.shape(r_mea)
                #print 
                STS1_r_mea = numpy.append(STS1_r_mea, r_mea)
                print '-----------------------------------'
                #Y = r_mea
                #plt.plot(X, Y, linewidth=2.0)
                
#                 
#                 
#             else:
#                 r_mea = (bl_e_mea - e_mea) / bl_e_mea
#                 print experiment
#                 print 'BL:', bl_e_mea
#                 print 'ei:', e_mea
#                 print 'ri:', r_mea
#                 print '-----------------------------------'
#                 Y = r_mea
#                 plt.plot(X, Y, linewidth=2.0)
                
            
                #numpy.mean(test_errors,0)
                #Y     = numpy.mean(test_errors,0)
                #Y_err = numpy.std (test_errors,0)
                #print_row_errors2(Y,Y_err,p)
                #print_row(Y,Y_err)
                #plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
                #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
             # helps to distinguish overlapping error bars
            
                #plt.plot(X, Y, linewidth=2.0, linestyle=linestyles[ei], color=colors[ei])
                
            
#             else:
#                 # run t-test
#                 nr_columns = numpy.shape(e_mea)[0]
#                 p = -numpy.ones((nr_columns))
#                 for ci in range(5):
#                     res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
#                     p[ci] = res['result']
            
#             print_row_errors2(e_mea,e_std,p)
#             #zzz
#             
#             nrs_induction_samples = numpy.array(training_data_fractions)*xmax
#             X = nrs_induction_samples + plot_shift
#             #Y     = numpy.mean(test_errors,0)
#             #Y_err = numpy.std (test_errors,0)
#             print_row_errors2(Y,Y_err,p)
#             #print_row(Y,Y_err)
#             plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
#             #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#             plot_shift += xmax/700 # helps to distinguish overlapping error bars
#         
    alfs=20
    #plt.legend(legend,prop={'size':alfs},loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(legend,prop={'size':alfs-10}) #,loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(legend,prop={'size':alfs-10},loc='lower left') #, bbox_to_anchor=(1, 0.5))
    plt.legend(legend,prop={'size':alfs-10},loc='upper right') #, bbox_to_anchor=(1, 0.5))
    #plt.legend(bbox_to_anchor=(2.1, 2.05))
    plt.xlabel('Number of training samples',fontsize=20)
    #plt.ylabel('(lower the better) Avg. Error',rotation=90,fontsize=20)
    plt.ylabel('relative improvement',rotation=90,fontsize=20)
    
    TLu_r_mea = TLu_r_mea.reshape(2, nr_td_fractions)
    TLu_r_max = numpy.amax(TLu_r_mea, axis=0)
    TLu_r_min = numpy.amin(TLu_r_mea, axis=0)
    plt.plot(X, TLu_r_max, linewidth=2.0, color='blue', linestyle= '-' )
    #plt.plot(X, TLu_r_min, linewidth=2.0, color='red', linestyle= '-' )
    #plt.fill_between(X, TLu_r_max, TLu_r_min, color='grey', alpha='0.5')
    
    TLs_r_mea = TLs_r_mea.reshape(2, nr_td_fractions)
    TLs_r_max = numpy.amax(TLs_r_mea, axis=0)
    TLs_r_min = numpy.amin(TLs_r_mea, axis=0)
    plt.plot(X, TLs_r_max, linewidth=2.0, color='blue', linestyle= '-' )
    plt.plot(X, TLs_r_min, linewidth=2.0, color='blue', linestyle= '-' )
    #plt.fill_between(X, TLs_r_max, TLs_r_min, color='brown', alpha='0.5')
    
    STS1_r_mea = STS1_r_mea.reshape(2, nr_td_fractions)
    STS1_r_max = numpy.amax(STS1_r_mea, axis=0)
    STS1_r_min = numpy.amin(STS1_r_mea, axis=0)
    plt.plot(X, STS1_r_max, linewidth=2.0, color='blue', linestyle= '-' )
    #plt.plot(X, STS1_r_min, linewidth=2.0, color='red', linestyle= '-' )
    #plt.fill_between(X, STS1_r_max, STS1_r_min, color='orange', alpha='0.5')
    
    plt.fill_between(X, STS1_r_max, TLu_r_max, color='orange', alpha='0.5')
    plt.fill_between(X, TLu_r_max, TLs_r_max, color='grey', alpha='0.5')
    plt.fill_between(X, TLs_r_max, TLs_r_min, color='brown', alpha='0.5')
     
    
    Y = numpy.zeros(len(e_mea))
    plt.plot(X, Y, linewidth=2.0, color='black', linestyle= '-.' )
    print nrs_induction_samples
    plt.xticks(nrs_induction_samples,rotation=90,fontsize=20)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll,fontsize=20)
#   plt.title(title)
    #plt.xlim(xmax,xmax)
    plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()



def plot_e3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=None,loc='upper right'):
    """
    Error plots.
    """
#     mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
# #     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
# #     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
#     mlab.stop() # in case the server is already running 
#     mlab.start()
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers,legend_ei,linestyle,linewidth,color] = experiment
        
        legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            if source_dataset is None:
                # baseline or source experiment
#                 target_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,target_dataset_name,training_data_fraction,nr_rotations)
                target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
            else:
                # reusability experiment
#                 target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                
                target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                 .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
                 
            outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
            #outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
            
            for ri in range(nr_reps):
                outputs = outputs_list[ri]
                test_error = outputs['ft_test_err']
                test_errors[ri,fi] = test_error
            
        e_mea = numpy.mean(test_errors,0)
        e_std = numpy.std (test_errors,0)
        #print e_mea
        #print e_std
        p = None
#         if ei == 0:
#             # store baseline stats
#             bl_e_mea = e_mea
#             bl_e_std = e_std
#         else:
#             # run t-test
#             nr_columns = numpy.shape(e_mea)[0]
#             p = -numpy.ones((nr_columns))
#             for ci in range(5):
#                 res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
#                 p[ci] = res['result']
        
        print_row_errors2(e_mea,e_std,p)
        #zzz
        
        nrs_induction_samples = numpy.array(training_data_fractions)*xmax
        X = nrs_induction_samples + plot_shift
        Y     = numpy.mean(test_errors,0)
        Y_err = numpy.std (test_errors,0)
        print_row_errors2(Y,Y_err,p)
        #print_row(Y,Y_err)
        #plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        plot_shift += xmax/200 # helps to distinguish overlapping error bars
        
    alfs=15
    plt.legend(legend,prop={'size':alfs})
    plt.xlabel('$N$')
    plt.ylabel('$\overline{\\varepsilon}$',rotation=0,fontsize=alfs)
    print nrs_induction_samples
    plt.xticks(nrs_induction_samples,rotation=90)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll)
#   plt.title(title)
    plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight')
    plt.show()
    plt.close()
    
    
#         nrs_design_samples = numpy.array(training_data_fractions)*xmax #n_ds_max/c
#         X = nrs_design_samples #+ plot_shift
#         Y     = e_mea
#         Y_err = e_std
#         plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#         #print test_errors
#         #plt.boxplot(test_errors[0],widths=0.1)
#         # this helps to avoid overlapping error bars
#         #plot_shift += nrs_design_samples[-1]/300
#         pass
#         
#     alfs=15
#     plt.legend(legend,prop={'size':alfs},loc=loc)
#     plt.xlabel('$n_{ds.ori}/c$')
#     plt.ylabel('$\overline{\\varepsilon}$',rotation=0,fontsize=alfs)
#     plt.xticks(nrs_design_samples,rotation=90)
#     yticks,locs = plt.yticks()
#     ll = ['%.2f' % a for a in yticks]
#     plt.yticks(yticks,ll)
#     #plt.title(title)
#     plt.xlim(n_ds_max/40,n_ds_max+n_ds_max/40)
#     if ylim is not None:
#         plt.ylim(ylim)
#     else:
#         ax = plt.gca()
#         ax.margins(None,0.03)
#         print 'ylim=',plt.ylim()
#     plt.tight_layout()
#     plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
#     plt.boxplot(test_errors, widths=0.01)
#     plt.show()
#     plt.close()
     
   # mlab.stop()
    
def plot_e2_tune(results_dir,experiments,Ls,n_ds_max,c,nr_reps,file_name,ylim=None,loc='upper right'):
    """
    Error plots.
    """
    mlab = Matlab(matlab='/usr/local/MATLAB/R2012a/bin/matlab')
#     mlab = Matlab(matlab='/opt/matlab/bin/matlab')
#     #mlab = Matlab(matlab='/usr/local/bin/matlab/matlab')
    mlab.stop() # in case the server is already running 
    mlab.start()
    
    nr_experiments = len(experiments)
    nr_nn = len(Ls)
    plot_shift = 0
    legend = list()
    bl_e_mea = None
    bl_e_std = None
    bl_es    = None
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset_name,source_dataset_name,source_reuse_mode,nr_rotations,legend_ei,linestyle,linewidth,color] = experiment
        
        legend.append(legend_ei)
        test_errors = -numpy.ones((nr_reps,nr_nn))
        
        for fi in range(nr_nn):
            
            L = Ls[fi]
            
            if source_dataset_name is None:
                # baseline or source experiment
                target_outputs_dir = '{}{}_{:1d}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,L,nr_rotations)
            else:
                # reusability experiment
                target_outputs_dir = '{}{}_reusing_{}_{}_{:1d}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,L,nr_rotations)
                    
            outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
            #outputs_list = load_outputs(target_outputs_dir)
            
            for ri in range(nr_reps):
                outputs = outputs_list[ri]
                test_error = outputs['ft_test_err']
                test_errors[ri,fi] = test_error
            
        e_mea = numpy.mean(test_errors,0)
        e_std = numpy.std (test_errors,0)
        p = None
        if ei == 0:
            # store baseline stats
            bl_e_mea = e_mea
            bl_e_std = e_std
        else:
            # run t-test
            nr_columns = numpy.shape(e_mea)[0]
            p = -numpy.ones((nr_columns))
            for ci in range(nr_nn):
                res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
                p[ci] = res['result']
        print_row_errors(e_mea,e_std,p)
        
        Ls = [1,2,3,4,5]
        Ls = [1,2,3]
        nrs_design_samples = numpy.array(Ls)#*n_ds_max/c
        X = nrs_design_samples + plot_shift
        Y     = e_mea
        Y_err = e_std
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        # this helps to avoid overlapping error bars
        plot_shift += nrs_design_samples[-1]/300
        pass
        
    alfs=15
    plt.legend(legend,prop={'size':alfs},loc=loc)
    plt.xlabel('Number of times the [100, 100] hidden layer neural network is multiplied')
    #plt.xlabel('Number of hidden layers')
    plt.ylabel('$\overline{\\varepsilon}$',rotation=0,fontsize=alfs)
    plt.xticks(nrs_design_samples,rotation=90)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll)
#   plt.title(title)
#   plt.xlim(n_ds_max/40,n_ds_max+n_ds_max/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()
     
    mlab.stop()

def plot_t(results_dir,experiments,training_data_fractions,n_ds_max,c,nr_reps,file_name,ylim=None):
    """
    Total training time plots.
    """
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    figure = plt.figure(figsize=( 8.,3.)) # default is 8.,6.
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset_name,source_dataset_name,source_reuse_mode,nr_rotations,legend_ei,linestyle,linewidth,color] = experiment
        legend.append(legend_ei)
        times_t_pt = numpy.zeros((nr_reps,nr_td_fractions))
        times_t_ft = numpy.zeros((nr_reps,nr_td_fractions))
        times_s_pt = numpy.zeros((nr_reps,nr_td_fractions))
        times_s_ft = numpy.zeros((nr_reps,nr_td_fractions))
        times_tota = numpy.zeros((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            if source_dataset_name is None:
                # baseline or source experiment
                target_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,training_data_fraction,nr_rotations)
                target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                cpu_model_name = target_outputs_list[0]['cpu_model_name']
                if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
                    # Passmark CPU Mark 5661
                    cpu_factor = 1.00
                elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
                    # Passmark CPU Mark 8561 = 1.51*5661
                    cpu_factor = 1.51
                elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
                    # Passmark CPU Mark 9652 = 1.51*5661
                    cpu_factor = 1.70
                for ri in range(nr_reps):
                    target_outputs = target_outputs_list[ri]
                    target_pt_trai_times = target_outputs['pt_trai_times']
                    target_pt_trai_time = cpu_factor*numpy.sum(target_pt_trai_times)
                    target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = target_pt_trai_time + target_ft_trai_time
                    times_t_pt[ri,fi] = target_pt_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_tota[ri,fi] = time
            else:
                # reusability experiment
                source_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,source_dataset_name,training_data_fraction,nr_rotations)
                source_outputs_list = load_outputs(source_outputs_dir,prefix='short_outputs_')
                target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                cpu_model_name = target_outputs_list[0]['cpu_model_name']
                if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
                    # Passmark CPU Mark 5661
                    cpu_factor = 1.00
                elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
                    # Passmark CPU Mark 8561 = 1.51*5661
                    cpu_factor = 1.51
                elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
                    # Passmark CPU Mark 9652 = 1.51*5661
                    cpu_factor = 1.70
                for ri in range(nr_reps):
                    source_outputs = source_outputs_list[ri]
                    source_pt_trai_times = source_outputs['pt_trai_times']
                    source_pt_trai_time = cpu_factor*numpy.sum(source_pt_trai_times)
                    source_ft_trai_time = cpu_factor*source_outputs['ft_trai_time'] 
                    target_outputs = target_outputs_list[ri]
                    target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = source_pt_trai_time + source_ft_trai_time + target_ft_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_s_pt[ri,fi] = source_pt_trai_time
                    times_s_ft[ri,fi] = source_ft_trai_time
                    times_tota[ri,fi] = time
            
        nrs_design_samples = numpy.array(training_data_fractions)*n_ds_max/c
        X = nrs_design_samples + plot_shift
        Y     = numpy.mean(times_tota,0)
        Y_err = numpy.std (times_tota,0)
        #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        # this helps to avoid overlapping error bars
        plot_shift += nrs_design_samples[-1]/300
        
        print 's_pt',
        print_row_times(numpy.mean(times_s_pt,0),numpy.std(times_s_pt,0))
        print 's_ft',
        print_row_times(numpy.mean(times_s_ft,0),numpy.std(times_s_ft,0))
        print 't_pt',
        print_row_times(numpy.mean(times_t_pt,0),numpy.std(times_t_pt,0))
        print 't_ft',
        print_row_times(numpy.mean(times_t_ft,0),numpy.std(times_t_ft,0))
        print 'tota',
        print_row_times(numpy.mean(times_tota,0),numpy.std(times_tota,0))
        
#     alfs=15
#     plt.xlabel('$n_{ds.ori}/c$')
#     plt.ylabel('$\overline{t}$',rotation=0,fontsize=alfs)
#     plt.xticks(nrs_design_samples,rotation=90)
# #   plt.title(title)
# #   plt.xlim(xmax/40,xmax+xmax/40)
#     if ylim is not None:
#         plt.ylim(ylim)
#     else:
#         ax = plt.gca()
#         ax.margins(None,0.03)
#         print 'ylim=',plt.ylim()
#     plt.tight_layout()
#     plt.legend(legend,loc='upper left',prop={'size':12})
# #   plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
# #   plt.show()
#     plt.close()
    
def plot_t2(results_dir,experiments,training_data_fractions,n_ds_max,c,nr_reps,file_name,ylim=None):
    """
    Total training time plots.
    """
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    figure = plt.figure(figsize=( 8.,3.)) # default is 8.,6.
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset_name,source_dataset_name,source_reuse_mode,nr_rotations,legend_ei,linestyle,linewidth,color] = experiment
        legend.append(legend_ei)
        times_t_pt = numpy.zeros((nr_reps,nr_td_fractions))
        times_t_ft = numpy.zeros((nr_reps,nr_td_fractions))
        times_s_pt = numpy.zeros((nr_reps,nr_td_fractions))
        times_s_ft = numpy.zeros((nr_reps,nr_td_fractions))
        times_tota = numpy.zeros((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            if source_dataset_name is None:
                # baseline or source experiment
                target_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,training_data_fraction,nr_rotations)
                target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                cpu_model_name = target_outputs_list[0]['cpu_model_name']
                if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
                    # Passmark CPU Mark 5661
                    cpu_factor = 1.00
                elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
                    # Passmark CPU Mark 8561 = 1.51*5661
                    cpu_factor = 1.51
                elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
                    # Passmark CPU Mark 9652 = 1.51*5661
                    cpu_factor = 1.70
                else:
                    cpu_factor = 1.70
                for ri in range(nr_reps):
                    target_outputs = target_outputs_list[ri]
                    target_pt_trai_times = target_outputs['pt_trai_times']
                    target_pt_trai_time = cpu_factor*numpy.sum(target_pt_trai_times)
                    target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = target_pt_trai_time + target_ft_trai_time
                    times_t_pt[ri,fi] = target_pt_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_tota[ri,fi] = time
            else:
                # reusability experiment
                source_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,source_dataset_name,training_data_fraction,nr_rotations)
                source_outputs_list = load_outputs(source_outputs_dir,prefix='short_outputs_')
                #source_outputs_list = load_outputs(source_outputs_dir,prefix='outputs_')
                target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                #target_outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                cpu_model_name = target_outputs_list[0]['cpu_model_name']
                if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
                    # Passmark CPU Mark 5661
                    cpu_factor = 1.00
                elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
                    # Passmark CPU Mark 8561 = 1.51*5661
                    cpu_factor = 1.51
                elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
                    # Passmark CPU Mark 9652 = 1.51*5661
                    cpu_factor = 1.70
                else:
                    cpu_factor = 1.70
                for ri in range(nr_reps):
                    source_outputs = source_outputs_list[ri]
                    source_pt_trai_times = source_outputs['pt_trai_times']
                    source_pt_trai_time = cpu_factor*numpy.sum(source_pt_trai_times)
                    source_ft_trai_time = cpu_factor*source_outputs['ft_trai_time'] 
                    target_outputs = target_outputs_list[ri]
                    target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = source_pt_trai_time + source_ft_trai_time + target_ft_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_s_pt[ri,fi] = source_pt_trai_time
                    times_s_ft[ri,fi] = source_ft_trai_time
                    times_tota[ri,fi] = time
            
        nrs_design_samples = numpy.array(training_data_fractions)*n_ds_max/c
        X = nrs_design_samples + plot_shift
        Y     = numpy.mean(times_tota,0)
        Y_err = numpy.std (times_tota,0)
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        # this helps to avoid overlapping error bars
        plot_shift += nrs_design_samples[-1]/300
        
        print 's_pt',
        print_row_times(numpy.mean(times_s_pt,0),numpy.std(times_s_pt,0))
        print 's_ft',
        print_row_times(numpy.mean(times_s_ft,0),numpy.std(times_s_ft,0))
        print 't_pt',
        print_row_times(numpy.mean(times_t_pt,0),numpy.std(times_t_pt,0))
        print 't_ft',
        print_row_times(numpy.mean(times_t_ft,0),numpy.std(times_t_ft,0))
        print 'tota',
        print_row_times(numpy.mean(times_tota,0),numpy.std(times_tota,0))
        
    alfs=15
    plt.xlabel('$n_{ds.ori}/c$')
    plt.ylabel('$\overline{t}$',rotation=0,fontsize=alfs)
    plt.xticks(nrs_design_samples,rotation=90)
#   plt.title(title)
#   plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.legend(legend,loc='upper left',prop={'size':12})
#   plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
#   plt.show()
    plt.close()
    
def plot_t3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=None):
    """
    Total training time plots.
    """
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    figure = plt.figure(figsize=( 8.,3.)) # default is 8.,6.
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers,legend_ei,linestyle,linewidth,color] = experiment
        legend.append(legend_ei)
        times_t_pt = numpy.zeros((nr_reps,nr_td_fractions))
        times_t_ft = numpy.zeros((nr_reps,nr_td_fractions))
        times_s_pt = numpy.zeros((nr_reps,nr_td_fractions))
        times_s_ft = numpy.zeros((nr_reps,nr_td_fractions))
        times_tota = numpy.zeros((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            if source_dataset is None:
                # baseline or source experiment
                target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,target_dataset, training_data_fraction)
#                 target_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,target_dataset_name,training_data_fraction,nr_rotations)
                #target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                target_outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
#                 cpu_model_name = None
#                 #cpu_model_name = target_outputs_list[0]['cpu_model_name']
#                 if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
#                     # Passmark CPU Mark 5661
#                     cpu_factor = 1.00
#                 elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
#                     # Passmark CPU Mark 8561 = 1.51*5661
#                     cpu_factor = 1.51
#                 elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
#                     # Passmark CPU Mark 9652 = 1.51*5661
#                     cpu_factor = 1.70
#                 else:
#                     cpu_factor = 1.70
                for ri in range(nr_reps):
                    target_outputs = target_outputs_list[ri]
                    target_pt_trai_times = target_outputs['pt_trai_times']
                    target_pt_trai_time = numpy.sum(target_pt_trai_times)
                    target_ft_trai_time = target_outputs['ft_trai_time'] 
#                     target_pt_trai_time = cpu_factor*numpy.sum(target_pt_trai_times)
#                     target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = target_pt_trai_time + target_ft_trai_time
                    times_t_pt[ri,fi] = target_pt_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_tota[ri,fi] = time
            else:
                # reusability experiment
                source_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,source_dataset,training_data_fraction)
#                 source_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,source_dataset_name,training_data_fraction,nr_rotations)
#                 source_outputs_list = load_outputs(source_outputs_dir,prefix='short_outputs_')
                source_outputs_list = load_outputs(source_outputs_dir,prefix='outputs_')
                target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
#                 target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
#                     .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
#                 target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                target_outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
#                 cpu_model_name = target_outputs_list[0]['cpu_model_name']
#                 if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
#                     # Passmark CPU Mark 5661
#                     cpu_factor = 1.00
#                 elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
#                     # Passmark CPU Mark 8561 = 1.51*5661
#                     cpu_factor = 1.51
#                 elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
#                     # Passmark CPU Mark 9652 = 1.51*5661
#                     cpu_factor = 1.70
#                 else:
#                     cpu_factor = 1.70
                for ri in range(nr_reps):
                    source_outputs = source_outputs_list[ri]
                    source_pt_trai_times = source_outputs['pt_trai_times']
                    source_pt_trai_time = numpy.sum(source_pt_trai_times)
                    source_ft_trai_time = source_outputs['ft_trai_time'] 
#                     source_pt_trai_time = cpu_factor*numpy.sum(source_pt_trai_times)
#                     source_ft_trai_time = cpu_factor*source_outputs['ft_trai_time'] 
                    target_outputs = target_outputs_list[ri]
                    target_ft_trai_time = target_outputs['ft_trai_time'] 
#                     target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = source_pt_trai_time + source_ft_trai_time + target_ft_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_s_pt[ri,fi] = source_pt_trai_time
                    times_s_ft[ri,fi] = source_ft_trai_time
                    times_tota[ri,fi] = time
                    
                #print_row_errors2(e_mea,e_std,p)
        #zzz
        
        nrs_induction_samples = numpy.array(training_data_fractions)*xmax
        X = nrs_induction_samples + plot_shift
        Y     = numpy.mean(times_tota,0)
        Y_err = numpy.std (times_tota,0)
        #print_row_errors2(Y,Y_err)
        #print_row(Y,Y_err)
        #plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        plot_shift += xmax/200 # helps to distinguish overlapping error bars
        
        print 's_pt',
        print_row_times(numpy.mean(times_s_pt,0),numpy.std(times_s_pt,0))
        print 's_ft',
        print_row_times(numpy.mean(times_s_ft,0),numpy.std(times_s_ft,0))
        print 't_pt',
        print_row_times(numpy.mean(times_t_pt,0),numpy.std(times_t_pt,0))
        print 't_ft',
        print_row_times(numpy.mean(times_t_ft,0),numpy.std(times_t_ft,0))
        print 'tota',
        print_row_times(numpy.mean(times_tota,0),numpy.std(times_tota,0))
        
    alfs=15
    #plt.legend(legend,prop={'size':alfs})
    plt.xlabel('$N$')
    plt.ylabel('$\overline{t}$',rotation=0,fontsize=alfs)
    print nrs_induction_samples
    plt.xticks(nrs_induction_samples,rotation=90)
#     yticks,locs = plt.yticks()
#     ll = ['%.2f' % a for a in yticks]
#     plt.yticks(yticks,ll)
#   plt.title(title)
#   plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()
            
#         nrs_design_samples = numpy.array(training_data_fractions)*n_ds_max/c
#         X = nrs_design_samples + plot_shift
#         Y     = numpy.mean(times_tota,0)
#         Y_err = numpy.std (times_tota,0)
#         plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#         # this helps to avoid overlapping error bars
#         plot_shift += nrs_design_samples[-1]/300
#         
#         print 's_pt',
#         print_row_times(numpy.mean(times_s_pt,0),numpy.std(times_s_pt,0))
#         print 's_ft',
#         print_row_times(numpy.mean(times_s_ft,0),numpy.std(times_s_ft,0))
#         print 't_pt',
#         print_row_times(numpy.mean(times_t_pt,0),numpy.std(times_t_pt,0))
#         print 't_ft',
#         print_row_times(numpy.mean(times_t_ft,0),numpy.std(times_t_ft,0))
#         print 'tota',
#         print_row_times(numpy.mean(times_tota,0),numpy.std(times_tota,0))
#         
#     alfs=15
#     plt.xlabel('$n_{ds.ori}/c$')
#     plt.ylabel('$\overline{t}$',rotation=0,fontsize=alfs)
#     plt.xticks(nrs_design_samples,rotation=90)
# #   plt.title(title)
# #   plt.xlim(xmax/40,xmax+xmax/40)
#     if ylim is not None:
#         plt.ylim(ylim)
#     else:
#         ax = plt.gca()
#         ax.margins(None,0.03)
#         print 'ylim=',plt.ylim()
#     plt.tight_layout()
#     plt.legend(legend,loc='upper left',prop={'size':12})
# #   plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
# #   plt.show()
#     plt.close()
    
def plot_t2_tune(results_dir,experiments,Ls,n_ds_max,c,nr_reps,file_name,ylim=None):
    """
    Total training time plots.
    """
    
    nr_experiments = len(experiments)
    nr_nn = len(Ls)
    plot_shift = 0
    legend = list()
    figure = plt.figure(figsize=( 8.,3.)) # default is 8.,6.
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset_name,source_dataset_name,source_reuse_mode,nr_rotations,legend_ei,linestyle,linewidth,color] = experiment
        legend.append(legend_ei)
        times_t_pt = numpy.zeros((nr_reps,nr_nn))
        times_t_ft = numpy.zeros((nr_reps,nr_nn))
        times_s_pt = numpy.zeros((nr_reps,nr_nn))
        times_s_ft = numpy.zeros((nr_reps,nr_nn))
        times_tota = numpy.zeros((nr_reps,nr_nn))
        
        for fi in range(nr_nn):
            
            nr_nn = Ls[fi]
            
            if source_dataset_name is None:
                # baseline or source experiment
                target_outputs_dir = '{}{}_{:1d}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,nr_nn,nr_rotations)
                target_outputs_list = load_outputs(target_outputs_dir)
                #target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                cpu_model_name = target_outputs_list[0]['cpu_model_name']
                if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
                    # Passmark CPU Mark 5661
                    cpu_factor = 1.00
                elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
                    # Passmark CPU Mark 8561 = 1.51*5661
                    cpu_factor = 1.51
                elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
                    # Passmark CPU Mark 9652 = 1.51*5661
                    cpu_factor = 1.70
                for ri in range(nr_reps):
                    target_outputs = target_outputs_list[ri]
                    target_pt_trai_times = target_outputs['pt_trai_times']
                    target_pt_trai_time = cpu_factor*numpy.sum(target_pt_trai_times)
                    target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = target_pt_trai_time + target_ft_trai_time
                    times_t_pt[ri,fi] = target_pt_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_tota[ri,fi] = time
            else:
                # reusability experiment
                source_outputs_dir = '{}{}_{:1d}_x{:02d}/'\
                    .format(results_dir,source_dataset_name,training_data_fraction,nr_rotations)
                source_outputs_list = load_outputs(source_outputs_dir,prefix='short_outputs_')
                target_outputs_dir = '{}{}_reusing_{}_{}_{:1d}_x{:02d}/'\
                    .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
                cpu_model_name = target_outputs_list[0]['cpu_model_name']
                if   cpu_model_name == 'Intel(R) Core(TM) i7 CPU         950  @ 3.07GHz':
                    # Passmark CPU Mark 5661
                    cpu_factor = 1.00
                elif cpu_model_name == 'Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz':
                    # Passmark CPU Mark 8561 = 1.51*5661
                    cpu_factor = 1.51
                elif cpu_model_name == 'Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz':
                    # Passmark CPU Mark 9652 = 1.51*5661
                    cpu_factor = 1.70
                for ri in range(nr_reps):
                    source_outputs = source_outputs_list[ri]
                    source_pt_trai_times = source_outputs['pt_trai_times']
                    source_pt_trai_time = cpu_factor*numpy.sum(source_pt_trai_times)
                    source_ft_trai_time = cpu_factor*source_outputs['ft_trai_time'] 
                    target_outputs = target_outputs_list[ri]
                    target_ft_trai_time = cpu_factor*target_outputs['ft_trai_time'] 
                    time = source_pt_trai_time + source_ft_trai_time + target_ft_trai_time
                    times_t_ft[ri,fi] = target_ft_trai_time
                    times_s_pt[ri,fi] = source_pt_trai_time
                    times_s_ft[ri,fi] = source_ft_trai_time
                    times_tota[ri,fi] = time
        
        Ls = [1,2,3,4,5]  
        Ls = [1,2,3]
        nrs_design_samples = numpy.array(Ls)#*n_ds_max/c
        X = nrs_design_samples + plot_shift
        Y     = numpy.mean(times_tota,0)
        Y_err = numpy.std (times_tota,0)
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        # this helps to avoid overlapping error bars
        plot_shift += nrs_design_samples[-1]/300
        
        print 's_pt',
        print_row_times(numpy.mean(times_s_pt,0),numpy.std(times_s_pt,0))
        print 's_ft',
        print_row_times(numpy.mean(times_s_ft,0),numpy.std(times_s_ft,0))
        print 't_pt',
        print_row_times(numpy.mean(times_t_pt,0),numpy.std(times_t_pt,0))
        print 't_ft',
        print_row_times(numpy.mean(times_t_ft,0),numpy.std(times_t_ft,0))
        print 'tota',
        print_row_times(numpy.mean(times_tota,0),numpy.std(times_tota,0))
        
    alfs=15
    plt.xlabel('Number of times the [100, 100] hidden layer neural network is multiplied')
    #plt.xlabel('Number of hidden layers')
    plt.ylabel('$\overline{t}$',rotation=0,fontsize=alfs)
    plt.xticks(nrs_design_samples,rotation=90)
#   plt.title(title)
#   plt.xlim(xmax/40,xmax+xmax/40)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.tight_layout()
    plt.legend(legend,loc='upper left',prop={'size':12})
#   plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
#   plt.show()
    plt.close()
    
def plot_De(results_dir,experiments,training_data_fractions,n_ds_max,c,nr_reps,file_name,ylim=None):
    """
    Plots of relative improvement in error.
    """
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    plot_shift = 0
    legend = list()
    xticks = numpy.array([])
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [target_dataset_name,source_dataset_name,source_reuse_mode,nr_rotations,legend_ei,linestyle,linewidth,color] = experiment
        legend.append(legend_ei)
        baseli_test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        target_test_errors = -numpy.ones((nr_reps,nr_td_fractions))
        
        for fi in range(nr_td_fractions):
            
            training_data_fraction = training_data_fractions[fi]
            
            baseli_outputs_dir = '{}{}_{:4.2f}_x{:02d}/'\
                .format(results_dir,target_dataset_name,training_data_fraction,1)
            target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x{:02d}/'\
                .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction,nr_rotations)
                    
            baseli_outputs_list = load_outputs(baseli_outputs_dir,prefix='short_outputs_')
            target_outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
            
            for ri in range(nr_reps):
                baseli_outputs = baseli_outputs_list[ri]
                baseli_test_error = baseli_outputs['ft_test_err']
                baseli_test_errors[ri,fi] = baseli_test_error
                target_outputs = target_outputs_list[ri]
                target_test_error = target_outputs['ft_test_err']
                target_test_errors[ri,fi] = target_test_error
            
        rel_err_imps = (baseli_test_errors-target_test_errors)/baseli_test_errors
            
        nrs_design_samples = numpy.array(training_data_fractions)*n_ds_max/c
        X = nrs_design_samples + plot_shift
        xticks = numpy.append(xticks,nrs_design_samples,axis=0)
        Y     = numpy.mean(rel_err_imps,0)
        Y_err = numpy.std (rel_err_imps,0)
        print_row_errors(Y,Y_err)
        plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
        # this helps to avoid overlapping error bars
        plot_shift += nrs_design_samples[-1]/300
        
    alfs=15
    pl = plt.legend(legend,prop={'size':alfs})
#   pl.get_frame().set_alpha(0.5)
    plt.xlabel('$n_{ds.ori}/c$')
    plt.ylabel('$\overline{\Delta\\varepsilon_{r}}$',rotation=90,fontsize=alfs)
    xticks = numpy.unique(xticks)
    plt.xticks(xticks,rotation=90)
    yticks,locs = plt.yticks()
    ll = ['%.2f' % a for a in yticks]
    plt.yticks(yticks,ll)
#   plt.title(title)
#   plt.xlim(75,3075)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        ax = plt.gca()
        ax.margins(None,0.03)
        print 'ylim=',plt.ylim()
    plt.axhline(linewidth=2,color='k')
    plt.tight_layout()
    plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()
    
def plot_class_es(results_dir,experiments,c,nr_reps,figsize,xticklabels,file_name):
    """
    Plot intra-class errors.
    """
    
    nr_experiments = len(experiments)
    training_data_fraction = 0.3
    fig,ax = plt.subplots(figsize=figsize) # default is 8.,6.
    ind = numpy.arange(c)
    width = 0.25
    colors =  ('white','white','black')
    hatches = (''     ,'//'   ,''     )
    legend = list()
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        print 'experiment=',experiment
        color = colors [ei]
        hatch = hatches[ei]
        
        [target_dataset_name,source_dataset_name,source_reuse_mode,legend_ei] = experiment
        legend.append(legend_ei)
    
        if source_dataset_name is None:
            # baseline or source experiment
            target_outputs_dir = '{}{}_{:4.2f}_x01/'\
                .format(results_dir,target_dataset_name,training_data_fraction)
        else:
            # reusability experiment
            target_outputs_dir = '{}{}_reusing_{}_{}_{:4.2f}_x01/'\
                .format(results_dir,target_dataset_name,source_dataset_name,source_reuse_mode,training_data_fraction)
                
        outputs_list = load_outputs(target_outputs_dir,prefix='short_outputs_')
        
        class_es = -numpy.ones((nr_reps,c))
    
        for ri in range(nr_reps):
            outputs = outputs_list[ri]
            cm = outputs['ft_confusion_matrix']
            for ci in range(c):
                nr_hits    = cm[ci,ci]
                nr_samples = numpy.sum(cm[ci,:])
                class_e = 1-(nr_hits/nr_samples)
                class_es[ri,ci] = class_e
        
        class_es_mean = numpy.mean(class_es,0)
        class_es_std  = numpy.std (class_es,0)
    
#       for ci in range(c):
#           print '{:4.1f}\t'.format(class_es_mean[ci]*100),
#       print '\n'

        rects = ax.bar(ind+ei*width,class_es_mean,width,color=color,hatch=hatch,yerr=class_es_std)
        
    ax.set_xticks(ind+width)
    ax.set_xticklabels(xticklabels)
    plt.xlim((-0.5,c+4.5))
    plt.ylim((0,0.9))
    plt.xlabel('class label')
    plt.ylabel('intra-class error',rotation=90)
    plt.legend(legend)
    plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
    plt.show()


def plot_train_info(results_dir,experiments,nr_reps,file_name,ylim=None):
    """
    Plot traing info such as Pretraining training cost.
    """
    from sklearn.metrics import confusion_matrix
    nr_experiments = len(experiments)
    training_data_fraction = 1.0
#     fig,ax = plt.subplots(figsize=figsize) # default is 8.,6.
#     
#     
#     linestyles = ['--','-',':','--',':','--','-',':','--',':','--','-',':','--',':','--','-',':','--',':']
#     linewidths = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#     colors=['black','red','green','blue','cyan','red','green','blue','cyan','red','green','blue','cyan','red','green','red','green','blue','cyan']
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        #legend.append(legend_ei)
        
                             
        if source_dataset is None:
        # baseline or source experiment
            target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
        
        outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
        #print outputs_list
        for ri in range(nr_reps):
            outputs = outputs_list[ri]
            pt_trai_costs_vs_stage = outputs['pt_trai_costs_vs_stage']
            pt_ep = numpy.array(pt_trai_costs_vs_stage)
            y_test_pred = outputs['y_test_pred']
            y_test = outputs['y_test']
            val_epochs_rep = outputs['val_epochs_rep']
            val_epochs_errs_rep = outputs['val_epochs_errs_rep']
            test_epochs_rep = outputs['test_epochs_rep']
            test_epochs_errs_rep = outputs['test_epochs_errs_rep'] 
            
    f1 = open(results_dir+'bbbc+feat3_1.00/bbbc+feat3_cell_metadata.pkl', 'rb')
    cell_metadata = pickle.load(f1)
    cell_metadata = cell_metadata.astype(int)
    f1.close()
    f2 = open(results_dir+'bbbc+feat3_1.00/bbbc+feat3_tes_met.pkl', 'rb')
    tes_met = pickle.load(f2)
    tes_met = tes_met.astype(int)
    f2.close()
    
    print 'len y_test_pred', len(y_test_pred)
    print 'type y_test_pred', type(y_test_pred)
    
    print 'processing the individual cells ... '
    #identify the full image and cell labels
    Image_nrs = set(tes_met[0:,1])
    Total_tables = 0
    for Image_nr in Image_nrs:
        #print 'Image_nr', Image_nr
        Tabel_nrs = set(tes_met[numpy.where(tes_met[0:,1] == Image_nr)][0:,0])
        Total_tables = Total_tables + len(Tabel_nrs)
    
    predictions = numpy.zeros(([Total_tables,4]), dtype=numpy.int)     
    
    count = 0
    for Image_nr in Image_nrs:
        #print 'Image_nr', Image_nr
        im = tes_met[numpy.where(tes_met[0:,1] == Image_nr)]
        #print im
        
        Tabel_nrs = set(im[0:,0])
        for Tabel_nr in Tabel_nrs:
            #print 'Tabel_nr', Tabel_nr
            #print 'Image_nr', Image_nr
            cells = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr)))] #[0,2]
            #print 'cells', set(cells[0:,2]) #cells
            cell_nrs = set(cells[0:,2]) 
            
            cell_value = numpy.zeros(([len(cell_nrs),2]), dtype=numpy.int)  
            #cell_value = numpy.zeros([len(cell_nrs),1])
            for idx, cell_nr in enumerate(cell_nrs):
                #print 'cell', cell_nr
                 
                cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,3]
                cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
                #print y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
                #cell_value[idx] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
    #        
            cell_value_true = cell_value[0:,0]
            cell_value_pred = cell_value[0:,1]
    
    #         print 'Prediction of cell in a Image', Image_nr, Tabel_nrs
            #print 'Prediction of cells in a Image', stats.itemfreq(cell_value_pred) 
    #         print 'Highest prediction in a Image', stats.itemfreq(cell_value_pred)[0,0] 
            freq = stats.itemfreq(cell_value_pred)
            #freq[numpy.argmax(freq[0:,1])][0].astype(int)
            
            #print Predictions
            #print Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
            predictions[count] = Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
    
            
            count += 1
    #         print 'count', count
    #     if count > 10:
    #         break
    
    
    
    true_val = predictions[0:,2]
    pred_val = predictions[0:,3]
    from sklearn.metrics import accuracy_score
    avg_acc = accuracy_score(true_val, pred_val) 
    print 'Average the prediction of individual cells'
    print 'Accuracy', avg_acc
    print 'Test error', 1-avg_acc   
    
    print
    print
    print 'Prediction of individual cells'
    aa = y_test_pred.flatten()
    bb = y_test.flatten()
    ind_acc = accuracy_score(bb, aa)
    print 'Accuracy', ind_acc
    print 'Test error', 1-ind_acc   
    

       
    
    print 'len y_test_pred', len(y_test_pred)
    print 'type y_test_pred', type(y_test_pred)
    #print 'y_test_pred', y_test_pred[0:,0:100]
    print 
    
    print "Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4)
    avg_cm = confusion_matrix(true_val, pred_val)    
    print(avg_cm)
    plt.matshow(avg_cm)
    #plt.title('Confusion matrix with Averaging predictions')
    plt.title("Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(results_dir+'avg_cm.png',bbox_inches='tight',dpi=200)
    
    
    print "Confusion matrix (Ind cells): Acc = (%.2f)" % round(ind_acc*100,4)
    ind_cm = confusion_matrix(bb,aa)    
    print(ind_cm)
    plt.matshow(ind_cm)
    plt.title("Confusion matrix (Ind cells): Acc = (%.2f)" % round(ind_acc*100,4))
    #plt.title('Confusion matrix with Individual predictions')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(results_dir+'ind_cm.png',bbox_inches='tight',dpi=200)
 
    file_name = 'pt_3layer.png'
    f, axarr = plt.subplots(len(pt_ep), sharex=True) 
    for k in range(len(pt_ep)):
        axarr[k].plot(range(len(pt_ep[k])), pt_ep[k], '--', linewidth=2, label='layer '+str(k))
        axarr[k].set_ylabel('reconstruction cost',rotation=90)
        axarr[k].legend()
       
    axarr[0].set_title('Pre-training reconstruction cost')
    plt.xlabel('Pt epochs')
    plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
    plt.show()
    
def plot_train_batch_info(results_dir,experiments,nr_reps,file_name,label_type, ylim=None):
    """
    Plot traing info such as Pretraining training cost.
    """
    from sklearn.metrics import confusion_matrix
    nr_experiments = len(experiments)
    training_data_fraction = 1.0
#     fig,ax = plt.subplots(figsize=figsize) # default is 8.,6.
#     
#     
#     linestyles = ['--','-',':','--',':','--','-',':','--',':','--','-',':','--',':','--','-',':','--',':']
#     linewidths = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#     colors=['black','red','green','blue','cyan','red','green','blue','cyan','red','green','blue','cyan','red','green','red','green','blue','cyan']
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        #legend.append(legend_ei)
        
                             
        if source_dataset is None:
        # baseline or source experiment
            target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
        elif source_dataset is not None:
         # reusability experiment
#                 source_outputs_dir = '{0}{1}_{2:4.2f}/'\
#                     .format(results_dir,source_dataset,training_data_fraction)
#                 source_outputs_list = load_outputs(source_outputs_dir,prefix='outputs_')
                target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
                #target_outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
        print 'target_outputs_dir', target_outputs_dir
        
        outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
        #print outputs_list
        for ri in range(nr_reps):
            outputs = outputs_list[ri]
            pt_trai_costs_vs_stage = outputs['pt_trai_costs_vs_stage']
            pt_ep = numpy.array(pt_trai_costs_vs_stage)
            y_test_pred = outputs['y_test_pred']
            y_test = outputs['y_test']
            val_epochs_rep = outputs['val_epochs_rep']
            val_epochs_errs_rep = outputs['val_epochs_errs_rep']
            test_epochs_rep = outputs['test_epochs_rep']
            test_epochs_errs_rep = outputs['test_epochs_errs_rep'] 
    
    PC = 'hpc'
    if PC == 'pc':
        dir = results_dir+'meta_data/'
    elif PC == 'hpc':
        dir = '/home/aditya/store/Datasets/pickled/bbbc+feat4/'
        
    if label_type == 'moa':
        f1 = open(dir+'bbbc+moa_cell_metadata.pkl', 'rb')
        f2 = open(dir+'bbbc+moa_tes_met.pkl', 'rb')
    elif label_type == 'compound':
        f1 = open(dir+'bbbc+comp_cell_metadata.pkl', 'rb')
        f2 = open(dir+'bbbc+comp_tes_met.pkl', 'rb')
        
        
        
#     if label_type == 'moa':
#         f1 = open(results_dir+'meta_data/bbbc+moa_cell_metadata.pkl', 'rb')
#         f2 = open(results_dir+'meta_data/bbbc+moa_tes_met.pkl', 'rb')
#     elif label_type == 'compound':
#         f1 = open(results_dir+'meta_data/bbbc+comp_cell_metadata.pkl', 'rb')
#         f2 = open(results_dir+'meta_data/bbbc+comp_tes_met.pkl', 'rb')
        
    #f1 = open(results_dir+'meta_data/bbbc+comp_cell_metadata.pkl', 'rb')
    cell_metadata = pickle.load(f1)
    cell_metadata = cell_metadata.astype(int)
    f1.close()
    #f2 = open(results_dir+'meta_data/bbbc+comp_tes_met.pkl', 'rb')
    tes_met = pickle.load(f2)
    tes_met = tes_met.astype(int)
    f2.close()
    
    y_test_pred = y_test_pred.flatten()
    
    print 'len y_test_pred', len(y_test_pred)# len(y_test_pred)
    print 'type y_test_pred', type(y_test_pred) #type(y_test_pred)
    
    print 'processing the individual cells ... '
    #identify the full image and cell labels
    
    # match the batch processing prediction to meta data
    tes_met = tes_met[0:len(y_test_pred)]

#    print 
    Image_nrs = set(tes_met[0:,1])
    Total_tables = 0
    for Image_nr in Image_nrs:
        #print 'Image_nr', Image_nr
        Tabel_nrs = set(tes_met[numpy.where(tes_met[0:,1] == Image_nr)][0:,0])
        Total_tables = Total_tables + len(Tabel_nrs)
    
    predictions = numpy.zeros(([Total_tables,4]), dtype=numpy.int)     
        
    count = 0
    for Image_nr in Image_nrs:
        #print 'Image_nr', Image_nr
        im = tes_met[numpy.where(tes_met[0:,1] == Image_nr)]
        #print im
        
        Tabel_nrs = set(im[0:,0])
        for Tabel_nr in Tabel_nrs:
            #print 'Tabel_nr', Tabel_nr
            #print 'Image_nr', Image_nr
            cells = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr)))] #[0,2]
            #print 'cells', set(cells[0:,2]) #cells
            cell_nrs = set(cells[0:,2]) 
            
            cell_value = numpy.zeros(([len(cell_nrs),2]), dtype=numpy.int)  
            #cell_value = numpy.zeros([len(cell_nrs),1])
            for idx, cell_nr in enumerate(cell_nrs):
                #print 'cell', cell_nr
                #print idx 
                #print
                
                if label_type == 'moa':
                    cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,3]
                    cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0] #,0]
                elif label_type == 'compound':
                    cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,4]
                    cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0] #,0]
                #print y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
                #cell_value[idx] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
    #        
            cell_value_true = cell_value[0:,0]
            cell_value_pred = cell_value[0:,1]
    
    #         print 'Prediction of cell in a Image', Image_nr, Tabel_nrs
            #print 'Prediction of cells in a Image', stats.itemfreq(cell_value_pred) 
    #         print 'Highest prediction in a Image', stats.itemfreq(cell_value_pred)[0,0] 
            freq = stats.itemfreq(cell_value_pred)
            #freq[numpy.argmax(freq[0:,1])][0].astype(int)
            
            #print Predictions
            #print Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
            predictions[count] = Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
    
            
            count += 1
#             print 'count', count
#             if count > 1:
#                 break
    
    
    
    true_val = predictions[0:,2]
    pred_val = predictions[0:,3]
    from sklearn.metrics import accuracy_score
    avg_acc = accuracy_score(true_val, pred_val) 
    print 'Average the prediction of individual cells'
    print 'Accuracy', avg_acc
    print 'Test error', 1-avg_acc   
    
    print
    print
    print 'Prediction of individual cells'
    aa = y_test_pred.flatten()
    bb = y_test.flatten()
    ind_acc = accuracy_score(bb, aa)
    print 'Accuracy', ind_acc
    print 'Test error', 1-ind_acc   
    

       
    
    print 'len y_test_pred', len(y_test_pred)
    print 'type y_test_pred', type(y_test_pred)
    #print 'y_test_pred', y_test_pred[0:,0:100]
    print 
    
    
    #import os
    #output_folder = '/media/aditya/nimi1/repos/Clean_code/results/BL_bbbc_feat4/' + target_dataset +'_1.00/' 
    #output_folder = results_dir+target_dataset
#     if not os.path.isdir(output_folder):
#         os.makedirs(output_folder)
#     os.chdir(output_folder) 
    
    print "Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4)
    avg_cm = confusion_matrix(true_val, pred_val)    
    print(avg_cm)
    plt.matshow(avg_cm)
    #plt.title('Confusion matrix with Averaging predictions')
    plt.title("Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(target_outputs_dir+'avg_cm.png',bbox_inches='tight',dpi=200)
    #plt.savefig(results_dir+'avg_cm.png',bbox_inches='tight',dpi=200)
    
    
    print "Confusion matrix (Ind cells): Acc = (%.2f)" % round(ind_acc*100,4)
    ind_cm = confusion_matrix(bb,aa)    
    print(ind_cm)
    plt.matshow(ind_cm)
    plt.title("Confusion matrix (Ind cells): Acc = (%.2f)" % round(ind_acc*100,4))
    #plt.title('Confusion matrix with Individual predictions')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(target_outputs_dir+'ind_cm.png',bbox_inches='tight',dpi=200)
    #plt.savefig(results_dir+'ind_cm.png',bbox_inches='tight',dpi=200)
    
    
    if approach == 'BL':
        file_name = 'pt_'+str(len(pt_ep))+'layer.png'
        f, axarr = plt.subplots(len(pt_ep), sharex=True) 
        for k in range(len(pt_ep)):
            axarr[k].plot(range(len(pt_ep[k])), pt_ep[k], '--', linewidth=2, label='layer '+str(k))
            axarr[k].set_ylabel('reconstruction cost',rotation=90)
            axarr[k].legend()
        
        axarr[0].set_title('Pre-training reconstruction cost')
        plt.xlabel('Pt epochs')
        plt.savefig(target_outputs_dir+file_name,bbox_inches='tight',dpi=200)
        #plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
    plt.show()
    


def plot_train_batch_info2(results_dir,experiments,nr_reps,file_name,label_type, ylim=None):
    """
    Plot traing info such as Pretraining training cost.
    """
    from sklearn.metrics import confusion_matrix
    nr_experiments = len(experiments)
    training_data_fraction = 1.0
#     fig,ax = plt.subplots(figsize=figsize) # default is 8.,6.
#     
#     
#     linestyles = ['--','-',':','--',':','--','-',':','--',':','--','-',':','--',':','--','-',':','--',':']
#     linewidths = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#     colors=['black','red','green','blue','cyan','red','green','blue','cyan','red','green','blue','cyan','red','green','red','green','blue','cyan']
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        #legend.append(legend_ei)
        
                             
        if source_dataset is None:
        # baseline or source experiment
            target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
        elif source_dataset is not None:
         # reusability experiment
#                 source_outputs_dir = '{0}{1}_{2:4.2f}/'\
#                     .format(results_dir,source_dataset,training_data_fraction)
#                 source_outputs_list = load_outputs(source_outputs_dir,prefix='outputs_')
                target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
                #target_outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
        print 'target_outputs_dir', target_outputs_dir
        
        outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
        
        PC = 'hpc'
        if PC == 'pc':
            dir = results_dir+'meta_data/'
        elif PC == 'hpc':
            dir = '/home/aditya/store/Datasets/pickled/bbbc+feat4/'
            
        if label_type == 'moa':
            f1 = open(dir+'bbbc+moa_cell_metadata.pkl', 'rb')
            f2 = open(dir+'bbbc+moa_tes_met.pkl', 'rb')
        elif label_type == 'compound':
            f1 = open(dir+'bbbc+comp_cell_metadata.pkl', 'rb')
            f2 = open(dir+'bbbc+comp_tes_met.pkl', 'rb')
        
        
        cell_metadata = pickle.load(f1)
        cell_metadata = cell_metadata.astype(int)
        f1.close()
       
        tes_met = pickle.load(f2)
        tes_met = tes_met.astype(int)
        f2.close()
        
        
        
        avg_accs = []
        avg_cms = []
        ind_accs = []
        ind_cms = []
        pt_times = []
        ft_times = []
        #print outputs_list
        for ri in range(nr_reps):
            outputs = outputs_list[ri]
            pt_trai_costs_vs_stage = outputs['pt_trai_costs_vs_stage']
            pt_ep = numpy.array(pt_trai_costs_vs_stage)
            y_test_pred = outputs['y_test_pred']
            y_test = outputs['y_test']
            val_epochs_rep = outputs['val_epochs_rep']
            val_epochs_errs_rep = outputs['val_epochs_errs_rep']
            test_epochs_rep = outputs['test_epochs_rep']
            test_epochs_errs_rep = outputs['test_epochs_errs_rep'] 
            
            avg_acc = outputs['avg_acc']
            avg_cm = outputs['avg_cm'] 
            ind_acc = outputs['ind_acc']
            ind_cm = outputs['ind_cm']
            
            avg_accs.append(avg_acc)
            avg_cms.append(avg_cm)
            ind_accs.append(ind_acc)
            ind_cms.append(ind_cm) 
            
            # pre-training time for each hidden layer
            pt_time = outputs['pt_trai_times']
            # fine-tuning time
            ft_time = outputs['ft_trai_time']
            pt_times.append(pt_time)
            ft_times.append(ft_time)
            
            #print avg_cm
        
        print 'avg cms'
        m = numpy.matrix.round(numpy.mean(avg_cms,axis=0)).astype(int)
        print m
        fig = plt.figure()
        plt.imshow(m, interpolation='nearest', cmap=plt.cm.Blues)
        #plt.imshow(m, interpolation='nearest',cmap='Reds')
        #plt.imshow(m, interpolation='nearest',cmap='Greys')
        
        ## cmaps
        ##Spectral, summer, coolwarm, Wistia_r, pink_r, Set1, Set2, Set3, brg_r,
        ## Dark2, prism, PuOr_r, afmhot_r, terrain_r, PuBuGn_r, RdPu, gist_ncar_r,
        ## gist_yarg_r, Dark2_r, YlGnBu, RdYlBu, hot_r, gist_rainbow_r, gist_stern,
        ## PuBu_r, cool_r, cool, gray, copper_r, Greens_r, GnBu, gist_ncar, spring_r,
        ## gist_rainbow, gist_heat_r, Wistia, OrRd_r, CMRmap, bone, gist_stern_r,
        ## RdYlGn, Pastel2_r, spring, terrain, YlOrRd_r, Set2_r, winter_r, PuBu, RdGy_r,
        ## spectral, rainbow, flag_r, jet_r, RdPu_r, gist_yarg, BuGn, Paired_r, hsv_r, bwr,
        ## cubehelix, Greens, PRGn, gist_heat, spectral_r, Paired, hsv, Oranges_r, prism_r,
        ## Pastel2, Pastel1_r, Pastel1, gray_r, jet, Spectral_r, gnuplot2_r, gist_earth, YlGnBu_r,
        ## copper, gist_earth_r, Set3_r, OrRd, gnuplot_r, ocean_r, brg, gnuplot2, PuRd_r, bone_r,
        ## BuPu, Oranges, RdYlGn_r, PiYG, CMRmap_r, YlGn, binary_r, gist_gray_r, Accent, BuPu_r, 
        ## gist_gray, flag, bwr_r, RdBu_r, BrBG, Reds, Set1_r, summer_r, GnBu_r, BrBG_r, Reds_r, RdGy,
        ## PuRd, Accent_r, Blues, autumn_r, autumn, cubehelix_r, nipy_spectral_r, ocean, PRGn_r, Greys_r,
        ## pink, binary, winter, gnuplot, RdYlBu_r, hot, YlOrBr, coolwarm_r, rainbow_r, Purples_r, PiYG_r,
        ## YlGn_r, Blues_r, YlOrBr_r, seismic, Purples, seismic_r, RdBu, Greys, BuGn_r, YlOrRd, PuOr, PuBuGn,
        ## nipy_spectral, afmhot

       
        #deviations = numpy.matrix.round(numpy.std(avg_cms,axis=0))
        #print deviations
        #plt.imshow(deviations, interpolation='nearest')
        #plt.matshow(deviations)
        
        #plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
        for x_val, y_val in zip(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1])):
            for idy, val in enumerate(m[x_val]):
                if val != 0:
                    plt.text(idy, x_val, m[x_val,idy], fontsize = '12',  va='center', ha='center')
            #plt.text(x_val, y_val, m[x_val,y_val], fontsize = 'smaller',  va='center', ha='center')
        
        #print "Confusion matrix (Collective): Acc = (%.2f)" % numpy.mean(avg_accs)*100
        #plt.matshow(m)
        #plt.text(numpy.arange(numpy.shape(m)[0]), numpy.arange(numpy.shape(m)[1]), m, va='center', ha='center')
        plt.title("Confusion matrix (Collective): Acc = (%.2f)" % (numpy.mean(avg_accs)*100))
        #plt.title("Confusion matrix (Collective): Acc = (%.2f)" % round(numpy.mean(avg_accs)*100,4))
        #plt.title('Confusion matrix with Averaging predictions of Collective')
        #plt.title("Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4))
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(target_outputs_dir+'avg_coll_cm.png',bbox_inches='tight',dpi=200)
        #plt.savefig(results_dir+'avg_cm.png',bbox_inches='tight',dpi=200)
        plt.close(fig)
        print "test ......"
        print "Confusion matrix (Collective): Acc = (%.2f)" % (numpy.mean(avg_accs)*100)
        
        print 'Collective the prediction of individual cells'
        print 'Coll Accuracy =' + str(avg_accs)
        print 'Coll mean Accuracy = %.2f(%.2f)' % (numpy.mean(avg_accs)*100, numpy.std(avg_accs)*100)  
        print 'Prediction of individual cells'
        print 'Ind Accuracy =' + str(ind_accs)
        print 'Ind mean Accuracy = %.2f(%.2f)' % (numpy.mean(ind_accs)*100, numpy.std(ind_accs)*100) 
        
        print 'Time take for train pt layers in sec = ', pt_times
        print 'Time to train pt layers = mean %.2f(%.2f)s' % (numpy.mean(pt_times), numpy.std(pt_times))
        print 'Time take for train ft layers in sec = ', ft_times
        print 'Time to train ft layers = mean %.2f(%.2f)s' % (numpy.mean(ft_times), numpy.std(ft_times))
        
        
        
        
        
#         
#             y_test_pred = y_test_pred.flatten()
#     
#             print 'len y_test_pred', len(y_test_pred)
#             print 'type y_test_pred', type(y_test_pred)
#     
#             print 'processing the individual cells ... '
#             #identify the full image and cell labels
#             
#             # match the batch processing prediction to meta data
#             tes_met = tes_met[0:len(y_test_pred)]
#         
#         #    print 
#             Image_nrs = set(tes_met[0:,1])
#             Total_tables = 0
#             for Image_nr in Image_nrs:
#                 #print 'Image_nr', Image_nr
#                 Tabel_nrs = set(tes_met[numpy.where(tes_met[0:,1] == Image_nr)][0:,0])
#                 Total_tables = Total_tables + len(Tabel_nrs)
#             
#             predictions = numpy.zeros(([Total_tables,4]), dtype=numpy.int)     
#                 
#             count = 0
#             for Image_nr in Image_nrs:
#                 #print 'Image_nr', Image_nr
#                 im = tes_met[numpy.where(tes_met[0:,1] == Image_nr)]
#                 #print im
#                 
#                 Tabel_nrs = set(im[0:,0])
#                 for Tabel_nr in Tabel_nrs:
#                     #print 'Tabel_nr', Tabel_nr
#                     #print 'Image_nr', Image_nr
#                     cells = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr)))] #[0,2]
#                     #print 'cells', set(cells[0:,2]) #cells
#                     cell_nrs = set(cells[0:,2]) 
#                     
#                     cell_value = numpy.zeros(([len(cell_nrs),2]), dtype=numpy.int)  
#                     #cell_value = numpy.zeros([len(cell_nrs),1])
#                     for idx, cell_nr in enumerate(cell_nrs):
#                         #print 'cell', cell_nr
#                         #print idx 
#                         #print
#                         
#                         if label_type == 'moa':
#                             cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,3]
#                             cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0] #,0]
#                         elif label_type == 'compound':
#                             cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,4]
#                             cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0] #,0]
#                         #print y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
#                         #cell_value[idx] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
#             #        
#                     cell_value_true = cell_value[0:,0]
#                     cell_value_pred = cell_value[0:,1]
#             
#             #         print 'Prediction of cell in a Image', Image_nr, Tabel_nrs
#                     #print 'Prediction of cells in a Image', stats.itemfreq(cell_value_pred) 
#             #         print 'Highest prediction in a Image', stats.itemfreq(cell_value_pred)[0,0] 
#                     freq = stats.itemfreq(cell_value_pred)
#                     #freq[numpy.argmax(freq[0:,1])][0].astype(int)
#                     
#                     #print Predictions
#                     #print Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
#                     predictions[count] = Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
#             
#                     
#                     count += 1
#         #             print 'count', count
#         #             if count > 1:
#         #                 break
#             
#             
#             
#             true_val = predictions[0:,2]
#             pred_val = predictions[0:,3]
#             from sklearn.metrics import accuracy_score
#             avg_acc = accuracy_score(true_val, pred_val) 
#             print 'Average the prediction of individual cells'
#             print 'Accuracy', avg_acc
#             print 'Test error', 1-avg_acc   
#             
#             print
#             print
#             print 'Prediction of individual cells'
#             aa = y_test_pred.flatten()
#             bb = y_test.flatten()
#             ind_acc = accuracy_score(bb, aa)
#             print 'Accuracy', ind_acc
#             print 'Test error', 1-ind_acc   
#             
#         
#                
#             
#             print 'len y_test_pred', len(y_test_pred)
#             print 'type y_test_pred', type(y_test_pred)
#             #print 'y_test_pred', y_test_pred[0:,0:100]
#             print 
#             
#             
#             #import os
#             #output_folder = '/media/aditya/nimi1/repos/Clean_code/results/BL_bbbc_feat4/' + target_dataset +'_1.00/' 
#             #output_folder = results_dir+target_dataset
#         #     if not os.path.isdir(output_folder):
#         #         os.makedirs(output_folder)
#         #     os.chdir(output_folder) 
#             
#             print "Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4)
#             avg_cm = confusion_matrix(true_val, pred_val)    
#             print(avg_cm)
#             plt.matshow(avg_cm)
#             #plt.title('Confusion matrix with Averaging predictions')
#             plt.title("Confusion matrix (avg cells): Acc = (%.2f)" % round(avg_acc*100,4))
#             plt.colorbar()
#             plt.ylabel('True label')
#             plt.xlabel('Predicted label')
#             plt.savefig(target_outputs_dir+'avg_cm.png',bbox_inches='tight',dpi=200)
#             #plt.savefig(results_dir+'avg_cm.png',bbox_inches='tight',dpi=200)
#             
#             
#             print "Confusion matrix (Ind cells): Acc = (%.2f)" % round(ind_acc*100,4)
#             ind_cm = confusion_matrix(bb,aa)    
#             print(ind_cm)
#             plt.matshow(ind_cm)
#             plt.title("Confusion matrix (Ind cells): Acc = (%.2f)" % round(ind_acc*100,4))
#             #plt.title('Confusion matrix with Individual predictions')
#             plt.colorbar()
#             plt.ylabel('True label')
#             plt.xlabel('Predicted label')
#             plt.savefig(target_outputs_dir+'ind_cm.png',bbox_inches='tight',dpi=200)
#             #plt.savefig(results_dir+'ind_cm.png',bbox_inches='tight',dpi=200)
#             
#             
#             if approach == 'BL':
#                 file_name = 'pt_'+str(len(pt_ep))+'layer.png'
#                 f, axarr = plt.subplots(len(pt_ep), sharex=True) 
#                 for k in range(len(pt_ep)):
#                     axarr[k].plot(range(len(pt_ep[k])), pt_ep[k], '--', linewidth=2, label='layer '+str(k))
#                     axarr[k].set_ylabel('reconstruction cost',rotation=90)
#                     axarr[k].legend()
#                 
#                 axarr[0].set_title('Pre-training reconstruction cost')
#                 plt.xlabel('Pt epochs')
#                 plt.savefig(target_outputs_dir+file_name,bbox_inches='tight',dpi=200)
#                 #plt.savefig(results_dir+file_name,bbox_inches='tight',dpi=200)
#     plt.show()


def plot_layerwise_features(results_dir,experiments,training_data_fractions,nr_reps):
    """
    Plot traning info such as layerwise features.
    """
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    
    for ei in range(nr_experiments):
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach
        
        if approach == 'MSTS':  # Cyclic STS
            num_of_STS = 3
            total_transfers = num_of_STS *2
            for transfer in range(1,total_transfers+1):
                for fi in range(nr_td_fractions):
                    training_data_fraction = training_data_fractions[fi]
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                                 .format(results_dir,
                                         target_dataset,
                                         source_dataset,
                                         str(transfer))
                    outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                    for ri in range(nr_reps):
                        outputs = outputs_list[ri]
                
        else:            
            for fi in range(nr_td_fractions):
                training_data_fraction = training_data_fractions[fi]
                if source_reuse_mode == None:
                    target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
                elif source_reuse_mode == 'Join':
                    target_outputs_dir = '{0}{1}_{2}_{3}_{4:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_reuse_mode,
                            source_dataset,
                            training_data_fraction)
                elif source_reuse_mode is not None and source_reuse_mode is not 'Join':
                    # reusability experiment
                    target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            source_dataset,
                            source_reuse_mode,
                            str(retrain_ft_layers),
                            training_data_fraction)
                outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                for ri in range(nr_reps):
                    outputs = outputs_list[ri]


        results_dir2 = results_dir+'01_features/'
        target_outputs_dir = target_outputs_dir.replace(results_dir,results_dir2)[:-1]
        print target_outputs_dir
        from utils import tile_raster_images
        import PIL.Image
        import numpy 
        print 'restoring source problem fine-tunned weights'
        sda_reuse_ft_model = outputs['sda_reuse_ft_model']
        layer = 1
        for ids in range(len(sda_reuse_ft_model.params)-2):
            #if ids%2 == 0:
            if ids == 0:
                print 'ids', ids
                xx = sda_reuse_ft_model.params[ids].get_value()
                print 'shape xx', numpy.shape(xx)
                a = numpy.sqrt(numpy.shape(xx[0:,1]))
                X=sda_reuse_ft_model.params[ids].get_value(borrow=True).T
                X_mea = numpy.mean(X, axis=1)
                print 'shape  X_mea', numpy.shape(X_mea)
                print 'shape  X_mea', X_mea
                
#                 image = PIL.Image.fromarray(tile_raster_images(
#                                 X=sda_reuse_ft_model.params[ids].get_value(borrow=True).T,
#                                 img_shape=(a,a), tile_shape=(5, 5),
#                                 tile_spacing=(1, 1)))
#                 image.save(target_outputs_dir+'_L'+str(layer)+'_.png')
#                 #layer = layer + 1
                            
                    

def plot_BBBC_sts(results_dir,experiments,training_data_fractions,nr_reps,file_name,ylim=None):
    """
    Avg Error for STS (BBBC data)
    """
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)

    bl_e_mea = None
    bl_e_std = None
    bl_es    = None   
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach

        avg_accs = -numpy.ones((nr_reps,nr_td_fractions))
        
        
        if approach == 'MSTS':  # Cyclic STS
            num_of_STS = 10
            total_transfers = num_of_STS *2
            
            avg_accs_MSTS = -numpy.ones((total_transfers,nr_reps,nr_td_fractions))
            
            for transfer in range(1,total_transfers+1):
                
                for fi in range(nr_td_fractions):
                    training_data_fraction = training_data_fractions[fi]
                    
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                                 .format(results_dir,
                                         target_dataset,
                                         source_dataset,
                                         str(transfer))
                    outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                    
                    for ri in range(nr_reps):
                        outputs = outputs_list[ri]
                        avg_acc = outputs['avg_acc']

                        avg_accs[ri,fi] = avg_acc
                        avg_accs_MSTS[transfer-1,ri,fi] = avg_acc
                        
                a_mea = numpy.mean(avg_accs,0)
                a_std = numpy.std (avg_accs,0)
                print_row_errors3(a_mea,a_std)
                
            a_mea_MSTS = -numpy.ones((total_transfers))
            a_std_MSTS = -numpy.ones((total_transfers))
            for l in range(total_transfers):
                a_mea_MSTS[l] = numpy.mean(avg_accs_MSTS[l],0)
                a_std_MSTS[l] = numpy.std(avg_accs_MSTS[l],0)
            print_row_errors3(a_mea_MSTS,a_std_MSTS)
                

                
#                 nrs_induction_samples = numpy.array(training_data_fractions)*xmax
#                 X = nrs_induction_samples + plot_shift
#                 Y     = numpy.mean(test_errors,0)
#                 Y_err = numpy.std (test_errors,0)
#                 print_row_errors2(Y,Y_err,p)
#                 plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei])
#                 plot_shift += xmax/200 # helps to distinguish overlapping error bars
                
                             
    
                
        else:            
            for fi in range(nr_td_fractions):
                
                training_data_fraction = training_data_fractions[fi]
                      
                if source_reuse_mode == None:
                
                    target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)                     
                outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')

                
#                 for ri in range(nr_reps):
#                     outputs = outputs_list[ri]
#                     test_error = outputs['ft_test_err']
#                     test_errors[ri,fi] = test_error
#             
#             e_mea = numpy.mean(test_errors,0)
#             e_std = numpy.std (test_errors,0)
#             print 'test_errors'
#             #print test_errors
#             print e_mea
            #print e_std
#             nrs_induction_samples = numpy.array(training_data_fractions)*xmax
#             X = nrs_induction_samples + plot_shift
#             plot_shift += xmax/300
#             p = None
#             if approach == 'BL' and source_reuse_mode == None:
#             #if ei == 0:
#                 # store baseline stats
#                 print experiment
#                 print 'baseline'
#                 bl_e_mea = e_mea
#                 bl_e_std = e_std
#                 print 'BL:', bl_e_mea
#                 print 'ei:', e_mea
#             else:
#                 r_mea = (bl_e_mea - e_mea) / bl_e_mea
#                 print experiment
#                 print 'BL:', bl_e_mea
#                 print 'ei:', e_mea
#                 print 'ri:', r_mea
#                 print '-----------------------------------'
#                 Y = r_mea
#                 plt.plot(X, Y, linewidth=2.0)
#                 
#             
#                   #numpy.mean(test_errors,0)
#                 #Y     = numpy.mean(test_errors,0)
#                 #Y_err = numpy.std (test_errors,0)
#                 #print_row_errors2(Y,Y_err,p)
#                 #print_row(Y,Y_err)
#                 #plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
#                 #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
#              # helps to distinguish overlapping error bars
#             
#                 #plt.plot(X, Y, linewidth=2.0, linestyle=linestyles[ei], color=colors[ei])
#                 
#             
# #             else:
# #                 # run t-test
# #                 nr_columns = numpy.shape(e_mea)[0]
# #                 p = -numpy.ones((nr_columns))
# #                 for ci in range(5):
# #                     res = mlab.run('ttest2ms.m', {'n':nr_reps,'m1':bl_e_mea[ci],'s1':bl_e_std[ci],'m2':e_mea[ci],'s2':e_std[ci]})
# #                     p[ci] = res['result']
#             
# #             print_row_errors2(e_mea,e_std,p)
# #             #zzz
# #             
# #             nrs_induction_samples = numpy.array(training_data_fractions)*xmax
# #             X = nrs_induction_samples + plot_shift
# #             #Y     = numpy.mean(test_errors,0)
# #             #Y_err = numpy.std (test_errors,0)
# #             print_row_errors2(Y,Y_err,p)
# #             #print_row(Y,Y_err)
# #             plt.errorbar(X,Y,Y_err,linestyle=linestyles[ei],linewidth=linewidths[ei],color=colors[ei],elinewidth=2,capsize=2)
# #             #plt.errorbar(X,Y,Y_err,linestyle=linestyle,linewidth=linewidth,color=color)
# #             plot_shift += xmax/700 # helps to distinguish overlapping error bars
# #         
#     alfs=20
#     #plt.legend(legend,prop={'size':alfs},loc='center left', bbox_to_anchor=(1, 0.5))
#     #plt.legend(legend,prop={'size':alfs-10}) #,loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.legend(legend,prop={'size':alfs-5},loc='upper right') #, bbox_to_anchor=(1, 0.5))
#     #plt.legend(legend,prop={'size':alfs-5},loc='lower left') #, bbox_to_anchor=(1, 0.5))
#     #plt.legend(bbox_to_anchor=(2.1, 2.05))
#     plt.xlabel('Number of training samples',fontsize=20)
#     #plt.ylabel('(lower the better) Avg. Error',rotation=90,fontsize=20)
#     plt.ylabel('relative improvement',rotation=90,fontsize=20)
#     Y = numpy.zeros(len(e_mea))
#     plt.plot(X, Y, linewidth=1.0, color='black', linestyle= '-.' )
#     print nrs_induction_samples
#     plt.xticks(nrs_induction_samples,rotation=90,fontsize=20)
#     yticks,locs = plt.yticks()
#     ll = ['%.2f' % a for a in yticks]
#     plt.yticks(yticks,ll,fontsize=20)
# #   plt.title(title)
#     plt.xlim(xmax/40,xmax+xmax/40)
#     if ylim is not None:
#         plt.ylim(ylim)
#     else:
#         ax = plt.gca()
#         ax.margins(None,0.03)
#         print 'ylim=',plt.ylim()
#     plt.tight_layout()
#     plt.savefig(results_dir+file_name,bbox_inches='tight', dpi=300)
#     plt.show()
#     plt.close()
    

def plot_BBBC_sts2(results_dir,experiments,training_data_fractions,nr_reps,file_name,ylim=None):
    """
    Avg Error for STS (BBBC data)
    """
    
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)

    bl_e_mea = None
    bl_e_std = None
    bl_es    = None   
    
    for ei in range(nr_experiments):
        
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers] = experiment
        print 'approach =====', approach

        avg_accs = -numpy.ones((nr_reps,nr_td_fractions))
        
        
        if approach == 'MSTS':  # Cyclic STS
            num_of_STS = 10
            total_transfers = (num_of_STS *2)
            
            avg_accs_MSTS = -numpy.ones((total_transfers+1,nr_reps,nr_td_fractions))
            
            for transfer in range(1,total_transfers+1):
                
                for fi in range(nr_td_fractions):
                    training_data_fraction = training_data_fractions[fi]
                    
                    if transfer == 1:
                        bl_outputs_dir = '{0}{1}_{2:4.2f}/'\
                        .format(results_dir,
                                target_dataset,
                                training_data_fraction)                     
                        bl_outputs_list = load_outputs(bl_outputs_dir,prefix='outputs_')
                        for ri in range(nr_reps):
                            bl_outputs = bl_outputs_list[ri]
                            avg_acc = bl_outputs['avg_acc']
                            avg_accs[ri,fi] = avg_acc
                            avg_accs_MSTS[transfer-1,ri,fi] = avg_acc
                        print_row_errors3(numpy.mean(avg_accs,0),numpy.std (avg_accs,0))
                    
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                                 .format(results_dir,
                                         target_dataset,
                                         source_dataset,
                                         str(transfer))
                    outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
                    
                    for ri in range(nr_reps):
                        outputs = outputs_list[ri]
                        avg_acc = outputs['avg_acc']
                        avg_accs[ri,fi] = avg_acc
                        avg_accs_MSTS[transfer,ri,fi] = avg_acc
                        
                a_mea = numpy.mean(avg_accs,0)
                a_std = numpy.std (avg_accs,0)
                print_row_errors3(a_mea,a_std)
                
            a_mea_MSTS = -numpy.ones((total_transfers+1))
            a_std_MSTS = -numpy.ones((total_transfers+1))
            for l in range(total_transfers+1):
                a_mea_MSTS[l] = numpy.mean(avg_accs_MSTS[l],0)
                a_std_MSTS[l] = numpy.std(avg_accs_MSTS[l],0)
            print_row_errors3(a_mea_MSTS,a_std_MSTS)
            
            t_stat = -numpy.ones((total_transfers))
            p_val = -numpy.ones((total_transfers))
            for l in range(total_transfers+1):
                if l == 0:
                    baseline = avg_accs_MSTS[l]
                    print 'baseline', numpy.mean(avg_accs_MSTS[l],0)
                else:
                    follow_up = avg_accs_MSTS[l]
                    print 'follow_up', numpy.mean(avg_accs_MSTS[l],0)
                    paired_sample = stats.ttest_rel(baseline, follow_up)
                    t_stat[l-1], p_val[l-1] = paired_sample
                    print "The t-statistic is %.3f and the p-value is %.3f." % paired_sample
             
            print_row_errors3(a_mea_MSTS[1:],a_std_MSTS[1:],p_val)


def plot_ETL(results_dir,experiments,training_data_fractions,nr_reps,file_name,ylim=None):
    """
    Avg Error for ETL (BBBC data)
    """
    nr_experiments = len(experiments)
    nr_td_fractions = len(training_data_fractions)
    for ei in range(nr_experiments):
        experiment = experiments[ei]
        [approach, target_dataset,source_dataset,source_reuse_mode,retrain_ft_layers,tranferred_layers] = experiment
        print 'approach =====', approach
        
        for fi in range(nr_td_fractions):
            training_data_fraction = training_data_fractions[fi]
            if source_dataset is None:
                # baseline or source experiment
                target_outputs_dir = '{0}{1}_{2:4.2f}/'\
                    .format(results_dir,
                            target_dataset,
                            training_data_fraction)
            else:
                # reusability experiment
                target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5}_{6:4.2f}/'\
                 .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         str(tranferred_layers),
                         training_data_fraction)
            outputs_list = load_outputs(target_outputs_dir,prefix='outputs_')
        
        for repetition in range(nr_reps):
            outputs = outputs_list[repetition]
            ground_truth = outputs['y_test']
            ground_truth = ground_truth.flatten()

        En = []
        r = []
        import numpy as np
        Ensemble = np.zeros((len(ground_truth),12))
        for repetition in range(nr_reps):
            outputs = outputs_list[repetition]
            tmp = outputs['y_test_class_prob']
            conv = np.vstack(tmp.flatten()).astype(np.float32)
            En.append(conv)
            print np.shape(En)
            r.append(np.mean(ground_truth == np.array(En[repetition]).argmax(axis=1)))
            print r[repetition]
            Ensemble = np.add(Ensemble,En[repetition])
        
        Ensemble_error = np.mean(ground_truth == np.array(Ensemble).argmax(axis=1))
        print 'Ensemble_error = ', Ensemble_error
        print 'Avg test error = %.2f(%.2f)' % (np.mean(r), np.std(r))
       
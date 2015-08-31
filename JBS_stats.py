from stat_calculations import load_majvote_accus_cv, box_plot_performance, load_classification_report_cv
''' Set2 Experiments '''
target_name = 'set2'
source_name = 'set1'

# # ''' Set1 Experiments '''
# target_name = 'set1'
# source_name = 'set2'






nr_reps = 10
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+target_name+'a/MFC7_'+target_name+'_1.00/'
# baseline = load_majvote_accus_cv(result_dir,nr_reps)
# print 'baseline', baseline


# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[1, 1, 1, 1, 1, 1, 1, 1]_1.00/'
# DTL_11111111 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_11111111', DTL_11111111
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 1, 1, 1, 1, 1, 1]_1.00/'
# DTL_00111111 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_00111111', DTL_00111111
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 1, 1, 1, 1]_1.00/'
# DTL_00001111 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_00001111', DTL_00001111
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 0, 0, 1, 1]_1.00/'
# DTL_00000011 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_00000011', DTL_00000011

# Prepare classification report
result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+target_name+'a/MFC7_'+target_name+'_1.00/'
baseline = load_classification_report_cv(result_dir,nr_reps)
print 'baseline', baseline
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[1, 1, 1, 1, 1, 1, 1, 1]_1.00/'
# DTL_11111111 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_11111111', DTL_11111111
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 1, 1, 1, 1, 1, 1]_1.00/'
# DTL_00111111 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_00111111', DTL_00111111
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 1, 1, 1, 1]_1.00/'
# DTL_00001111 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_00001111', DTL_00001111
# result_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+source_name+'a/MFC7_'+target_name+'_reusing_MFC7_'+source_name+'_PT+FT_[1, 1, 1, 1, 1, 1, 1, 1]_[0, 0, 0, 0, 0, 0, 1, 1]_1.00/'
# DTL_00000011 = load_majvote_accus_cv(result_dir,nr_reps)
# print 'DTL_00000011', DTL_00000011


# target_outputs_dir = '/home/aditya/store/Theano/DTL_HPC2/results/BL_'+target_name+'a/'
# 
# data = [baseline, DTL_00000011, DTL_00001111, DTL_00111111, DTL_11111111]
# from scipy import stats
# 
# 
# 
# paired_sample = stats.ttest_rel(baseline, DTL_11111111)
# print "The DTL_11111111 t-statistic is %.3f and the p-value is %.3f." % paired_sample
# paired_sample = stats.ttest_rel(baseline, DTL_00111111)
# print "The DTL_00111111 t-statistic is %.3f and the p-value is %.3f." % paired_sample
# paired_sample = stats.ttest_rel(baseline, DTL_00001111)
# print "The DTL_00001111 t-statistic is %.3f and the p-value is %.3f." % paired_sample
# paired_sample = stats.ttest_rel(baseline, DTL_00000011)
# print "The DTL_00000011 t-statistic is %.3f and the p-value is %.3f." % paired_sample
# print '====== ====== ====== ====== ====== ====== ======'

#box_plot_performance(target_outputs_dir, target_name, data)
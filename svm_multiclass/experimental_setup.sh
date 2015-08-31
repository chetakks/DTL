#!/bin/bash

=========================================== RBF with no PCA based only 10% data
Running Stopped

Set1
taskset -c 0 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console_set1_rep1.txt 2>&1 &

Set2
taskset -c 3 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console_set2_rep1.txt 2>&1 &


Repetition2
Set1
taskset -c 2 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console_set1_rep2.txt 2>&1 &

Set2
taskset -c 4 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console_set2_rep2.txt 2>&1 &




Running test errors restart
Set1
taskset -c 2 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console2_set1_rep1.txt 2>&1 &

Set2
taskset -c 4 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console2_set2_rep1.txt 2>&1 &



=========================================== RBF with no PCA based only 1% data, Cost = 100
Running

Set1
taskset -c 0 nohup python run_rbf_msvm.py all_results/data1/c100_g0.001 100.0 2 0.001 > all_results/data1/c100_g0.001/console_set1_rep1.txt 2>&1 &

Set2
taskset -c 3 nohup python run_rbf_msvm.py all_results/data1/c100_g0.001 100.0 2 0.001 > all_results/data1/c100_g0.001/console_set2_rep1.txt 2>&1 &

=========================================== RBF with no PCA based only 1% data, Cost = 1000
Running

Set1
taskset -c 2 nohup python run_rbf_msvm.py all_results/data1/c1000_g0.001 1000.0 2 0.001 > all_results/data1/c1000_g0.001/console_set1_rep1.txt 2>&1 &

Set2
taskset -c 5 nohup python run_rbf_msvm.py all_results/data1/c1000_g0.001 1000.0 2 0.001 > all_results/data1/c1000_g0.001/console_set2_rep1.txt 2>&1 &



===========================================






taskset -c 5 nohup python run_msvm.py experiments/default2 1.0 0 > experiments/default2/console.txt 2>&1 &

taskset -c 1 nohup python run_msvm.py experiments/cost_50000 50000 0 > experiments/cost_50000/console.txt 2>&1 &




===========================================RBF cost, gamma
Running Done
taskset -c 4 nohup python run_rbf_msvm.py experiments/c_100rbf_0.001 100.0 2 0.001 > experiments/c_100rbf_0.001/console.txt 2>&1 &

taskset -c 5 nohup python run_rbf_msvm.py experiments/c_100rbf_0.0001 100.0 2 0.0001 > experiments/c_100rbf_0.0001/console.txt 2>&1 &

taskset -c 6 nohup python run_rbf_msvm.py experiments/c_100rbf_0.00001 100.0 2 0.00001 > experiments/c_100rbf_0.00001/console.txt 2>&1 &

taskset -c 0 nohup python run_rbf_msvm.py experiments/rbf_0.001 1.0 2 0.001 > experiments/rbf_0.001/console.txt 2>&1 &

taskset -c 1 nohup python run_rbf_msvm.py experiments/rbf_0.0001 1.0 2 0.0001 > experiments/rbf_0.0001/console.txt 2>&1 &

taskset -c 2 nohup python run_rbf_msvm.py experiments/rbf_0.00001 1.0 2 0.00001 > experiments/rbf_0.00001/console.txt 2>&1 &

=========================================== RBF with no PCA based
Running stopped

ALLN
taskset -c 0 nohup python run_rbf_msvm.py experiments/data100/c100_g0.001 100.0 2 0.001 > experiments/data100/c100_g0.001/console_ALLN.txt 2>&1 &

Campthothecin
taskset -c 1 nohup python run_rbf_msvm.py experiments/data100/c100_g0.001 100.0 2 0.001 > experiments/data100/c100_g0.001/console_Campthothecin.txt 2>&1 &

taxol
taskset -c 2 nohup python run_rbf_msvm.py experiments/data100/c100_g0.001 100.0 2 0.001 > experiments/data100/c100_g0.001/console_taxol.txt 2>&1 &

emetine
taskset -c 3 nohup python run_rbf_msvm.py experiments/data100/c100_g0.001 100.0 2 0.001 > experiments/data100/c100_g0.001/console_emetine.txt 2>&1 &

AZ-A
taskset -c 4 nohup python run_rbf_msvm.py experiments/data100/c100_g0.001 100.0 2 0.001 > experiments/data100/c100_g0.001/console_AZ-A.txt 2>&1 &

vincristine
taskset -c 5 nohup python run_rbf_msvm.py experiments/data100/c100_g0.001 100.0 2 0.001 > experiments/data100/c100_g0.001/console_vincristine.txt 2>&1 &


=========================================== RBF with PCA based
Running ... DONE
taskset -c 3 nohup python run_rbf_msvm.py experiments/pca_c_100rbf_0.001 100.0 2 0.001 > experiments/pca_c_100rbf_0.001/console.txt 2>&1 &

taskset -c 0 nohup python run_rbf_msvm.py experiments/pca_c100_rbf0.001_x0.05 100.0 2 0.001 > experiments/pca_c100_rbf0.001_x0.05/console.txt 2>&1 &


taskset -c 2 nohup python run_rbf_msvm.py experiments/pca_c200_rbf0.001_x0.05 200.0 2 0.001 > experiments/pca_c200_rbf0.001_x0.05/console.txt 2>&1 &

taskset -c 4 nohup python run_rbf_msvm.py experiments/pca_c500_rbf0.001_x0.05 500.0 2 0.001 > experiments/pca_c500_rbf0.001_x0.05/console.txt 2>&1 &

taskset -c 5 nohup python run_rbf_msvm.py experiments/pca_c1000_rbf0.001_x0.05 1000.0 2 0.001 > experiments/pca_c1000_rbf0.001_x0.05/console.txt 2>&1 &

=========================================== RBF with no PCA based
Running ... DONE
taskset -c 1 nohup python run_rbf_msvm.py experiments/c_100rbf_0.001_x0.05 100.0 2 0.001 > experiments/c_100rbf_0.001_x0.05/console.txt 2>&1 &

TODO
taskset -c 6 nohup python run_rbf_msvm.py experiments/c_200rbf_0.001_x0.05 200.0 2 0.001 > experiments/c_200rbf_0.001_x0.05/console.txt 2>&1 &

taskset -c 7 nohup python run_rbf_msvm.py experiments/c_500rbf_0.001_x0.05 500.0 2 0.001 > experiments/c_500rbf_0.001_x0.05/console.txt 2>&1 &

===========================================

taskset -c 4 nohup python run_rbf_msvm.py experiments/c_100rbf_0.001 100.0 2 0.001 > experiments/c_100rbf_0.001/console.txt 2>&1 &

taskset -c 5 nohup python run_rbf_msvm.py experiments/c_100rbf_0.0001 100.0 2 0.0001 > experiments/c_100rbf_0.0001/console.txt 2>&1 &

taskset -c 6 nohup python run_rbf_msvm.py experiments/c_100rbf_0.00001 100.0 2 0.00001 > experiments/c_100rbf_0.00001/console.txt 2>&1 &


===========================================

taskset -c 0 nohup python run_rbf_msvm.py experiments/rbf_0.001 1.0 2 0.001 > experiments/rbf_0.001/console.txt 2>&1 &

taskset -c 1 nohup python run_rbf_msvm.py experiments/rbf_0.0001 1.0 2 0.0001 > experiments/rbf_0.0001/console.txt 2>&1 &

taskset -c 2 nohup python run_rbf_msvm.py experiments/rbf_0.00001 1.0 2 0.00001 > experiments/rbf_0.00001/console.txt 2>&1 &



===========================================

taskset -c 0 nohup python run_msvm.py experiments/cost_100000 100000 0 > experiments/cost_100000/console.txt 2>&1 &

taskset -c 1 nohup python run_msvm.py experiments/cost_100000 200000 0 > experiments/cost_200000/console.txt 2>&1 &

taskset -c 2 nohup python run_msvm.py experiments/cost_100000 300000 0 > experiments/cost_300000/console.txt 2>&1 &

taskset -c 3 nohup python run_msvm.py experiments/cost_100000 400000 0 > experiments/cost_400000/console.txt 2>&1 &

taskset -c 4 nohup python run_msvm.py experiments/cost_100000 500000 0 > experiments/cost_500000/console.txt 2>&1 &




===========================================

taskset -c 3 nohup python run_msvm.py experiments/poly_default 1.0 1 > experiments/poly_default/console.txt 2>&1 &

===========================================

taskset -c 0 nohup python run_msvm.py experiments/cost_0.1 0.1 0 > experiments/cost_0.1/console.txt 2>&1 &

taskset -c 5 nohup python run_msvm.py experiments/cost_0.01 0.01 0 > experiments/cost_0.01/console.txt 2>&1 &

taskset -c 6 nohup python run_msvm.py experiments/cost_0.001 0.001 0 > experiments/cost_0.001/console.txt 2>&1 &

taskset -c 7 nohup python run_msvm.py experiments/cost_0.0001 0.0001 0 > experiments/cost_0.0001/console.txt 2>&1 &

taskset -c 8 nohup python run_msvm.py experiments/cost_0.00001 0.00001 0 > experiments/cost_0.00001/console.txt 2>&1 &

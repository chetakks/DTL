#!/bin/bash

<< COMMENT

echo "to run a python code:1";
nvidia-smi

CODE=$HOME/repos/Convolution_code/code
# TODO cd $CODE

echo "to run a python code 1">> foo.out
nvidia-smi
python $CODE/reuse_cnn1.py MAHDBase_250  10 0  mnist   0,0,0,0 2>> foo.err >> foo.out
echo "to run a python code 2">> foo.out
COMMENT

date1=$(date +"%s")
time ./svm-train -c 10 /home/aditya_1t/BBBC_data/libsvm-3.20/libsvm_data/MFC7_set1_ALLN_train.dat
date2=$(date +"%s")
diff=$(($date2-$date1))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."

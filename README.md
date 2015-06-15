# Setup:
```sh
echo "prepare the dataset in accordance to LOOV cross-validation";
python two_MFC7_data2.py <path-to-csv-data>
```
# DTL

#!/bin/bash


CODE=$HOME/store/Theano/DTL_JBS
# TODO cd $CODE

echo "prepare the dataset in accordance to LOOV cross-validation";
python $CODE/two_MFC7_data2.py '/home/aditya/store/Datasets/pickled/MFC7/ljosa_data/'

<< COMMENT


taskset -c 0 python $CODE/experiment_set1.py > results/BL_set1a/set1_rep1_5.txt 2>&1 &
taskset -c 1 python $CODE/experiment_set1.py > results/BL_set1a/set1_rep6_10.txt 2>&1 &
taskset -c 2 python $CODE/experiment_set2.py > results/BL_set2a/set2_rep1_5.txt 2>&1 &
taskset -c 3 python $CODE/experiment_set2.py > results/BL_set2a/set2_rep6_10.txt 2>&1 &
COMMENT

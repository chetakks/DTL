# HPC syetem requirements:
The following steps are important to setup the HPC machine to run on GPU
http://deeplearning.net/software/theano/install.html#install
```sh
Linux operating system
Python >= 2.6
g++, python-dev
NumPy >= 1.6.2
SciPy >= 0.11
Open BLAS
nose
Sphinx >= 0.5.1, pygments
Git
pydot
NVIDIA CUDA drivers and SDK
libgpuarray
Matplotlib
sklearn
```

# Easy Installation of an Optimized Theano on Ubuntu 
http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

# Download the data
```sh
wget http://www.deepnets.ineb.up.pt/files/ljosa_data.zip
```

# Prepare the data in accordance to LOOV cross-validation:
Eg: python two_MFC7_data2.py '/home/aditya/store/Datasets/pickled/MFC7/ljosa_data/'
```sh
$ python two_MFC7_data2.py <path-to-downloaded-ljosa-data>
```

# DTL experiments for 10 repetition
```sh
taskset -c 0 nohup python experiment_set1.py > results/BL_set1a/set1_rep1_5.txt 2>&1 &
taskset -c 1 nohup python experiment_set1.py > results/BL_set1a/set1_rep6_10.txt 2>&1 &
taskset -c 2 nohup python experiment_set2.py > results/BL_set2a/set2_rep1_5.txt 2>&1 &
taskset -c 3 nohup python experiment_set2.py > results/BL_set2a/set2_rep6_10.txt 2>&1 &
```

# Edit JBS_results.py file:
Edit these file path to appropriately
* `result_dir = ` <path-of-the-stored-result>
* `approach = `
* `source_reuse_mode = `
* `target_dataset = `
* `fold = `
* `source_dataset = `
* `source_fold = `
* `nr_reps = `

# Results and Confusion matrix
```sh
python JBS_results.py
```



# Install multiclass SVM
Install multiclass SVM light to run the above experiment. Note the above code is for SVM with RBF kernel. The dataset need to be in prefered format of the SVMlight. https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html

# Experiments with SVM
```sh
taskset -c 0 nohup python run_rbf_msvm.py all_results/data10/c100_g0.001 100.0 2 0.001 > all_results/data10/c100_g0.001/console_set1_rep1.txt 2>&1 &
```


# RLST-git-proj
Source code in this project is pushed as a supplementary material for a scientific paper. Details will be released after the submission deadline of related conference.
---------------------------------------
For paper reviewers:

The project is run under maskrcnn_benchmark conda environment, check https://github.com/facebookresearch/maskrcnn-benchmark for installation details. Code itself is not well organized at current stage, we will refactor it if the paper get accepted and the link will be updated here.

After installation of maskrcnn_benchmark, run **python3 train.py** to get the model trained and run **python3 eval_act_acc.py** to get single-step evaluation of your trained model. The path of the trained model needs to be specified in ./predict_next.py;

1.  rlf_dataset_v3.py || rlf_dataset.py
Containing all functions for building our RL operations, e.g. initialization, get state, take action, get reward, etc. Main difference between v3 and the initial version is the way to generate distorted ground truth. English and Chinese are commented in the code for readers to better understand. 

For now, rlf_dataset.py is imported as text environment for train.py, eval_acc_acc.py and predict_next.py;

-----------------------------------------
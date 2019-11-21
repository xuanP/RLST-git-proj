import time
import math
import os 
import numpy as np
import torch
from torch.autograd import Variable
from rlf_dataset import text_detection_env
from predict_next import predict_next

eval_num = 4000

def evaluation(te, eval_num):
    ic = 0
    icc = 0
    for i in range(eval_num):
        te.reset()
        img,focus,_ = te[i]
        dxs = predict_next(img,focus)
        allre = te.get_all_reward()
        pmax = dxs.argmax()
        rmax = np.asarray(allre).argmax()
        #print("real wd:",np.asarray(allre))
        if pmax==rmax:
            ic = ic+1
        if allre[pmax]>0:
            icc = icc + 1
        print(i, ic, icc, pmax,rmax, te.actlist[pmax], te.actlist[rmax], pmax==rmax)
    return float(ic/eval_num),float(icc/eval_num)
    
        
if __name__ == "__main__":
    #root_image_path = '/home/user/data/icdar2015/raw'
    #root_label_path = '/home/user/data/icdar2015/gt'
    root_image_path = '/home/user/data/icdar2015/test_img'
    root_label_path = '/home/user/data/icdar2015/test_gt'
    te = text_detection_env(root_image_path,root_label_path)
    rate1,rate2 = evaluation(te,eval_num)
    print("rate1:",rate1, "rate2:",rate2)

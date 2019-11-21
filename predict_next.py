import time
import math
import os 
import numpy as np
import torch
from torch.autograd import Variable
#from cnnmodel import Policy
#from focus_cnn_v4 import FocusPolicy
from progressive_v10 import FocusPolicy
#from cnnmodel import Policy as FocusPolicy
from rlf_dataset import text_detection_env

checkpoint_path = './checkpoints/model_100_0.134.pth'

model = FocusPolicy()
model = model.cuda()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
    
def predict_next(im,focus):
    start = time.time()
    #print(im_fn)
    im = Variable(torch.from_numpy(im)).cuda()
    im = im.unsqueeze(0)   # 模型是用batch输入的，这里我们只有一张图
    focus = Variable(torch.from_numpy(focus)).cuda()
    focus = focus.unsqueeze(0)
    dxs = model(im,focus)
    dxs = dxs.data.cpu().numpy()
    during = time.time() - start
    #print('net time :{:.6f}'.format(during))
    return dxs
        
if __name__ == "__main__":
    root_image_path = '/home/user/data/icdar2015/raw'
    root_label_path = '/home/user/data/icdar2015/gt'
    te = text_detection_env(root_image_path,root_label_path)
    img, focus, reward = te[1]
    print('input img:',img.shape)
    print('focus:',focus.shape)
    dxs=predict_next(img,focus)
    np.set_printoptions(precision=2)
    print("predict:",dxs)
    allre = te.get_all_reward()
    print("real wd:",np.asarray(allre))

import torch
from torch.autograd import Variable
import os 
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
#from focus_cnn_v4 import FocusPolicy
#from cnnmodel import Policy as FocusPolicy
from progressive_v10 import FocusPolicy
#main diff between v1 and v3 is the initialization
#from rlf_dataset_v3 import text_detection_env #v3
from rlf_dataset import text_detection_env #v1
import time
from utils import weight_init
#from tensorboardX import SummaryWriter

#writer = SummaryWriter()


def train(epochs, model, trainloader, crit, optimizer,
         scheduler, save_step, weight_decay):

    for e in range(epochs):
        print('Epoch {} / {}'.format(e + 1, epochs))
        model.train()
        start = time.time()
        loss = 0.0
        total = 0.0
        scheduler.step()
        for i, (img, box, label) in enumerate(trainloader):
            optimizer.zero_grad()
            img = Variable(img.cuda())
            box = Variable(box.cuda())
            label = Variable(label.cuda().float())
            f_label = model(img,box)
            loss1 = crit(f_label,label)
                        
            loss1.backward()
            optimizer.step()
            
            loss += loss1
            ml = loss/((i+1)*32)
            if (i + 1) % 50 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (50*10)/(time.time() - start)
                start = time.time()
                print('Step {:06d}, model loss {:.5f},  {:.2f} seconds/step, {:.2f} examples/second'.format(
                    i, ml,  avg_time_per_step, avg_examples_per_second))
        print()   
        #writer.add_scalar('loss', loss / len(trainloader), e)
        
        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/model_{:03d}_{:.3f}.pth'.format(e + 1,ml ))
        

def main():
    root_image_path = '/home/user/data/icdar2015/raw'
    root_label_path = '/home/user/data/icdar2015/gt'
    trainset = text_detection_env(root_image_path,root_label_path)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=16)
    model = FocusPolicy()
    model = model.cuda()
    model.load_state_dict(torch.load('./checkpoints/bak/model_098_0.177.pth'))
    #model.apply(weight_init)
    
    crit = torch.nn.L1Loss(reduction='sum')
    #crit = torch.nn.SmoothL1Loss(reduction='sum')
    
    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.045e-4)
                                #  weight_decay=1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, 
                                    gamma=0.94)   
    
    train(epochs=100, model=model, trainloader=trainloader,
          crit=crit, optimizer=optimizer,scheduler=scheduler, 
          save_step=2, weight_decay=weight_decay)

    #writer.close()

if __name__ == "__main__":
    main()
    
from skimage.io import imread
from skimage.util import crop
from skimage.transform import rotate,resize,rescale
from skimage import io,transform,data
import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon
from torch.utils.data import Dataset, DataLoader
import math

class text_detection_env(Dataset):
    def __init__(self, image_root_path, label_root_path, if_debug=False):
        self.image_root_path = image_root_path
        self.label_root_path = label_root_path
        self.imglist = list(filter(lambda fn:fn.lower().endswith('.jpg') or fn.lower().endswith('.png') ,
                                   os.listdir(self.image_root_path)))
        #step length and initialization, might be optimized using different values
        self.prompt = True
        self.imgname = ''
        self.conwid = 256  # context width
        self.conhei = 196
        self.maxattwid = 196 # maximum attention width
        self.minattwid = 32 # minimum attention width
        self.atthei = 32
        self.minatthei = 24
        self.maxatthei = 64
        self.maxscale = 4.0
        self.minscale = 0.25
        self.scaledt = 1.1
        self.shiftdt = 10
        self.heightstep= 10
        self.widthstep= 10
        self.rangle = 0.0
        self.maxangle = 20.0
        self.angledt = 2.0
        self.curiou = 0.0 #current iou
        self.actlist = ['up','down','left','right','ahu','ahd','awu','awd','su','sd','rr','rl']
        self.sample_num_per_image = 1
        self.total_len = 20000  
        self.counter = 0  # how many get is called 
        self.att_hei_list = list(np.arange(self.minatthei, self.maxatthei, 8))
        self.att_wid_list = list(np.arange(self.minattwid, self.maxattwid, 8))
        self.dispose = None
        self.reset()
        
    
    '''
    During each step we can obtain a reward. The problem of RL will
    become Regression.
    由于每一步都可以由reward，所以learning会退化成Rregression的问题。
    在当前的position下面，获取每个action的reward。
    '''
    def get_all_reward(self):
        rewardlist = []
        for act in self.actlist:
            self.posi = (self.atthei,self.attwid, self.cx, self.cy, self.scaleratio)  # 记录当前位置 record current position
            _, reward = self.action(act)
            rewardlist.append(reward)
            self.atthei,self.attwid, self.cx, self.cy, self.scaleratio = self.posi  # 恢复当前的位置 revert to previous position
            self.curiou = self.get_max_iou()  # 恢复当前的iou revert to previous IoU
        return rewardlist    
    
    '''
    Randomly select an image, attention initialized to a random size.
    随机挑选一个图片，attention调整到一个随机值
    如果可以提示，就把中心调整到text box 的某个点。
    '''
    def reset(self):
        self.attwid = self.minattwid  # in pixels
        self.imgname = random.choice(self.imglist)
        self.impath = os.path.join(self.image_root_path,self.imgname)
        self.img = imread(self.impath)
        self.cx = self.img.shape[1]//2    # 初始化的时候放在中间，以后也可以根据 hot map来初始化
        self.cy = self.img.shape[0]//2
        self.readlabel()
        # select a random attenion
        self.atthei = random.choice(self.att_hei_list[1:-1])
        self.attwid = random.choice(self.att_wid_list[1:-1]) #  初始化不要太长
        #self.scaleratio = 1.0
        self.scaleratio = random.uniform(0.25*1.1,4.0/1.1)  
        #self.rangle = 0.0  
        self.rangle = random.uniform(20-2.0,20+2.0)  
        if self.prompt:
            for i in range(100):
                poly = random.choice(self.txtboxes)
                #self.cx,self.cy = int(poly.centroid.coords[0][0]), int(poly.centroid.coords[0][1])
                bs = [int(it) for it in poly.bounds]
                #print('bounds:', poly.bounds)
                cx1,cy1 = random.randint(bs[0],bs[2]), random.randint(bs[1],bs[3])   # 文字框中任意一点
                #self.cx,self.cy =  (bs[0]+bs[2])//2, (bs[1]+bs[3])//2   # 文字框中心点
                #self.cx,self.cy =  (self.cx+cx1)//2, (self.cy+cy1)//2
                self.cx,self.cy =  cx1,cy1
                if self.if_in_range():
                    break
        
        if self.dispose is None:
            self.dispose = dispose(self.img, debug = False)
        else:
            self.dispose.setimage(self.img)
        self.curiou = self.get_max_iou()
    
    def readlabel(self):
        self.txtboxes = []
        base=os.path.basename(self.impath)
        base=os.path.splitext(base)[0]
        fname = os.path.join(self.label_root_path, "gt_"+base+".txt")
        #print(fname)
        if not os.path.exists(fname):
            return
        fanswer = codecs.open(fname, mode='r', encoding='utf-8-sig') # icdar是utf-8-sig编码的。
        for line in fanswer:
            s=line.split(",")
            #print(s)
            if len(s)>=9 and s[8]!="###":
                # Create a Polygon
                s = [int(it) for it in s[0:8]]
                coords = [(s[0], s[1]), (s[2], s[3]), (s[4], s[5]), (s[6], s[7])]
                poly = Polygon(coords)
                self.txtboxes.append(poly)
        fanswer.close()
                
    # 根据当前attention在图像中的位置，计算最大的iou。
    # According to current position of attention, calculate maximum IoU
    def get_max_iou_box(self):
        ioulist = []
        maxiou = 0.0
        pa,_ = self.cal_attention()
        attpoly = Polygon([(pa[0],pa[1]),(pa[2],pa[3]),(pa[4],pa[5]),(pa[6],pa[7])])
        for pp in self.txtboxes:
            if pp.intersects(attpoly):
                pu = attpoly.union(pp)
                pint  = attpoly.intersection(pp)
                ioulist.append(pint.area/pu.area)
        if len(ioulist)>0:
            maxiou = max(ioulist)
        return maxiou
    
    # 根据当前中心点位置，找到中心点最近的box，计算iou。
    def get_max_iou(self):
        ioulist = []
        maxiou = 0.0
        pa,_ = self.cal_attention()
        attpoly = Polygon([(pa[0],pa[1]),(pa[2],pa[3]),(pa[4],pa[5]),(pa[6],pa[7])])
        for pp in self.txtboxes:
            if pp.intersects(attpoly):
                pu = attpoly.union(pp)
                pint  = attpoly.intersection(pp)
                ioulist.append(pint.area/pu.area)
        if len(ioulist)>0:
            maxiou = max(ioulist)
        return maxiou
    
    # 计算contex在背景图片中的实际位置
    # Calc the position of context within the background
    def cal_context(self):  
        in_range = True
        cwid = int( 0.5 * self.conwid /  self.scaleratio)
        chei = int( 0.5 * self.conhei /  self.scaleratio)
        sina = math.sin(self.rangle*math.pi/180)
        cosa = math.cos(self.rangle*math.pi/180)
        x1 = self.cx - cwid*cosa - chei*sina
        y1 = self.cy - cwid*sina + chei*cosa
        x2 = self.cx + cwid*cosa - chei*sina
        y2 = self.cy + cwid*sina + chei*cosa
        x3 = self.cx + cwid*cosa + chei*sina
        y3 = self.cy + cwid*sina - chei*cosa
        x4 = self.cx - cwid*cosa + chei*sina
        y4 = self.cy - cwid*sina - chei*cosa
        if self.cx<0 or self.cx>=self.img.shape[1] or self.cy<0 or self.cy>=self.img.shape[0]:
            in_range = False
        return [x1,y1,x2,y2,x3,y3,x4,y4], in_range

    # 计算attention在背景图片中的实际位置
    # Calc the position of attention within the background
    def cal_attention(self):
        in_range = True
        cwid = int( 0.5 * self.attwid / self.scaleratio)
        chei = int( 0.5 * self.atthei / self.scaleratio)
        sina = math.sin(self.rangle*math.pi/180)
        cosa = math.cos(self.rangle*math.pi/180)
        x1 = self.cx - cwid*cosa - chei*sina
        y1 = self.cy + cwid*sina - chei*cosa
        x2 = self.cx + cwid*cosa - chei*sina
        y2 = self.cy - cwid*sina - chei*cosa
        x3 = self.cx + cwid*cosa + chei*sina
        y3 = self.cy - cwid*sina + chei*cosa
        x4 = self.cx - cwid*cosa + chei*sina
        y4 = self.cy + cwid*sina + chei*cosa
        if self.cx<0 or self.cx>=self.img.shape[1] or self.cy<0 or self.cy>=self.img.shape[0]:
            in_range = False
        return [x1,y1,x2,y2,x3,y3,x4,y4],in_range   
    
      
    def if_in_range(self):
        if self.cx>=self.img.shape[1] or self.cx<=0:
            return False
        if self.cy>=self.img.shape[0] or self.cy<=0:
            return False
        return True
    
    # 根据背景图片，当前的cx,cy，从中取出固定尺寸的context
    # According to the background image and current cx cy, extract current context image
    def outState(self):
        iferror = False
        img = None
        try:
            img, in_range = self.dispose.cut(self.cx,self.cy, self.rangle, self.conwid/self.scaleratio, self.conhei/self.scaleratio)
        except:
            print("*** cut error.",self.impath,self.cx,self.cy,self.rangle,self.scaleratio)
            iferror  = True
        if img is None or img.shape[0]==0 or img.shape[1]==0 or in_range == False:
            img = np.zeros([100, 100, 3], np.uint8)+255
            iferror = True
        #print(self.cx,self.cy, self.rangle)
        imgout = cv2.resize(img, dsize=(self.conwid , self.conhei), interpolation=cv2.INTER_CUBIC)
        #imgout = cv2.rectangle(imgout, (self.conwid//2-self.attwid//2,self.conhei//2-self.atthei//2), 
        #              (self.conwid//2+self.attwid//2,self.conhei//2+self.atthei//2), (0,255,0), 1)
        #imgout = cv2.rectangle(imgout, (self.conwid//2-self.attwid//2,self.conhei//2-self.atthei//2), 
        #                          (self.conwid//2+self.attwid//2,self.conhei//2+self.atthei//2), (255,255,255), -1)  # fill with white
        #overlay = img.copy()
        #cv2.rectangle(overlay, (self.conwid//2-self.attwid//2,self.conhei//2-self.atthei//2), 
        #               (self.conwid//2+self.attwid//2,self.conhei//2+self.atthei//2), (0,125,125), -1)
        #alpha = 0.05  # Transparency factzor.
        # Following line overlays transparent rectangle over the image
        #imgout = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        if imgout.shape[0]!=self.conhei or imgout.shape[1]!=self.conwid or imgout.shape[2]!=3:
            print("*** out stat error.",self.impath,self.cx,self.cy,self.rangle,self.scaleratio)
        return imgout, iferror 
    
    def action(self, act):
        if_out_of_bound = False
        if act=="up":  # up
            self.cy = self.cy - int(self.shiftdt)
            if not self.if_in_range():
                self.cy = self.cy + int(self.shiftdt)
                if_out_of_bound = True
        elif act=="down":   # down
            self.cy = self.cy + int(self.shiftdt)
            if not self.if_in_range():
                self.cy = self.cy - int(self.shiftdt)
                if_out_of_bound = True
        elif act=="left":  # left
            self.cx = self.cx - int(self.shiftdt)
            if not self.if_in_range():
                self.cx = self.cx + int(self.shiftdt)
        elif act=="right":  # right
            self.cx = self.cx + int(self.shiftdt)
            if not self.if_in_range():
                self.cx = self.cx - int(self.shiftdt)
        elif act=="rr":  # rotate right
            self.rangle = self.rangle - self.angledt
            if not (self.if_in_range() and abs(self.rangle)<self.maxangle):
                self.rangle = self.rangle + self.angledt
        elif act=="rl":  # rotate left
            self.rangle = self.rangle + self.angledt
            if not (self.if_in_range() and abs(self.rangle)<self.maxangle):
                self.rangle = self.rangle - self.angledt
        elif act=="awu":  # attention width up
            if self.attwid+self.widthstep<=self.maxattwid:
                self.attwid = self.attwid+self.widthstep
        elif act=="awd":  # attention width down
            if self.attwid-self.widthstep>=self.minattwid:
                self.attwid = self.attwid-self.widthstep
        elif act=="ahu":  # attention height up
            if self.atthei+self.heightstep<=self.maxatthei:
                self.atthei = self.atthei+self.heightstep
        elif act=="ahd":  # attention height down
            if self.atthei-self.heightstep>=self.minatthei:
                self.atthei = self.atthei-self.heightstep
        elif act=="su":  # scale up
            self.scaleratio = self.scaleratio*self.scaledt
            if not (self.if_in_range() and self.scaleratio<self.maxscale):
                self.scaleratio = self.scaleratio/self.scaledt
                if_out_of_bound = True
            
        elif act=="sd":  # scale down
            self.scaleratio = self.scaleratio/self.scaledt
            if not (self.if_in_range() and self.scaleratio>self.minscale):
                self.scaleratio = self.scaleratio*self.scaledt
                if_out_of_bound = True
        else:
            pass 
        
        img, iferror = self.outState()
        reward = 0.0 
        if iferror==False:
            reward = self.cal_reward()
        #print(reward)
        self.curiou = self.get_max_iou()
        return img, reward

    def cal_reward(self):
        iou = self.get_max_iou()
        reward = iou - self.curiou
        reward = reward*2.0   # seems in -0.5 - 0.5
        if reward>0.5:
            reward = 0.5
        if reward<-0.5:
            reward = -0.5
        return reward
        
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # the idx is ignored here.
        
        #COMMENT START##### Needs to be commented during Inference ########
        
        self.counter = self.counter + 1
        if (self.counter %  self.sample_num_per_image) == 0:
            self.reset() # 换个图片 Change an image
        
        act = random.choice(self.actlist)   # select a random action
        self.action(act)
        #COMMENT END######################################################
        img, _ = self.outState()
        allreward = self.get_all_reward()
        #x1 = (self.conwid/2-self.attwid/2) / float(self.conwid)
        #x2 = (self.conwid/2+self.attwid/2) / float(self.conwid)
        #y1 = (self.conhei/2-self.atthei/2) / float(self.conhei)
        #y2 = (self.conhei/2+self.atthei/2) / float(self.conhei)
        x1 = (self.conwid/2-self.attwid/2) 
        x2 = (self.conwid/2+self.attwid/2) 
        y1 = (self.conhei/2-self.atthei/2)
        y2 = (self.conhei/2+self.atthei/2) 
        box1 = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        box2 = np.asarray([x1*0.80, y1*0.80, x2*0.80+self.conwid*0.20, y2*0.80+self.conhei*0.20], dtype=np.float32)
        box3 = np.asarray([x1*0.67, y1*0.67, x2*0.67+self.conwid*0.33, y2*0.67+self.conhei*0.33], dtype=np.float32)
        box4 = np.asarray([x1*0.33, y1*0.33, x2*0.33+self.conwid*0.67, y2*0.33+self.conhei*0.67], dtype=np.float32)
        box5 = np.asarray([self.conwid*0.01, self.conhei*0.01, self.conwid*0.99, self.conhei*0.99], dtype=np.float32)
        box6 = np.asarray([x1*0.80+self.conwid*0.5*0.20, y1*0.80+self.conhei*0.5*0.20, x2*0.80+self.conwid*0.5*0.20, y2*0.80+self.conhei*0.5*0.20], dtype=np.float32)
        box7 = np.asarray([x1*0.67+self.conwid*0.5*0.33, y1*0.67+self.conhei*0.5*0.33, x2*0.67+self.conwid*0.5*0.33, y2*0.67+self.conhei*0.5*0.33], dtype=np.float32)
        box = np.concatenate((box1,box2,box3,box4,box5,box6,box7),axis=0)
        return img, box, np.asarray(allreward,dtype=np.float32)


# 用于切图的类 class for cutting images
class dispose:
    def __init__(self,image,debug):
        self.debug=debug
        self.setimage(image)
    def setimage(self,image):    
        if isinstance(image,np.ndarray):
            self.image=image
        else:
            self.image=io.imread(image)      
    def cut(self,cx,cy,theta,w,h):      
        #io.imshow(img)
        img=self.image
        x1=cx-w/2
        y1=cy+h/2
        x2=cx+w/2
        y2=cy+h/2
        x3=cx-w/2
        y3=cy-h/2
        x4=cx+w/2
        y4=cy-h/2
        nx1=math.cos(theta*math.pi/180)*(x1-cx)-math.sin(theta*math.pi/180)*(y1-cy)+cx
        ny1=math.sin(theta*math.pi/180)*(x1-cx)+math.cos(theta*math.pi/180)*(y1-cy)+cy
        nx2=math.cos(theta*math.pi/180)*(x2-cx)-math.sin(theta*math.pi/180)*(y2-cy)+cx
        ny2=math.sin(theta*math.pi/180)*(x2-cx)+math.cos(theta*math.pi/180)*(y2-cy)+cy
        nx3=math.cos(theta*math.pi/180)*(x3-cx)-math.sin(theta*math.pi/180)*(y3-cy)+cx
        ny3=math.sin(theta*math.pi/180)*(x3-cx)+math.cos(theta*math.pi/180)*(y3-cy)+cy
        nx4=math.cos(theta*math.pi/180)*(x4-cx)-math.sin(theta*math.pi/180)*(y4-cy)+cx
        ny4=math.sin(theta*math.pi/180)*(x4-cx)+math.cos(theta*math.pi/180)*(y4-cy)+cy
        minx=round(min(nx1,nx2,nx3,nx4))        
        miny=round(min(ny1,ny2,ny3,ny4))  
        maxx=round(max(nx1,nx2,nx3,nx4))        
        maxy=round(max(ny1,ny2,ny3,ny4))
        (maxr,maxc,dimen)=img.shape
        #print(maxc,maxr)
      #  print(minx,miny,maxx,maxy)
        if cx>=maxc or cx<=0 or cy>=maxr or cy<=0:
            z=np.zeros([10,10,3],dtype=int)
            z[:,:,:]=255
            label=-1
            if self.debug==1:
                 print("the mid point is not in the background picture!")
            return z,label
        elif minx<=0 and miny<=0 and 0<=maxx and maxx<=maxc and 0<=maxy and maxy<=maxr:
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255  
            z[-miny:maxy-miny,-minx:maxx-minx,:]=img[0:maxy,0:maxx,:]
           # print(1)
          #  io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
         #   io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
           # io.imshow(z3)
            label=1
            return z3,label
        elif minx<=0 and  miny>=0 and 0<maxx and maxx<=maxc and 0<maxy and maxy<=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255  
            z[0:maxy-miny,-minx:maxx-minx,:]=img[miny:maxy,0:maxx,:]
         #   print(2)
          #  io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
          #  io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
         #   io.imshow(z3)
            label=1
            return z3,label
        elif minx<=0 and maxy>=maxr and 0<=miny and miny<=maxr and 0<maxx and maxx<=maxc :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[0:maxr-miny,-minx:maxx-minx,:]=img[miny:maxr,0:maxx,:]
          #  print(3)
         #   io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
          #  io.imshow(z3)
            label=1
            return z3,label
        elif maxc>=minx and minx>=0 and miny<=0 and 0<=maxx and maxx<=maxc and 0<=maxy and maxy<=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny,0:maxx-minx,:]=img[0:maxy,minx:maxx,:]
          #  print(4)
          #  io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
          #  io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
         #   io.imshow(z3)
            label=1
            return z3,label
        elif maxc>=minx and minx>=0 and maxy>=maxr and 0<=maxx and maxx<=maxc and 0<miny and miny<=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[0:maxr-miny,0:maxx-minx,:]=img[miny:maxr,minx:maxx,:]
          #  print(5)
         #   io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
         #   io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
         #   io.imshow(z3)
            label=1
            return z3,label
        elif maxx>=maxc and miny<=0 and maxr>=maxy and maxy>=0 and 0<=minx and minx<=maxc :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny,0:maxc-minx,:]=img[0:maxy,minx:maxc,:]
          #  print(6)
         #   io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)
            label=1
            return z3,label
        elif maxc>=minx and minx>=0 and maxr>=miny and miny>=0 and 0<=maxy and maxy<=maxr and maxx>=maxc :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[0:maxy-miny,0:maxc-minx,:]=img[miny:maxy,minx:maxc,:]
          #  print(7)
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
       #     io.imshow(z3)
            label=1
            return z3,label
        elif maxx>=maxc and maxr>=miny and miny>=0 and maxy>=maxr and 0<=minx and minx<=maxc :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[0:maxr-miny,0:maxc-minx,:]=img[miny:maxr,minx:maxc,:]
          #  print(8)
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif minx<=0 and maxx>=maxc and miny<=0 and 0<=maxy<=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny,-minx:maxx-minx-(maxx-maxc),:]=img[0:maxy,:,:]
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif minx<=0 and maxx>=maxc and maxy>=miny>=0 and 0<=maxy<=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[0:maxy-miny,-minx:maxx-minx-(maxx-maxc),:]=img[miny:maxy,:,:]
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif minx<=0 and maxx>=maxc and 0<=miny<=maxr and maxy>=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[0:maxr-miny,-minx:maxx-minx-(maxx-maxc),:]=img[miny:maxr,:,:]
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif minx<=0 and 0<=maxx<=maxc and 0>=miny and maxy>=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny-(maxy-maxr),-minx:maxx-minx,:]=img[0:maxr,0:maxx,:]
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif maxc>=minx>=0 and 0<=maxx<=maxc and 0>=miny and maxy>=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny-(maxy-maxr),0:maxx-minx,:]=img[0:maxr,minx:maxx,:]
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif maxc>=minx>=0 and maxx>=maxc and 0>=miny and maxy>=maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny-(maxy-maxr),0:maxc-minx,:]=img[0:maxr,minx:maxc,:]
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        elif minx<0 and maxx>maxc and miny<0 and maxy>maxr :
            z = np.zeros([maxy-miny,maxx-minx,3], dtype = 'uint8')
            z[:,:,:]=255 
            z[-miny:maxy-miny-(maxy-maxr),-minx:maxx-minx-(maxx-maxc),:]=img[:,:,:]
           # io.imsave('tmp.jpg', z)
           # z1=io.imread('tmp.jpg')
        #    io.imshow(z1)
            z2=transform.rotate(z,-theta,resize=True)
        #    io.imshow(z2)
            (row,col,x)=z2.shape  
            xmin=int(col/2-w/2)
            xmax=int(col/2+w/2)
            ymin=int(row/2-h/2)
            ymax=int(row/2+h/2)
            z3=z2[ymin:ymax,xmin:xmax,:]
        #    io.imshow(z3)     
            label=1
            return z3,label
        else:
             cut=img[miny:maxy,minx:maxx,:]
             img2=transform.rotate(cut,-theta,resize=True)
             (row1,col1,x1)=cut.shape
             (row,col,x)=img2.shape  
             xmin=int(col/2-w/2)
             xmax=int(col/2+w/2)
             ymin=int(row/2-h/2)
             ymax=int(row/2+h/2)
             img3=img2[ymin:ymax,xmin:xmax,:]
       #      io.imshow(img3)
             label=1
             return img3,label

if __name__ == "__main__":  
    #root_image_path = '/home/user/data/ICPR_MTWI_2018/image_1000'
    #root_label_path = '/home/user/data/ICPR_MTWI_2018/txt_1000'
    root_image_path = '/home/user/data/icdar2015/raw'
    root_label_path = '/home/user/data/icdar2015/gt'

    te = text_detection_env(root_image_path,root_label_path)

    print(te.txtboxes[0])
    te.reset()
    print(te.txtboxes[0])
    print(te.get_all_reward())
    img, box, rwd = te[10]
    print(img.shape,box.shape,rwd.shape)
    print(box)

# -*- coding: utf-8 -*-

import  torch
import torchvision.datasets as dset
from PIL import ImageFile,Image
import os,glob
import random,csv
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
import numpy
import torchvision
import visdom

class myDataset(Dataset):
    def __init__(self,root,resize,mode):
        super(myDataset,self).__init__()
        self.root=root
        self.resize=resize
        self.name2lable={}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2lable[name]=len(self.name2lable.keys())
        self.images,self.labels=self.load_csv('images.csv')
        if mode =='train':
            self.images=self.images[:int(len(self.images))]
            self.labels=self.labels[:int(len(self.labels))]
        if mode=='test':
            self.images=self.images[:64]
            self.labels=self.labels[:64]
 
    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images=[]
            n=len(self.name2lable.keys())
            img=[0for i in range(n)]
            for name in self.name2lable.keys():
                img[self.name2lable[name]]=glob.glob(os.path.join(self.root,name,'*.jpg'))
            for i in range(n):
                random.shuffle(img[i])
            ratio=[1,1,1,1,1,1,1,1,1,1]
            for i in range(n):
                img[i]=img[i][:int(len(img[i])*ratio[i])]
                images+=img[i]
            random.shuffle(images)
            with open (os.path.join(self.root,filename),mode='w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                random.shuffle(images)
                for img in images:
                    name=img.split(os.sep)[-2]
                    label=self.name2lable[name]
                    writer.writerow([img,label])
                print('writer',filename)
                
        images,labels=[],[]
        with open(os.path.join(self.root,filename),encoding='utf-8') as f:
            reader=csv.reader(f)
            for row  in reader:
                img, label=row
                label=int(label)
                images.append(img)
                labels.append(label)
        assert len(images)==len(labels)
        return images,labels
        
    def denormalize(self,x_hat):
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
        mean=torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x=x_hat*std+mean
        return  x
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idex):
        img,label=self.images[idex],self.labels[idex]
        tf=transforms.Compose([#cut
                lambda x:Image.open(x).convert('RGB'),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]
                                 )
                ])
        tf1=transforms.Compose([#resize
                lambda x:Image.open(x).convert('RGB'),
                transforms.Resize((self.resize,self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]
                                 )
                ])

        img=tf1(img)   
        label=torch.tensor(label)
        return img,label

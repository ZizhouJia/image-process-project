import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import random
import torch.utils.data as Data
try:
    import cPickle as pickle
except ImportError:
    import pickle

transform_img = transforms.Compose([
                    transforms.CenterCrop(178),
                    transforms.Resize(128),
                    transforms.ToTensor(),
                    ])

class celebadataset(Data.Dataset):
    def __init__(self,img_dir='img_align_celeba',attrpath='list_attr_celeba.txt',identipath='identity_CelebA.txt',transform=transform_img,mode='train',load_data=True):
        self.img_dir=img_dir
        self.attrpath=attrpath
        self.identipath=identipath
        self.transform=transform
        self.mode=mode
        self.load_data=load_data
        self.train=[]
        self.test=[]
        self.attr2idx={}
        self.idx2attr={}


        if(self.load_data==True):
            self.preprocess()
        else:
            print('loading data..')
            loadfile=open('./dataset/processdata.pkl','rb')
            self.train=pickle.load(loadfile)
            self.test=pickle.load(loadfile)
            self.attr2idx=pickle.load(loadfile)
            self.idx2attr=pickle.load(loadfile)
        if(self.mode=='train'):
            self.selected=[]
            for i in range(len(self.train)):
                self.selected.append(i)
        else:
            self.selected=[]
            for i in range(len(self.test)):
                self.selected.append(i)
        self.index=len(self.selected)

    def preprocess(self):
        print("process dataset...")
        alldata=[]
        #identity
        imgidentity=[]
        lines = [line.rstrip() for line in open(self.identipath, 'r')]
        for i, line in enumerate(lines):
            split = line.split()
            id=int(split[1])
            id_onehot=np.zeros((10177,),dtype=np.int)
            id_onehot[id-1]=1
            imgidentity.append(id_onehot)

        #print(max(imgidentity))
        #attribute
        lines = [line.rstrip() for line in open(self.attrpath, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for idx in range(len(self.idx2attr)):
                if(values[idx] == '1'):
                    label.append(1)
                else:
                    label.append(0)

            #image=Image.open(os.path.join(self.img_dir, filename))


            alldata.append([filename,label,imgidentity[i]])

        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(alldata):
            if(i<2000):
                self.test.append(alldata[i])
            else:
                self.train.append(alldata[i])
        print("finished processing dataset.")
        #print(self.train[10:12])
        #print(self.test[10:12])


    def __getitem__(self,index):
        if(self.index<10):
            if(self.mode=='train'):
                self.selected=[]
                for i in range(len(self.train)):
                    self.selected.append(i)
            else:
                self.selected=[]
                for i in range(len(self.test)):
                    self.selected.append(i)
            self.index=len(self.selected)
        self.index-=2
        select_list_len=len(self.selected)

        list_index=random.randint(0, select_list_len-1)
        img1_index=self.selected[list_index]
        self.selected.remove(img1_index)

        list_index=random.randint(0, select_list_len-2)
        img2_index=self.selected[list_index]
        self.selected.remove(img2_index)

        if(self.mode=='train'):
            img1name,img1_label,img1_id=self.train[img1_index]
            img2name,img2_label,img2_id=self.train[img2_index]
        else:
            img1name,img1_label,img1_id=self.test[img1_index]
            img2name,img2_label,img2_id=self.test[img2_index]

        img1=Image.open(os.path.join(self.img_dir, img1name))
        img2=Image.open(os.path.join(self.img_dir, img2name))
        if self.transform is not None:
            img1=self.transform(img1)
            img2=self.transform(img2)

        return img1*2-1,img2*2-1,torch.Tensor(img1_id),torch.Tensor(img2_id),torch.Tensor(img1_label),torch.Tensor(img2_label)


    def __len__(self):
        if(self.mode=='train'):
            return len(self.train)/2-10
        else:
            return len(self.test)/2-10


if __name__ == '__main__':
    train_dataset=celebadataset(img_dir='img_align_celeba',attrpath='list_attr_celeba.txt',identipath='identity_CelebA.txt',transform=transform_img,mode='train',load_data=True)
    train_dataloader=Data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0)
    test_dataset=celebadataset(img_dir='img_align_celeba',attrpath='list_attr_celeba.txt',identipath='identity_CelebA.txt',transform=transform_img,mode='test',load_data=False)
    test_dataloader=Data.DataLoader(dataset=test_dataset,batch_size=32,shuffle=True,num_workers=0)
    for i,(img1,label1,id1,img2,label2,id2) in enumerate(train_dataloader):
        print(img1,label1,id1,img2,label2,id2)
        print("train")
        break
    for i,(img1,label1,id1,img2,label2,id2) in enumerate(test_dataloader):
        print(img1,label1,id1,img2,label2,id2)
        print("test")
        break

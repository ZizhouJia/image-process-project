
from model import GAN_module_face
from dataset import celebadataset
import torch.utils.data as Data
from utils.common_tools import *
from utils.data_provider import *
import GAN_solver

models=[]
models.append(GAN_module_face.Encoder())
models.append(GAN_module_face.Decoder())
models.append(GAN_module_face.Discriminator())

for i in range(0,len(models)):
    models[i]=nn.DataParallel(models[i],device_ids=[0,1])

lrs=[0.0001,0.0001,0.0001]

optimizers=generate_optimizers(models,lrs,optimizer_type="adam",weight_decay=0.001)
function=weights_init(init_type='xavier')
solver=GAN_solver.GAN_solver(models,'celeba',optimizers,save_path="checkpoints")
solver.init_models(function)
solver.cuda()
train_dataset=celebadataset.celebadataset(img_dir='./dataset/img_align_celeba',attrpath='./dataset/list_attr_celeba.txt',identipath='./dataset/identity_CelebA.txt',transform=celebadataset.transform_img,mode='train',load_data=True)
train_provider_dataset=celebadataset.celebadataset(img_dir='./dataset/img_align_celeba',attrpath='./dataset/list_attr_celeba.txt',identipath='./dataset/identity_CelebA.txt',transform=celebadataset.transform_img,mode='train',load_data=True)

train_dataprovider=data_provider(train_provider_dataset ,batch_size=16, is_cuda=False)
train_dataloader=Data.DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0)
solver.train_loop(train_dataloader,train_dataprovider,None,epochs=100)

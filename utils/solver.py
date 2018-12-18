import os
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision

def get_time_string():
    dt=datetime.now()
    return dt.strftime("%Y%m%d%H%M")

class solver(object):
    def __init__(self,models,model_name,optmizers,save_path="checkpoints"):
        self.models=models
        self.model_name=model_name
        self.save_path=save_path
        self.time_string=get_time_string()
        self.optimizers=optmizers
        self.writer=SummaryWriter()

    def get_models(self):
        return self.models

    def init_models(self,init_func):
        for model in self.models:
            init_func(model)

    def cuda(self):
        for i in range(0,len(self.models)):
            self.models[i]=self.models[i].cuda()

    def parallel(self,device_ids=[0,1]):
        for i in range(0,len(self.models)):
            self.models[i]=nn.DataParallel(self.models[i],device_ids)
        #if(self.optimizers is not None):
        for i in range(0,len(self.optimizers)):
            self.optimizers[i]=nn.DataParallel(self.optimizers[i],device_ids)



    def cpu(self):
        for i in range(0,len(self.models)):
            self.models[i]=self.models[i].cpu()

    def set_optimizers(self,optimizers):
        self.optimizers=optimizers

    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def train_mode(self):
        for model in self.models:
            model.train()

    def eval_mode(self):
        for model in self.models:
            model.eval()

    def write_log(self,loss,index):
        for key in loss:
            self.writer.add_scalar("saclar/"+key,loss[key],index)

    def output_loss(self,loss,epoch,iteration):
        print("in epoch %d iteration %d "%(epoch,iteration))
        print(loss)

    def write_log_image(self,image_dict,index):
        for key in image_dict:
            self.writer.add_image("image",torchvision.utils.make_grid(image_dict[key]),index)


    def save_models(self,epoch=-1):
        path=self.save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path=os.path.join(path,self.model_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        path=os.path.join(path,self.time_string)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(epoch!=-1):
            path=os.path.join(path,str(epoch))
            if(not os.path.exists(path)):
                os.mkdir(path)

        file_name="model"
        for i in range(0,len(self.models)):
            torch.save(self.models[i].state_dict(),os.path.join(path,file_name+"-"+str(i)+".pkl"))

        print("the model "+self.model_name+" has already been saved "+self.time_string)

    def restore_models(self,save_path,model_name,time_string,epoch=-1):
        path=save_path
        path=os.path.join(path,model_name)
        path=os.path.join(path,time_string)
        if(epoch!=-1):
            path=os.path.join(path,epoch)
        file_name="model"

        for i in range(0,len(self.models)):
            self.models[i].load_state_dict(torch.load(os.path.join(path,file_name+"-"+str(i)+".pkl")))

    def update_optimizers(self,epoch):
        pass

    def train_one_batch(self,input_dict):
        raise NotImplementedError

    def test_one_batch(self,input_dict):
        raise NotImplementedError

    def train_loop(self,dataloader,param_dict,epochs=100):
        raise NotImplementedError

    def test_all(self,dataloader):
        pass

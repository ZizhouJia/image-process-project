import utils.solver as solver
import utils.loss_function as loss_function
import torch.nn.functional as F
import numpy as np
import os
import cv2
import torch

class GAN_solver(solver.solver):
    def __init__(self,models,model_name,optimizers,save_path="checkpoints"):
        super(GAN_solver,self).__init__(models,model_name,optimizers)
        self.encoder,self.decoder,self.discriminator,self.classifier=self.models
        self.e_opt,self.d_opt,self.dis_opt,self.cls_opt=optimizers

    def train_one_batch(self,input_dict):
        x1=input_dict["x1"]
        x2=input_dict["x2"]

        id1=input_dict["id1"]
        id2=input_dict["id2"]
        emotion1=input_dict["emotion1"]
        emotion2=input_dict["emotion2"]

        d_x=input_dict["d_x"]
        d_id=input_dict["d_id"]
        d_emotion=input_dict["d_emotion"]

        epoch=input_dict["epoch"]

        iteration=input_dict["iteration"]

        feature_id1,feature_emotion1=self.encoder(x1)
        feature_id2,feature_emotion2=self.encoder(x2)

        fake_x1=self.decoder(feature_id1,feature_emotion2)
        fake_x2=self.decoder(feature_id2,feature_emotion1)

        rct_x1=self.decoder(feature_id1,feature_emotion1)
        rct_x2=self.decoder(feature_id2,feature_emotion2)

        feature_id1_fake,feature_emotion2_fake=self.encoder(fake_x1)
        feature_id2_fake,feature_emotion1_fake=self.encoder(fake_x2)

        cyc_x1=self.decoder(feature_id1_fake,feature_emotion1_fake)
        cyc_x2=self.decoder(feature_id2_fake,feature_emotion2_fake)

        real_emo=self.classifier(d_x)
        real_emo_label=d_emotion
        fake_emo=self.classifier(torch.cat((fake_x1,fake_x2),0))
        fake_emo_label=torch.cat((emotion2,emotion1),0)

        dis_real_x=self.discriminator(torch.cat((x1,x2),0))
        dis_fake_x=self.discriminator(torch.cat((fake_x1,fake_x2,rct_x1,rct_x2),0))


        d_real_loss=loss_function.D_real_loss(dis_real_x,"wgan")
        d_fake_loss=loss_function.D_fake_loss(dis_fake_x,"wgan")
        d_loss=d_real_loss+d_fake_loss

        g_loss=loss_function.G_fake_loss(dis_fake_x,"wgan")

        rct_image_loss=loss_function.l1_loss(x1,rct_x1)+loss_function.l1_loss(x2,rct_x2)
        rct_image_loss=rct_image_loss/2

        cyc_rct_image_loss=loss_function.l1_loss(x1,cyc_x1)+loss_function.l1_loss(x2,cyc_x2)
        cyc_rct_image_loss=cyc_rct_image_loss/2

        d_classify_emo_loss=F.binary_cross_entropy_with_logits(real_emo,real_emo_label,size_average=False)/real_emo.size()[0]

        g_classify_emo_loss=F.binary_cross_entropy_with_logits(fake_emo,fake_emo_label,0),size_average=False)/fake_emo.size()[0]

        #compute loss for gradient gradient_penalty
        alpha = torch.rand(x1.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * x1.data + (1 - alpha) * fake_x1.data).requires_grad_(True)
        out_src= self.discriminator(x_hat)
        d_loss_gp = loss_function.gradient_penalty(out_src, x_hat)

        total_g_loss=g_loss+\
        10*cyc_rct_image_loss+ \
        10*rct_image_loss+ \
        1.0*g_classify_emo_loss

        total_d_loss=d_loss+d_classify_emo_loss+10.0*d_loss_gp
        retain=False
        
        if(iteration%3==0):
            retain=True
        total_d_loss.backward(retain_graph=retain)
        self.dis_opt.step()
        self.cls_opt.step()
        self.zero_grad_for_all()

        if(iteration%3==0):
            total_g_loss.backward()
            self.e_opt.step()
            self.d_opt.step()
            self.zero_grad_for_all()


        loss={}
        loss["g_loss"]=g_loss.detach().cpu().item()
        loss["rct_image_loss"]=rct_image_loss.detach().cpu().item()
        loss["cyc_rct_image_loss"]=cyc_rct_image_loss.detach().cpu().item()
        #loss["rct_id_loss"]=rct_feature_id_loss.detach().cpu().item()
        #loss["rct_feature_loss"]=rct_feature_emo_loss.detach().cpu().item()
        #loss["g_classify_id_loss"]=g_classify_id_loss.detach().cpu().item()
        loss["g_classify_emo_loss"]=g_classify_emo_loss.detach().cpu().item()
        loss["d_loss"]=d_loss.detach().cpu().item()
        #loss["d_classify_id_loss"]=d_classify_id_loss.detach().cpu().item()
        loss["d_Classify_emo_loss"]=d_classify_emo_loss.detach().cpu().item()
        loss["total_g_loss"]=total_g_loss.detach().cpu().item()
        loss["total_d_loss"]=total_d_loss.detach().cpu().item()-loss["d_classify_emo_loss"]
        loss["d_loss_gp"]=d_loss_gp.detach().cpu().item()
        return loss

    def test_one_batch(self,input_dict):
        x1=input_dict["x1"].cuda()
        x2=input_dict["x2"].cuda()
        batch_size=x1.size()[0]
        feature_id1,feature_emotion1=self.encoder(x1)
        feature_id2,feature_emotion2=self.encoder(x2)

        fake_x1=self.decoder(feature_id1,feature_emotion1)
        fake_x2=self.decoder(feature_id1,feature_emotion2)

        x1=(x1+1)/2
        fake_x1=(fake_x1+1)/2
        fake_x2=(fake_x2+1)/2
        x2=(x2+1)/2

        # x1=x1.cpu().detach().numpy()
        # x2=x2.cpu().detach().numpy()
        # fake_x1=fake_x1.cpu().detach().numpy()
        # fake_x2=fake_x2.cpu().detach().numpy()
        out_images=torch.zeros((batch_size,3,x1.shape[2],x1.shape[3]*4))
        out_images[:,:,0:x1.shape[2],0:x1.shape[3]]=x1.cpu()
        out_images[:,:,0:x1.shape[2],x1.shape[3]:x1.shape[3]*2]=fake_x1.cpu()
        out_images[:,:,0:x1.shape[2],x1.shape[3]*2:x1.shape[3]*3]=fake_x2.cpu()
        out_images[:,:,0:x1.shape[2],x1.shape[3]*3:x1.shape[3]*4]=x2.cpu()
        return out_images

    def train_loop(self,param_dict,epochs=100):
        iteration_count=0
        dataloader=param_dict["loader"]
        d_dataprovider=param_dict["provider"]
        for i in range(0,epochs):
            for step,(x1,x2,id1,id2,emotion1,emotion2) in enumerate(dataloader):
                d_x1,d_x2,d_id1,d_id2,d_emotion1,d_emotion2=d_dataprovider.next()
                input_dict={}
                input_dict["x1"]=x1.cuda()
                input_dict["x2"]=x2.cuda()
                input_dict["id1"]=id1.cuda()
                input_dict["id2"]=id2.cuda()
                input_dict["emotion1"]=emotion1.cuda()
                input_dict["emotion2"]=emotion2.cuda()
                input_dict["d_x"]=torch.cat((d_x1.cuda(),d_x2.cuda()),0)
                input_dict["d_id"]=torch.cat((d_id1.cuda(),d_id2.cuda()),0)
                input_dict["d_emotion"]=torch.cat((d_emotion1.cuda(),d_emotion2.cuda()),0)
                input_dict["epoch"]=i
                input_dict["iteration"]=iteration_count
                loss=self.train_one_batch(input_dict)
                iteration_count+=1
                if(iteration_count%1==0):
                    self.write_log(loss,iteration_count)
                    self.output_loss(loss,i,iteration_count)
                if(iteration_count%10==0):
                    out_images=self.test_one_batch(input_dict)
                    images={}
                    images["image"]=out_images
                    self.write_log_image(images,int(iteration_count/100))
            if(i%1==0):
                self.save_models(epoch=i)

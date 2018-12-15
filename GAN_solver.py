import utils.solver.solver as solver
import utils.loss_function as loss_function
import torch.nn.functional as F

class GAN_solver(solver):
    def __init__(self,models,model_name,optimizers,save_path="checkpoints"):
        super(GAN_solver,self).__init__(models,model_name,optimizers)
        self.encoder,self.decoder,self.discriminator=self.models
        self.e_opt,self.d_opt,self.dis_opt=optimizers

    def train_one_batch(self,input_dict):
        x1=input_dict["x1"]
        x2=input_dict["x2"]

        id1=input_dict["id1"]
        id2=input_dict["id2"]
        emotion1=input_dict["emotion1"]
        emotion2=input_dict["emothon2"]

        feature_id1,feature_emotion1=self.encoder(x1)
        feature_id2,feature_emotion2=self.encoder(x2)

        fake_x1=self.decoder(feature_id1,feature_emotion2)
        fake_x2=self.decoder(feature_id2,feature_emotion1)

        rct_x1=self.decoder(feature_id1,feature_emotion1)
        rct_x2=self.decoder(feature_id2,feature_emotion2)

        dis_fake_x1,dis_f_id1,dis_f_emo2=self.discriminator(fake_x1)
        dis_fake_x2,dis_f_id2,dis_f_emo1=self.discriminator(fake_x2)
        dis_real_x1,dis_r_id1,dis_r_emo1=self.discriminator(x1)
        dis_real_x2,dis_r_id2,dis_r_emo2=self.discriminator(x2)

        feature_id1_fake,feature_emotion2_fake=self.encoder(fake_x1)
        feature_id2_fake,feature_emotion1_fake=self.encoder(fake_x2)

        cyc_x1=self.decoder(feature_id1_fake,feature_emotion1_fake)
        cyc_x2=self.decoder(feature_id2_fake,feature_emotion2_fake)

        d_real_loss=(loss_function.D_real_loss(dis_real_x1,"lsgan")+loss_function.D_real_loss(dis_real_x2,"lsgan"))
        d_fake_loss=(loss_function.D_fake_loss(dis_fake_x1,"lsgan")+loss_function.D_fake_loss(dis_fake_x2,"lsgan"))
        d_loss=d_real_loss+d_fake_loss

        g_loss=(loss_function.G_fake_loss(dis_fake_x1,"lsgan")+loss_function.G_fake_loss(dis_fake_x2,"lsgan"))/2

        rct_image_loss=loss_function.l1_loss(x1,rct_x1)+loss_function.l1_loss(x2,rct_x2)
        rct_image_loss=rct_image_loss/2

        cyc_rct_image_loss=loss_function.l1_loss(x1,cyc_x1)+loss_function(x2,cyc_x2)
        cyc_rct_image_loss=cyc_rct_image_loss/2

        rct_feature_id_loss=loss_function.l1_loss(feature_id1,feature_id1_fake)+loss_function.l1_loss(feature_id2,feature_id2_fake)
        rct_feature_id_loss=rct_feature_id_loss/2

        rct_feature_emo_loss=loss_function.l1_loss(feature_emotion1,feature_emotion1_fake)+loss_function.l1_loss(feature_emotion2,feature_emotion2_fake)
        rct_feature_emo_loss=rct_feature_emo_loss/2

        d_classify_id_loss=F.binary_cross_entropy_with_logits(dis_r_id1,id1,size_average=False)/dis_r_id1.size()[0]+F.binary_cross_entropy_with_logits(dis_r_id2,id2,size_average=False)/dis_r_id2.size()[0]
        d_classify_id_loss=d_classify_id_loss/2

        d_classify_emo_loss=F.binary_cross_entropy_with_logits(dis_r_emo1,emotion1,size_average=False)/dis_r_emo1.size()[0]+F.binary_cross_entropy_with_logits(dis_r_emo2,emotion2,size_average=False)/dis_r_emo2.size()[0]
        d_classify_emo_loss=d_classify_emo_loss/2

        g_classify_id_loss=F.binary_cross_entropy_with_logits(dis_f_id1,id1,size_average=False)/dis_f_id1.size()[0]+F.binary_cross_entropy_with_logits(dis_f_id2,id2,size_average=False)/dis_f_id2.size()[0]
        g_classify_id_loss=g_classify_id_loss/2

        g_classify_emo_loss=F.binary_cross_entropy_with_logits(dis_f_emo1,emotion1,size_average=False)/dis_f_emo1.size()[0]+F.binary_cross_entropy_with_logits(dis_f_emo2,emotion2,size_average=False)/dis_f_emo2.size()[0]
        g_classify_emo_loss=g_classify_emo_loss/2

        total_g_loss=g_loss+\
        1.0*rct_feature_id_loss+ \
        1.0*rct_feature_emo_loss+ \
        10*cyc_rct_image_loss+ \
        10*rct_image_loss+ \
        1.0*g_classify_id_loss+ \
        1.0*g_classify_emo_loss+ \

        total_d_loss=d_loss+ \
        1.0*d_classify_id_loss+ \
        1.0*d_classify_emo_loss

        total_d_loss.backward(retain_graph=True)
        self.dis_opt.step()
        self.zero_grad_for_all()

        total_g_loss.backward(retain_graph=True)
        self.e_opt.step()
        self.d_opt.step()
        self.zero_grad_for_all()
        loss={}
        loss["g_loss"]=g_loss.detach().cpu().item()
        loss["rct_image_loss"]=rct_image_loss.detach().cpu().item()
        loss["cyc_rct_image_loss"]=cyc_rct_image_loss.detach().cpu().item()
        loss["rct_id_loss"]=rct_feature_id_loss.detach().cpu().item()
        loss["rct_feature_loss"]=rct_feature_emo_loss.detach().cpu().item()
        loss["g_classify_id_loss"]=g_classify_id_loss.detach().cpu().item()
        loss["g_classify_emo_loss"]=g_classify_emo_loss.detach().cpu().item()
        loss["d_loss"]=d_loss.detach().cpu().item()
        loss["d_classify_id_loss"]=d_classify_id_loss.detach().cpu().item()
        loss["d_Classify_emo_loss"]=d_classify_emo_loss.detach().cpu().item()
        loss["total_g_loss"]=total_g_loss.detach().cpu().item()
        loss["total_d_loss"]=total_d_loss.detach().cpu().item()
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
        fake_x1=(fake_x1)/2
        fake_x2=(fake_x2)/2
        x2=(x2+1)/2

        x1=x1.permute(0,2,3,1).cpu().detach().numpy()
        x2=x2.permute(0,2,3,1).cpu().detach().numpy()
        fake_x1=fake_x1.permute(0,2,3,1).cpu().detach().numpy()
        fake_x2=fake_x2.permute(0,2,3,1).cpu().detach().numpy()

        for i in range(0,batch_size):
            image=np.zeros(x1.shape[2],x1.shape[3]*4,3)
            image[0:x1.shape[2],0:x1.shape[3],3]=x1[i,:,:,:]
            image[0:x1.shape[2],x1.shape[3]:x1.shape[3]*2,3]=fake_x1[i,:,:,:]
            image[0:x1.shape[2],x1.shape[3]*2:x1.shape[3]*3,3]=fake_x2[i,:,:,:]
            image[0:x1.shape[2],x1.shape[3]*3:x1.shape[3]*4,3]=x2[i,:,:,:]
            image=image*255
            image=image.astype(np.int32)
            cv2.imwrite(os.path.join("test_output","test_"+str(i)+".jpg"),image)

    def train_loop(self,dataloader,param_dict,epochs=100):
        iteration_count=0
        for i in range(0,epochs):
            for step,(x1,x2,id1,id2,emotion1,emothon2) in enumerate(dataloader):
                input_dict={}
                input_dict["x1"]=x1.cuda()
                input_dict["x2"]=x2.cuda()
                input_dict["id1"]=id1.cuda()
                input_dict["id2"]=id2.cuda()
                input_dict["emothon1"]=emothon1.cuda()
                input_dict["emothon2"]=emothon2.cuda()
                loss=self.train_one_batch(input_dict)
                iteration_count+=1
                if(iteration_count%100==0):
                    self.write_log(loss,iteration_count)
                    self.output_loss(loss,i,iteration_count)
    

import cv2
import numpy as np
import torch
import GAN_solver
from model import GAN_module_face
from utils.common_tools import *


testimg_emo=["test1.jpg","test2.jpg","test3.jpg","test4.jpg","test5.jpg"]
testimg_id=["test6.jpg","test7.jpg","test8.jpg","test9.jpg","test10.jpg"]
output_image_matrix=np.zeros((128*6,128*6,3))
output_image_name="output.jpg"
testimg_path="testimg/"

for i in range(0,len(testimg_emo)):
    image=cv2.resize(cv2.imread(testimg_path+testimg_emo[i])[20:-20,:,:],(128,128)).astype(np.float32)/255.0
    testimg_emo[i]=image.transpose([2,0,1])
    output_image_matrix[0:128,(i+1)*128:(i+2)*128,:]=image
    testimg_emo[i]=(testimg_emo[i]+1)/2

for i in range(0,len(testimg_id)):
    image=cv2.resize(cv2.imread(testimg_path+testimg_id[i])[20:-20,:,:],(128,128)).astype(np.float32)/255.0
    testimg_id[i]=image.transpose([2,0,1])
    output_image_matrix[(i+1)*128:(i+2)*128,0:128,:]=image
    testimg_id[i]=(testimg_id[i]+1)/2

models=[]
models.append(GAN_module_face.Encoder())
models.append(GAN_module_face.Decoder())
models.append(GAN_module_face.Discriminator())
models.append(GAN_module_face.Classifier())
for i in range(0,len(models)):
    models[i]=nn.DataParallel(models[i],device_ids=[0,1])
lrs=[0.0001,0.0001,0.0001,0.001]

optimizers=generate_optimizers(models,lrs,optimizer_type="adam",weight_decay=0.0001)

solver=GAN_solver.GAN_solver(models,'celeba',optimizers,save_path="checkpoints")

for i in range(52):
    epoch=str(i)
    solver.restore_models("checkpoints","celeba","201812302253",epoch=epoch)
    solver.cuda()
    solver.eval_mode()

    for i in range(0,len(testimg_id)):
        image_id=torch.Tensor(testimg_id[i]).unsqueeze(0)
        for j in range(0,len(testimg_emo)):
            image_emo=torch.Tensor(testimg_emo[j]).unsqueeze(0)
            input_dict={}
            input_dict["x1"]=image_id
            input_dict["x2"]=image_emo
            images=solver.test_one_batch(input_dict).detach().cpu().numpy().transpose([0,2,3,1]).squeeze()
            output_image_matrix[(i+1)*128:(i+2)*128,(j+1)*128:(j+2)*128,:]=images[:,128*2:128*3,:]

    output_tmp=(output_image_matrix*255.0).astype(np.uint8)

    cv2.imwrite(testimg_path+epoch+output_image_name,output_tmp)

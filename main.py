import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import pickle
import os
import datetime
import torchvision
import torchvision.utils as vutils
import visdom

from model import *
from pytorch_ssim import *
from loss import *
from dataloader import *
from misc import *
from tqdm import tqdm
import io
import sys

img_size=128  #the size of the input image
batch_size_train=16  #batchsize during training
batch_size_test=8 #batchsize during testing
num_age=10  #the number of age
test_photo_num=8 



data=myDataset('./data_all_part',img_size,'train') #dataloader
dataloader=DataLoader(data,batch_size=batch_size_train,shuffle=True)
data_test=myDataset('./data_all_part',img_size,'test')
dataloader_test=DataLoader(data_test,batch_size=batch_size_test)
cuda = torch.cuda.is_available() #cuda is available?
print("cuda: %s"%cuda_is_available)
# set cuda
net_E = Res_Encoder().cuda()
net_D_img = Dimg().cuda()
net_D_z  = Dz().cuda()
net_G = Generator().cuda()
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).cuda()

# weight initialization
net_E.apply(weights_init)
net_D_img.apply(weights_init)
net_D_z.apply(weights_init)
net_G.apply(weights_init)

# set optimizer
optimizer_E = optim.Adam(net_E.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D_z = optim.Adam(net_D_z.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D_img = optim.Adam(net_D_img.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_G = optim.Adam(net_G.parameters(),lr=0.0002,betas=(0.5,0.999))


# build criterions to calculate loss
BCE = nn.BCELoss().cuda()
L1  = nn.L1Loss().cuda()
CE = nn.CrossEntropyLoss().cuda()
MSE = nn.MSELoss().cuda()
SSIM_loss=SSIM().cuda()

# make label for the generated image
out_l = -torch.ones(num_age*test_photo_num*num_age).view(test_photo_num*num_age,num_age)
for i,l in enumerate(out_l):
    l[i//test_photo_num] = 1
out_l_v = Variable(out_l)
out_l_v = out_l_v.cuda()

#the file path of the result
result='./result'
if not os.path.exists(result):
    os.mkdir(result)

# the number of iterations
niter=100 

# the visualization of the result 
viz=visdom.Visdom() 
step=0
for epoch in range(niter):
    for i,(img_data,img_label) in enumerate(dataloader):
        if i%10==0:
            print("epoch:%s,%s"%(epoch,i))    
        img_age = img_label
        
        #visilization of the input image 
        viz.images(img_data,win='img',nrow=8,opts=dict(title='image'))
        viz.text(str(img_age.numpy()),win='age',opts=dict(title='age'))
        viz.text(str(img_gender.numpy()),win='gender',opts=dict(title='gender'))
        img_data_v = Variable(img_data)
        img_age_v = Variable(img_age).view(-1,1)
        img_data_v = img_data_v.cuda()
        img_age_v = img_age_v.cuda()

        # make one hot encoding version of label
        batchSize = img_data_v.size(0)
        age_ohe = one_hot(img_age,batchSize,n_l,use_cuda)

        # prior distribution z_star, real_label, fake_label
        z_star = Variable(torch.FloatTensor(batchSize*n_z).uniform_(-1,1)).view(batchSize,n_z)
        real_label = Variable(torch.ones(batchSize).fill_(1)).view(-1,1) 
        fake_label = Variable(torch.ones(batchSize).fill_(0)).view(-1,1) 
        z_star, real_label, fake_label = z_star.cuda(),real_label.cuda(),fake_label.cuda()

        # loss function
        netE.zero_grad()
        netG.zero_grad()

        #traing the E,G
        # ssim loss 
        z = netE(img_data_v)
        reconst = netG(z,age_ohe) 
        ssim_loss=-SSIM_loss(reconst,img_data_v)

        # VGG loss
        img_fm,_ = feature_extractor(img_data_v)
        img_fm_v = Variable(img_fm)
        reconst_fm, _ = feature_extractor(reconst)
        reconst_fm_v = Variable(reconst_fm)
        VGG_L1_loss = L1(img_fm_v, reconst_fm_v)

        # D_z loss 
        Dz = netD_z(z)
        Ez_loss = BCE(Dz,real_label)

        # D_img loss
        z = netE(img_data_v)
        reconst = netG(z,age_ohe,img_gender_v)
        D_reconst,_ = netD_img(reconst,age_ohe.view(batchSize,n_l,1,1))
        G_img_loss = BCE(D_reconst,real_label)

        # TV loss
        reconst = netG(z.detach(),age_ohe)
        G_tv_loss = TV_LOSS(reconst,img_size=img_size)
        EG_loss = a1*ssim_loss + a2*G_img_loss + a3*Ez_loss + a4*G_tv_loss + a5*VGG_L1_loss # a1~5 is the parameters
        EG_loss.backward()
        optimizerE.step()
        optimizerG.step()

        ## train D_z D_age D_img
        netD_z.zero_grad()
        netD_img.zero_grad()
        Dz_prior = netD_z(z_star)
        Dz = netD_z(z.detach())
        Dz_loss = BCE(Dz_prior,real_label)+BCE(Dz,fake_label)
        D_img,D_age_real = netD_img(img_data_v,age_ohe.view(batchSize,n_l,1,1))
        D_reconst,D_age_reconst = netD_img(reconst.detach(),age_ohe.view(batchSize,n_l,1,1))
        D_loss = BCE(D_img,real_label)+BCE(D_reconst,fake_label)+BCE(D_age_real,age_ohe)+BCE(D_age_reconst,age_ohe)
        Dz_loss.backward()
        optimizerD_z.step()
        D_loss.backward()
        optimizerD_img.step()

        if i==0:
            print("z",z.shape)
            print("D_reconst",D_reconst.shape)
            print("Dz",Dz.shape)
            print("reconst",reconst.shape)
            
       viz.line(Y=torch.Tensor([ssim_loss.item()]),X=torch.Tensor([step]),win='ssim_loss',update='append',opts=dict(title='ssim_loss'))
       viz.line(Y=torch.Tensor([VGG_L1_loss.item()]),X=torch.Tensor([step]),win='VGG_L1_loss',update='append',opts=dict(title='VGG_L1_loss'))
       viz.line(Y=torch.Tensor([G_img_loss.item()]),X=torch.Tensor([step]),win='G_img_loss',update='append',opts=dict(title='G_img_loss'))
       viz.line(Y=torch.Tensor([Ez_loss.item()]),X=torch.Tensor([step]),win='Ez_loss',update='append',opts=dict(title='Ez_loss'))
       viz.line(Y=torch.Tensor([G_tv_loss.item()]),X=torch.Tensor([step]),win='G_tv_loss',update='append',opts=dict(title='G_tv_loss'))
       viz.line(Y=torch.Tensor([EG_loss.item()]),X=torch.Tensor([step]),win='EG_loss',update='append',opts=dict(title='EG_loss'))
       viz.line(Y=torch.Tensor([Dz_loss.item()]),X=torch.Tensor([step]),win='Dz_loss',update='append',opts=dict(title='Dz_loss'))
       viz.line(Y=torch.Tensor([D_loss.item()]),X=torch.Tensor([step]),win='D_loss',update='append',opts=dict(title='D_loss'))
       step=step+1
    
    i=0
    #test
    for img_data_test, img_label_test in tqdm(dataloader_test):
        fixed_noise = img_data_test[:8].repeat(num_age,1,1,1)
        fixed_img_v = Variable(fixed_noise)
        fixed_img_v = fixed_img_v.cuda()
        fixed_z = netE(fixed_img_v)
        fixed_fake = netG(fixed_z,fixed_l_v) 
        vutils.save_image(fixed_fake.data,
                    '%s/reconst_epoch%03d_%03d.png' % (outf,epoch+1,i),
                    normalize=True)
        i=i+1

    #save
    if epoch%10==0:
        torch.save(netE.state_dict(),"%s/netE_%03d.pth"%(outf,epoch+1))
        torch.save(netG.state_dict(),"%s/netG_%03d.pth"%(outf,epoch+1))
        torch.save(netD_img.state_dict(),"%s/netD_img_%03d.pth"%(outf,epoch+1))
        torch.save(netD_z.state_dict(),"%s/netD_z_%03d.pth"%(outf,epoch+1))



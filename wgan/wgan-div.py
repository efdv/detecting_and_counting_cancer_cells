# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:06:11 2023

@author: edgar
"""

import argparse
import os
import numpy as np
import math
import sys
import cv2
import glob

#import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder
from torchvision.io import read_image 
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

liststages = ['Anaphase']
# liststages = ['Prophase', 'Metaphase', 'Anaphase', 'Telophase']
# adamlist = [0.0001, 0.0001, 0.0001, 0.00001]
adamlist = [0.0001]
# n_criticlist = [10, 4, 4, 4]
n_criticlist = [4]

for stage in range(len(liststages)):

    namedir = liststages[stage]+'_wgandiv'
    os.makedirs(namedir, exist_ok=True)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=adamlist[stage], help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.01, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=n_criticlist[stage] , help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)
    
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    
    cuda = True if torch.cuda.is_available() else False
    
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
    
            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
    
            self.model = nn.Sequential(
                *block(opt.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )
    
        def forward(self, z):
            img = self.model(z)
            img = img.view(img.shape[0], *img_shape)
            return img
    
    
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
    
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
            )
    
        def forward(self, img):
            img_flat = img.view(img.shape[0], -1)
            validity = self.model(img_flat)
            return validity
    
    
    k = 2
    p = 6
    
    
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
    
    # ------------------------------ Configure data loader ---------------------------------------
    # path = "../../../datasets/cellcycle_dataset_ch4/"+liststages[stage]+'/'
    # dirfile =  glob.glob(path+'/*')
    # dataset = []
    # for ipath in dirfile:
    #     # I = cv2.imread(ipath)
    #     I = read_image(ipath)
    #     I = (I.float() /127.5) - 1
    #     # I[:,:,0] = (I[:,:,0]/127.5) - 1
    #     # I[:,:,1] = (I[:,:,1]/127.5) - 1
    #     # I[:,:,2] = (I[:,:,2]/127.5) - 1
    #     # I = cv2.resize(I, (64, 64))
    #     #I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    #     # I = np.transpose(I, (2, 0, 1))
    #     dataset.append(I)
     
    # class CustomDataset(Dataset):
    #     def __init__(self, data):
    #         self.data = data
    
    #     def __len__(self):
    #         return len(self.data)
    
    #     def __getitem__(self, idx):
    #         return self.data[idx]
       
    # custom_dataset = CustomDataset(dataset)
    
    # batch_size = opt.batch_size  # Tamaño del lote
    # dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    # dataloader = DataLoader(custom_dataset, batch_size=opt.batch_size, shuffle=True)
    #------------------------------------ fin del dataloader ----------------------------------------------
    
    transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
    ])
    
    # Ruta al directorio que contiene tus imágenes
    # data_path = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/cellcycle_dataset_ch4/"
    data_path = "E:/OneDrive - Universidad de Guanajuato/EF-Duque-Vazquez-Doctorado/datasets/cellcycle_dataset_ch4/wgan-div/Anaphase/"    
    # Crear un objeto Dataset utilizando ImageFolder
    dataset = datasets.ImageFolder(root=data_path, transform=transform, target_transform=lambda x: 0 if x == dataset.class_to_idx[liststages[stage]] else -1) 
    # Definir el DataLoader para cargar lotes de datos
    batch_size = opt.batch_size  # Tamaño del lote
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # ----------
    #  Training
    # ----------
    
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs,_) in enumerate(dataloader):
    
            # Configure input
            real_imgs = Variable(imgs.type(Tensor), requires_grad=True)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images
            fake_imgs = generator(z)
    
            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
    
            # Compute W-div gradient penalty
            real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(
                real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    
            fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(
                fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    
            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
    
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
    
            d_loss.backward()
            optimizer_D.step()
    
            optimizer_G.zero_grad()
    
            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
    
                # -----------------
                #  Train Generator
                # -----------------
    
                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
    
                g_loss.backward()
                optimizer_G.step()
    
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                
                if batches_done % opt.sample_interval == 0:
                    #
                    #save_image(fake_imgs.data[:25], namedir+"/%d.png" % batches_done, nrow=5, normalize=True)
                    #cv2.imwrite(namedir+"/%d.png", fake_imgs.data[:25])
                    
                    for i, fake_img in enumerate(fake_imgs.data[:25]):
                        filename = namedir + "/%d.png" % (batches_done + i)
                        # save_image(fake_img, filename, normalize=True)
                        # fake_img = np.transpose(fake_img, (1, 2, 0))
                        # fake_img = fake_img.cpu().numpy()
                        # if fake_img.shape[0] == 1:
                        #     fake_img = fake_img.squeeze(0)
                        # fake_img = cv2.normalize(fake_img, None, 0, 255, cv2.NORM_MINMAX)
                        # gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
                        # fake_img = cv2.merge((gray, gray, gray))
                        # fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
                        save_image(fake_img, filename, nrow=5, normalize=True)
                        # fake_img = np.uint8(fake_img)
                        # cv2.imwrite(filename, fake_img)
                        # save_image(fake_img, filename, nrow=5, normalize=True)
    
                # if batches_done % opt.sample_interval == 0:
                #     save_image(fake_imgs.data[:25], namedir+"/%d.png" % batches_done, nrow=5, normalize=True)
    
                batches_done += opt.n_critic
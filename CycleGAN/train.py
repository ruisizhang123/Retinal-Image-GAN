#!/usr/bin/python3

import itertools
import config as config

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from torchvision.utils import save_image
from utils import weights_init_normal
from datasets import ImageDataset

def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

def main():
    netG_A2B = Generator(config.input_nc, config.output_nc)
    netG_B2A = Generator(config.output_nc, config.input_nc)
    netD_A = Discriminator(config.input_nc)
    netD_B = Discriminator(config.output_nc)

    netG_A2B = torch.nn.DataParallel(netG_A2B, device_ids=[0, 1])
    netG_B2A = torch.nn.DataParallel(netG_B2A, device_ids=[0, 1])
    netD_A = torch.nn.DataParallel(netD_A, device_ids=[0, 1])
    netD_B = torch.nn.DataParallel(netD_B, device_ids=[0, 1])

    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=config.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=config.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step)

    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(config.batchSize, config.input_nc, config.size, config.size)
    input_B = Tensor(config.batchSize, config.output_nc, config.size, config.size)
    target_real = Variable(Tensor(config.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(config.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [ transforms.CenterCrop(config.crop_size),
                    transforms.Resize(config.size),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    dataset = ImageDataset(config.dataroot, transforms_=transforms_, unaligned=True)
    dataloader = DataLoader(dataset, batch_size=config.batchSize, shuffle=True, num_workers=config.n_cpu)

    for epoch in range(config.epoch, config.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            optimizer_G.zero_grad()
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0

            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            pred_fake=torch.squeeze(pred_fake)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            pred_fake=torch.squeeze(pred_fake)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()

            optimizer_D_A.zero_grad()

            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(torch.squeeze(pred_real), target_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(torch.squeeze(pred_fake), target_fake)

            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            if (i+1)%config.log_iter==0:
                print('epoch ',epoch, ' batch ',i+1 ,'loss_G:', loss_G.item(), 'loss_G_identity:', (loss_identity_A + loss_identity_B).item(), 'loss_G_GAN:' ,(loss_GAN_A2B + loss_GAN_B2A).item(),
                        'loss_G_cycle:', (loss_cycle_ABA + loss_cycle_BAB).item(), 'loss_D:', (loss_D_A + loss_D_B).item() )
        sample_path = os.path.join(config.sample_dir, '{}-images.jpg'.format(i + 1))
        x_concat = torch.cat([real_A, fake_A, real_B, fake_B], dim=3)
        save_image(denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        torch.save(netG_A2B, '../output/netG_A2B.ckpt')
        torch.save(netG_B2A, '../output/netG_B2A.ckpt')
        torch.save(netD_A, '../output/netD_A.ckpt')
        torch.save(netD_B, '../output/netD_B.ckpt')

if __name__ == '__main__':
    main()
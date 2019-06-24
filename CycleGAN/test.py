#!/usr/bin/python3

import argparse
import sys
import os
import config as config

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset
import config as opt
from utils import tensor2image


netG_A2B = Generator(config.input_nc, config.output_nc)
netG_B2A = Generator(config.output_nc, config.input_nc)

netG_A2B.cuda()
netG_B2A.cuda()

netG_A2B = torch.load('../output/netG_A2B.ckpt')
netG_B2A = torch.load('../output/netG_B2A.ckpt')

netG_A2B.eval()
netG_B2A.eval()

Tensor = torch.cuda.FloatTensor
input_A = Tensor(config.batchSize, config.input_nc, config.size, config.size)
input_B = Tensor(config.batchSize, config.output_nc, config.size, config.size)

transforms_ = [transforms.CenterCrop(config.crop_size),
               transforms.Resize(config.size),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(config.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=config.batchSize, shuffle=False, num_workers=config.n_cpu)

if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')
if not os.path.exists('output/substract'):
    os.makedirs('output/substract')

for i, batch in enumerate(dataloader):
    real_A = Variable(input_A.copy_(batch['A']))
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    real_A = 0.5*(real_A + 1.0)

    image_concat = torch.cat([real_A,fake_B,fake_B-real_A], dim=3)
    save_image(fake_B, 'output/B/%04d.png' % (i+1))
    save_image(image_concat, 'output/substract/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')


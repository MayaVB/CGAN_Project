import argparse
import os
import numpy as np
import math

from dag import DAG
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pickle

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
import torch

from dataset_loader import MnistMixup
from generator import Generator, Discriminator
from train import train

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

if __name__ == "__main__":
    dataset_path = "data/MNIST"
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Loss functions

    # Initialize generator and discriminator
    generator = Generator(img_shape, opt.n_classes, opt.latent_dim)
    discriminator = Discriminator(opt.n_classes, img_shape)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        # adversarial_loss.cuda()

    mixup_dataset = MnistMixup(dataset_path, to_mix=True, size=10000, transform=True)
    normal_dataset = MnistMixup(dataset_path, to_mix=False, size=10000, transform=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Augmentation Run
    print("Starting Augmentation Run")
    datasets = [normal_dataset, mixup_dataset]
    datasets = ConcatDataset(datasets)

    train(opt.n_epochs, opt.n_classes, opt.latent_dim, DataLoader(datasets, shuffle=True, batch_size=opt.batch_size),
          generator, discriminator, optimizer_G, optimizer_D)

    # Baseline run
    print("Starting Baseline Run")
    datasets = normal_dataset

    train(opt.n_epochs, opt.n_classes, opt.latent_dim, DataLoader(datasets, shuffle=True, batch_size=opt.batch_size),
          generator, discriminator, optimizer_G, optimizer_D)

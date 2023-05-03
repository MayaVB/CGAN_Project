import os.path
import numpy as np
import torch
from torchmetrics import MeanSquaredLogError
from scipy.linalg import sqrtm
from const import FloatTensor, LongTensor
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import matplotlib.pyplot as plt

from metric.franchest import Franchest
from utils import plot_data


def calculate_fid_score_epoch(generated_images, real_images, batch_size=32, device="cuda"):
    # Concatenate generated and real images into numpy arrays
    gen_imgs = np.concatenate(generated_images, axis=0)
    real_imgs = np.concatenate(real_images, axis=0)

    # Rescale pixel values to [0, 1] range
    gen_imgs = (gen_imgs + 1) / 2
    real_imgs = (real_imgs + 1) / 2

    # Calculate mu_gen and mu_real
    mu_gen = np.mean(gen_imgs, axis=0)
    mu_real = np.mean(real_imgs, axis=0)

    # Calculate cov_gen and cov_real
    cov_gen = np.cov(gen_imgs.reshape(-1, gen_imgs.shape[-1]), rowvar=False)
    cov_real = np.cov(real_imgs.reshape(-1, real_imgs.shape[-1]), rowvar=False)

    cov_mean, _ = sqrtm(np.dot(cov_gen, cov_real), disp=False)

    # calculate fid score
    fid_score = np.sum((mu_gen - mu_real) ** 2) + np.trace(cov_gen + cov_real - 2 * cov_mean)

    return fid_score


def sample_image(n_row, latent_dim, generator, num_epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])

    labels = Variable(LongTensor(labels))
    labels = torch.eye(10)[labels - 1]

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"images/batch_{num_epoch}.png", nrow=n_row, normalize=True)


def train(n_epochs, n_classes, latent_dim, dataloader, generator, discriminator
          , optimizer_G, optimizer_D, save_weights_directory='inseption_weights/', save_images_path="images/mixup_10000"):
    adversarial_loss = MeanSquaredLogError()
    generated_images = []
    real_images = []
    fid_score_list = []
    d_loss_list = []
    g_loss_list = []
    franchest = Franchest()
    for epoch in range(n_epochs):
        d_loss_agg = 0
        g_loss_agg = 0
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
            gen_labels = torch.eye(10)[gen_labels - 1]

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = (adversarial_loss(validity, valid))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = (adversarial_loss(validity_real, valid))

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = (adversarial_loss(validity_fake, fake))

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            generated_images.append(gen_imgs.detach().cpu().numpy())
            real_images.append(real_imgs.cpu().numpy())

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                  % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

            # d_loss_list.append(d_loss.detach().numpy())
            d_loss_agg += d_loss.item()

            # g_loss_list.append(g_loss.detach().numpy())
            g_loss_agg += g_loss.item()

            eval(generator, imgs, save_images_path, n_classes, latent_dim, franchest)

        d_loss_agg /= i
        d_loss_list.append(d_loss_agg)

        g_loss_agg /= i
        g_loss_list.append(g_loss_agg)

        # FID calculation
        fid_score = calculate_fid_score_epoch(generated_images, real_images)
        # print("FID score: ", fid_score)
        fid_score_list.append(fid_score)

        # save loss

    # round(batch_size * 0.25)
    print("100% FID score is: ", sum(fid_score_list) / len(fid_score_list))
    print("25% FID score is: ", sum(fid_score_list[0:4]) / 5)

    # plot loss vs epoch
    plt.figure()
    plt.plot(d_loss_list, linewidth=3, color='blue')
    plt.title('discriminator loss vs epoch')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('discriminator loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('discriminator_loss.png')

    plt.figure()
    plt.plot(d_loss_list, linewidth=3, color='blue')
    plt.title('generator loss vs epoch')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('generator loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('generator_loss.png')

    # sample_image(n_row=n_classes, latent_dim=latent_dim, generator=generator, num_epoch=epoch)


def eval(generator, images_real, save_images_path="", n_classes=10, latent_dim=10, franchest=None):
    # SAVE IMAGE IN PATH #
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_classes ** 2, latent_dim))))

    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_classes) for num in range(n_classes)])

    labels = Variable(LongTensor(labels))
    labels = torch.eye(10)[labels - 1]

    gen_imgs = generator(z, labels)
    score = franchest.compute_fid(gen_imgs, images_real)
    print(score)

    #save_image(gen_imgs.data, save_images_path, nrow=n_classes, normalize=True)

    return score


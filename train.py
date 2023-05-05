import numpy as np
import torch
from torchmetrics import MeanSquaredLogError
from const import FloatTensor, LongTensor
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import matplotlib.pyplot as plt

from metric.franchest import Franchest
from utils import plot_data


def eval(generator, images_real, save_images_path="", n_classes=10, latent_dim=10, franchest=None):
    # SAVE IMAGE IN PATH #
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_classes ** 2, latent_dim))))

    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_classes) for num in range(n_classes)])

    labels = Variable(LongTensor(labels))
    labels = torch.eye(10)[labels - 1]

    gen_imgs = generator(z, labels)
    score = franchest.compute_fid(gen_imgs, images_real)
    # print(score)

    #save_image(gen_imgs.data, save_images_path, nrow=n_classes, normalize=True)

    return score


def sample_image(n_row, latent_dim, generator, num_epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])

    labels = Variable(LongTensor(labels))
    labels = torch.eye(10)[labels - 1]

    gen_imgs = generator(z, labels)

    save_image(gen_imgs.data, f"images/epoch_num_{num_epoch}.png", nrow=n_row, normalize=True)


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
        fid_score = 0
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

            # print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #       % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

            # d_loss_list.append(d_loss.detach().numpy())
            d_loss_agg += d_loss.item()

            # g_loss_list.append(g_loss.detach().numpy())
            g_loss_agg += g_loss.item()

            # FID estimation
            curr_fid_score = eval(generator, imgs, save_images_path, n_classes, latent_dim, franchest)
            fid_score += curr_fid_score.item()

        fid_score /= i
        fid_score_list.append(fid_score)
        # print("FID score: ", fid_score)

        d_loss_agg /= i
        d_loss_list.append(d_loss_agg)

        g_loss_agg /= i
        g_loss_list.append(g_loss_agg)

    print("FID score is: ", sum(fid_score_list) / len(fid_score_list))

    # plot loss vs epoch
    plt.figure()
    plt.plot(d_loss_list, linewidth=3, color='blue', label='d_loss')
    plt.plot(g_loss_list, linewidth=3, color='orange', label='g_loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epochs', fontsize=12)
    plt.savefig('d_loss_and_g_loss.png')

    sample_image(n_row=n_classes, latent_dim=latent_dim, generator=generator, num_epoch=epoch)



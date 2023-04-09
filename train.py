import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from const import FloatTensor, LongTensor


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
          , optimizer_G, optimizer_D):
    adversarial_loss = torch.nn.MSELoss()

    for epoch in range(n_epochs):
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
            validity, validity_dag = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real, _ = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake, _ = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        sample_image(n_row=10, latent_dim=latent_dim, generator=generator, num_epoch=epoch)

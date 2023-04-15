import numpy as np
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, img_shape, n_classes, latent_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Linear(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape, n_augments=1):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Linear(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

        #self.linear = nn.Linear(32, 1)
        #self.n_augments = n_augments
        #self.linears_dag = []
        #for i in range(self.n_augments):
        #    self.linears_dag.append(nn.Linear(32, 1))
        #self.linears_dag = nn.ModuleList(self.linears_dag)


    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)

        # dag
        #feature = validity.view(-1, 32)
        #outputs_dag = []
        #for i in range(self.n_augments):
        #    outputs_dag.append(self.linears_dag[i](feature))

        #return validity, outputs_dag
        return validity

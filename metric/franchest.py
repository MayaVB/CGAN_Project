# Copyright: MIT Licence.
# Chen Liu (chen.liu.cl2482@yale.edu)
# https://github.com/ChenLiu-1996/GAN-IS-FID-evaluator
import numpy as np
import scipy
import torch
import torch.utils.data
from scipy.linalg import sqrtm
from torch import optim
from metric.inseption import train, Net


class Franchest:
    def __init__(self, to_train=False):
        self.model = Net()
        PATH = "metric/inseption_weights/inseption.pt"
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        if to_train:
            train(30, self.model, optimizer)
            torch.save(self.model, PATH)
        else:
            model = torch.load(PATH)
            model.eval()

    def compute_fid(self, real, fake) -> float:
        subset_real = self.get_activations(real)
        real_mean = torch.mean(subset_real, axis=0)
        real_cov = torch.cov(subset_real)

        subset_fake = self.get_activations(fake)
        fake_mean = torch.mean(subset_fake, axis=0)
        fake_cov = torch.cov(subset_fake)

        fid_value = self.frechet_distance(real_mean, fake_mean, real_cov, fake_cov)
        return fid_value

    @staticmethod
    def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
        """
        Function for returning the Fréchet distance between multivariate Gaussians,
        parameterized by their means and covariance matrices.
        Parameters:
            mu_x: the mean of the first Gaussian, (n_features)
            mu_y: the mean of the second Gaussian, (n_features)
            sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
            sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
        """

        return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * torch.trace(
            matrix_sqrt(sigma_x @ sigma_y))

    def get_activations(self, data):
        activations = self.model.get_activation(data).view(data.shape[0], -1)

        return activations


def matrix_sqrt(x):
    """
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    """
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)

    return torch.Tensor(y.real, device=x.device)

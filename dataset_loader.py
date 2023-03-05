import os
import random
import numpy as np
import pandas as pd
import idx2numpy
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from skimage import io
import gzip
import numpy as np
import torchvision.transforms as transforms

image_size = 32

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(image_size),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])


def read_images(path_dir, size=60000):
    data_images = idx2numpy.convert_from_file(path_dir)

    return data_images[:size]


def read_labels(path_dir, size=60000):
    labels = idx2numpy.convert_from_file(path_dir)

    return labels[:size]


class MnistMixup(Dataset):
    def __init__(self, image_directory, to_mix=True, size=60000, transform=None):
        """
        Args:
            image_directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_directory = image_directory
        self.transform = transform
        self.data = (read_images(os.path.join(self.image_directory,
                                              "train-images-idx3-ubyte.gz"), size))

        self.labels = (read_labels(os.path.join(self.image_directory,
                                                "train-labels-idx1-ubyte.gz"), size))

        self.to_mix = to_mix
        self.train = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]

        # Create a one hot label
        label = torch.zeros(10)
        label[self.labels[idx]] = 1.

        # Transform the image by converting to tensor and normalizing it
        if self.transform:
            image = transform(self.data[idx])

        # If data is for training, perform mixup, only perform mixup roughly on 1 for every 5 images
        if self.to_mix and idx > 0:

            # Choose another image/label randomly
            mixup_idx = random.randint(0, len(self.data) - 1)
            mixup_label = torch.zeros(10)
            mixup_label[self.labels[mixup_idx]] = 1.
            if self.transform:
                mixup_image = transform(self.data[mixup_idx])

            # Select a random number from the given beta distribution
            # Mixup the images accordingly
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            image = lam * image + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label

        return image, label


if __name__ == "__main__":
    mixup_dataset = MnistMixup('.\\data\\MNIST', to_mix=True, size=10000, transform=True)
    normal_dataset = MnistMixup('.\\data\\MNIST', to_mix=False, size=10000, transform=True)

    print(len(mixup_dataset))
    datasets = [mixup_dataset, normal_dataset]
    datasets = ConcatDataset(datasets)
    loader = DataLoader(
        datasets,
        shuffle=True,
        num_workers=1,
        batch_size=64
    )
    for i, (imgs, labels) in enumerate(loader):
        print(imgs.shape)

    import matplotlib.pyplot as plt

    print(mixup_dataset[4][1])
    plt.imshow(mixup_dataset[4][0].squeeze())
    plt.show()

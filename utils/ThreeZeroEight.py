from __future__ import print_function

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

class_directories = {
    "clap": "clap",
    #"kick": "kick",
    #"pickup": "pickup",
    #"run": "run",
    #"sitdown": "sitdown",
    #"standup": "standup",
    #"walk": "walk",
    #"wavehand": "wavehand"
}


def get_custom_data_loader(data_dir, train_ratio=0.8, transform=None):
    dataset = ImageFolder(root=data_dir, transform=transform)
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_test = num_samples - num_train

    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    return train_dataset, test_dataset  # 返回 Dataset 对象



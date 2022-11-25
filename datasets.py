import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from sklearn.datasets import make_blobs
import torchvision
import numpy as np
from copy import deepcopy


dataset_transforms = {
    "cifar10": 
        {
            "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]), 
            "test": transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        },
    "cifar100": 
        {
            "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]), 
            "test": transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        }
}


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, download, train, transform, num_samples=None):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, download=download,
                                                    train=train, transform=transform)
        self.num_samples = num_samples
        self.targets = torch.tensor(self.cifar10.targets)
        if num_samples:
            subset_rand_indices = torch.randperm(len(self.cifar10))[:num_samples]
            self.cifar10 = torch.utils.data.Subset(self.cifar10, subset_rand_indices)
            self.targets = self.targets[subset_rand_indices]
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

    def set_cifar10_transform(self, transform):
        if self.num_samples:
            self.cifar10.dataset.transform = transform
        else:
            self.cifar10.transform = transform
    
    def get_cifar10_transform(self):
        print(self.transform)
    
    transform = property(fset=set_cifar10_transform, fget=get_cifar10_transform)



class CustomCIFAR100Dataset(Dataset):
    def __init__(self, root, download, train, transform, num_samples=None):
        self.cifar100 = torchvision.datasets.CIFAR100(root=root, download=download,
                                                    train=train, transform=transform)
        self.num_samples = num_samples
        self.targets = torch.tensor(self.cifar100.targets)
        if num_samples:
            subset_rand_indices = torch.randperm(len(self.cifar100))[:num_samples]
            self.cifar100 = torch.utils.data.Subset(self.cifar100, subset_rand_indices)
            self.targets = self.targets[subset_rand_indices]
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)

    def set_cifar100_transform(self, transform):
        if self.num_samples:
            self.cifar100.dataset.transform = transform
        else:
            self.cifar100.transform = transform
    
    transform = property(fset=set_cifar100_transform)


dataset_classes = {
    "cifar10": CustomCIFAR10Dataset,
    "cifar100": CustomCIFAR100Dataset
}


def get_dataset(dataset, data_path, num_samples, train_percentage):
    
    train_transform = dataset_transforms[dataset]["train"]
    test_transform = dataset_transforms[dataset]["test"]
    dataset_class = dataset_classes[dataset]
    
    full_dataset = dataset_class(root=data_path, train=True,
                                                    download=True, transform=None, num_samples=num_samples)
    num_train_samples = int(len(full_dataset) * train_percentage)
    num_val_samples = len(full_dataset) - num_train_samples
    test_dataset = dataset_class(root=data_path, train=False,
                                                download=True, transform=test_transform)

    train_indices, val_indices = random_split(range(len(full_dataset)), [num_train_samples, num_val_samples], generator=torch.Generator().manual_seed(42))

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = train_transform

    train_dataset.dataset = deepcopy(full_dataset)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = test_transform
    return full_dataset, train_dataset, val_dataset, test_dataset

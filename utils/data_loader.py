import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from utils.fashion_mnist import MNIST, FashionMNIST
from utils.ThreeZeroEight import get_custom_data_loader

def get_data_loader(args):

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == '308':
        trans = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        data_dir = '/root/autodl-tmp/pytorch-wgan-master/datasets/308'
        train_set, test_set = get_custom_data_loader(data_dir, transform=trans)  # 注意这里返回的是 Dataset
        #train_loader = data_utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        #test_loader = data_utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        return train_loader, test_loader

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, split='train', download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot,  split='test', download=args.download, transform=trans)

    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

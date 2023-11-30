from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader



class TCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def store_masked_loaders(args, train_dataset, test_dataset, t):
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    N_CLASSES_PER_TASK = args.dataset.n_classes_per_task
    i = t * N_CLASSES_PER_TASK

    train_mask = np.logical_and(np.array(train_dataset.targets) >= i,
                                np.array(train_dataset.targets) < i + N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= i,
                               np.array(test_dataset.targets) < i + N_CLASSES_PER_TASK)
    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_dataloader = DataLoader(train_dataset,
                              batch_size=args.train.batch_size, shuffle=True, num_workers=args.dataset.num_workers)
    dataloader_test = DataLoader(test_dataset,
                             batch_size=args.train.batch_size, shuffle=False, num_workers=args.dataset.num_workers)

    return train_dataloader, dataloader_test, len(train_dataset)


def get_cifar100_dataloaders(args, t, dataloaders_test):

    cifar100_norm = [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]]
    train_transform = transforms.Compose([
         # transforms.RandomCrop(args.dataset.image_size, padding=4),
         transforms.RandomResizedCrop(args.dataset.image_size),
         # transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(*cifar100_norm)])
    test_transform = transforms.Compose([
         transforms.Resize(args.dataset.image_size),
         transforms.ToTensor(),
         transforms.Normalize(*cifar100_norm)])

    train_dataset = MyCIFAR100(args.cifar100_dir, train=True,
                            download=True, transform=train_transform)
    test_dataset = TCIFAR100(args.cifar100_dir, train=False,
                             download=True, transform=test_transform)

    train_dataloader, dataloader_test, data_train_nums = store_masked_loaders(args, train_dataset, test_dataset, t)
    dataloaders_test.append(dataloader_test)
    print('class name:', t, 'number of training sets:', data_train_nums,
          'number of testing sets:', len(test_dataset))

    return train_dataloader, dataloaders_test, data_train_nums

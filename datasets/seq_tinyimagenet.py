import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        classnames2labels = {}
        classnames = []
        with open(f'{self.root}/wnids.txt') as wnid:
            for line in wnid:
                classnames.append(line.strip('\n'))
        for label, classname in enumerate(classnames):
            classnames2labels[classname] = label

        self.data, self.targets = [], []
        if self.train:
            for label, classname in enumerate(classnames):
                filepaths = list((self.root / "train" / classname / "images").glob("*.JPEG"))
                self.data.extend(filepaths)
                labels = [label] * len(filepaths)
                self.targets.extend(labels)
        else:
            val_all_classnames = []
            with open(f'{self.root}/val/val_annotations.txt') as valtxt:
                for line in valtxt:
                    filename = line.strip('\n').split('\t')[0]
                    filepath = os.path.join(self.root, 'val', 'images', filename)
                    self.data.append(filepath)
                    val_all_classnames.append(line.strip('\n').split('\t')[1])
            for val_classname in val_all_classnames:
                self.targets.append(classnames2labels[val_classname])

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]
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


def get_tinyimagenet_dataloaders(args, t, dataloaders_test):
    tinyimagenet_norm = [[0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]]

    train_transform = transforms.Compose([
         # transforms.RandomCrop(args.dataset.image_size, padding=4),
         transforms.RandomResizedCrop(args.dataset.image_size),
         # transforms.Resize(args.dataset.image_size),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(*tinyimagenet_norm)])
    test_transform = transforms.Compose([
         transforms.Resize(args.dataset.image_size),
         transforms.ToTensor(),
         transforms.Normalize(*tinyimagenet_norm)])

    train_dataset = TinyImagenet(args.tinyimagenet_dir, train=True, transform=train_transform)
    test_dataset = TinyImagenet(args.tinyimagenet_dir, train=False, transform=test_transform)

    train_dataloader, dataloader_test, data_train_nums = store_masked_loaders(args, train_dataset, test_dataset, t)
    dataloaders_test.append(dataloader_test)
    print('class name:', t, 'number of training sets:', data_train_nums,
          'number of testing sets:', len(test_dataset))

    return train_dataloader, dataloaders_test, data_train_nums
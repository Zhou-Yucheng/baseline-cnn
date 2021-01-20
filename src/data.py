#!/usr/bin/python3.7

import os
import socket

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def get_image_folder_data_loader(data_dir, input_size, batch_size):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomPerspective(),
            # transforms.RandomRotation(45),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    image_dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

    n_worker = 1 if socket.gethostname() == 'ThinkPad-zyc' else 4
    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    valid_loader = DataLoader(image_dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_worker)

    return train_loader, valid_loader

# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms as T



import os
from random import randint
import urllib
import zipfile


# Define device to use (CPU or GPU). CUDA = GPU support for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

DATA_DIR = 'D:\\D_downloads\\tiny-imagenet-200\\tiny-imagenet-200' # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'val')

# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(make_grid(images))  # Using Torchvision.utils make_grid function


def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = randint(0, len(images) - 1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')


# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name, transform, batch_size):
    if data is None:
        return None

    # Read image files to pytorch dataset using ImageFolder, a generic data
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(name == "train"),
                            **kwargs)

    return dataloader


# Unlike training folder where images are already arranged in sub folders based
# on their labels, images in validation folder are all inside a single folder.
# Validation folder comes with images folder and val_annotations txt file.
# The val_annotation txt file comprises 6 tab separated columns of filename,
# class label, x and y coordinates, height, and width of bounding boxes
val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt',
                       sep='\t',
                       header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])

print(val_data.head())


# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(VALID_DIR, 'images')

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
print('Display first 10 entries of resulting val_img_dict dictionary\n')
print( {k: val_img_dict[k] for k in list(val_img_dict)[:10]})

# Create subfolders (if not present) for validation images based on label ,
# and move images into the respective folders
for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

# Save class names (for corresponding labels) as dict from words.txt file
class_to_name_dict = dict()
fp = open(os.path.join(DATA_DIR, 'words.txt'), 'r')
data = fp.readlines()
for line in data:
    words = line.strip('\n').split('\t')
    class_to_name_dict[words[0]] = words[1].split(',')[0]
fp.close()

# Display first 20 entries of resulting dictionary
print('Display first 20 entries of resulting dictionary \n')
print({k: class_to_name_dict[k] for k in list(class_to_name_dict)[:20]})

# Define transformation sequence for image pre-processing
# If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
# If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225])
preprocess_transform = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #
])

preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

# Define batch size for data loaders
batch_size = 64

train_loader = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform, batch_size=batch_size)

# Display batch of training set images
# show_batch(train_loader)


# Create train loader for pre-trained models (normalized based on specific requirements)
train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform_pretrain, batch_size = batch_size)

# Display batch of pre-train normalized images
# show_batch(train_loader_pretrain)


# Create dataloaders for validation data (depending if model is pretrained)
val_loader = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform, batch_size = batch_size)

val_loader_pretrain = generate_dataloader(val_img_dir, "val",
                                 transform=preprocess_transform_pretrain, batch_size = batch_size)


print("")
# Display batch of validation images
# show_batch(val_loader)





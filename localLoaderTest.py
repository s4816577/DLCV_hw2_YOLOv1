import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image

DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')

class AERIAL(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.images_filenames = glob.glob(os.path.join(root, '*.jpg'))
        self.transform = transform
        self.len = len(self.images_filenames)
        
        #windows
        self.images_filenames = sorted(self.images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
    def __getitem__(self, index):
        image_name = self.images_filenames[index]
        
        image = Image.open(image_name)
        image = image.resize((448,448), Image.ANTIALIAS)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image
        
    def __len__(self):
        return self.len 

def test_loader():
    trainset = AERIAL('hw2_train_val/train15000/images', transform=transforms.ToTensor())
    '''
    list = [0 for i in range(16)]
    print(list)
    iteration = 0
    count = 0
    for img, labels in trainset:
        iteration += 1
        for i in range(len(labels)):
            list[labels[i][4]] += 1
        if iteration % 100 == 0:
            count += 1
            print(count)
    print(list)
    '''
    img1 = trainset[0]
    print(img1.shape)

if __name__ == '__main__':
    test_loader()

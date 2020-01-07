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
        self.images_filenames = glob.glob(os.path.join(root, 'images', '*.jpg'))
        self.labels_filenames = glob.glob(os.path.join(root, 'labelTxt_hbb', '*.txt'))
        self.transform = transform
        self.len = len(self.images_filenames)
        
        #windows
        self.images_filenames = sorted(self.images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        self.labels_filenames = sorted(self.labels_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
    def __getitem__(self, index):
        image_name = self.images_filenames[index]
        label_name = self.labels_filenames[index]
        
        image = Image.open(image_name)
        image = image.resize((448,448), Image.ANTIALIAS)
        label = parse_label(label_name)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
        
    def __len__(self):
        return self.len
        
def parse_label(filename):
    result = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 10:
                continue
            x1 = int(float(token[0]) * 448 / 512)
            y1 = int(float(token[1]) * 448 / 512)
            x2 = int(float(token[4]) * 448 / 512)
            y2 = int(float(token[5]) * 448 / 512)
            cls = int(DOTA_CLASSES.index(token[8]))
            #prob = float(token[9])
            result.append([x1, y1, x2, y2, cls])
    return result    

def test_loader():
    trainset = AERIAL('hw2_train_val/train15000', transform=transforms.ToTensor())
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
    img1, label1 = trainset[0]
    print(img1.shape, label1)

if __name__ == '__main__':
    test_loader()

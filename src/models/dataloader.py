import os
import numpy as np
import pandas as pd
import itertools

# Import required libraries
import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class CaptchaDataset(Dataset): 
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
    
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 

        if self.transform:
            image = self.transform(image)
        
        return image, label

PATH = os.path.join('data', 'captcha_images_v2')
images = []
labels = []


for f in os.listdir(PATH):
    file = os.path.join(PATH, f)
    label = f.split('.')[0]

    if not os.path.isfile(file) and not f.endswith('.png'):
        continue

    # Read the image
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    tensor = transform(image)
    images.append(image)
    labels.append(label)
    
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

X_test, X_valid, y_test, y_valid = train_test_split(
    images, labels, test_size=0.5, random_state=42)

train = CaptchaDataset(X_train, y_train)
test = CaptchaDataset(X_test, y_test)
validation = CaptchaDataset(X_valid, y_valid)

train = DataLoader(train, batch_size=24, shuffle=True)
test = DataLoader(test, batch_size=24, shuffle=True)
valid = DataLoader(validation, batch_size=24, shuffle=True)

image, label = next(iter(train))
print(image.shape)
print(label)
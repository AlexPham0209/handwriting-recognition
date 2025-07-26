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
    def __init__(self, images, labels, idx_to_char, char_to_idx):
        self.images = images
        self.labels = labels

        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        
        self.transform = transforms.Compose([
            # transforms.RandomRotation(10),
            transforms.RandomAdjustSharpness(5),
            transforms.ColorJitter(brightness=(0.5, 1.0))
        ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(list(map(lambda x: self.char_to_idx[x], list(self.labels[idx]))), dtype=torch.float32)

        image = image / 255.
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)
    
        return image, label


def load_data(batch_size=16, test_size = 0.2, random_state=11):
    PATH = os.path.join('data', 'captcha_images_v2')
    images = []
    labels = []
    chars = set()
    
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
        chars.update(list(label))
        
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state)

    X_test, X_valid, y_test, y_valid = train_test_split(
        images, labels, test_size=0.5, random_state=random_state)

    chars = ['-'] + sorted(list(chars))
    idx_to_char = {i:symbol for i, symbol in enumerate(chars)}
    char_to_idx = {symbol:i for i, symbol in enumerate(chars)}

    train = CaptchaDataset(X_train, y_train, idx_to_char, char_to_idx)
    test = CaptchaDataset(X_test, y_test, idx_to_char, char_to_idx)
    valid = CaptchaDataset(X_valid, y_valid, idx_to_char, char_to_idx)

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=1, shuffle=True)
    valid = DataLoader(valid, batch_size=batch_size, shuffle=True)

    return train, test, valid, idx_to_char, char_to_idx

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size).to(device)
        self.batch_norm = nn.BatchNorm2d(out_channels).to(device)
        self.relu = nn.ReLU().to(device)
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2)).to(device)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pooling(x)

        return x 

class CaptchaDetectionModel(nn.Module):
    def __init__(self, num_chars, device):
        super(CaptchaDetectionModel, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(3, 32, device).to(device),
            ConvBlock(32, 64, device).to(device),
            ConvBlock(64, 128, device).to(device),
        ).to(device)

        self.fc1 = nn.Linear(in_features=92, out_features=32).to(device)
        self.dropout = nn.Dropout(p=0.2).to(device)
        
        self.lstm = nn.LSTM(32, 64, 2, bidirectional=True, batch_first=True).to(device)
        self.fc2 = nn.Linear(128, num_chars).to(device)

    def forward(self, x):
        # Input Shape: (N, c, w, h)
        x = self.conv(x)

        # Reshape into (N, c, w * h)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc1(x)
        x = self.dropout(x)

        # Input Shape: (N, w * h, 32) where w * h represents the sequence length and 32 is the input_size
        # Output Shape: (N, w * h, 64)
        x, (_, _) = self.lstm(x)
        x = self.fc2(x)
        
        return x.permute(1, 0, 2)
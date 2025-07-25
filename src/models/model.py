from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dropout=0.2):
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return x 

class CaptchaDetectionModel(nn.Module):
    def __init__(self, features, in_channels):
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.fc = nn.Linear(in_features=128, out_features=32)
        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(32, 64, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64, features)

    def forward(self, x):
        # Input Shape: (c, w, h)
        x = self.conv(x)

        # Shape into (c, w * h)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.dropout(x)

        x = self.lstm(x)
        x = self.fc(x)
        self.fc
        
        return x
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, kernel_size=(3, 3), dropout=0.2):
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size).to(device)
        self.batch_norm = nn.BatchNorm2d(out_channels).to(device)
        self.relu = nn.ReLU().to(device)
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2)).to(device)
        self.dropout = nn.Dropout(p=dropout).to(device)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return x 

class CaptchaDetectionModel(nn.Module):
    def __init__(self, num_symbols, device):
        self.conv = nn.Sequential(
            ConvBlock(3, 32).to(device),
            ConvBlock(32, 64).to(device),
            ConvBlock(64, 128).to(device),
        ).to(device)

        self.fc = nn.Linear(in_features=128, out_features=32).to(device)
        self.dropout = nn.Dropout(p=0.2).to(device)

        self.lstm = nn.LSTM(32, 64, 2, bidirectional=True, batch_first=True).to(device)
        self.fc = nn.Linear(64, num_symbols + 1).to(device)

    def forward(self, x):
        # Input Shape: (N, c, w, h)
        x = self.conv(x)

        # Transpose into (N, w, h, c) and then reshape into (N, w * h, c)
        x = x.transpose(0, 2, 3, 1).view(x.shape[0], -1, x.shape[-1])
        x = self.fc(x)
        x = self.dropout(x)

        # Input Shape: (N, w * h, 32) where w * h represents the sequence length and 32 is the input_size
        # Output Shape: (N, w * h, 64)
        x = self.lstm(x)

        x = self.fc(x)
        self.fc
        
        return x
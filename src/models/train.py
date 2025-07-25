from models.dataloader import load_data
import matplotlib.pyplot as plt
from models.model import CaptchaDetectionModel
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, criterion, epochs=10):
    pass

def validation(model):
    pass 


if __name__ == "__main__":
    training, test, valid, chars_to_idx, idx_to_chars = load_data(batch_size=16)
    model = CaptchaDetectionModel(len(chars_to_idx), 3, device=DEVICE).to(DEVICE)
    criterion = nn.CTCLoss().to(DEVICE)

    
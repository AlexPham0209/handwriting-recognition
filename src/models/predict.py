import os
from matplotlib import pyplot as plt
from models.dataloader import load_data
from models.model import CaptchaDetectionModel
import torch
from utils.ctc_decoding import ctc_greedy, ctc_beam_search
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

with open(os.path.join(CONFIG_PATH, 'model.yaml'), 'r') as file:
    config = yaml.safe_load(file)

train, test, valid, idx_to_char, char_to_idx = load_data()
load_path = config['testing']['load_path']

model = CaptchaDetectionModel(len(char_to_idx), device=DEVICE).to(DEVICE)
checkpoint = torch.load(load_path, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

num_rows = 4
num_cols = 5
figure, axis = plt.subplots(num_rows, num_cols)

for i in range(num_rows):
    for j in range(num_cols):
        image, label = next(iter(test))
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        out = model(image)
        predicted = ctc_greedy(out, idx_to_char) if not config['testing']['beam_search'] else ctc_beam_search(out, idx_to_char)
        
        axis[i, j].set_title(f"{predicted}")
        axis[i, j].imshow(image[0].permute(1, 2, 0).cpu())
        axis[i, j].axis('off') 
    
plt.show()
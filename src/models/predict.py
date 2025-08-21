import os
from matplotlib import pyplot as plt
from models.dataloader import load_data
from models.model import CaptchaDetectionModelV2, CaptchaDetectionModelV3
import torch
from utils.ctc_decoding import ctc_greedy, ctc_beam_search
import tensorflow as tf
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = "configs"

with open(os.path.join(CONFIG_PATH, 'model.yaml'), 'r') as file:
    config = yaml.safe_load(file)

train, test, valid, idx_to_char, char_to_idx = load_data()
load_path = config['testing']['load_path']

# model = CaptchaDetectionModelV2(len(char_to_idx), device=DEVICE).to(DEVICE)
model = CaptchaDetectionModelV3(2, len(char_to_idx), 512, device=DEVICE).to(DEVICE)
checkpoint = torch.load(load_path, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

num_rows = 2
num_cols = 5
figure, axis = plt.subplots(num_rows, num_cols)

for i in range(num_rows):
    for j in range(num_cols):
        image, label = next(iter(test))
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        out = model(image)
        T, N, C = out.shape
        
        out = out.detach().cpu().numpy()
        input_lengths = torch.full(size=(N,), fill_value=T).detach().cpu().numpy()

        # predicted = ctc_greedy(out, idx_to_char) if not config['testing']['beam_search'] else ctc_beam_search(out, idx_to_char)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=out,
            sequence_length=input_lengths,
            beam_width=9
        )

        decoded = list(tf.sparse.to_dense(decoded[0]).numpy()[0])
        predicted = ''.join([idx_to_char[elem] for elem in decoded]).replace('-', '')
            
        axis[i, j].set_title(f"{predicted}")
        axis[i, j].imshow(image[0].permute(1, 2, 0).cpu())
        axis[i, j].axis('off') 
    
plt.show()
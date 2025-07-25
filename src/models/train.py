from models.dataloader import load_data
import matplotlib.pyplot as plt
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    pass

if __name__ == "__main__":
    train, test, valid, symbols = load_data(batch_size=1)
    # while True:
    #     image, label = next(iter(train))
        
    #     image = image[0]
    #     label = label[0]

    #     print(image.shape)
    #     imgplot = plt.imshow(image.transpose(1, 2, 0))
    #     plt.show()
    
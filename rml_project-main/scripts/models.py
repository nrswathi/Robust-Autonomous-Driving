# Pytorch model
import torch
import torch.nn as nn
from torchsummary import summary 

FREEZE_RATIO = 0.6

class Flatten(nn.Module):
    def forward(self, x):
        x = x.permute(0,2,3,1)
        return x.reshape(x.shape[0], -1)
    
def create_model(freeze_ratio=0):
    model = nn.Sequential(nn.Conv2d(3, 64, 3, padding='valid', stride=1), nn.ReLU(),
                        nn.MaxPool2d(2,2, padding = 0),
                        nn.Conv2d(64, 32, 3, padding='valid', stride=1), nn.ReLU(),
                        nn.MaxPool2d(2,2, padding = 0),
                        nn.Conv2d(32, 32, 3, padding='valid', stride=1), nn.ReLU(),
                        nn.MaxPool2d(2,2, padding = 0),
                        nn.Conv2d(32, 32, 3, padding='valid', stride=1), nn.ReLU(),
                        nn.MaxPool2d(2,2, padding = 0),
                        Flatten(),
                        nn.Linear(1760, 128),nn.ReLU(),
                        nn.Linear(128, 3),
                        nn.Softmax(dim =1))
    
    total_layers = len(list(model.parameters()))
    count = 0
    split_idx = int(freeze_ratio*total_layers)
    for param in model.parameters():
        if count >= split_idx:
            break
        param.requires_grad = False
        count = count + 1
    return model

if __name__ == "__main__":
    print("Creating the model")
    model = create_model(FREEZE_RATIO)
    summary(model, (3,110,220))
import os
import pandas as pd
import torch
import torch.nn as nn
from models import create_model
from dataset import Custom_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from attacks import pgd

BATCH_SIZE = 128
IMG_SHAPE = (110, 220, 3)
N_EPOCHS = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epoch(loader, model, opt=None, device='cpu'):
    """Standard training/evaluation epoch over the dataset"""
    running_loss = 0
    total_loss = 0
    correct = 0
    total_samples = 0
    CSE_loss = nn.CrossEntropyLoss()
    for i , data in tqdm(enumerate(loader)):
        inputs, labels = data
        if opt:
            opt.zero_grad()
        pred = model(inputs.to(device))
        loss = CSE_loss(pred, labels.to(device))
        if opt:
            loss.backward()
            opt.step()

        probabilities = nn.Softmax(dim=1)(pred)
        predicted_label = torch.argmax(probabilities, dim= 1)
        correct += (predicted_label == labels.to(device)).sum().item()
        total_samples += labels.size(0)

        running_loss += loss.item()
        total_loss += loss.item()
        if i % 20 == 0:    # print every 2000 mini-batches
            print(f'loss: {running_loss / 20:.4f}')
            running_loss = 0.0  

    avg_loss = total_loss / (i+1)
    accuracy = (correct/ total_samples)

    return accuracy, avg_loss

def epoch_adv(loader, model, attack, opt=None, device='cpu', **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    running_loss = 0
    correct = 0
    total_samples = 0
    CSE_loss = nn.CrossEntropyLoss()
    for i , data in tqdm(enumerate(loader)):
        inputs, labels = data
        adv_pert = attack(model, inputs, labels, device=device)
        adv_inputs = inputs + adv_pert
        combined_inputs = torch.concat([inputs, adv_inputs], axis =0)
        combined_labels = torch.concat([labels, labels], axis = 0)
        if opt:
            opt.zero_grad()
        pred = model(combined_inputs.to(device))
        loss = CSE_loss(pred, combined_labels.to(device))
        if opt:
            loss.backward()
            opt.step()

        probabilities = nn.Softmax(dim=1)(pred)
        predicted_label = torch.argmax(probabilities, dim= 1)
        correct += (predicted_label == combined_labels.to(device)).sum().item()
        total_samples += combined_labels.size(0)

        running_loss += loss.item()

    avg_loss = running_loss / (i+1)
    accuracy = (correct/ total_samples)
    return accuracy, avg_loss


def train(model, train_loader, val_loader, opt, n_epochs, out_path="./checkpoints/model_weights_default_save.pth", device='cpu'):
    min_loss = 1e9
    for n in range(n_epochs):
        train_acc, train_loss = epoch(train_loader, model, opt, device=device)
        print(f"Epoch:{n},\t Training Loss:{train_loss}, Training accuracy:{train_acc}")
        val_acc, val_loss = epoch(val_loader, model, device=device)
        print(f"Epoch:{n},\t Validation Loss:{val_loss}, Validation accuracy:{val_acc}")
    
        if val_loss<min_loss:
            print("Saving the model weuights")
            torch.save(model.state_dict(), out_path)
            min_loss = val_loss

def train_adv(model, train_loader, val_loader, opt, n_epochs, attack=pgd, out_path="./checkpoints/adv_trained_model_weights_default_save.pth", device='cpu'):
    min_loss = 1e9
    for n in range(n_epochs):
        train_acc, train_loss = epoch_adv(train_loader, model, attack, opt, device=device)
        print(f"Epoch:{n},\t Training Loss:{train_loss}, Training accuracy:{train_acc}")
        val_acc, val_loss = epoch_adv(val_loader, model, attack, device=device)
        print(f"Epoch:{n},\t Validation Loss:{val_loss}, Validation accuracy:{val_acc}")
    
    torch.save(model.state_dict(), out_path)



if __name__ == "__main__":
    print("Creating Training Pipeline")
    print("Device:", device)
    train_csv = "./data/csv/roi_cropped/train_csv.csv"
    val_csv = "./data/csv/roi_cropped/val_csv.csv"
    img_folder = "./data/roicropped/roicropped"
    assert os.path.exists(train_csv) and os.path.exists(val_csv), "Error: CSV file does not exists. Check the path"
    assert os.path.exists(img_folder), "Error: Image Folder does not exists. Check the path"
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print("Distribution of labels in Training set:", train_df['label'].value_counts())
    print("Distribution of labels in Validation set:", val_df['label'].value_counts())
    
    train_dataset = Custom_Dataset(train_df, img_folder)
    val_dataset = Custom_Dataset(val_df, img_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    print("Start training base classifier")
    train(model, train_dataloader, val_dataloader, opt, N_EPOCHS, device=device)
    print("Start Adversarial trainingf")
    train_adv(model, train_dataloader, val_dataloader, opt, N_EPOCHS, pgd, device=device)
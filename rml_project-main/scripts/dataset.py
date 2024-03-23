from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import pandas as pd


BATCH_SIZE = 128

class Custom_Dataset(Dataset):
    """Custom dataset to read from dataframe and return image and its respective labels

    Args:
        Dataset (torch.data.Dataset): Subclass the pytorch Dataset class
    """
    def __init__(self, df, image_folder):
        self.image_names = df['img_file'].values
        self.labels = df['label'].values
        self.image_folder = image_folder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_folder, image_name)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255
        img_tensor = torch.tensor(img).permute(2,0,1).float()
        label = self.labels[index]
        return img_tensor, label
    

if __name__ == "__main__":
    print("Creating Custom Dataset")
    csv_path = "./data/csv/roi_cropped/val_csv.csv"
    img_folder = "./data/roicropped/roicropped"
    assert os.path.exists(csv_path), "Error: CSV file does not exists. Check the path"
    assert os.path.exists(img_folder), "Error: Image Folder does not exists. Check the path"

    df = pd.read_csv(csv_path)
    print("Distribution of labels:", df['label'].value_counts())
    dataset = Custom_Dataset(df, img_folder)
    print("Number of samples in dataset:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for i, l in dataloader:
        print("Shape of single input batch:", i.shape)
        print("Shape of sinle label batch:",l.shape)
        print("Batch of Labels:",l)
        break

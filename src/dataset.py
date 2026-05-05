import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_SIZE = 224


class LicensePlateDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=IMG_SIZE):
        self.df       = pd.read_csv(csv_path)
        self.img_dir  = img_dir
        self.img_size = img_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img     = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, row['filename'])),
            cv2.COLOR_BGR2RGB,
        )
        orig_w, orig_h = row['width'], row['height']
        img     = cv2.resize(img, (self.img_size, self.img_size))
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        bbox    = torch.tensor([
            (row['xmin'] * scale_x) / self.img_size,
            (row['ymin'] * scale_y) / self.img_size,
            (row['xmax'] * scale_x) / self.img_size,
            (row['ymax'] * scale_y) / self.img_size,
        ], dtype=torch.float32)
        return self.transform(img), bbox

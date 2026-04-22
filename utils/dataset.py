import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Filter the dataframe to only include images that exist on your disk
        # .iloc[:, 0] refers to the first column (the image filenames)
        self.df = self.df[self.df.iloc[:, 0].apply(
            lambda x: os.path.exists(os.path.join(self.root_dir, x))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Using iloc ensures we get data from the filtered dataframe correctly
        img_id = self.df.iloc[index, 0]
        caption = self.df.iloc[index, 1]

        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption

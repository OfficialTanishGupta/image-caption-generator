import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image


class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, vocab, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.annotations = data['annotations']
        self.images = {img['id']: img['file_name'] for img in data['images']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        caption = ann['caption']
        img_id = ann['image_id']
        img_path = os.path.join(self.root_dir, self.images[img_id])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numerical_caption = [self.vocab.stoi["<sos>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<eos>"])

        return image, torch.tensor(numerical_caption)
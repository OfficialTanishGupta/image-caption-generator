import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import json

from utils.vocab import Vocabulary
from utils.dataloader import MyCollate
from models.model import ImageCaptionModel
from utils.coco_dataset import CocoDataset

def train():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/annotations/captions_train2017.json", "r") as f:
        data = json.load(f)

    captions = [ann["caption"] for ann in data["annotations"]]

    vocab = Vocabulary()
    vocab.build_vocab(captions)

    vocab_size = len(vocab)
    pad_idx = vocab.stoi["<pad>"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CocoDataset(
        root_dir="data/train2017",
        annotation_file="data/annotations/captions_train2017.json",
        vocab=vocab,
        transform=transform
    )
    
    val_dataset = CocoDataset(
        root_dir="data/val2017",
        annotation_file="data/annotations/captions_val2017.json",
        vocab=vocab,
        transform=transform
    )
    
    indices = list(range(5000))
    train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    embed_size = 256
    hidden_size = 256
    num_epochs = 20
    model = ImageCaptionModel(embed_size, hidden_size, vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "encoder" in name and param.requires_grad:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = optim.Adam([
        {"params": encoder_params, "lr": 1e-5},
        {"params": decoder_params, "lr": 3e-4}
    ])
    
    print("Training started...")

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (imgs, caps) in enumerate(train_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)

            # Forward pass
            outputs = model(imgs, caps[:, :-1])

            # Loss
            loss = criterion(
                outputs.reshape(-1, vocab_size), 
                caps[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    print("Training Complete!")
    
    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, caps in val_loader:
            imgs = imgs.to(device)
            caps = caps.to(device)

            outputs = model(imgs, caps[:, :-1])

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                caps[:, 1:].reshape(-1)
            )
            val_loss += loss.item() 

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

if __name__ == "__main__":
    train()

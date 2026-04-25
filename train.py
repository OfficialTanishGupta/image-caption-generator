import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd

from utils.dataset import FlickrDataset
from utils.vocab import Vocabulary
from utils.dataloader import MyCollate
from models.model import ImageCaptionModel

def train():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # 1. Build Vocabulary
    # ======================
    print("Building vocab...")
    vocab = Vocabulary()
    df = pd.read_csv("data/captions.txt")
    vocab.build_vocab(df["caption"].tolist())

    vocab_size = len(vocab)
    pad_idx = vocab.stoi["<pad>"]
    print(f"Vocab size: {vocab_size}")

    # ======================
    # 2. Transforms & Data
    # ======================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(
        root_dir="data/images",
        captions_file="data/captions.txt",
        vocab=vocab,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2, 
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    # ======================
    # 3. Model Setup
    # ======================
    embed_size = 256
    hidden_size = 256
    learning_rate = 3e-4
    num_epochs = 20
    model = ImageCaptionModel(embed_size, hidden_size, vocab_size).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "encoder" in name and param.requires_grad:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": 1e-5},
        {"params": decoder_params, "lr": 3e-4}
    ])
    
    
    
        

    # ======================
    # 4. Training Loop
    # ======================
    print("Training started...")

    for epoch in range(num_epochs):
        for batch_idx, (imgs, caps) in enumerate(loader):
            imgs = imgs.to(device)
            caps = caps.to(device)

            # Forward pass
            outputs = model(imgs, caps[:, :-1])

            # Loss calculation
            loss = criterion(
                outputs.reshape(-1, vocab_size), 
                caps[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")

        # Save the model state
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    print("Training Complete!")
    
    
    
    

# This block is required on Windows to use num_workers > 0
if __name__ == "__main__":
    train()

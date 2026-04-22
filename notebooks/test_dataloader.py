import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.dataset import FlickrDataset
from utils.vocab import Vocabulary
from utils.dataloader import MyCollate

# Build vocab
dataset_temp = FlickrDataset(
    root_dir="data/images",
    captions_file="data/captions.txt",
    vocab=Vocabulary()
)

captions = [dataset_temp.df.iloc[i, 1] for i in range(len(dataset_temp))]
vocab = Vocabulary()
vocab.build_vocab(captions)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataset = FlickrDataset(
    root_dir="data/images",
    captions_file="data/captions.txt",
    vocab=vocab,
    transform=transform
)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=MyCollate(pad_idx=vocab.stoi["<pad>"])
)

# Test one batch
for imgs, caps in loader:
    print("Images shape:", imgs.shape)
    print("Captions shape:", caps.shape)
    print(caps)
    break
from utils.dataset import FlickrDataset
from utils.vocab import Vocabulary

vocab = Vocabulary(freq_threshold=5)

dataset = FlickrDataset(
    root_dir="data/images",
    captions_file="data/captions.txt",
    vocab=vocab
)

print(f"Building vocab from {len(dataset)} items...")
captions = dataset.df.iloc[:, 1].tolist()
dataset.vocab.build_vocab(captions)

# 4. Now __getitem__ will work correctly
img, cap = dataset[0]

print("Vocab Size:", len(dataset.vocab))
print("Numerical Caption (Tensor):", cap)

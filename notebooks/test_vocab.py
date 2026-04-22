from utils.dataset import FlickrDataset
from utils.vocab import Vocabulary

dataset = FlickrDataset(
    root_dir="data/images",
    captions_file="data/captions.txt"
)

captions = [dataset[i][1] for i in range(len(dataset))]

vocab = Vocabulary(freq_threshold=5)
vocab.build_vocab(captions)

print("Vocab size:", len(vocab))

sample = captions[0]
print("Original:", sample)
print("Numericalized:", vocab.numericalize(sample))
from utils.dataset import FlickrDataset

dataset = FlickrDataset(
    root_dir="data/images",
    captions_file="data/captions.txt"
)

print(len(dataset))

img, cap = dataset[0]

print(cap)
import torch
from torch.nn.utils.rnn import pad_sequence

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images)

        captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return images, captions
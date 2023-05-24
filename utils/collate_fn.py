import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        caption_lengths = [len(caption) for caption in targets]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        multilabel = torch.tensor(np.array([item[2] for item in batch]))
        
        return imgs, targets, caption_lengths, multilabel
    
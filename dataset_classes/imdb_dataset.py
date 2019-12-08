import torch
from torch.utils.data import Dataset
import numpy as np
from utils.file_utils import unpickler


class IMDBDataset(Dataset):
    def __init__(self, config, is_training):
        self.config = config
        if is_training:
            self.dataset = unpickler(config.training_data)
        else:
            self.dataset = unpickler(config.dev_data)

        if self.config.small_data_size > 0:
            self.dataset = self.dataset[: self.config.small_data_size]

    def __len__(self):
        return len(self.dataset)

    def get_gt(self, indices=None):
        if indices is None:
            indices = range(len(self.dataset))
        out_list = []
        for idx in indices:
            gt = self.dataset[idx]["label"]
            out_list.append(gt)
        return out_list

    def get_seq_len(self, indices=None):
        if indices is None:
            indices = range(len(self.dataset))
        out_list = []
        for idx in indices:
            gt = self.dataset[idx]["num_tokens"]
            out_list.append(gt)
        return out_list

    def __get__(self, index):

        out_dict = {
            "features": torch.tensor(self.dataset[index]["text"]),
            "word_mask": torch.tensor(
                self.dataset[index]["word_mask"], dtype=torch.float
            ),
            "label": torch.tensor(self.data[index]["label"]),
            "index": torch.tensor(index),
        }

        return out_dict

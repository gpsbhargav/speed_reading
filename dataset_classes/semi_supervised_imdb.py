import torch
from torch.utils.data import Dataset
import numpy as np
from utils.file_utils import unpickler

import pdb


class LabeledDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_all = unpickler(config.training_data)
        self.labeled_indices = self.init_labeled_indices(self.data_all)

    def init_labeled_indices(self, data_all):
        labeled_indices = []
        for i, data in enumerate(data_all):
            if data["label"] != -1:
                labeled_indices.append(i)
        return labeled_indices

    def is_everything_labeled(self):
        if len(self.labeled_indices) >= len(self.data_all):
            return True
        else:
            return False

    def __len__(self):
        return len(self.labeled_indices)

    def get_gt(self, indices=None):
        if indices is None:
            indices = range(len(self.labeled_indices))
        out_list = []
        for idx in indices:
            gt = self.data_all[self.labeled_indices[idx]]["label"]
            assert gt in [0, 1]
            out_list.append(gt)
        return torch.tensor(out_list)

    def get_seq_len(self, indices=None):
        if indices is None:
            indices = range(len(self.labeled_indices))
        out_list = []
        for idx in indices:
            gt = self.data_all[self.labeled_indices[idx]]["num_tokens"]
            out_list.append(gt)
        return torch.tensor(out_list, dtype=torch.float)

    def __getitem__(self, index):
        out_dict = {
            "features": torch.tensor(
                self.data_all[self.labeled_indices[index]]["text"]
            ),
            "word_mask": torch.tensor(
                self.data_all[self.labeled_indices[index]]["word_mask"],
                dtype=torch.float,
            ),
            "label": torch.tensor(self.data_all[self.labeled_indices[index]]["label"]),
            "indices": torch.tensor(index),
        }

        return out_dict

    def add_to_labeled_set(self, newly_labeled_indices, labels):
        assert len(newly_labeled_indices) == len(labels)
        if len(labels) == 0:
            return
        for i, idx in enumerate(newly_labeled_indices):
            self.data_all[idx]["label"] = labels[i]
            self.labeled_indices.append(idx)


class UnLabeledDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_all = unpickler(config.training_data)
        self.unlabeled_indices = self.init_unlabeled_indices(self.data_all)

    def init_unlabeled_indices(self, data_all):
        unlabeled_indices = []
        for i, data in enumerate(data_all):
            if data["label"] == -1:
                unlabeled_indices.append(i)
        return unlabeled_indices

    def __len__(self):
        return len(self.unlabeled_indices)

    def get_seq_len(self, indices=None):
        if indices is None:
            indices = range(len(self.unlabeled_indices))
        out_list = []
        for idx in indices:
            gt = self.data_all[self.unlabeled_indices[idx]]["num_tokens"]
            out_list.append(gt)
        return torch.tensor(out_list, dtype=torch.float)

    def __getitem__(self, index):
        out_dict = {
            "features": torch.tensor(
                self.data_all[self.unlabeled_indices[index]]["text"]
            ),
            "word_mask": torch.tensor(
                self.data_all[self.unlabeled_indices[index]]["word_mask"],
                dtype=torch.float,
            ),
            "indices": torch.tensor(index),
        }

        return out_dict

    def get_unlabeled_indices(self):
        return self.unlabeled_indices

    def remove_from_unlabeled_set(self, newly_labeled_indices):
        if len(newly_labeled_indices) == 0:
            return
        new_unlabeled_list = [
            i for i in self.unlabeled_indices if i not in newly_labeled_indices
        ]
        self.unlabeled_indices = new_unlabeled_list


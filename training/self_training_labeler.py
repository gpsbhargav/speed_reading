import sys

sys.path.append("../")

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_classes.semi_supervised_imdb import LabeledDataset, UnLabeledDataset

import pdb


class SelfTrainingLabeler:
    def __init__(self, config):
        self.config = config
        self.dataset = UnLabeledDataset(config)
        self.model = None
        self.batch_size = config.dev_batch_size
        self.device = torch.device("cuda:{}".format(0))

    def set_model(self, model):
        self.model = model

    def get_labels(self):
        assert self.model is not None

        self.model.eval()

        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=self.config.data_loader_num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
        )

        all_class_probabilities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                model_outputs = self.model(batch)
                model_outputs = (
                    model_outputs["class_probabilities"].detach().cpu().numpy()
                )
                all_class_probabilities.append(model_outputs)

        all_class_probabilities = np.concatenate(all_class_probabilities, axis=0)

        predicted_classes = np.argmax(all_class_probabilities, axis=-1)
        max_probs = np.max(all_class_probabilities, axis=-1)

        unlabeled_indices = self.dataset.get_unlabeled_indices()

        newly_labeled_indices = []
        new_labels = []

        for i in range(len(predicted_classes)):
            if max_probs[i] >= self.config.pseudo_label_threshold:
                newly_labeled_indices.append(unlabeled_indices[i])
                new_labels.append(predicted_classes[i])

        self.dataset.remove_from_unlabeled_set(newly_labeled_indices)

        return {
            "newly_labeled_indices": newly_labeled_indices,
            "new_labels": new_labels,
        }


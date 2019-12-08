import os

from dataset_classes.imdb_dataset import IMDBDataset
from models.models import Model1
from utils.file_utils import create_dir, Logger
from utils.metrics import compute_accuracy

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json

import pdb


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.dataset = IMDBDataset(config, is_training=False)
        print("Dev set len: {}".format(len(self.dataset)))
        self.model = None
        self.batch_size = config.dev_batch_size
        self.device = torch.device("cuda:{}".format(0))

    def set_model(self, model):
        self.model = model

    def run_model(self):
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
        gt_classes = self.dataset.get_gt()

        classification_accuracy = compute_accuracy(
            pred=predicted_classes, gt=gt_classes
        )

        return {"classification_accuracy": classification_accuracy}

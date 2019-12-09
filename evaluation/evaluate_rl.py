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

    def form_batch_times_seq_len_tensor(self, tensor_list_in):
        assert len(tensor_list_in[0].shape) == 1
        tensor_out = [t.unsqueeze(1) for t in tensor_list_in]
        tensor_out = torch.cat(tensor_out, dim=1)
        assert tensor_out.shape[0] == tensor_list_in[0].shape[0]
        assert tensor_out.shape[1] == len(tensor_list_in)
        return tensor_out

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
        all_num_read = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                model_outputs = self.model(batch, supervised=False)
                class_probabilities = (
                    model_outputs["class_probabilities"].detach().cpu().numpy()
                )
                read_tokens = model_outputs["actions"]
                read_tokens = self.form_batch_times_seq_len_tensor(read_tokens)
                num_read = read_tokens.sum(axis=-1)
                all_num_read.append(num_read)
                all_class_probabilities.append(class_probabilities)

        all_class_probabilities = np.concatenate(all_class_probabilities, axis=0)

        all_num_read = np.concatenate(all_num_read, axis=-1)
        num_gt_words = self.dataset.get_seq_len()
        fraction_of_words_read = all_num_read / num_gt_words
        avg_read_fraction = fraction_of_words_read.mean().item()

        predicted_classes = np.argmax(all_class_probabilities, axis=-1)
        gt_classes = self.dataset.get_gt()

        classification_accuracy = compute_accuracy(
            pred=predicted_classes, gt=gt_classes
        )

        return {
            "classification_accuracy": classification_accuracy,
            "avg_read_fraction": avg_read_fraction,
        }

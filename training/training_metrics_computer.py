import os

from dataset_classes.imdb_dataset import IMDBDataset
from models.models import Model1
from utils.file_utils import create_dir, Logger
from utils.metrics import compute_accuracy
from training.rl_classes import ActorCriticForFullTrainingSet

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json

import pdb


class SupervisedTrainingMetricComputer:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.dev_batch_size
        self.device = torch.device("cuda:{}".format(0))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def run_model(self, model, dataset):
        model.eval()

        data_loader = DataLoader(
            dataset,
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
                model_outputs = model(batch)
                model_outputs = model_outputs["class_probabilities"].detach()
                all_class_probabilities.append(model_outputs)

        all_class_probabilities = torch.cat(all_class_probabilities, dim=0)

        predicted_classes = torch.argmax(all_class_probabilities, dim=-1)
        gt_classes = dataset.get_gt()

        loss = self.cross_entropy_loss(
            all_class_probabilities, gt_classes.to(all_class_probabilities.device)
        )

        classification_accuracy = compute_accuracy(
            pred=predicted_classes, gt=gt_classes
        )

        return {"classification_accuracy": classification_accuracy, "loss": loss.item()}


class RLTrainingMetricComputer:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.dev_batch_size
        self.device = torch.device("cuda:{}".format(0))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def run_model(self, model, dataset):
        ac_metrics_computer = ActorCriticForFullTrainingSet(self.config, dataset)
        model.eval()

        data_loader = DataLoader(
            dataset,
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

        metrics = {
            "classification_reward": 0,
            "sum_cumulative_rewards": 0,
            "classification_accuracies": 0,
            "fraction_of_words_read": 0,
            "total_loss": 0,
            "actor_loss": 0,
            "critic_loss": 0,
            "entropy_loss": 0,
            "classification_loss": 0,
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                model_outputs = model(batch, supervised=False)
                m = ac_metrics_computer.compute_loss(
                    class_probabilities_in=model_outputs["class_probabilities"],
                    actions_in=model_outputs["actions"],
                    action_log_probs_in=model_outputs["action_log_probs"],
                    state_values_in=model_outputs["state_values"],
                    entropies_in=model_outputs["entropies"],
                    indices_in=batch["indices"],
                )
                for key, value in m.items():
                    metrics[key] += value

        for key, value in metrics.items():
            metrics[key] = value / len(dataset)

        return metrics

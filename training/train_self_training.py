import argparse
import random
import time
import os
import glob
import sys

sys.path.append("../")

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AdamW
from torch.optim import RMSprop


from models.models import Model1

# from dataset_classes.imdb_dataset import IMDBDataset
from dataset_classes.semi_supervised_imdb import LabeledDataset, UnLabeledDataset
from utils.file_utils import create_dir, Logger, JSONLLogger
from evaluation.evaluate_supervised import Evaluator
from training.self_training_labeler import SelfTrainingLabeler
from utils.metrics import compute_accuracy
from training.training_metrics_computer import SupervisedTrainingMetricComputer

import pdb

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


parser = argparse.ArgumentParser()


parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--dev_batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=5e-4)
# parser.add_argument("--epochs", type=int, default=10)
# parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
# parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=0.1)
parser.add_argument("--word_embedding_size", type=int, default=300)
parser.add_argument("--lstm_hidden_size", type=int, default=128)
parser.add_argument("--state_vector_size", type=int, default=25)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--train_word_embeddings", action="store_true")

parser.add_argument(
    "--optimizer", type=str, choices=["adamw", "rmsprop"], required=True
)

parser.add_argument("--patience_for_adding_data", type=int, default=4)
parser.add_argument("--patience_before_adding_data", type=int, default=4)

parser.add_argument("--pseudo_label_threshold", type=float, default=0.7)

parser.add_argument("--resume_training", action="store_true")
parser.add_argument("--training_data", type=str, required=True)
parser.add_argument("--dev_data", type=str, required=True)
parser.add_argument("--embedding_matrix", type=str, required=True)
parser.add_argument("--checkpoint_name", type=str, default="snapshot.pt")

parser.add_argument("--fp16", action="store_true")
parser.add_argument("--log_every", type=int, default=5)
# parser.add_argument("--save_every", type=int, default=500)
# parser.add_argument("--early_stopping_patience", type=int, default=4)
parser.add_argument("--small_data_size", type=int, default=-1)
parser.add_argument("--eval_only", action="store_true")
# parser.add_argument("--training_topup", action="store_true")

parser.add_argument("--data_loader_num_workers", type=int, default=8)
parser.add_argument("--model_to_use", type=int, choices=[1], default=1)

parser.add_argument("--max_seq_len", type=int, default=250)

config = parser.parse_args()

config.n_gpu = torch.cuda.device_count()


if config.fp16:
    try:
        import apex
        from apex import amp
    except ImportError:
        raise ImportError("Please install nvidia apex to use fp16 training.")


def save_model(config, logger, model, optimizer, iterations, best_acc):
    logger.write_log("Saving model")
    snapshot_prefix = os.path.join(config.save_dir, config.checkpoint_name)
    snapshot_path = snapshot_prefix
    # save model without the "DataParallel" part
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()
    state = {
        "iteration": iterations,
        "model_state_dict": model_state_dict,
        "best_acc": best_acc,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if config.fp16:
        state["amp_state_dict"] = amp.state_dict()
    torch.save(state, snapshot_path)
    for f in glob.glob(snapshot_prefix + "*"):
        if f != snapshot_path:
            os.remove(f)
    logger.write_log("Model saved")


def load_model(config, logger, model, optimizer, device):
    if os.path.isfile(os.path.join(config.save_dir, config.checkpoint_name)):
        logger.write_log("=> loading checkpoint")
        checkpoint = torch.load(
            os.path.join(config.save_dir, config.checkpoint_name), map_location="cpu"
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        best_dev_accuracy = checkpoint["best_acc"]
        iterations = checkpoint["iteration"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if config.fp16:
            amp.load_state_dict(checkpoint["amp_state_dict"])

        logger.write_log(
            "=> loaded checkpoint. Resuming iteration {}".format(
                checkpoint["iteration"]
            )
        )

        return {"iterations": iterations, "best_dev_accuracy": best_dev_accuracy}


create_dir(config.save_dir)

logger = Logger(config.save_dir + "training_log_all.log")

if not config.eval_only:
    training_metrics_logger = JSONLLogger(
        config.save_dir + "training_metrics_log.jsonl"
    )
    dev_metrics_logger = JSONLLogger(config.save_dir + "dev_metrics_log.jsonl")
    epoch_logger = JSONLLogger(config.save_dir + "epoch_logs.jsonl")

logger.write_log("Reading data")

labeled_dataset = LabeledDataset(config)

self_training_labeler = SelfTrainingLabeler(config)

training_set_metrics_computer = SupervisedTrainingMetricComputer(config)

evaluator = Evaluator(config)

model = Model1(config)

if config.optimizer == "rmsprop":
    optimizer = RMSprop(model.parameters(), lr=config.lr)
elif config.optimizer == "adamw":
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)


cross_entropy_loss = nn.CrossEntropyLoss()

logger.write_log("Labeled training data size:{}".format(len(labeled_dataset)))

total_loss_since_last_time = 0
stop_training_flag = False
iterations = 0
best_dev_accuracy = -1
patience_counter_for_adding_data = 0
patience_counter_before_adding_data = 0
epoch_counter = 0


device = torch.device("cuda:{}".format(0))

model.to(device)

if config.fp16:
    # apex.amp.register_half_function(torch, "einsum")
    amp.register_float_function(torch, "sigmoid")
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if config.resume_training:
    if os.path.isfile(os.path.join(config.save_dir, config.checkpoint_name)):
        logger.write_log("=> loading checkpoint")
        checkpoint = torch.load(
            os.path.join(config.save_dir, config.checkpoint_name), map_location="cpu"
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        best_dev_accuracy = checkpoint["best_acc"]
        iterations = checkpoint["iteration"]
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if config.fp16:
            amp.load_state_dict(checkpoint["amp_state_dict"])

        logger.write_log(
            "=> loaded checkpoint. Resuming iteration {}".format(
                checkpoint["iteration"]
            )
        )

# model = nn.DataParallel(model)

if config.eval_only:
    evaluator.set_model(model)
    dev_metrics = evaluator.run_model()
    dev_accuracy = dev_metrics["classification_accuracy"]
    print("Dev acc: {}".format(dev_accuracy))
    stop_training_flag = True


if not config.eval_only:
    logger.write_log("Training now")

optimizer.zero_grad()


start = time.time()

while patience_counter_for_adding_data <= config.patience_for_adding_data:

    if config.eval_only:
        break

    dataloader = DataLoader(
        labeled_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=config.data_loader_num_workers,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )

    patience_counter_before_adding_data = 0

    best_dev_acc_at_this_stage = best_dev_accuracy

    while patience_counter_before_adding_data <= config.patience_before_adding_data:

        for batch_idx, batch in enumerate(dataloader):
            model.train()

            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            model_outputs = model(batch)

            all_class_probabilities = model_outputs["class_probabilities"]

            total_loss = cross_entropy_loss(
                model_outputs["class_probabilities"], batch["label"]
            )

            if torch.isnan(total_loss).item():
                logger.write_log(
                    "Loss became nan in iteration {}. Training stopped".format(
                        iterations
                    )
                )
                stop_training_flag = True
                break

            training_metrics_logger.write_log({"training_loss": total_loss.item()})

            total_loss_since_last_time += total_loss.item()

            if config.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), config.max_grad_norm
                )
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            iterations += 1

            if (batch_idx + 1) % config.log_every == 0:
                avg_loss_since_last_time = total_loss_since_last_time / config.log_every
                total_loss_since_last_time = 0

                indices = batch["indices"].detach().cpu()
                predicted_classes = torch.argmax(all_class_probabilities, dim=-1)
                gt_classes = labeled_dataset.get_gt(indices)
                classification_accuracy = compute_accuracy(
                    pred=predicted_classes, gt=gt_classes
                )
                logger.write_log("- - - - - - - - - - - - - - - - - - - -")
                logger.write_log(
                    "Time:{:.1f}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, Acc: {}".format(
                        time.time() - start,
                        iterations,
                        avg_loss_since_last_time,
                        total_loss.item(),
                        classification_accuracy,
                    )
                )

        evaluator.set_model(model)
        dev_metrics = evaluator.run_model()
        dev_accuracy = dev_metrics["classification_accuracy"]
        dev_metrics_logger.write_log({"classification_accuracy": dev_accuracy})

        training_set_metrics = training_set_metrics_computer.run_model(model, labeled_dataset)

        log_dict = training_set_metrics
        log_dict["dev_classification_accuracy"] = dev_metrics["classification_accuracy"]

        epoch_logger.write_log(log_dict)

        logger.write_log("================================")
        logger.write_log("Dev acc: {}".format(dev_accuracy))
        logger.write_log("================================")

        if dev_accuracy < best_dev_accuracy:
            patience_counter_before_adding_data += 1
        else:
            best_dev_accuracy = dev_accuracy
            patience_counter_before_adding_data = 0
            save_model(config, logger, model, optimizer, iterations, best_dev_accuracy)

        if stop_training_flag is True:
            break

    if stop_training_flag is True:
        break

    if not labeled_dataset.is_everything_labeled():
        self_training_labeler.set_model(model)
        new_labels_info = self_training_labeler.get_labels()
        labeled_dataset.add_to_labeled_set(
            newly_labeled_indices=new_labels_info["newly_labeled_indices"],
            labels=new_labels_info["new_labels"],
        )
        logger.write_log("* * * * * * * * * * * *")
        logger.write_log(
            "Added {} new datapoints to labeled set".format(
                len(new_labels_info["new_labels"])
            )
        )
        logger.write_log("Labeled data size: {}".format(len(labeled_dataset)))
        logger.write_log("* * * * * * * * * * * *")
    else:
        logger.write_log("* * * * * * * * * * * *")
        logger.write_log("No more unlabeled data left")
        logger.write_log("* * * * * * * * * * * *")

    # is current best_dev_acc more than previous dev acc + 0.5 ?
    # This means there's no significant benefit of adding new data
    if best_dev_accuracy < best_dev_acc_at_this_stage + 1.0:
        patience_counter_for_adding_data += 1
        load_model(config, logger, model, optimizer, device)
    else:
        patience_counter_for_adding_data = 0


logger.write_log("================================")
logger.write_log("Best Dev acc: {}".format(best_dev_accuracy))
logger.write_log("================================")

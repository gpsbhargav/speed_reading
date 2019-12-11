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
from dataset_classes.imdb_dataset import IMDBDataset
from utils.file_utils import create_dir, Logger
from evaluation.evaluate_rl import Evaluator
from training.rl_classes import ActorCritic1

# from utils.metrics import compute_accuracy

import pdb

import pdb

random.seed(42)


parser = argparse.ArgumentParser()


parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--dev_batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_grad_norm", type=float, default=0.1)
parser.add_argument("--word_embedding_size", type=int, default=300)
parser.add_argument("--lstm_hidden_size", type=int, default=128)
parser.add_argument("--state_vector_size", type=int, default=25)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--train_word_embeddings", action="store_true")

parser.add_argument(
    "--optimizer", type=str, choices=["adamw", "rmsprop"], required=True
)

parser.add_argument("--resume_training", action="store_true")
parser.add_argument("--training_data", type=str, required=True)
parser.add_argument("--dev_data", type=str, required=True)
parser.add_argument("--embedding_matrix", type=str, required=True)
parser.add_argument("--checkpoint_name", type=str, default="snapshot.pt")

parser.add_argument("--fp16", action="store_true")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--early_stopping_patience", type=int, default=20)
parser.add_argument("--small_data_size", type=int, default=-1)
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--training_topup", action="store_true")

parser.add_argument("--data_loader_num_workers", type=int, default=0)
parser.add_argument("--model_to_use", type=int, choices=[1], default=1)

parser.add_argument("--max_seq_len", type=int, default=250)

parser.add_argument("--warm_start_checkpoint", type=str, default="")
parser.add_argument("--sent_skip_reward_multiplier", type=float, default=0.5)
parser.add_argument("--step_reward_multiplier", type=float, default=0.1)

parser.add_argument("--loss_weight_entropy", type=float, default=0.1)
parser.add_argument("--loss_weight_actor", type=float, default=10.0)
parser.add_argument("--loss_weight_critic", type=float, default=1.0)
parser.add_argument("--loss_weight_classification", type=float, default=1.0)

parser.add_argument("--greedy_action", action="store_true")

parser.add_argument("--train_only_rl_agents", action="store_true")


parser.add_argument("--rl_algo", type=str, choices=["ac"], required=True)


config = parser.parse_args()

config.n_gpu = torch.cuda.device_count()

if config.fp16:
    try:
        import apex
        from apex import amp
    except ImportError:
        raise ImportError("Please install nvidia apex to use fp16 training.")

create_dir(config.save_dir)

logger = Logger(config.save_dir + "training_log.log")

logger.write_log("Reading data")

dataset = IMDBDataset(config, is_training=True)

logger.write_log("Creating dataloader")

dataloader = DataLoader(
    dataset,
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

logger.write_log("Building model")

model = Model1(config)

num_train_steps = int(
    (len(dataset) / config.train_batch_size / config.gradient_accumulation_steps)
    * config.epochs
)


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

logger.write_log("Training data size:{}".format(len(dataset)))

total_loss_since_last_time = 0
# dev_predictions_best_model = None
stop_training_flag = False
num_evaluations_since_last_best_dev_acc = 0

iterations = 0
best_dev_accuracy = -1
start_epoch = 0

best_dev_metrics = {}

device = torch.device("cuda:{}".format(0))

model.to(device)

if config.fp16:
    # apex.amp.register_half_function(torch, "einsum")
    amp.register_float_function(torch, "sigmoid")
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if config.warm_start_checkpoint != "":
    logger.write_log("[Warm start] Loading model parameters")
    if os.path.isfile(config.warm_start_checkpoint):
        checkpoint = torch.load(config.warm_start_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.fp16:
            amp.load_state_dict(checkpoint["amp_state_dict"])

if config.resume_training:
    if os.path.isfile(os.path.join(config.save_dir, config.checkpoint_name)):
        logger.write_log("=> loading checkpoint")
        checkpoint = torch.load(
            os.path.join(config.save_dir, config.checkpoint_name), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if not config.training_topup:
            best_dev_metrics = checkpoint["best_dev_metrics"]
            start_epoch = checkpoint["epoch"]
            best_dev_accuracy = checkpoint["best_acc"]
            iterations = checkpoint["iteration"]
            # if config.use_lr_sched:
            #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if config.fp16:
            amp.load_state_dict(checkpoint["amp_state_dict"])

        logger.write_log(
            "=> loaded checkpoint. Resuming epoch {}, iteration {}".format(
                checkpoint["epoch"] + 1, checkpoint["iteration"]
            )
        )

model = nn.DataParallel(
    model,
    #    device_ids=available_gpus[1:],
    #    output_device=available_gpus[0],
)

evaluator = Evaluator(config)

if config.rl_algo == "ac":
    rl_algorithm = ActorCritic1(config, dataset)

if not config.eval_only:
    logger.write_log("Training now")

optimizer.zero_grad()

start = time.time()

for epoch in range(start_epoch, config.epochs):
    for batch_idx, batch in enumerate(dataloader):

        if config.eval_only:
            break

        if iterations > num_train_steps:
            logger.write_log("Reached maximum number of iterations")
            stop_training_flag = True
            break

        model.train()

        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        model_outputs = model(batch, supervised=False)

        rl_algorithm_output = rl_algorithm.compute_loss(
            class_probabilities_in=model_outputs["class_probabilities"],
            actions_in=model_outputs["actions"],
            action_log_probs_in=model_outputs["action_log_probs"],
            state_values_in=model_outputs["state_values"],
            entropies_in=model_outputs["entropies"],
            indices_in=batch["indices"],
        )

        total_loss = rl_algorithm_output["loss"]
        mean_classification_accuracy = rl_algorithm_output[
            "mean_classification_accuracy"
        ]
        avg_read_fraction = rl_algorithm_output["avg_read_fraction"]

        if torch.isnan(total_loss).item():
            logger.write_log(
                "Loss became nan in iteration {}. Training stopped".format(iterations)
            )
            stop_training_flag = True
            break

        # if config.n_gpu > 1:
        #     total_loss = total_loss.mean()

        if config.gradient_accumulation_steps > 1:
            total_loss = total_loss / config.gradient_accumulation_steps

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

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            # if config.use_lr_sched:
            #     scheduler.step()
            optimizer.zero_grad()
            iterations += 1

        if (batch_idx + 1) % config.log_every == 0:
            avg_loss_since_last_time = total_loss_since_last_time / config.log_every
            total_loss_since_last_time = 0

            logger.write_log("- - - - - - - - - - - - - - - - - - - -")
            logger.write_log(
                "Time:{:.1f}, Epoch:{}/{}, Iteration:{}, Avg_train_loss:{:.4f}, batch_loss:{:.4f}, Acc: {:.2f}, Avg read fraction: {:.4f}".format(
                    time.time() - start,
                    epoch + 1,
                    config.epochs,
                    iterations,
                    avg_loss_since_last_time,
                    total_loss.item(),
                    mean_classification_accuracy,
                    avg_read_fraction,
                )
            )

        if (batch_idx + 1) % config.save_every == 0:
            logger.write_log("Saving model")
            snapshot_prefix = os.path.join(config.save_dir, config.checkpoint_name)
            snapshot_path = snapshot_prefix
            # save model without the "DataParallel" part
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            state = {
                "epoch": epoch,
                "iteration": iterations,
                "model_state_dict": model_state_dict,
                "best_acc": best_dev_accuracy,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_metrics": best_dev_metrics,
            }
            # if config.use_lr_sched:
            #     state["scheduler_state_dict"] = scheduler.state_dict()
            if config.fp16:
                state["amp_state_dict"] = amp.state_dict()
            torch.save(state, snapshot_path)
            for f in glob.glob(snapshot_prefix + "*"):
                if f != snapshot_path:
                    os.remove(f)

    if stop_training_flag == True:
        break

    logger.write_log("Evaluating on dev set")

    evaluator.set_model(model)

    dev_metrics = evaluator.run_model()

    logger.write_log("==================================")
    logger.write_log("Dev set:")
    logger.write_log("Metrics: {}".format(dev_metrics))
    logger.write_log("==================================")

    dev_accuracy = dev_metrics["classification_accuracy"]

    # update best validation set accuracy
    if dev_accuracy > best_dev_accuracy:

        best_dev_metrics = dev_metrics

        num_evaluations_since_last_best_dev_acc = 0

        # found a model with better validation set accuracy

        best_dev_accuracy = dev_accuracy
        snapshot_prefix = os.path.join(config.save_dir, "best_snapshot")
        snapshot_path = snapshot_prefix + "_dev_f1_{}_iter_{}_model.pt".format(
            dev_accuracy, iterations
        )

        # save model, delete previous 'best_snapshot' files
        # save model without the "DataParallel" part
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()
        state = {
            "epoch": epoch,
            "iteration": iterations,
            "model_state_dict": model_state_dict,
            "best_acc": best_dev_accuracy,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dev_metrics": best_dev_metrics,
        }
        # if config.use_lr_sched:
        #     state["scheduler_state_dict"] = scheduler.state_dict()
        if config.fp16:
            state["amp_state_dict"] = amp.state_dict()
        logger.write_log("Saving model")
        torch.save(state, snapshot_path)
        for f in glob.glob(snapshot_prefix + "*"):
            if f != snapshot_path:
                os.remove(f)

    else:
        num_evaluations_since_last_best_dev_acc += 1

    if num_evaluations_since_last_best_dev_acc > config.early_stopping_patience:
        logger.write_log(
            "Training stopped because dev acc hasn't increased in {} epochs.".format(
                config.early_stopping_patience
            )
        )
        logger.write_log("Best dev set metrics = {}".format(best_dev_metrics))
        break

    if config.eval_only:
        break

logger.write_log("=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")
logger.write_log("Best dev metrics: {}".format(best_dev_metrics))
logger.write_log("=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")

logger.write_log("Done")


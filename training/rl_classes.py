import torch
import torch.nn as nn
import json

import pdb


class ActorCritic1:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.log_file = config.save_dir + "reward_logs.jsonl"

        if not self.config.training_topup:
            self.create_log_file()

    def create_log_file(self):
        with open(self.log_file, "w") as out_file:
            pass

    def write_log(self, dict_in):
        with open(self.log_file, "a") as out_file:
            json_str = json.dumps(dict_in)
            out_file.write(json_str)
            out_file.write("\n")

    def compute_accuracy(self, class_probabilities, indices):
        predicted_classes = torch.argmax(class_probabilities, dim=-1)
        gt_classes = self.dataset.get_gt(indices=indices)

        accuracies = []
        for i in range(len(predicted_classes)):
            if predicted_classes[i] == gt_classes[i]:
                accuracies.append(1.0)
            else:
                accuracies.append(0.0)
        return torch.tensor(accuracies)

    def compute_classification_reward(self, class_probabilities, indices, accuracies):
        gt_classes = self.dataset.get_gt(indices=indices)
        classification_rewards = torch.zeros_like(
            accuracies, device=class_probabilities.device
        )
        for i in range(len(accuracies)):
            if accuracies[i] > 0:
                classification_rewards[i] = 1.0
            else:
                classification_rewards[i] = class_probabilities[i][gt_classes[i]]

        return classification_rewards

    def compute_step_rewards(self, actions):
        """
        actions: list of length num_sents. Each item is torch tensor of length batch_size
        """
        batch_size = actions[0].shape[0]
        seq_len = len(actions)
        step_rewards = torch.zeros(
            [batch_size, seq_len], dtype=torch.float, device=actions[0].device
        )
        for j in range(seq_len):
            for i in range(batch_size):
                action_taken = actions[j][i]
                # The actions could be floats or ints and i want to know if they are 0 or 1.
                # Comparing with 0.5 is one way to do it.
                if action_taken > 0.5:
                    reward = -1 / seq_len
                else:
                    reward = -self.config.sent_skip_reward_multiplier / seq_len
                step_rewards[i][j] = reward

        return step_rewards

    def compute_cumulative_reward(self, step_rewards, classification_rewards):
        """
        step_rewards: torch tensor. batch_size x num_sents
        classification_rewards: torch tensor of length batch_size
        """

        # for each item in batch,
        #### reverse step_rewards and compute cumsum and reverse again
        #### multiply above vector by config.step_reward_multiplier
        #### add classification_rewards to all items in above vector
        #### store above vector in a list
        # concatenate list into a batch_size x seq_len tensor and return it

        assert step_rewards.shape[0] == len(classification_rewards)

        cumulative_reward = []
        for i in range(len(classification_rewards)):
            stp_rwrds = step_rewards[i]

            # reverse stp_rwrds
            inv_idx = torch.arange(stp_rwrds.size(0) - 1, -1, -1).long()
            stp_rwrds = stp_rwrds.index_select(0, inv_idx.to(step_rewards.device))

            # calc cumsum
            cum_sum = stp_rwrds.cumsum(dim=-1)

            # reverse cum_sum
            inv_idx = torch.arange(cum_sum.size(0) - 1, -1, -1).long()
            cum_sum = cum_sum.index_select(0, inv_idx.to(step_rewards.device))

            cum_reward = (
                self.config.step_reward_multiplier * cum_sum + classification_rewards[i]
            )

            cumulative_reward.append(cum_reward.unsqueeze(0))

        cumulative_reward = torch.cat(cumulative_reward, dim=0)

        assert cumulative_reward.shape[0] == step_rewards.shape[0]
        assert cumulative_reward.shape[1] == step_rewards.shape[1]

        return cumulative_reward

    def form_batch_times_seq_len_tensor(self, tensor_list_in):
        assert len(tensor_list_in[0].shape) == 1
        tensor_out = [t.unsqueeze(1) for t in tensor_list_in]
        tensor_out = torch.cat(tensor_out, dim=1)
        assert tensor_out.shape[0] == tensor_list_in[0].shape[0]
        assert tensor_out.shape[1] == len(tensor_list_in)
        return tensor_out

    def compute_actor_loss(self, cumulative_rewards, action_log_probs, values):
        """
        cumulative_rewards: torch tensor. batch_size x num_sents
        action_log_probs: torch tensor. batch_size x num_sents
        values: torch tensor. batch_size x num_sents
        """
        values_detached = values.detach()
        reduced_cumulative_rewards = cumulative_rewards - values_detached
        loss = reduced_cumulative_rewards * action_log_probs
        # - because I want to maximize the log probabilities of good actions
        loss = -(loss.sum(dim=-1)).mean()
        return loss

    def compute_critic_loss(self, cumulative_rewards_in, values_in):
        """
        cumulative_rewards: torch tensor. batch_size x num_sents
        values: torch tensor. batch_size x num_sents
        """
        mse = self.mse_loss(input=values_in, target=cumulative_rewards_in)
        return mse

    def compute_entropy_loss(self, entropies):
        """
        entropies: torch tensor. batch_size x num_sents
        """
        return entropies.mean()

    def compute_classification_loss(self, class_probabilities, indices):
        gt_classes = self.dataset.get_gt(indices=indices)
        cross_entropy_loss = self.cross_entropy_loss(class_probabilities, gt_classes)
        return cross_entropy_loss

    def compute_loss(
        self,
        class_probabilities_in,
        actions_in,
        action_log_probs_in,
        state_values_in,
        entropies_in,
        indices_in,
    ):

        # compute sf reward for each example
        # compute reward r_t in {-1/num_sents , -sent_skip_reward_multiplier/num_sents} for each time step
        # compute cumulative reward for each step.
        # add sf reward to each cumulative reward
        # detach value tensor. subtract value from corresponding cumulative reward
        # multiply action log probs with corresponding (reward-value)
        # sum each row in the above matrix. Then compute the mean to get actor loss l_actor
        # flatten cumulative reward matrix. flatten value matrix. compute MSE between these to get critic loss l_critic
        # average all entropies to get l_entropy
        # take weighted sum of l_actor, l_critic and l_entropy to get l_total
        # return l_total, mean sf F1, mean sf recall

        classification_accuracies = self.compute_accuracy(
            class_probabilities=class_probabilities_in, indices=indices_in
        )

        mean_classification_accuracy = classification_accuracies.mean()

        classification_rewards = self.compute_classification_reward(
            class_probabilities=class_probabilities_in,
            indices=indices_in,
            accuracies=classification_accuracies,
        )

        action_log_probs = self.form_batch_times_num_sents_tensor(action_log_probs_in)
        state_values = self.form_batch_times_num_sents_tensor(state_values_in)
        entropies = self.form_batch_times_num_sents_tensor(entropies_in)
        actions = self.form_batch_times_num_sents_tensor(actions_in)

        num_words_read = torch.sum(actions, dim=-1).to(torch.float)
        num_gt_words = self.dataset.get_seq_len(indices_in)
        fraction_of_words_read = num_words_read / num_gt_words
        avg_read_fraction = fraction_of_words_read.mean().item()

        actor_loss = self.compute_actor_loss(
            cumulative_rewards, action_log_probs, state_values
        )
        critic_loss = self.compute_critic_loss(cumulative_rewards, state_values)

        entropy_loss = self.compute_entropy_loss(entropies)

        classification_loss = self.compute_classification_loss(
            class_probabilities_in, indices_in
        )

        total_loss = (
            (self.config.loss_weight_classification * classification_loss)
            + (self.config.loss_weight_actor * actor_loss)
            + (self.config.loss_weight_critic * critic_loss)
            + (self.config.loss_weight_entropy * entropy_loss)
        )

        self.write_log(
            {
                "mean_classification_accuracy": mean_classification_accuracy,
                "avg_read_fraction": avg_read_fraction,
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "classification_loss": classification_loss.item(),
            }
        )

        return {
            "loss": total_loss,
            "avg_read_fraction": avg_read_fraction,
            "mean_classification_accuracy": mean_classification_accuracy,
        }


import torch
import torch.nn as nn
from torch.distributions import Bernoulli

import numpy as np

import pdb


class Model1(nn.Module):
    """
    Based on https://openreview.net/forum?id=B1xf9jAqFQ
    """

    def __init__(self, config):
        super(Model1, self).__init__()
        self.config = config

        # load embedding matrix
        embedding_matrix = np.load(config.embedding_matrix)

        # Init word embeddings
        self.word_embedding = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1]
        )
        self.word_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.word_embedding.weight.requires_grad = False

        # uses current input, previous hidden state and one hot version of previous action to form state vector for current timestep
        self.fc_relu_state = nn.Sequential(
            nn.Linear(
                config.word_embedding_size + config.lstm_hidden_size + 1,
                config.state_vector_size,
            ),
            nn.ReLU(),
        )

        # takes RL state vector and produces sentence probability
        self.fc_read_probs = nn.Sequential(
            nn.Linear(config.state_vector_size, 2), nn.Softmax(dim=-1)
        )

        self.fc_classifier = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, 2), nn.Softmax(dim=-1)
        )

        self.value_function = nn.Linear(config.state_vector_size, 1)

        self.dropout = nn.Dropout(config.dropout)

        self.state_lstm_cell = nn.LSTMCell(
            input_size=config.word_embedding_size,
            hidden_size=config.lstm_hidden_size,
            bias=True,
        )

        self.h0 = nn.Parameter(
            torch.randn([1, config.lstm_hidden_size], requires_grad=True),
            requires_grad=True,
        )
        self.c0 = nn.Parameter(
            torch.randn([1, config.lstm_hidden_size], requires_grad=True),
            requires_grad=True,
        )

    def forward_supervised(self, data_in):
        """ 
        data_in = {
            "features":  batch x max_seq_len tensor,
        }
        """
        embeddings = self.word_embedding(data_in["features"])
        embeddings = self.dropout(embeddings)

        # expand h0 and c0 to batch_size x hidden_dim
        batch_size = data_in["features"].shape[0]
        old_h = torch.cat(
            [self.h0.clone() for i in range(batch_size)], dim=0
        ).contiguous()
        old_c = torch.cat(
            [self.c0.clone() for i in range(batch_size)], dim=0
        ).contiguous()

        assert old_h.shape[0] == batch_size
        assert old_h.shape[1] == self.config.model_hidden_size
        assert old_c.shape[0] == batch_size
        assert old_c.shape[1] == self.config.model_hidden_size

        # transpose to seq_len x batch x hidden_dim
        seq_len = data_in["features"].shape[1]
        embeddings = embeddings.permute(1, 0, 2).contiguous()

        assert embeddings.shape[0] == seq_len
        assert embeddings.shape[1] == batch_size

        for time_step in range(seq_len):
            new_h, new_c = self.state_lstm_cell(embeddings[time_step], (old_h, old_c))
            old_h = new_h
            old_c = new_c

        new_h = self.dropout(new_h)
        class_probabilities = self.fc_classifier(new_h)

        return {"class_probabilities": class_probabilities}

    def forward_rl(self, data_in):
        """ 
        data_in = {
            "features":  batch x max_num_sents tensor,
            "word_mask": batch x [0,0,0,0,1,1,1,1], # 0 means token is padding.
        }
        """
        embeddings = self.word_embedding(data_in["features"])
        embeddings = self.dropout(embeddings)

        # expand h0 and c0 to batch_size x hidden_dim
        batch_size = data_in["features"].shape[0]
        old_h = torch.cat(
            [self.h0.clone() for i in range(batch_size)], dim=0
        ).contiguous()
        old_c = torch.cat(
            [self.c0.clone() for i in range(batch_size)], dim=0
        ).contiguous()

        assert old_h.shape[0] == batch_size
        assert old_h.shape[1] == self.config.model_hidden_size
        assert old_c.shape[0] == batch_size
        assert old_c.shape[1] == self.config.model_hidden_size

        old_action = torch.zeros((batch_size, 1), device=old_h.device)

        # transpose to seq_len x batch x hidden_dim
        seq_len = data_in["features"].shape[1]
        embeddings = embeddings.permute(1, 0, 2).contiguous()

        word_masks = data_in["word_mask"].permute(1, 0)

        assert embeddings.shape[0] == seq_len
        assert embeddings.shape[1] == batch_size

        all_read_probabilities = []
        all_predicted_state_values = []
        all_action_log_probs = []
        all_actions = []
        all_entropies = []
        for time_step in range(seq_len):

            state_vector = self.fc_relu_state(
                torch.cat([embeddings[time_step], old_h, old_action], dim=1)
            )

            val = self.value_function(state_vector).squeeze(-1)

            new_h, new_c = self.state_lstm_cell(embeddings[time_step], (old_h, old_c))

            read_probs = self.fc_read_probs(state_vector)

            assert read_probs.shape[0] == batch_size
            assert read_probs.shape[1] == 2

            # let the first dimension be the probability of selecting the sentence
            read_probs = read_probs[:, 1]

            # dim: batch_size
            assert len(read_probs.shape) == 1
            assert read_probs.shape[0] == data_in["features"].shape[0]

            read_prob_dist = Bernoulli(probs=read_probs)
            entropy = read_prob_dist.entropy()

            action = read_prob_dist.sample()

            action_log_probs = read_prob_dist.log_prob(action)
            action_log_probs = action_log_probs * word_masks[time_step]

            actions_masked = action * word_masks[time_step]
            actions_masked = actions_masked.unsqueeze(1)

            old_h = actions_masked * old_h + (1 - actions_masked) * old_h
            new_c = actions_masked * new_c + (1 - actions_masked) * old_c

            old_h = new_h
            old_c = new_c

            old_action = actions_masked

            assert old_action.shape[0] == batch_size
            assert old_action.shape[1] == 1

            all_read_probabilities.append(read_probs)
            all_predicted_state_values.append(val)
            all_action_log_probs.append(action_log_probs)
            all_actions.append(actions_masked.type(torch.int32).squeeze(-1))
            all_entropies.append(entropy)

        new_h = self.dropout(new_h)
        class_probabilities = self.fc_classifier(new_h)

        # all lists are seq_len x batch_size
        return {
            "class_probabilities": class_probabilities,
            "read_probabilities": all_read_probabilities,
            "action_log_probs": all_action_log_probs,
            "state_values": all_predicted_state_values,
            "actions": all_actions,
            "entropies": all_entropies,
        }

    def forward(self, data_in, supervised=True):
        if supervised is True:
            return self.forward_supervised(data_in)
        else:
            return self.forward_rl(data_in)


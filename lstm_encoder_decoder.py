import numpy as np
import random
import os, errno
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import RGCNConv, GraphConv

n_speakers = 2


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)

    def forward(self, x_input):
        """

        :param x_input: input shape (seq_len,batch_size,input_size)
        :return: lstm_out gives all hidden states in teh sequence; hidden gives the hidden state
        and cell gives the last state for the last element in teh sequence
        """

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        """

        :param batch_size: x_input.shape[1]
        :return: zeroed hidden state and cell state
        """
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class GCN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim):
        super(GCN, self).__init__()
        self.num_relations = 2 * n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_type, edge_norm):
        x = self.conv1(node_features, edge_index, edge_type, edge_norm)
        x = self.conv2(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class Seq2Seq(nn.Module):

    def __init__(self, input_size, hidden_size, g_dim, h1_dim, h2_dim):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self.graph_processing = GCN(g_dim=g_dim, h1_dim=h1_dim, h2_dim=h2_dim)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size)

    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size,
                    training_prediction="recursive", teacher_forcing_ratio=0.5, learning_rate=0.01, dynamic_tf=False):
        """
        train lstm encoder-decoder

        : param input_tensor:              input data with shape (batch,seq_len, number features); PyTorch tensor
        : param target_tensor:             target data with shape (batch, seq_len, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        """
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        n_batches = int(input_tensor.shape[1] / batch_size)
        print(n_batches)
        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0
                batch_loss_tf = 0
                batch_loss_no_tf = 0
                num_tf = 0
                num_no_tf = 0

                for b in range(n_batches):
                    input_batch = input_tensor[:, b:b + batch_size, :]
                    print("input_batch shape")
                    print(input_batch.shape)
                    target_batch = target_tensor[:, b:b + batch_size, :]

                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)
                    optimizer.zero_grad()
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # ----------------------------------------
                    # For only decoder
                    # print(encoder_output.shape)
                    # print(encoder_hidden[1].shape)
                    #
                    # decoder_input = target_batch[-1, :, :]
                    # decoder_hidden = encoder_hidden
                    # -----------------------------------------
                    print(encoder_hidden[0].size())
                    graph_out = self.graph_processing(encoder_hidden[0].size[0], encoder_hidden[0].size[1], encoder_hidden[0].size[2])
                    decoder_input = target_batch[-1, :, :]
                    decoder_hidden = graph_out
                    if training_prediction == "recursive":
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                batch_loss = batch_loss / n_batches
                losses[it] = batch_loss
                print(losses)
        return losses

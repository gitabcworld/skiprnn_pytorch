"""
Graph creation functions.
"""


from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

from rnn_cells.base_rnn import custom_LSTM
from rnn_cells.custom_cells import CBasicLSTMCell, CBasicGRUCell, \
                                            CSkipLSTMCell, CSkipGRUCell, \
                                            CMultiSkipLSTMCell, CMultiSkipGRUCell


from util.misc import *

def create_model(model, input_size, hidden_size, num_layers):
    """
    Returns a tuple of (cell, initial_state) to use with dynamic_rnn.
    If num_cells is an integer, a single RNN cell will be created. If it is a list, a stack of len(num_cells)
    cells will be created.
    """
    if not model in ['nn_lstm', 'custom_lstm', 'nn_gru', 'custom_gru', 'skip_lstm', 'skip_gru']:
        raise ValueError('The specified model is not supported. Please use {lstm, gru, skip_lstm, skip_gru}.')
    #if model == 'lstm':
    #    cells = custom_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    #    # cells = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    if model == 'nn_lstm':
        cells = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    if model == 'nn_gru':
        cells = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    if model == 'custom_lstm':
        if num_layers == 1:
            cells = CBasicLSTMCell(input_size=input_size, hidden_size=hidden_size, batch_first = True)
        else:
            cells = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    if model == 'custom_gru':
        if num_layers == 1:
            cells = CBasicGRUCell(input_size=input_size, hidden_size=hidden_size, batch_first = True)
        else:
            cells = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    if model == 'skip_lstm':
        if num_layers == 1:
            cells = CSkipLSTMCell(input_size=input_size, hidden_size=hidden_size, batch_first = True)
        else:
            cells = CMultiSkipLSTMCell(input_size=input_size, hidden_size=hidden_size,
                                       batch_first=True, num_layers=num_layers)
    if model == 'skip_gru':
        if num_layers == 1:
            cells = CSkipGRUCell(input_size=input_size, hidden_size=hidden_size, batch_first = True)
        else:
            cells = CMultiSkipGRUCell(input_size=input_size, hidden_size=hidden_size,
                                       batch_first=True, num_layers=num_layers)

    return cells


def split_rnn_outputs(model, rnn_outputs):
    """
    Split the output into the actual RNN outputs and the state update gate
    """
    if using_skip_rnn(model):
        return rnn_outputs[0], rnn_outputs[1], rnn_outputs[2]
    else:
        return rnn_outputs[0], rnn_outputs[1], None

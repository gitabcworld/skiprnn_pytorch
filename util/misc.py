"""
Generic functions that are used in different scripts.
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import types
from decimal import Decimal


FLAGS = {}

def create_generic_flags():
    """
    Create flags which are shared by all experiments
    """
    # Generic flags
    FLAGS['cuda'] = True
    FLAGS['model'] = 'skip_lstm'      # Select RNN cell: {nn_lstm, nn_gru,
                                                            # custom_lstm, custom_gru,
                                                            # skip_lstm, skip_gru}
    FLAGS['rnn_cells'] = 110            # Number of RNN cells (or hidden_size)
    FLAGS['rnn_layers'] = 1             # Number of RNN layers.
    FLAGS['batch_size'] = 256           # Batch size
    FLAGS['learning_rate'] = 1e-4       # Learning rate.
    FLAGS['learning_rate_patience'] = 1e4  # Learning rate.
    FLAGS['grad_clip'] = 1.0            # grad clip. Clip gradients at this value. Set to <=0 to disable clipping
    # Flags for the Skip RNN cells
    FLAGS['cost_per_sample'] = 0.       # Cost per used sample. Set to 0 to disable this option

def print_setup(task_specific_setup=None):
    """
    Print experimental setup
    :param task_specific_setup: (optional) function printing task-specific parameters
    """
    flags_models = {'lstm', 'gru', 'skip_lstm', 'skip_gru'}
    print('\n\n\tExperimental setup')
    print('\t------------------\n')
    print('\tModel: %s' % FLAGS['model'])
    print('\tNumber of layers: %d' % FLAGS['rnn_layers'])
    print('\tNumber of cells: %d' % FLAGS['rnn_cells'])
    print('\tBatch size: %d' % FLAGS['batch_size'])
    print('\tLearning rate: %.2E' % Decimal(FLAGS['learning_rate']))

    if FLAGS['grad_clip'] > 0:
        print('\tGradient clipping: %.1f' % FLAGS['grad_clip'])
    else:
        print('\tGradient clipping: No')

    if FLAGS['model'].lower().startswith('skip'):
        print('\tCost per sample: %.2E' % Decimal(FLAGS['cost_per_sample']))

    if isinstance(task_specific_setup, types.FunctionType):
        print('')
        task_specific_setup()

    print('\n\n')


def compute_used_samples(update_state_gate):
    """
    Compute number of used samples (i.e. number of updated states)
    :param update_state_gate: values for the update state gate
    :return: number of used samples
    """
    return update_state_gate.sum() / update_state_gate.shape[0]
    '''
    batch_size = update_state_gate.shape[0]
    steps = 0.
    for idx in range(batch_size):
        for idt in range(update_state_gate.shape[1]):
            steps += update_state_gate[idx, idt]
    return steps / batch_size
    '''

def using_skip_rnn(model):
    """
    Helper function determining whether a Skip RNN models is being used
    """
    return model.lower() == 'skip_lstm' or model.lower() == 'skip_gru'

def compute_budget_loss(model, iscuda, loss, updated_states, cost_per_sample):
    """
    Compute penalization term on the number of updated states (i.e. used samples)
    """
    if using_skip_rnn(model):
        return torch.mean(torch.sum(cost_per_sample * updated_states,1),0)
    else:
        if iscuda:
            return Variable(torch.zeros(loss.shape).cuda(),requires_grad=True)
        else:
            return Variable(torch.zeros(loss.shape), requires_grad=True)

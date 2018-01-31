"""
Train RNN models on the frequency discrimination task. Sine waves with period in [1, 100] are randomly generated and
the network has to classify those with period in [5, 6].

Batches are stratified. Validation is performed on data generated on the fly.
"""

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from util.misc import *
from util.graph_definition import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger import Logger

# Task-independent flags
create_generic_flags()

# Task-specific flags
FLAGS['sampling_period'] = 1.           # Sampling period, in milliseconds
FLAGS['signal_duration'] = 100.         # Signal duration, in milliseconds
FLAGS['validation_batches'] = 15        # How many batches to use for validation metrics.
FLAGS['evaluate_every'] = 300           # How often is the model evaluated.

# Constants
START_PERIOD = 0
END_PERIOD = 100
START_TARGET_PERIOD = 5
END_TARGET_PERIOD = 6
INPUT_SIZE = 1
OUTPUT_SIZE = 2
SEQUENCE_LENGTH = int(FLAGS['signal_duration'] / FLAGS['sampling_period'])


def task_setup():
    print('\tSignal duration: %.1fms' % FLAGS['signal_duration'])
    print('\tSampling period: %.1fms' % FLAGS['sampling_period'])
    print('\tSequence length: %d' % SEQUENCE_LENGTH)
    print('\tTarget periods: (%.0f, %.0f)' % (START_TARGET_PERIOD, END_TARGET_PERIOD))
    print('\tDistractor periods: (%.0f, %.0f) U (%.0f, %.0f)' % (START_PERIOD, START_TARGET_PERIOD,
                                                                 END_TARGET_PERIOD, END_PERIOD))


def generate_example(t, frequency, phase_shift):
    return np.cos(2 * np.pi * frequency * t + phase_shift)


def random_disjoint_interval(start, end, avoid_start, avoid_end):
    """
    Sample a value in [start, avoid_start] U [avoid_end, end] with uniform probability
    """
    val = random.uniform(start, end - (avoid_end - avoid_start))
    if val > avoid_start:
        val += (avoid_end - avoid_start)
    return val


def generate_batch(batch_size, sampling_period, signal_duration, start_period, end_period,
                   start_target_period, end_target_period):
    """
    Generate a stratified batch of examples. There are two classes:
        class 0: sine waves with period in [start_target_period, end_target_period]
        class 1: sine waves with period in [start_period, start_target_period] U [end_target_period, end_period]
    :param batch_size: number of samples per batch
    :param sampling_period: sampling period in milliseconds
    :param signal_duration: duration of the sine waves in milliseconds

    :return x: batch of examples
    :return y: batch of labels
    """
    seq_length = int(signal_duration / sampling_period)

    n_elems = 1
    x = np.empty((batch_size, seq_length, n_elems))
    y = np.empty(batch_size, dtype=np.int64)

    t = np.linspace(0, signal_duration - sampling_period, seq_length)

    for idx in range(int(batch_size/2)):
        period = random.uniform(start_target_period, end_target_period)
        phase_shift = random.uniform(0, period)
        x[idx, :, 0] = generate_example(t, 1./period, phase_shift)
        y[idx] = 0
    for idx in range(int(batch_size/2), batch_size):
        period = random_disjoint_interval(start_period, end_period,
                                          start_target_period, end_target_period)
        phase_shift = random.uniform(0, period)
        x[idx, :, 0] = generate_example(t, 1./period, phase_shift)
        y[idx] = 1
    return x, y

class cellModule(nn.Module):

    def __init__(self, cells, model):
        super(cellModule, self).__init__()
        self.model = model
        self.rnn = cells
        self.d1 = nn.Linear(FLAGS['rnn_cells'],OUTPUT_SIZE)

    def forward(self, input, hx=None):
        if hx is not None:
            output = self.rnn(input, hx)
        else:
            output = self.rnn(input)
        output, hx, updated_state = split_rnn_outputs(self.model, output)
        output = self.d1(output[:,-1,:]) # Get the last output of the sequence
        return output, hx, updated_state

def train():
    logger = Logger('/tmp/skiprnn/' + FLAGS['model'], remove_previous_files=True)

    cells = create_model(model=FLAGS['model'],
                         input_size=INPUT_SIZE,
                         hidden_size=FLAGS['rnn_cells'],
                         num_layers=FLAGS['rnn_layers'])
    model_fn = cellModule(cells, model=FLAGS['model'])

    # Compute L2 loss
    mse_loss_fn = nn.CrossEntropyLoss()
    # Compute loss for each updated state
    budget_loss_fn = compute_budget_loss
    optimizer = torch.optim.Adam(params=model_fn.parameters(), lr=FLAGS['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1000, verbose=True)

    if FLAGS['cuda']:
        model_fn.cuda()
        mse_loss_fn.cuda()

    try:
        num_iters = 0
        while True:
            # Generate new batch and perform SGD update
            x, y = generate_batch(FLAGS['batch_size'],
                                  FLAGS['sampling_period'],
                                  FLAGS['signal_duration'],
                                  START_PERIOD, END_PERIOD,
                                  START_TARGET_PERIOD, END_TARGET_PERIOD)

            x = Variable(torch.from_numpy(x).float(), requires_grad=True)
            y = Variable(torch.from_numpy(y), requires_grad=False)
            if FLAGS['cuda']:
                x = x.cuda()
                y = y.cuda()

            output, hx, updated_states = model_fn(x)
            loss_mse = mse_loss_fn(output, y)

            loss_budget = budget_loss_fn(FLAGS['model'], FLAGS['cuda'], loss_mse, updated_states,
                                         FLAGS['cost_per_sample'])
            loss = loss_mse + loss_budget
            logger.log_value('train_loss', loss)
            train_accuracy = torch.mean((torch.max(output, 1)[1] == y).float())
            #print("Iteration %d, train loss: %.7f. train accuracy: %.7f" % (num_iters, loss, train_accuracy))
            optimizer.zero_grad()
            loss.backward()
            if FLAGS['grad_clip'] > 0:  # Gradient clipping
                torch.nn.utils.clip_grad_norm(model_fn.parameters(), FLAGS['grad_clip'])
            optimizer.step()
            # Reduce learning rate when a metric has stopped improving
            scheduler.step(loss)
            num_iters += 1

            # Evaluate on validation data generated on the fly
            if num_iters % FLAGS['evaluate_every'] == 0:
                valid_accuracy, valid_steps = 0., 0.
                for _ in range(FLAGS['validation_batches']):
                    valid_x, valid_y = generate_batch(FLAGS['batch_size'],
                                                      FLAGS['sampling_period'],
                                                      FLAGS['signal_duration'],
                                                      START_PERIOD, END_PERIOD,
                                                      START_TARGET_PERIOD, END_TARGET_PERIOD)
                    valid_x = Variable(torch.from_numpy(valid_x).float(), requires_grad=False)
                    valid_y = Variable(torch.from_numpy(valid_y), requires_grad=False)
                    if FLAGS['cuda']:
                        valid_x = valid_x.cuda()
                        valid_y = valid_y.cuda()

                    output, hx, updated_states = model_fn(valid_x)
                    # calculate accuracy with output
                    valid_iter_accuracy = torch.mean((torch.max(output, 1)[1] == valid_y).float()).data.cpu().numpy()
                    valid_accuracy += valid_iter_accuracy
                    if updated_states is not None:
                        valid_steps += compute_used_samples(updated_states).data.cpu().numpy()
                    else:
                        valid_steps += SEQUENCE_LENGTH
                valid_accuracy /= FLAGS['validation_batches']
                valid_steps /= FLAGS['validation_batches']
                print("Iteration %d, "
                      "validation accuracy: %.2f%%, "
                      "validation samples: %.2f (%.2f%%)" % (num_iters,
                                                             100. * valid_accuracy,
                                                             valid_steps,
                                                             100. * valid_steps / SEQUENCE_LENGTH))
    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup(task_setup)
    train()


if __name__ == '__main__':
    main()

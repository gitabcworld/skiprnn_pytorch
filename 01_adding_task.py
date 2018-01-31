"""
Train RNN models on the adding task. The network is given a sequence of (value, marker) tuples. The desired output is
the addition of the only two values that were marked with a 1, whereas those marked with a 0 need to be ignored.
Markers appear only in the first 10% and last 50% of the sequences.

Validation is performed on data generated on the fly.
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
FLAGS['validation_batches'] = 15    # How many batches to use for validation metrics.
FLAGS['evaluate_every'] = 100       # How often is the model evaluated.
FLAGS['sequence_length'] = 50       # Sequence length

# Constants
MIN_VAL = -0.5
MAX_VAL = 0.5
FIRST_MARKER = 10.
SECOND_MARKER = 50.
INPUT_SIZE = 2
OUTPUT_SIZE = 1

def task_setup():
    print('\tSequence length: %d' % FLAGS['sequence_length'])
    print('\tValues drawn from Uniform[%.1f, %.1f]' % (MIN_VAL, MAX_VAL))
    print('\tFirst marker: first %d%%' % FIRST_MARKER)
    print('\tSecond marker: last %d%%' % SECOND_MARKER)

def generate_example(seq_length, min_val, max_val):
    """
    Creates a list of (a,b) tuples where a is random[min_val,max_val] and b is 1 in only
    two tuples, 0 for the rest. The ground truth is the addition of a values for tuples with b=1.

    :param seq_length: length of the sequence to be generated
    :param min_val: minimum value for a
    :param max_val: maximum value for a

    :return x: list of (a,b) tuples
    :return y: ground truth
    """
    # Select b values: one in first X% of the sequence, the other in the second Y%
    b1 = random.randint(0, int(seq_length * FIRST_MARKER / 100.) - 1)
    b2 = random.randint(int(seq_length * SECOND_MARKER / 100.), seq_length - 1)

    b = [0.] * seq_length
    b[b1] = 1.
    b[b2] = 1.

    # Generate list of tuples
    x = [(random.uniform(min_val, max_val), marker) for marker in b]
    y = x[b1][0] + x[b2][0]

    return x, y


def generate_batch(seq_length, batch_size, min_val, max_val):
    """
    Generates batch of examples.

    :param seq_length: length of the sequence to be generated
    :param batch_size: number of samples in the batch
    :param min_val: minimum value for a
    :param max_val: maximum value for a

    :return x: batch of examples
    :return y: batch of ground truth values
    """
    n_elems = 2
    x = np.empty((batch_size, seq_length, n_elems))
    y = np.empty((batch_size, 1))

    for i in range(batch_size):
        sample, ground_truth = generate_example(seq_length, min_val, max_val)
        x[i, :, :] = sample
        y[i, 0] = ground_truth
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

    logger = Logger('/tmp/skiprnn/' + FLAGS['model'],remove_previous_files=True)

    cells = create_model(model=FLAGS['model'],
                                        input_size = INPUT_SIZE,
                                        hidden_size=FLAGS['rnn_cells'],
                                        num_layers=FLAGS['rnn_layers'])
    model_fn = cellModule(cells, model=FLAGS['model'])

    # Compute L2 loss
    mse_loss_fn = nn.MSELoss()
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
            model_fn.train()
            # Generate new batch and perform SGD update
            x, y = generate_batch(min_val=MIN_VAL, max_val=MAX_VAL,
                                  seq_length=FLAGS['sequence_length'],
                                  batch_size=FLAGS['batch_size'])
            x = Variable(torch.from_numpy(x).float(), requires_grad=True)
            y = Variable(torch.from_numpy(y).float(), requires_grad=False)

            if FLAGS['cuda']:
                x = x.cuda()
                y = y.cuda()

            output, hx, updated_states = model_fn(x)
            loss_mse = mse_loss_fn(output, y)

            loss_budget = budget_loss_fn(FLAGS['model'],FLAGS['cuda'], loss_mse, updated_states, FLAGS['cost_per_sample'])
            loss = loss_mse + loss_budget
            logger.log_value('train_loss', loss)
            #print("Iteration %d, train error: %.7f" % (num_iters, loss))
            optimizer.zero_grad()
            loss.backward()
            if FLAGS['grad_clip'] > 0: #Gradient clipping
                torch.nn.utils.clip_grad_norm(model_fn.parameters(), FLAGS['grad_clip'])
            optimizer.step()

            # Reduce learning rate when a metric has stopped improving
            scheduler.step(loss)

            num_iters += 1

            # Evaluate on validation data generated on the fly
            if num_iters % FLAGS['evaluate_every'] == 0:
                valid_error, valid_steps = 0., 0.
                model_fn.eval()
                for _ in range(FLAGS['validation_batches']):
                    valid_x, valid_y = generate_batch(min_val=MIN_VAL, max_val=MAX_VAL,
                                                      seq_length=FLAGS['sequence_length'],
                                                      batch_size=FLAGS['batch_size'])
                    valid_x = Variable(torch.from_numpy(valid_x).float(), requires_grad=False)
                    valid_y = Variable(torch.from_numpy(valid_y).float(), requires_grad=False)
                    if FLAGS['cuda']:
                        valid_x = valid_x.cuda()
                        valid_y = valid_y.cuda()

                    output, hx, updated_states = model_fn(valid_x)
                    loss_mse = mse_loss_fn(output, valid_y)
                    valid_iter_error = loss_mse
                    valid_error += valid_iter_error
                    if updated_states is not None:
                        valid_steps += compute_used_samples(updated_states).data.cpu().numpy()
                    else:
                        valid_steps += FLAGS['sequence_length']
                valid_error /= FLAGS['validation_batches']
                valid_steps /= FLAGS['validation_batches']
                print("Iteration %d, "
                      "validation error: %.7f, "
                      "validation samples: %.2f%%" % (num_iters,
                                                      valid_error,
                                                      100. * valid_steps / FLAGS['sequence_length']))
                logger.log_value('val_error', valid_error)
                logger.log_value('val_samples', 100. * valid_steps / FLAGS['sequence_length'])
            logger.step()


    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup(task_setup)
    train()


if __name__ == '__main__':
    main()

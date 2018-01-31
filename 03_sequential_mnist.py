"""
Train RNN models on the frequency discrimination task. Sine waves with period in [1, 100] are randomly generated and
the network has to classify those with period in [5, 6].

Batches are stratified. Validation is performed on data generated on the fly.
"""

from __future__ import absolute_import
from __future__ import print_function

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from util.misc import *
from util.graph_definition import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger import Logger

# Task-independent flags
create_generic_flags()

# Constants
OUTPUT_SIZE = 10
SEQUENCE_LENGTH = 784
MNIST_TRAIN_LENGTH = 60000
VALIDATION_SAMPLES = 5000
NUM_EPOCHS = 600

transform_train_test = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS['cuda'] else {}

# Split training into train and validation
indices = torch.randperm(MNIST_TRAIN_LENGTH)
train_indices = indices[:len(indices)-VALIDATION_SAMPLES]
val_indices = indices[len(indices)-VALIDATION_SAMPLES:]

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transform_train_test),
    sampler=SubsetRandomSampler(train_indices),
    batch_size=FLAGS['batch_size'], shuffle=False, **kwargs)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transform_train_test),
    sampler=SubsetRandomSampler(val_indices),
    batch_size=FLAGS['batch_size'], shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform_train_test),
    batch_size=FLAGS['batch_size'], shuffle=True, **kwargs)

TRAIN_ITERS = int(math.ceil((MNIST_TRAIN_LENGTH - VALIDATION_SAMPLES) / FLAGS['batch_size']))
VAL_ITERS = int(math.ceil(VALIDATION_SAMPLES / FLAGS['batch_size']))
TEST_ITERS = int(math.ceil(len(test_loader.dataset) / FLAGS['batch_size']))

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
                         input_size=1,
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
        for epoch in range(NUM_EPOCHS):
            model_fn.train()
            train_accuracy, train_loss = 0, 0
            for batch_idx, (x, y) in enumerate(train_loader):
                if FLAGS['cuda']:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)
                x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)

                output, hx, updated_states = model_fn(x)
                loss_mse = mse_loss_fn(output, y)

                loss_budget = budget_loss_fn(FLAGS['model'], FLAGS['cuda'], loss_mse, updated_states,
                                             FLAGS['cost_per_sample'])
                loss = loss_mse + loss_budget
                logger.log_value('train_loss', loss)
                train_loss += loss.data.cpu().numpy()
                train_accuracy += torch.mean((torch.max(output, 1)[1] == y).float()).data.cpu().numpy()
                #print("Epoch %d Iteration %d, train loss: %.7f. train accuracy: %.7f" %
                #      (epoch, batch_idx, loss, torch.mean((torch.max(output, 1)[1] == y).float())))
                optimizer.zero_grad()
                loss.backward()
                if FLAGS['grad_clip'] > 0:  # Gradient clipping
                    torch.nn.utils.clip_grad_norm(model_fn.parameters(), FLAGS['grad_clip'])
                optimizer.step()
            # Reduce learning rate when a metric has stopped improving
            scheduler.step(loss)
            train_loss /= TRAIN_ITERS + 1
            train_accuracy /= TRAIN_ITERS + 1
            #print("Epoch %d train loss: %.7f. train accuracy: %.7f" % (epoch, train_loss, train_accuracy))
            logger.log_value('train_loss', train_loss)
            logger.log_value('train_accuracy', train_accuracy)

            model_fn.eval()
            # Evaluate on validation data generated on the fly
            valid_accuracy, valid_steps = 0., 0.
            for batch_idx, (x, y) in enumerate(val_loader):
                if FLAGS['cuda']:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x, requires_grad = False), Variable(y, requires_grad = False)
                x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
                output, hx, updated_states = model_fn(x)
                valid_accuracy += torch.mean((torch.max(output, 1)[1] == y).float()).data.cpu().numpy()
                if updated_states is not None:
                    valid_steps += compute_used_samples(updated_states).data.cpu().numpy() / FLAGS['batch_size']
                else:
                    valid_steps += SEQUENCE_LENGTH
            valid_accuracy /= VAL_ITERS + 1
            valid_steps /= VAL_ITERS + 1
            logger.log_value('valid_accuracy', valid_accuracy)
            logger.log_value('valid_steps', valid_steps)

            model_fn.eval()
            # Evaluate on test data generated on the fly
            test_accuracy, test_steps = 0., 0.
            for batch_idx, (x, y) in enumerate(test_loader):
                if FLAGS['cuda']:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x, requires_grad = False), Variable(y, requires_grad = False)
                x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
                output, hx, updated_states = model_fn(x)
                test_accuracy += torch.mean((torch.max(output, 1)[1] == y).float()).data.cpu().numpy()
                if updated_states is not None:
                    test_steps += compute_used_samples(updated_states).data.cpu().numpy() / FLAGS['batch_size']
                else:
                    test_steps += SEQUENCE_LENGTH
            test_accuracy /= VAL_ITERS + 1
            test_steps /= VAL_ITERS + 1
            logger.log_value('test_accuracy', test_accuracy)
            logger.log_value('test_steps', test_steps)

            print(
                "Epoch %d/%d, "
                "training accuracy: %.2f%% , "
                "training loss: %.5f, "
                "validation accuracy: %.2f%%, "
                "validation samples: %.2f (%.2f%%), "
                "test accuracy: %.2f%%, "
                "test samples: %.2f (%.2f%%)" %
                                                (epoch + 1, NUM_EPOCHS,
                                                100. * train_accuracy,
                                                train_loss,
                                                100. * valid_accuracy,
                                                valid_steps, 100. * valid_steps / SEQUENCE_LENGTH,
                                                100. * test_accuracy,
                                                test_steps, 100. * test_steps / SEQUENCE_LENGTH))

            logger.step()

    except KeyboardInterrupt:
        pass

def main(argv=None):
    train()

if __name__ == '__main__':
    main()





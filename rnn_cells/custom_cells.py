import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform
from .basic_rnn_cells import BasicLSTMCell, BasicGRUCell
from .skip_rnn_cells import SkipLSTMCell, SkipGRUCell, MultiSkipLSTMCell, MultiSkipGRUCell
import math
import numpy as np

class CCellBase(RNNCellBase):

    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers = 1,
                    bias=True, batch_first = False, activation=F.tanh, layer_norm=False):
        super(CCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.cell = cell
        self.num_layers = num_layers
        self.weight_ih = []
        self.weight_hh = []
        self.bias_ih = []
        self.bias_hh = []

        for i in np.arange(self.num_layers):
            if i == 0:
                weight_ih = Parameter(xavier_uniform(torch.Tensor(learnable_elements * hidden_size, input_size)))
            else:
                weight_ih = Parameter(xavier_uniform(torch.Tensor(learnable_elements * hidden_size, hidden_size)))
            weight_hh = Parameter(xavier_uniform(torch.Tensor(learnable_elements * hidden_size, hidden_size)))
            self.weight_ih.append(weight_ih)
            self.weight_hh.append(weight_hh)
            if bias:
                bias_ih = Parameter(torch.zeros(learnable_elements * hidden_size))
                bias_hh = Parameter(torch.zeros(learnable_elements * hidden_size))
                self.bias_ih.append(bias_ih)
                self.bias_hh.append(bias_hh)
            else:
                self.register_parameter('bias_ih_' + str(i), None)
                self.register_parameter('bias_hh_' + str(i), None)
        self.weight_ih = nn.ParameterList(self.weight_ih)
        self.weight_hh = nn.ParameterList(self.weight_hh)
        if self.bias_ih:
            self.bias_ih = nn.ParameterList(self.bias_ih)
            self.bias_hh = nn.ParameterList(self.bias_hh)

        self.activation = activation
        self.layer_norm = layer_norm
        self.lst_bnorm_rnn = None


class CCellBaseLSTM(CCellBase):

    def forward(self, input, hx = None):

        if len(input.shape) == 3:
            if self.batch_first:
                input = input.transpose(0,1)
            sequence_length, batch_size, input_size = input.shape
        else:
            sequence_length = 1
            batch_size, input_size = input.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if input.is_cuda:
                hx = tuple([x.cuda() for x in hx])

        if len(input.shape) == 3:
            self.check_forward_input(input[0])
            self.check_forward_hidden(input[0], hx[0], '[0]')
            self.check_forward_hidden(input[0], hx[1], '[1]')
        else:
            self.check_forward_input(input)
            self.check_forward_hidden(input, hx[0], '[0]')
            self.check_forward_hidden(input, hx[1], '[1]')

        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            # Create gain and bias for input_gate, new_input, forget_gate, output_gate
            self.lst_bnorm_rnn = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(4)])
            if input.is_cuda:
                self.lst_bnorm_rnn = self.lst_bnorm_rnn.cuda()

        lst_output = []
        for t in np.arange(sequence_length):
            hx = self.cell(
                input[t], hx,
                self.weight_ih[0], self.weight_hh[0],
                self.bias_ih[0], self.bias_hh[0],
                activation=self.activation,
                lst_layer_norm=self.lst_bnorm_rnn
            )
            lst_output.append(hx[0])
        output = torch.stack(lst_output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output,hx

class CCellBaseGRU(CCellBase):

    def forward(self, input, hx = None):
        if len(input.shape) == 3:
            if self.batch_first:
                input = input.transpose(0,1)
            sequence_length, batch_size, input_size = input.shape
        else:
            sequence_length = 1
            batch_size, input_size = input.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if input.is_cuda:
                hx = hx.cuda()

        if len(input.shape) == 3:
            self.check_forward_input(input[0])
            self.check_forward_hidden(input[0], hx, '[0]')
        else:
            self.check_forward_hidden(input, hx, '[0]')

        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            # Create gain and bias for reset_gate and update_gate
            self.lst_bnorm_rnn = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(2)])
            if input.is_cuda:
                self.lst_bnorm_rnn = self.lst_bnorm_rnn.cuda()

        lst_output = []
        for t in np.arange(sequence_length):
            hx = self.cell(
                input[t], hx,
                self.weight_ih[0], self.weight_hh[0],
                self.bias_ih[0], self.bias_hh[0],
                activation=self.activation,
                lst_layer_norm=self.lst_bnorm_rnn
            )
            lst_output.append(hx)
        output = torch.stack(lst_output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output,hx

class CCellBaseSkipLSTM(CCellBase):

    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers = 1,
                    bias=True, batch_first = False, activation=F.tanh, layer_norm=False):
        super(CCellBaseSkipLSTM, self).__init__(cell, learnable_elements, input_size, hidden_size, num_layers,
                                                bias, batch_first, activation, layer_norm)
        self.weight_uh = Parameter(xavier_uniform(torch.Tensor(1, hidden_size)))
        if bias:
            self.bias_uh = Parameter(torch.ones(1))
        else:
            self.register_parameter('bias_uh', None)

    def forward(self, input, hx = None):

        if len(input.shape) == 3:
            if self.batch_first:
                input = input.transpose(0,1)
            sequence_length, batch_size, input_size = input.shape
        else:
            sequence_length = 1
            batch_size, input_size = input.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if input.is_cuda:
                if self.num_layers == 1:
                    hx = tuple([x.cuda() for x in hx])
                else:
                    hx = [tuple([j.cuda() if j is not None else None for j in i]) for i in hx]

        if len(input.shape) == 3:
            self.check_forward_input(input[0])
            if self.num_layers > 1:
                self.check_forward_hidden(input[0], hx[0][0], '[0]')
                self.check_forward_hidden(input[0], hx[0][1], '[1]')
            else:
                self.check_forward_hidden(input[0], hx[0], '[0]')
                self.check_forward_hidden(input[0], hx[1], '[1]')
        else:
            self.check_forward_input(input)
            if self.num_layers > 1:
                self.check_forward_hidden(input, hx[0][0], '[0]')
                self.check_forward_hidden(input, hx[0][1], '[1]')
            else:
                self.check_forward_hidden(input, hx[0], '[0]')
                self.check_forward_hidden(input, hx[1], '[1]')

        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = []
            for i in np.arange(self.num_layers):
                # Create gain and bias for input_gate, new_input, forget_gate, output_gate
                lst_bnorm_rnn_tmp = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(4)])
                if input.is_cuda:
                    lst_bnorm_rnn_tmp = lst_bnorm_rnn_tmp.cuda()
                self.lst_bnorm_rnn.append(lst_bnorm_rnn_tmp)
            self.lst_bnorm_rnn = torch.nn.ModuleList(self.lst_bnorm_rnn)


        lst_output = []
        lst_update_gate = []
        for t in np.arange(sequence_length):
            output, hx = self.cell(
                input[t], hx, self.num_layers,
                self.weight_ih, self.weight_hh, self.weight_uh,
                self.bias_ih, self.bias_hh, self.bias_uh,
                activation=self.activation,
                lst_layer_norm=self.lst_bnorm_rnn
            )
            new_h, update_gate = output
            lst_output.append(new_h)
            lst_update_gate.append(update_gate)
        output = torch.stack(lst_output)
        update_gate = torch.stack(lst_update_gate)
        if self.batch_first:
            output = output.transpose(0, 1)
            update_gate = update_gate.transpose(0, 1)
        return output, hx, update_gate

class CCellBaseSkipGRU(CCellBase):

    def __init__(self, cell, learnable_elements, input_size, hidden_size, num_layers = 1,
                    bias=True, batch_first = False, activation=F.tanh, layer_norm=False):
        super(CCellBaseSkipGRU, self).__init__(cell, learnable_elements, input_size, hidden_size, num_layers,
                                               bias, batch_first, activation, layer_norm)
        self.weight_uh = Parameter(xavier_uniform(torch.Tensor(1, hidden_size)))
        if bias:
            self.bias_uh = Parameter(torch.ones(1))
        else:
            self.register_parameter('bias_uh', None)

    def forward(self, input, hx = None):
        if len(input.shape) == 3:
            if self.batch_first:
                input = input.transpose(0,1)
            sequence_length, batch_size, input_size = input.shape
        else:
            sequence_length = 1
            batch_size, input_size = input.shape

        if hx is None:
            hx = self.init_hidden(batch_size)
            if input.is_cuda:
                if self.num_layers == 1:
                    hx = tuple([x.cuda() for x in hx])
                else:
                    hx = [tuple([j.cuda() if j is not None else None for j in i]) for i in hx]

        if len(input.shape) == 3:
            self.check_forward_input(input[0])
            if self.num_layers > 1:
                self.check_forward_hidden(input[0], hx[0][0], '[0]')
            else:
                self.check_forward_hidden(input[0], hx[0], '[0]')
        else:
            self.check_forward_input(input)
            if self.num_layers > 1:
                self.check_forward_hidden(input, hx[0][0], '[0]')
            else:
                self.check_forward_hidden(input, hx[0], '[0]')

        # Initialize batchnorm layers
        if self.layer_norm and self.lst_bnorm_rnn is None:
            self.lst_bnorm_rnn = []
            for i in np.arange(self.num_layers):
                # Create gain and bias for input_gate, new_input, forget_gate, output_gate
                lst_bnorm_rnn_tmp = torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in np.arange(2)])
                if input.is_cuda:
                    lst_bnorm_rnn_tmp = lst_bnorm_rnn_tmp.cuda()
                self.lst_bnorm_rnn.append(lst_bnorm_rnn_tmp)
            self.lst_bnorm_rnn = torch.nn.ModuleList(self.lst_bnorm_rnn)

        lst_output = []
        lst_update_gate = []
        for t in np.arange(sequence_length):
            output, hx = self.cell(
                input[t], hx, self.num_layers,
                self.weight_ih, self.weight_hh, self.weight_uh,
                self.bias_ih, self.bias_hh, self.bias_uh,
                activation=self.activation,
                lst_layer_norm=self.lst_bnorm_rnn
            )
            new_h, update_gate = output
            lst_output.append(new_h)
            lst_update_gate.append(update_gate)
        output = torch.stack(lst_output)
        update_gate = torch.stack(lst_update_gate)
        if self.batch_first:
            output = output.transpose(0, 1)
            update_gate = update_gate.transpose(0, 1)
        return output, hx, update_gate


class CBasicLSTMCell(CCellBaseLSTM):
    def __init__(self, *args, **kwargs):
        super(CBasicLSTMCell, self).__init__(cell=BasicLSTMCell, learnable_elements=4, num_layers = 1, *args, **kwargs)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.randn(batch_size, self.hidden_size)))

class CBasicGRUCell(CCellBaseGRU):
    def __init__(self, *args, **kwargs):
        super(CBasicGRUCell, self).__init__(cell=BasicGRUCell, learnable_elements=3, num_layers = 1, *args, **kwargs)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)))

class CSkipLSTMCell(CCellBaseSkipLSTM):
    def __init__(self, *args, **kwargs):
        super(CSkipLSTMCell, self).__init__(cell=SkipLSTMCell, learnable_elements=4, num_layers = 1,
                                            *args, **kwargs)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.ones(batch_size, 1),requires_grad=False),
                Variable(torch.zeros(batch_size, 1),requires_grad=False))

class CSkipGRUCell(CCellBaseSkipGRU):
    def __init__(self, *args, **kwargs):
        super(CSkipGRUCell, self).__init__(cell=SkipGRUCell, learnable_elements=3, num_layers = 1,
                                           *args, **kwargs)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)),
                Variable(torch.ones(batch_size, 1),requires_grad=False),
                Variable(torch.zeros(batch_size, 1),requires_grad=False))


class CMultiSkipLSTMCell(CCellBaseSkipLSTM):
    def __init__(self, *args, **kwargs):
        super(CMultiSkipLSTMCell, self).__init__(cell=MultiSkipLSTMCell, learnable_elements=4, *args, **kwargs)

    def init_hidden(self, batch_size):
        initial_states = []
        for i in np.arange(self.num_layers):
            initial_c = Variable(torch.randn(batch_size, self.hidden_size))
            initial_h = Variable(torch.randn(batch_size, self.hidden_size))
            if i == self.num_layers - 1: #last layer
                initial_update_prob = Variable(torch.ones(batch_size, 1),requires_grad=False)
                initial_cum_update_prob = Variable(torch.zeros(batch_size, 1),requires_grad=False)
            else:
                initial_update_prob = None
                initial_cum_update_prob = None
            initial_states.append((initial_c,initial_h,initial_update_prob,initial_cum_update_prob))
        return initial_states


class CMultiSkipGRUCell(CCellBaseSkipGRU):
    def __init__(self, *args, **kwargs):
        super(CMultiSkipGRUCell, self).__init__(cell=MultiSkipGRUCell, learnable_elements=3, *args, **kwargs)

    def init_hidden(self, batch_size):
        initial_states = []
        for i in np.arange(self.num_layers):
            initial_h = Variable(torch.randn(batch_size, self.hidden_size))
            if i == self.num_layers - 1: #last layer
                initial_update_prob = Variable(torch.ones(batch_size, 1),requires_grad=False)
                initial_cum_update_prob = Variable(torch.zeros(batch_size, 1),requires_grad=False)
            else:
                initial_update_prob = None
                initial_cum_update_prob = None
            initial_states.append((initial_h,initial_update_prob,initial_cum_update_prob))
        return initial_states

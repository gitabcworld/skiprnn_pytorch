import torch
import torch.nn.functional as F
from .basic_rnn_cells import BasicLSTMCell
from torch.nn._functions.rnn import LSTMCell, GRUCell
from torch.nn._functions.rnn import Recurrent
from torch.nn._functions.rnn import variable_recurrent_factory

def custom_StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward

def custom_AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None, flat_weight=None):

    if mode == 'LSTM':
        cell = LSTMCell
    elif mode == 'BasicLSTM':
        cell = BasicLSTMCell
    elif mode == 'GRU':
        cell = GRUCell
    elif mode == 'BasicGRU':
        # TODO: basicGRUCell
        cell = GRUCell
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = custom_StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM' or mode == 'BasicLSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward

def custom_RNN(*args, **kwargs):

    def forward(input, *fargs, **fkwargs):
        # Removed cudnn as there is not easy to extend the functionality
        #if cudnn.is_acceptable(input.data):
        #    func = CudnnRNN(*args, **kwargs)
        #else:
        func = custom_AutogradRNN(*args, **kwargs)

        # Hack for the tracer that allows us to represent RNNs as single
        # nodes and export them to ONNX in this form
        # It can be also used as a decorator at the higher level
        # Check the first argument explicitly to reduce the overhead of creating
        # the lambda
        import torch
        if torch._C._jit_is_tracing(input):
            import torch.onnx.symbolic
            func = torch.onnx.symbolic.RNN_symbolic_builder(*args, **kwargs)(func)

        return func(input, *fargs, **fkwargs)

    return forward
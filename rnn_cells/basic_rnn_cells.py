import torch.nn.functional as F
import torch.nn as nn

# Implementation from nn._functions.rnn.py
def BasicLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=F.tanh, lst_layer_norm=None):
    '''
    Parameters of a basic LSTM cell
    :param forget_bias: float, the bias added to the forget gates. The biases are for the forget gate in order to
    reduce the scale of forgetting in the beginning of the training. Default 1.0.
    :param activation: activation function of the inner states. Default: F.tanh()
    :param layer_norm: bool, whether to use layer normalization.
    :return: 
    '''
    hx, cx = hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    if lst_layer_norm:
        ingate = lst_layer_norm[0](ingate.contiguous())
        forgetgate = lst_layer_norm[1](forgetgate.contiguous())
        cellgate = lst_layer_norm[2](cellgate.contiguous())
        outgate = lst_layer_norm[3](outgate.contiguous())

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = activation(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * activation(cy)

    return hy, cy

###############################################################
###############################################################

# Implementation from nn._functions.rnn.py
def BasicGRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=F.tanh, lst_layer_norm=None):
    '''
    Parameters of a basic GRU cell
    :param forget_bias: float, the bias added to the forget gates. The biases are for the forget gate in order to
    reduce the scale of forgetting in the beginning of the training. Default 1.0.
    :param activation: activation function of the inner states. Default: F.tanh()
    :param layer_norm: bool, whether to use layer normalization.
    :return: 
    '''
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate_tmp = i_r + h_r
    inputgate_tmp = i_i + h_i
    if lst_layer_norm:
        resetgate_tmp = lst_layer_norm[0](resetgate_tmp.contiguous())
        inputgate_tmp = lst_layer_norm[1](inputgate_tmp.contiguous())

    resetgate = F.sigmoid(resetgate_tmp)
    inputgate = F.sigmoid(inputgate_tmp)
    newgate = activation(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy

###############################################################
###############################################################

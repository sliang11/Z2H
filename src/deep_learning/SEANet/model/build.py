# coding = utf-8

import imp
from torch import nn

from util.conf import Conf
from model.resnet import ResNet, ResFR, Res1d18
from model.inception import InceptionTime
from model.rnn import RNN
from model.transformer import TSTransformerEncoderClassiregressor


def getModel(conf: Conf) -> nn.Module:
    model_name = conf.getHP('model')

    if model_name == 'resnet':
        return ResNet(conf)
    elif model_name == 'resnetfr':
        return ResFR(conf)
    elif model_name == 'res1d18':
        return Res1d18(conf)
    elif model_name == 'incept':
        return InceptionTime(conf)
    elif model_name == 'rnn':
        return RNN(conf)
    elif model_name == 'transformer':
        feat_dim = conf.getHP('num_input_channels')
        max_seq_len = conf.getHP('dim_series')
        d_model = conf.getHP('transformer_d_model')
        n_heads = conf.getHP('transformer_n_heads')
        num_layers = conf.getHP('transformer_n_layers')
        dim_feedforward = conf.getHP('transformer_d_feedforward')
        num_classes = conf.getHP('num_class')
        dropout = conf.getHP('transformer_dropout')
        pos_encoding = conf.getHP('transformer_pos_encoding')
        activation = conf.getHP('transformer_activation')
        norm = conf.getHP('transformer_t_norm')
        freeze = conf.getHP('transformer_freeze')

        return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, d_model, n_heads, num_layers, dim_feedforward,
                                                   num_classes, dropout, pos_encoding, activation, norm, freeze,
                                                   conf)

    raise ValueError('invalid model name: {:s}'.format(model_name))

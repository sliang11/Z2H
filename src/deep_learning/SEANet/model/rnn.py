# coding = utf-8

import torch
from torch import nn, Tensor

from util.conf import Conf
from model.commons import Squeeze, Reshape
from util.commons import discretize


class RNN(nn.Module):
    def __init__(self, conf: Conf):
        super(RNN, self).__init__()

        # TODO fixed for binary classification
        assert conf.getHP('num_class') == 2
        num_class = 1

        model_type: str = conf.getHP('rnn_type')

        self.__threshold = conf.getHP('threshold')
        
        if model_type == 'gru':
            model_class = nn.GRU
        elif model_type == 'lstm':
            model_class = nn.LSTM
        else:
            raise ValueError('rnn {:s} is not supported'.format(model_type))

        dim_latent = conf.getHP('dim_latent')
        bidirectional = conf.getHP('if_rnn_bidirectional')
        num_directions = 2 if bidirectional else 1
        num_layers = conf.getHP('num_rnn_layers')
        dropout = conf.getHP('rnn_dropout')

        in_channels = conf.getHP('num_input_channels')

        self.__transform = model_class(
            input_size = in_channels,
            hidden_size = dim_latent,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout
        )

        # self.__map = model_class(
        #     input_size = dim_latent * (2 if bidirectional else 1),
        #     hidden_size = 1,
        #     num_layers = 1,
        #     batch_first = True
        # )

        self.__dim_infer = dim_latent * num_directions * num_layers

        self.__infer = nn.Sequential(
            nn.Linear(self.__dim_infer, num_class),

            Squeeze(),
            nn.Sigmoid()
        )


    def forward(self, input: Tensor) -> Tensor:
        input = torch.swapaxes(input, 1, 2)

        output, (h, c) = self.__transform(input)

        h = torch.swapaxes(h, 0, 1).reshape([-1, self.__dim_infer])
        
        return self.__infer(h)


    def infer(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            predictions = self.forward(input).detach().cpu().numpy()

        return discretize(predictions, self.__threshold)

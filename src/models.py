import torch
import torch.nn as nn

from sti_config import config


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Config
        input_size = config.getint('LSTM', 'input_size')
        hidden_layer_size = config.getint('LSTM', 'hidden_layer_size')
        output_size = config.getint('LSTM', 'output_size')

        # Network
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_output, (hn, cn) = self.lstm(input_seq)
        linear_output = self.linear(lstm_output[:, -1, :])
        return linear_output


class CnnLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Config
        in_channels = config.getint('CnnLSTM', 'in_channels')
        out_channels = config.getint('CnnLSTM', 'out_channels')
        kernel_size_h = config.getint('CnnLSTM', 'kernel_size_h')
        kernel_size_w = config.getint('CnnLSTM', 'kernel_size_w')
        dilation_h = config.getint('CnnLSTM', 'dilation_h')
        dilation_w = config.getint('CnnLSTM', 'dilation_w')
        stride_h= config.getint('CnnLSTM', 'stride_h')
        stride_w = config.getint('CnnLSTM', 'stride_w')
        padding = config.get('CnnLSTM', 'padding')
        input_size = config.getint('CnnLSTM', 'input_size')
        hidden_layer_size = config.getint('CnnLSTM', 'hidden_layer_size')
        output_size = config.getint('CnnLSTM', 'output_size')

        # Network
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=(kernel_size_h, kernel_size_w), dilation=(dilation_h, dilation_w),
                             stride=(stride_h, stride_w), padding=padding)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        input_seq = torch.unsqueeze(input_seq, 1)
        cnn_output = self.cnn(input_seq)
        cnn_output = torch.squeeze(cnn_output)
        cnn_output = cnn_output.swapaxes(1, 2)
        lstm_output, (hn, cn) = self.lstm(cnn_output)
        linear_output = self.linear(lstm_output[:, -1, :])
        return linear_output

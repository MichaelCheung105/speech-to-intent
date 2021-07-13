import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CnnMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        # Config
        in_channels = config.getint('CnnMaxPool', 'in_channels')
        out_channels = config.getint('CnnMaxPool', 'out_channels')
        kernel_size_h = config.getint('CnnMaxPool', 'kernel_size_h')
        kernel_size_w = config.getint('CnnMaxPool', 'kernel_size_w')
        dilation_h = config.getint('CnnMaxPool', 'dilation_h')
        dilation_w = config.getint('CnnMaxPool', 'dilation_w')
        stride_h= config.getint('CnnMaxPool', 'stride_h')
        stride_w = config.getint('CnnMaxPool', 'stride_w')
        padding = config.get('CnnMaxPool', 'padding')
        input_size = config.getint('CnnMaxPool', 'input_size')
        hidden_layer_size = config.getint('CnnMaxPool', 'hidden_layer_size')
        output_size = config.getint('CnnMaxPool', 'output_size')

        # Network
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=(kernel_size_h, kernel_size_w), dilation=(dilation_h, dilation_w),
                             stride=(stride_h, stride_w), padding=padding)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))
        # self.max_pool =
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 31)

    def forward(self, input_seq):
        input_seq = torch.unsqueeze(input_seq, 1)
        cnn_output = self.cnn(input_seq)
        cnn_output = self.max_pool(cnn_output)
        flattened_output = cnn_output.flatten(start_dim=1)
        flattened_output = F.relu(self.linear1(flattened_output))
        prediction = self.linear2(flattened_output)
        return prediction

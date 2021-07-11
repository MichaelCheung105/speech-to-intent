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
        kernel_size = config.getint('CnnLSTM', 'kernel_size')
        dilation = config.getint('CnnLSTM', 'dilation')
        stride = config.getint('CnnLSTM', 'stride')
        padding = config.get('CnnLSTM', 'padding')
        input_size = config.getint('CnnLSTM', 'input_size')
        hidden_layer_size = config.getint('CnnLSTM', 'hidden_layer_size')
        output_size = config.getint('CnnLSTM', 'output_size')

        # Network
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        swaped_input_seq = input_seq.swapaxes(1, 2)
        cnn_output = self.cnn(swaped_input_seq)
        swaped_cnn_output = cnn_output.swapaxes(1, 2)
        lstm_output, (hn, cn) = self.lstm(swaped_cnn_output)
        linear_output = self.linear(lstm_output[:, -1, :])
        return linear_output

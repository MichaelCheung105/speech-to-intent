import torch.nn as nn

from sti_config import config


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = config.getint('LSTM', 'input_size')
        hidden_layer_size = config.getint('LSTM', 'hidden_layer_size')
        output_size = config.getint('LSTM', 'output_size')
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_output, (hn, cn) = self.lstm(input_seq)
        linear_output = self.linear(lstm_output[:, -1, :])
        return linear_output

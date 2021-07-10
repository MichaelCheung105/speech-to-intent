import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_output, (hn, cn) = self.lstm(input_seq)
        linear_output = self.linear(lstm_output[:, -1, :])
        return linear_output

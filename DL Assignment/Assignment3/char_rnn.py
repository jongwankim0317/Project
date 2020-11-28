import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        ##### please implment here

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, d, hidden):
        
        ##### please implment here
        out, _ = self.lstm(d)
        out = self.fc(out[-1])
        output = self.softmax(out)
        return output

 

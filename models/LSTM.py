import torch
import torch.nn as nn


class LSTMpred(nn.Module):
    def __init__(self):
        super(LSTMpred, self).__init__()
        self.input_size = 30
        self.h = 1024
        self.hidden_size = 128
        self.num_layers = 2
        self.embedding = nn.Embedding(40, 128)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=256,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.gen_f = nn.Linear(256 * self.num_layers, self.hidden_size)
        self.projection = nn.Sequential(nn.Linear(self.hidden_size, self.h),
                                        nn.BatchNorm1d(self.h),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.h, self.hidden_size),
                                        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.long()
        x = self.embedding(x)
        x, (h_n, c_n) = self.lstm(x)
        output_forward = h_n[-2, :, :]
        output_backward = h_n[-1, :, :]
        out = torch.cat([output_forward, output_backward], dim=1)
        features = self.gen_f(out)
        projection = self.projection(features)
        return features, projection

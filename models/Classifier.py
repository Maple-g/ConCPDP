import torch.nn as nn
import torch
from models import lsoftmax


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.hidden_size = input_dim
        self.h = 1024
        self.lsoftmax = lsoftmax.LSoftmaxLinear(2, output_features=2, margin=4, device=torch.device("cuda"))
        self.softmax = nn.Softmax(dim=1)
        self.predictor = nn.Sequential(nn.Linear(self.hidden_size, self.h),
                                       nn.BatchNorm1d(self.h),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.h, self.hidden_size),
                                       )
        self.pre = nn.Sequential(nn.Linear(self.hidden_size, 2),

                                 )
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax.reset_parameters()

    def forward(self, x, target=None):
        if target is not None:
            prediction = self.predictor(x)
            predict = self.pre(prediction)
            logit = self.lsoftmax(input=predict, target=target)

        else:
            prediction = self.predictor(x)
            predict = self.pre(prediction)
            logit = self.softmax(predict)

        return prediction, logit



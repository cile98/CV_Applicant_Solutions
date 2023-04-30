"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(8)  # DO NOT MODIFY!
np.random.seed(8)  # DO NOT MODIFY!


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Conv = [nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1),
                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
                      nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)]

        for conv in self.Conv:
            torch.nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.linear = nn.Linear(256, 256, bias=True)
        self.batchnorm = lambda x: nn.BatchNorm2d(x)
        torch.nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.Conv[i](x)
            x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))
        x1 = self.Conv[9](x)
        x1 = torch.mul(x1, self.batchnorm(x1.shape[1])(self.Sigmoid(x1)))

        x = self.Conv[4](x)
        x2 = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))
        x = self.Conv[5](x2)
        x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))
        x = self.Conv[6](x)
        x3 = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))
        x = self.Conv[7](x3)
        x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))
        x = self.Conv[8](x)
        x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))
        x = torch.cat((x1, x2, x3, x), dim=1)

        x = self.Conv[10](x)
        x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))

        x4 = self.pooling(x)
        x4 = self.Conv[13](x4)
        x4 = torch.mul(x4, self.batchnorm(x4.shape[1])(self.Sigmoid(x4)))

        x = self.Conv[11](x)
        x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))

        x = self.Conv[12](x)
        x = torch.mul(x, self.batchnorm(x.shape[1])(self.Sigmoid(x)))

        x = torch.cat((x4, x), dim=1)
        x = torch.permute(x, [0, 2, 3, 1])
        shapes = x.shape
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.linear(x)
        x = x.reshape(*shapes)
        x = torch.permute(x, [0, 3, 1, 2])
        x = self.Sigmoid(x)
        return x


# write your code here ...
if __name__ == '__main__':
    model = Model()
    x = torch.randn(2, 3, 160, 320)
    out = model.forward(x)
    print(out.shape)

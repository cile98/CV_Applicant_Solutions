"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""
import numpy as np
import torch
import torch.nn as nn


# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.groups = groups
        self.GroupedConv = nn.ModuleList([nn.Conv2d(in_channels=in_channels // groups,
                                                    out_channels=out_channels // groups, kernel_size=kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation, bias=bias,
                                                    padding_mode=padding_mode, device=device, dtype=dtype, groups=1)
                                          for _ in range(self.groups)])

    def forward(self, x):
        x = torch.split(x, x.shape[1] // self.groups, dim=1)
        x = torch.cat([conv(x_input) for conv, x_input in zip(self.GroupedConv, x)], dim=1)
        return x


# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
if __name__ == '__main__':
    torch.manual_seed(8)  # DO NOT MODIFY!
    np.random.seed(8)  # DO NOT MODIFY!

    # random input (batch, channels, height, width)
    x = torch.randn(2, 64, 100, 100)

    # original 2d convolution
    grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

    # weights and bias
    w_torch = grouped_layer.weight
    b_torch = grouped_layer.bias
    y = grouped_layer(x)
    # split parameters of original convolution
    w_split = torch.split(w_torch, w_torch.shape[0] // 16, dim=0)
    b_split = torch.split(b_torch, b_torch.shape[0] // 16, dim=0)
    custom_grouped_conv = CustomGroupedConv2D(64, 128, 3, stride=1, padding=1, groups=16, bias=True)
    # set parameters of custom convolution
    for layer, ws, bs in zip(custom_grouped_conv.GroupedConv, w_split, b_split):
        layer.weight = torch.nn.Parameter(ws)
        layer.bias = torch.nn.Parameter(bs)
    y_custom = custom_grouped_conv.forward(x)
    print(torch.allclose(y.detach(), y_custom.detach(), atol=1e-6))

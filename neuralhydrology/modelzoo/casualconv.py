import torch.nn.functional as fun
class CasualConv1D(nn.Module):
    """
    Length preserving 1D convolution in which it only looks at the past
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super(CausalConv1d, self).__init__()
        self.conv_module = nn.Conv1D(in_channels=in_channels, out_channel=out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.padding = ((kernel_size-1)*dilation , 0)

    def forward(self, x):
        x = fun.pad(x, self.padding)
        return self.conv_module(x)


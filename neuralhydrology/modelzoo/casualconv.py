import torch.nn as nn
import torch.nn.functional as fun
class CasualConv1D(nn.Module):
    """
    Length preserving 1D convolution in which it only looks at the past
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super(CasualConv1D, self).__init__()
        self.conv_module = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.padding = ((kernel_size-1)*dilation , 0)

    def forward(self, x):
        """
         Receives tensor in shape B, N, C instead of B, C, N
        """
        x_trans = x.transpose(-2, -1)
        x_trans = fun.pad(x_trans, self.padding) 
        returned = self.conv_module(x_trans)
        # Transpose back so that channel now goes to the end
        return returned.transpose(-2, -1)


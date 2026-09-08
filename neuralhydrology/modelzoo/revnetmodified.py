
from  neuralhydrology.datasetzoo.basemultiseries import BaseMultiSeries
from neuralhydrology.datasetzoo.casualconv import CasualConv1D
import torch.nn as nn
from typing import List, Dict


class ModifiedRevnet(BaseMultiSeries):

    class Model1(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(ModifiedRevnet.Model1, self).__init__()
            self.lstm = nn.LSTM(input_size=inpu_size, hidden_size=hidden_size)

        def forward(self, x):
            return self.lstm(x["xd1"])


    class Model2(nn.Module):
        def __init__(self, input_size, hidden1, hidden2, output, kernel1, kernel2, dilation_factor=2):
            model2 = nn.Sequential(
                CasualConv1D(in_channels = input_size, out_channels = hidden1, kernel_size = kernel1 ),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2, dilation = dilation_factor),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2, dilation = dilation_factor * dilation_factor),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2, dilation = round(dilation_factor**3)),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2, dilation = round(dilation_factor**4)),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2, dilation = round(dilation_factor**5)),
                nn.GELU()
            )
            
        def forward(self, x):
            return self.model2(x["xfw1"])


    class Aggregator(nn.Module):
        def __init__(self, input1, input2, static, burn_area_resolution_days, output):
            super(ModifiedRevnet.Aggregator, self).__init__()
            self.w1 = nn.Parameter(torch.randn(input1, output), requires_grad=True)
            self.w2 = nn.Parameter(torch.randn(input2, output), requires_grad=True)
            self.w3 = nn.Parameter(torch.randn(static, output), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(output), requires_grad=True)
            self.burn_area_resolution_days = burn_area_resolution_days

        def foward(self, x):
            d1 = x["lstm"]@ self.w1
            d2 = x["conv"]@ self.w2
            s = x["xs"] @ self.w3
            d2 = torch.repeat_interleave(d2, repeats=self.burn_area_resolution_days, dim=-2)[..., :d1.shape[-2],: ]
            output = d1 + d2 + s + self.b
            output = GELU(output)
            return output

            

    def __init__(self, cfg: Config):
        model1 = ModifiedRevnet.Model1(self.embedding_net.output_size, cfg.hidden_size)
        hidden1, hidden2, kernel1, kernel2 = cfg.as_dict()["conv1d.hidden_size1"], cfg.as_dict()["conv1d.hidden_size2"], cfg.as_dict()["conv1d.kernel_size1"], cfg.as_dict()["conv1d.kernel_size2"]
        model2 = nn.Sequential(
            nn.CasualConv1D(in_channels = self.embedding_net.output_size, out_channels = hidden1, kernel_size = kernel1 ),
            nn.GELU(),
            nn.CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2),
            nn.GELU()
            )
        model_dictionary = nn.ModuleDict({"lstm": model1, "conv": model2})
        final_hidden_size = cfg.as_dict()["final_hidden_size"]
        static_size = len(cfg.static_attributes)
        burn_area_res = cfg.as_dict()["burn_area_resolution_days"]
        aggregator = ModifiedRevnet.Aggregator(cfg.hidden_size, hidden2, static_size, burn_area_res,  final_hidden_size)
        super(ModifiedRevnet, self).__init__(cfg, model_dictionary, aggregator)
        self.head = get_head(cfg=cfg, n_in=final_hidden_size, n_out=self.output_size)
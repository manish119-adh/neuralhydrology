
from  neuralhydrology.modelzoo.basemultiseries import BaseMultiSeries
from neuralhydrology.modelzoo.casualconv import CasualConv1D
from neuralhydrology.modelzoo.head import get_head
import torch.nn as nn
from typing import List, Dict
import torch



class ModifiedRevnet(BaseMultiSeries):

    class VisualWrapper(nn.Module):
        def __init__(self, base_model: ModifiedRevnet):
            super(ModifiedRevnet.VisualWrapper, self).__init__()
            self.base_model = base_model

        def forward(self, xd1, xfw1, xs):
            return self.base_model.forward({"xd_freq1D":xd1, "xburn":xfw1, "xs":xs})


    class Model1(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(ModifiedRevnet.Model1, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        def forward(self, xd_freq1D):
            output = self.lstm(xd_freq1D)
            return output


    class Model2(nn.Module):
        def __init__(self, input_size, hidden1, hidden2, kernel1, kernel2, dilation_factor=2):
            super(ModifiedRevnet.Model2, self).__init__()
            self.conv1d = nn.Sequential(
                CasualConv1D(in_channels = input_size, out_channels = hidden1, kernel_size = kernel1 ),
                nn.GELU(),
                CasualConv1D(in_channels = hidden1, out_channels = hidden2,kernel_size = kernel2),
                nn.GELU(),
                CasualConv1D(in_channels = hidden2, out_channels = hidden2,kernel_size = kernel2, dilation = dilation_factor),
                nn.GELU(),
                CasualConv1D(in_channels = hidden2, out_channels = hidden2,kernel_size = kernel2, dilation = dilation_factor * dilation_factor),
                nn.GELU(),
                CasualConv1D(in_channels = hidden2, out_channels = hidden2,kernel_size = kernel2, dilation = round(dilation_factor**3)),
                nn.GELU(),
                CasualConv1D(in_channels = hidden2, out_channels = hidden2,kernel_size = kernel2, dilation = round(dilation_factor**4)),
                nn.GELU(),
                CasualConv1D(in_channels = hidden2, out_channels = hidden2,kernel_size = kernel2, dilation = round(dilation_factor**5)),
                nn.GELU()
            )
            
        def forward(self, xburn):
            return self.conv1d(xfw1)


    class Aggregator(nn.Module):
        def __init__(self, input1, input2, static, burn_area_resolution_days, output):
            super(ModifiedRevnet.Aggregator, self).__init__()
            self.w1 = nn.Parameter(torch.randn(input1, output), requires_grad=True)
            self.w2 = nn.Parameter(torch.randn(input2, output), requires_grad=True)
            self.w3 = nn.Parameter(torch.randn(static, output), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(output), requires_grad=True)
            self.burn_area_resolution_days = burn_area_resolution_days

        def forward(self, lstm_output, conv_output, xs):
            d1 = lstm_output[0] @ self.w1
            d2 = conv_output @ self.w2
            s = xs @ self.w3
            d2 = torch.repeat_interleave(d2, repeats=self.burn_area_resolution_days, dim=-2)[..., :d1.shape[-2],: ]
            output = d1 + d2 + s[..., None, :] + self.b
            output = nn.GELU()(output)
            return output

            

    def __init__(self, cfg: Config):
        lstm_input, lstm_hidden = cfg.as_dict()["lstm.input_size"], cfg.as_dict()["lstm.hidden_size"]
        model1 = ModifiedRevnet.Model1(lstm_input, lstm_hidden)
        conv1dinput, hidden1, hidden2, kernel1, kernel2 = cfg.as_dict()["conv1d.input_size"], cfg.as_dict()["conv1d.hidden_size1"], cfg.as_dict()["conv1d.hidden_size2"], cfg.as_dict()["conv1d.kernel_size1"], cfg.as_dict()["conv1d.kernel_size2"]
        model2 = ModifiedRevnet.Model2(conv1dinput, hidden1, hidden2, kernel1, kernel2 )
        model_dictionary = nn.ModuleDict({"lstm": model1, "conv": model2})
        final_hidden_size = cfg.hidden_size
        static_size = len(cfg.static_attributes)
        burn_area_res = cfg.as_dict()["burn_area_resolution_days"]
        aggregator = ModifiedRevnet.Aggregator(lstm_hidden, hidden2, static_size, burn_area_res,  final_hidden_size)
        super(ModifiedRevnet, self).__init__(cfg, aggregator, model_dictionary)
        self.head = get_head(cfg=cfg, n_in=final_hidden_size, n_out=self.output_size)

    
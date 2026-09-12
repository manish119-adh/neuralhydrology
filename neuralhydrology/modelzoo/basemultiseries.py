
from neuralhydrology.modelzoo.basemodel import BaseModel
import torch.nn as nn
from typing import List, Dict
from neuralhydrology.utils.decorators import ignoreextraforward
import torch
class BaseMultiSeries(BaseModel):
    """
    Generic model that allows for combining multiple stream predictions from multiple channels and then combine
    outputting using a single dense layer. The outputs are combined using a single dense layer for
    each target time step. The multiple channels can have multiple temporal resolutions.
    The specifics of resolution is left to the subclass which is tasked with combining all the 
    hidden vectors of all models into a flat projection prediction_time_steps X final_dimension
    All models will output their hidden dimensions as final layer
    """

    def __init__(self,  cfg: Config, combiner: nn.Module, submodels: nn.ModuleDict):
        """
        The model takes config from models and replaces their head with an identity layer (no prediction)
        It them combines them into a downstream combiner model which produces. The inputs are expected
        All modules are expected to use consistent parameter names for same set of tensor in their forward method
        i.e If model[0] and model[1] both take static properties, both are expected to call it
        say ,xs in forward pass

        The combiner takes output from each module and combines them. The combiner's forward method
        will be expected to use keyword arguments for tensors that match the module names

        combiner: A model that combines outputs into a time series output
        submodels: Submodels each outputting a time series of hidden vectors that will be combined
        to make a prediction. Each submodels may output at different temporal resolutions
        It is on combiner to decide how to combine them

        It is suggested that you do not use the keys in input tensors for submodel outputs as
        the combiner also may use part of raw data e.g static inputs which will be input as required.
        Matching key names will cause the respective inputs from data dictionary be erased
        The parameter names in aggregator (combiner) model will be all input tensors with their respective
        names PLUS <model name>_output from all the submodels
        While we do not restrict what raw data the combiner can use we suggest that only static or
        in rare cases very low resolution data is used by combiner and all modules produce time series data

        The subclasses are also responsible for making their own prediction head which is not defined here

        """
        super(BaseMultiSeries, self).__init__(cfg=cfg)
        self.submodels = submodels
        self.combiner = combiner
        # 

    def forward(self, data:Dict[str, nn.Tensor]):
        # time series is input as a dictionary. Convert it to
        # tensor
        new_data = {}
        for seq_key in data:
            if isinstance(data[seq_key], dict):
                # convert all of them into tensors
                # Features make the last dimension
                new_data[seq_key] = torch.stack(data[seq_key].values(), dim=-1)
            else:
                new_data[seq_key] = data[seq_key] # pass unchanged
        data = new_data


        outputs = {f"{k}_output": ignoreextraforward(self.submodels[k])(**data) for k in self.submodels}
        # merge data and outputs
        
        combiner_output = ignoreextraforward(self.combiner)(**(data | outputs))
        # Finally add prediction head
        outputs = self.head(combiner_output)

        
        


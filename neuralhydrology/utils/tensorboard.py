import secrets
import torch
import torch.nn as nn
import string
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Every model we have accepts a heterogeneuous dictionary with string key and value of either tensor, nparray, list of strings or another dictionary of tensors
# It needs to be converted to a uniform string [Tensor] for visualization

def generate_separator_string():
    pool = string.ascii_letters + string.digits
    # Securely select characters and join them
    return ''.join([secrets.choice(pool) for _ in range(32)])

def flatten_for_graph(input_):
    maindict = {}
    for k in input_:
        if isinstance(input_[k], dict):
            sep = generate_separator_string()
            for k1 in input_[k]:
                maindict[f"__nested_{sep}_{k}_{sep}_{k1}"] = input_[k][k1]
        elif isinstance(input_[k], torch.Tensor):
            maindict[f"__tensor_{k}"] = input_[k]
        elif isinstance(input_[k], np.ndarray):
            maindict[f"__ndarray_{input_[k].dtype}:_{k}"] = torch.from_numpy(input_[k].astype(np.int64))
        elif isinstance(input_[k], list):
            maxlen = max(len(wd) for wd in input_[k])
            maindict[f"__list_{k}"] = torch.tensor([[ord(ch) for ch in wd] + [0]*(maxlen - len(wd)) for wd in input_[k]])
        else:
            raise TypeError("Unsupported type for visualization. All models must receive the input of dictionary of either a tensor, a ndarray (for dates), a list of strings (for feature list), or a dictionary of tensors (alternative for higher dimension tensor + list of features)")
    return maindict

def unflatten_after_graph(input_):
    maindict = {}
    def decode_nested(key):
        start = len("__nested_")
        end = key.index("_", start)
        sep = key[start:end]
        [k1, k2] = key[end+1:].split(f"_{sep}_")
        if k1 not in maindict:
            maindict[k1] = {}
        maindict[k1][k2] = input_[key]

    def decode_list(key):
        maindict[key[len("__list_"):]] = [bytes(list(t1)).decode('utf-8').rstrip('\x00') for t1 in input_[key]]

    def decode_array(key):
        dtype = key[len("__ndarray_"):key.index(":")]
        k1 = key[key.index(":") + 2:]
        maindict[k1] = input_[key].numpy().astype(dtype)
    


    for k in input_:
        if k.startswith("__tensor_"):
            maindict[k[len("__tensor_"):]] = input_[k]
        elif k.startswith("__nested_"):
            decode_nested(k)
        elif k.startswith("__ndarray_"):
            decode_array(k)
        elif k.startswith("__list_"):
            decode_list(k)
        else:
            raise ValueError(f"Cannot decode key {k}")
    return maindict

class WrappedModel(nn.Module):

    def __init__(self, main_model):
        super(WrappedModel, self).__init__()
        self.main_model = main_model

    def forward(self, input_):
        return unflatten_after_graph(input_)
            



def visualize_model(model, input_, filename):
    writer = SummaryWriter(filename )
    writer.add_graph(WrappedModel(model), flatten_for_graph(input_))
    writer.close()



         
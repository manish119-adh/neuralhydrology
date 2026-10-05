from torch.utils.data import Dataset
from typing import List, Dict, Union
from collections.abc import Iterable
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo.template import TemplateDataset
import random
import pandas as pd
from functools import partial, reduce
class DatasetView(Dataset):
    """
     This class represents a view dataset backed by one or more other datasets
     (does not require them to be of same type) but recommended

     This view can be created for merging datasets (e.g both CAMELSUS and CAMELSDE)
     together or by splitting (k fold cross validation)

     Note that it must by backed by backing datasets and will return the elements
     from backing datasets. Any modification in backing datasets will be reflected
    """
    
    

    def __init__(self, datasets: Union[List[Dataset] , Dict[str, Dataset]], mappings: Iterable, dataset_names: Union[None | List[str]] =None):
        """
        datasets must be either a list of backing datasets or
        a mapping name and dataset. if datasets is list and dataset_names
        is provided, mappings will be created

        mapping is an iterable of (dataset_key, index) tuple which maps index of this
        dataset into index of the backing datset. dataset_key is the dataset name if name 
        is provided or datasets is provided as dict otherwise it is the int index
        of the dataset, index is the index of original datset the data is drawn from
        """
        if isinstance(datasets, list):
            if not all(isinstance(dataset, Dataset) for dataset in datasets):
                raise TypeError("datasets must be either a list of datasets or mappings")
            if dataset_names is not None:
                if not (isinstance(dataset_names, list)) or not all(isinstance(name, str) for name in dataset_names) or len(datasets) != len(dataset_names) or len(set(dataset_names)) != len(dataset_names):
                    raise TypeError("dataset names must be strings and unique and of same length as datasets")
                self._datasets = dict(zip(dataset_names, datasets))
            else:
                self._datasets = dict(enumerate(datasets))
        elif isinstance(datasets, dict):
            if not all(isinstance(k, str) and isinstance(v, Dataset) for (k, v) in datasets.items()):
                raise TypeError("Dataset names must be string and value must be datasets")
            self._datasets = dict(datasets)
        else:
            raise TypeError(f"{type(datasets)} not allowed for datasets")

        
        for dataset_key, index in mappings:
            if dataset_key not in self._datasets:
                raise ValueError(f"No dataset {dataset_key}")
            if not (0 <= index < len(self._datasets[dataset_key])):
                raise IndexError(f"Index {index} out of range for dataset {dataset_key}")
        self._mappings = list(mappings)
            



    @property
    def datasets(self):
        return dict(self._datasets)

    @property
    def dataset_keys(self):
        return list(self._datasets.keys()) 

    def __len__(self):
        return len(self._mappings)

    def __getitem__(self, ind):
        key, index = self._mappings[ind]
        return self._datasets[key][index]

    @staticmethod
    def collate_fn(samples):
        return BaseDataset.collate_fn(samples)



    


def basinwise_kfold_cross_validation(dataset: BaseDataset, name, n_folds=5, randomize=False, stride=1):
    """
    Split datasets for basinwise k fold cross validation
    if randomize is true, the indices will be randomized otherwise it will be  
    kept in the same order 
    For each basin, split the indices individually
    """ 
    basin_wise_indices = {}
    for i in range(len(dataset)):
        (basin, _) = dataset.lookup_table[i]
        if basin not in basin_wise_indices:
            basin_wise_indices[basin] = []
        basin_wise_indices[basin].append(i)
    for basin in basin_wise_indices:
        # select a subset of samples by stride parameter
        basin_wise_indices[basin] = basin_wise_indices[basin][::stride]

    if randomize:
        # permute indices in basin_wise_indices
        for basin in basin_wise_indices:
            random.shuffle(basin_wise_indices[basin])

    def make_fold(basin_folds, name, fold_index):
        mappings = []
        train_indices = []
        validation_indices = []
        for basin in basin_folds:
            (start, end) = basin_folds[basin][fold_index]
            _train_indices = basin_wise_indices[basin][:start] +  basin_wise_indices[basin][end:]
            _validation_indices = basin_wise_indices[basin][start:end]
            train_indices.extend(_train_indices)
            validation_indices.extend(_validation_indices)
        return (DatasetView({name:dataset}, [(name, ind) for ind in train_indices]),
                DatasetView({name:dataset}, [(name, ind) for ind in validation_indices]))

    
    basin_fold_mappings = {}
    for basin in basin_wise_indices:
        #TODO Make folds more equal
        n_items = round(len(basin_wise_indices[basin])/n_folds)
        basin_fold_mappings[basin] = [(i, min(i+n_items, len(basin_wise_indices[basin]))) for i in range(0, len(basin_wise_indices[basin]), n_items)]
    return [make_fold(basin_fold_mappings, name, i) for i in range(n_folds)]

    

if __name__ == "__main__":
    from pathlib import Path
    from neuralhydrology.utils.config import Config
    from neuralhydrology.datasetzoo.camelsusburn import CamelsUSBurn
    cfg = Config(Path("notebooks/burndata/1_basin.yml"), dev_mode=True)
    # Add additional configuration
    dict_config = cfg.as_dict()
    dict_config.update({
        "daily_input_size":7, 
        "daily_hidden_size":20, 
        "burn_input_size":5,
        "burn_hidden_size1":4, 
        "burn_hidden_size2":3,
        "burn_kernel_size1":5, 
        "burn_kernel_size2":5, 
        # "burn_area_resolution_days":30, 
        "train_basin_file": "notebooks/burndata/1_basin.txt",
        "test_basin_file": "notebooks/burndata/1_basin.txt",
        "validation_basin_file": "notebooks/burndata/1_basin.txt",
        "data_dir":"data/CAMELS_US",
        "hidden_size":20, 
        "burn_start_dates":{"01013500":"1986-12"},
        "burn_end_dates":{"01013500":"2010-01"},
        "month_emb_dim":4,
        "model":"revnet",
        "dataset":"camels_us_burn"
        
        })
    cfg = Config(dict_config, dev_mode=True) # Update config
    dataset = CamelsUSBurn(cfg, is_train=True, period="train" )
    folds = basinwise_kfold_cross_validation(dataset, "camelsus", stride=50)
    gh = folds[1][0][len(folds[1][0]) - 4]
    gh

    




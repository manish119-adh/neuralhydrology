from neuralhydrology.datasetzoo.camelsus import CamelsUS
import glob
import pandas as pd
import xarray as xr
import numpy as np
from neuralhydrology.utils.config import Config
from pathlib import Path
from functools import reduce
from operator import iconcat
from neuralhydrology.utils.dateutils import get_month_year
from neuralhydrology.modelzoo.revnetmodified import ModifiedRevnet
import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Union
import sys
from functools import cache

from torch.nn.functional import pad as torch_pad

logger = logging.getLogger("Camels US")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class CamelsUSBurn(CamelsUS):
    burn_properties = [ "frac_low", "frac_moderate", "frac_high", "frac_inc_greenness", "frac_unburned_low"]
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):
        super(CamelsUSBurn, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)
        self.index_calls = 0 # count number of times __getitem__ is called


    def _get_burn_data_dates(self):
        """
        dates for burn data
        By default it starts 2 years before the first start date of the 
        given basin and ends after one month of the last end date for each basin
        however it can be changed for configuration

        If changed from default:
        There are two methods:
        single start and/or end dates: Same date applied for all basins
        per basin start and/or end dates: Each basin can have same start or end dates
        If only a subset of basins have per basin dates, default dates will be used for
        all the remaining basins

        Unlike forcing start dates which represent the end day of first
        samples of each prediction period and thus requires a warmup
        of sequence length the burn data start dates represent the days burn data
        is loaded so no model will be able to use any burn data beyond the
        start and end dates for a given basin

        No warmup period is applied on burn data so the program will only
        load burn data from the given start date
        """
        # Get burn data dates for this
        # if burn start and end dates are defined 
        # in the configuration files
        # use them otherwise it is 8 years before the start date of the first month
        # first start date till one month after last end date
        # initialize with default dates
        start_date = {basin:get_month_year(min(self.start_and_end_dates[basin]["start_dates"])) - 96 for basin in self.basins}    
        end_date = {basin:get_month_year(max(self.start_and_end_dates[basin]["end_dates"])) + 1 for basin in self.basins}
        def extract_basin_dates_from_config(val):
            if isinstance(val, dict):
                # if it is dict is is expected to have format {"basin":"basin_start_date"}
                # which will be extracted
                return {basin:get_month_year(val[basin]) for basin in val}
            else:
                # flat dates for all basins
                return {basin:get_month_year(val) for basin in self.basins}
        if f"{self.period}_burn_start_dates" in self.cfg.as_dict():
            val = self.cfg.as_dict()[f"{self.period}_burn_start_dates"]
            start_date.update(extract_basin_dates_from_config(val))
        if f"{self.period}_burn_end_dates" in self.cfg.as_dict():
            val = self.cfg.as_dict()[f"{self.period}_burn_end_dates"]
            end_date.update(extract_basin_dates_from_config(val))
        self._burn_dates = (start_date, end_date)
        # Get the sequence length for burn data default is 96 months or
        # 8 years upto the last month we have daily forcing data of
        self._burn_sequence_length = 96
        if "burn_sequence_length" in self.cfg.as_dict():
            self._burn_sequence_length = int(self.cfg.as_dict()["burn_sequence_length"])


    def _create_monthly_xarray_burn_data(self) -> xr.Dataset:
        self._get_burn_data_dates()
        fire_severity_data = self._load_burn_data()        
        x_burn = {}
        fire_months = {}
        def create_datset(basin): 
            start_date = self._burn_dates[0][basin]
            end_date = self._burn_dates[1][basin]
            month_year_list = list(map(get_month_year, range(start_date, end_date+1)))
            dict_ = {}
            for prop in  CamelsUSBurn.burn_properties:
                dict_[prop] = xr.DataArray(np.zeros(shape=(1, len(month_year_list))), 
                    dims=["basin", "month_year"],
                    coords = {
                        "month_year":month_year_list,
                        "basin":[basin]
                     }
                )
                for month_year in month_year_list:
                    if month_year in fire_months[basin]:
                        fire_data = fire_severity_data[basin][fire_months[basin][month_year]]
                        dict_[prop].loc[{"basin":basin, "month_year":month_year}] = fire_data["overlap_fraction"] * fire_data[prop]            
            ds = xr.Dataset(dict_)
            return ds          
        basin_datasets = [] 
        for basin in self.basins:
            fire_months[basin] = {}
            x_burn_parts = []
            # assume no fire if fire severity data for those basins 
            # do not exist
            if basin in fire_severity_data.index: 
                # collect the data for the months the fire occured and 
                # build reverse ,month to index mappping
                for i, fire in enumerate(fire_severity_data[basin]):
                    fire_date = get_month_year("-".join(fire["Ig_Date_str"].split("-")[:2]))     
                    fire_months[basin][fire_date] = i
            # I assume previous step already does all the sanity checks for dates
            basin_dataset = create_datset(basin)
            basin_datasets.append(basin_dataset)
        # merge them into single dataarray
        # concatenate all datasets so we get basin X months
        # Align so that
        # any mismatched dimensions will be filled with NaN 
        x_burn = xr.concat(basin_datasets, dim="basin", join="outer")
        x_burn = x_burn.sortby("month_year")
        # merge all basin DataArray's into a single xarray dataset
        return x_burn


    def _load_or_create_xarray_dataset(self) -> xr.Dataset:
        # Load or create xarray dataset overridden from basedataset
        # We will also merge burn dataset into the original dataset and save it 
        # if applicable
        
        if (self.cfg.train_data_file is not None) and (self.is_train):
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
                dataset = xr.Dataset.from_dict(d)
        else:
            # Merge the forcing dataset from basedataset
            logger.info("Loading burn timeseries data")
            forcing_dataset = super(CamelsUSBurn, self)._load_or_create_xarray_dataset()
            logger.info("Finished loading flow and forcings time series data. Preparing to load burn timeseries data")
            burn_dataset = self._create_monthly_xarray_burn_data()
            logger.info("Finished loading burn time series data")
            dataset = xr.merge([forcing_dataset, burn_dataset])
            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(dataset)
                logger.info("Training data saved to file")
        return dataset


    def _create_lookup_table(self,  xrds: xr.Dataset):
        # It calls _create_lookup_table from the base using only the original dataset (burn data removed)
        # because passing the combined dataset as unchanged caused an error
        xrds_without_burn = xrds[[name for name in xrds.data_vars if name not in CamelsUSBurn.burn_properties]]
        logger.info("Creating lookup tables for forcings and flow data")
        super(CamelsUSBurn, self)._create_lookup_table(xrds_without_burn)
        logger.info("Finished creating lookup tables for forcings and flow data")
        # Calculate indices corresponding to lookup table for burn datasets
        burn_lookup_table = []
        self._xburn = {}
        self._xburn_dates = {}
        xburn_date_index = {}
        ind1d = self.frequencies.index("1D") # index of 1D frequency in the self.frequecies
        logger.info("Enriching lookup tables with burn data")
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar):
            # compute the tensor for burn fractions and store them in self._x_burn
            filtered_data = {
                feature: (xrds[feature].loc[{
                    "basin":basin, 
                    "month_year":slice(self._burn_dates[0][basin],self._burn_dates[1][basin])
                    }]) for feature in CamelsUSBurn.burn_properties
                }
            self._xburn_dates[basin] = filtered_data["frac_low"].coords["month_year"]
            self._xburn[basin] = {feature: torch.from_numpy(filtered_data[feature].to_numpy().astype(np.float32)) for feature in CamelsUSBurn.burn_properties}
            # self._xburn_dates[basin] = xrds["frac_low"].coords["month_year"]
            # for the self._xburn_dates build a reverse index lookup table used to infer burn month index
            # of each sample in our lookup table
            # Numpy array receives dates as native int64 so it needs to convereted back
            # into MonthYear
            xburn_date_index[basin] = {get_month_year(date.data):i for (i, date) in enumerate(self._xburn_dates[basin])}
            # For each item in the lookup table find the month/year corresponding to
            # In the original lookup table, the index is the index of last
            # element of the sample for the given frequency and basin
            # Use the date associated with the day to infer the last index for burn
            # properties in the same sample
        logger.info("Adding indexes for burn data corresponding to the end days in forcing date for each sample")
        
        for i in tqdm(range(self.num_samples), file=sys.stdout, disable=self._disable_pbar):
            (basin, frequencies) = self.lookup_table[i]
            sampleind1d = frequencies[ind1d] # index of sample at  1 day frequency
            # infer the index of burn samples from sampleind1d
            end_month_year = get_month_year(self._dates[basin]["1D"][sampleind1d])
            burn_date_index = xburn_date_index[basin][end_month_year]
            burn_lookup_table.append(burn_date_index)
        self._burn_lookup_table = burn_lookup_table
        logger.info("Finished enriching lookup tables with burn data")
  
    
        
    def _load_burn_data(self):
        aggregates = load_burn_data(self.cfg.data_dir / "basin_burn" / "mbts_burn")
        basin_attributes = self._load_attributes()
        aggregate_basin_set = set(aggregates.index)
        attributes_basin_set = set(self.basins)
        common_basins = aggregate_basin_set & attributes_basin_set
        def add_actual_burn_area(basin):
            
            if basin not in aggregates.index:
                return {}
            row = aggregates[basin]
            start_date = self.start_and_end_dates[basin]["start_dates"]
            end_date = self.start_and_end_dates[basin]["end_dates"]
            basin_area_m2 = basin_attributes.area_gages2[basin]*1000000
            for firedata in row:
                firedata["overlap_fraction"] = firedata["overlap_area_m2"]/basin_area_m2 
            return row 
        # Add actual burn area in basins     
        list(map(add_actual_burn_area, self.basins))
        ffgh = aggregates
        return aggregates


    @cache
    def __getitem__(self, index):
        item = super(CamelsUSBurn, self).__getitem__(index)
        # Add burn item to the dictionary
        burn_end_index = self._burn_lookup_table[index]
        basin, indices = self.lookup_table[index]
        burn_start_index = burn_end_index + 1 - self._burn_sequence_length
        xburn_tensors = torch.stack([self._xburn[basin][feature][burn_start_index:burn_end_index+1] for feature in CamelsUSBurn.burn_properties], dim=-1)
        item["xburn"] = xburn_tensors
        item["end_month"] = torch.tensor(int(self._xburn_dates[basin][burn_end_index])%12)
        # calculate the number of days in each month
        if "x_d_1D" not in item:
            seq_len = item["x_d"]
        else:
            seq_len = item["x_d_1D"]
        k = next(iter(seq_len))
        seq_len = seq_len[k].shape[0]
        daily_end_index = indices[self.frequencies.index("1D")]
        assert item["end_month"] + 1 == get_month_year(self._dates[basin]["1D"][daily_end_index]).month
        
        all_dates = self._dates[basin]["1D"][daily_end_index + 1 - seq_len:daily_end_index + 1]
        month_days = [] # Number of days in each month
        for (i, date) in enumerate(all_dates):
            if i==0 or get_month_year(date).month != get_month_year(all_dates[i-1]).month:
                month_days.append(1)
            else:
                month_days[-1] += 1
        month_days = torch.tensor(month_days)
        # Zero pad to the left to match with burn_sequence_length
        item["month_days"] = torch_pad(month_days, ( self._burn_sequence_length - month_days.shape[0], 0))
        self.index_calls += 1
        return item # added xburn to the item
        # collect all tensors and slice them


    # @staticmethod
    # def collate_fn(samples: List[Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]]):
    #     # Number of month daya may not match across samples hence zero pad the samples to the left to make same shape
    #     feature_keys = [feature for feature in samples[0] if feature.endswith("_features")]
    #     non_feature_keys = [feature for feature in samples[0] if not feature.endswith("_features")]
    #     collated_item = CamelsUSBurn.__bases__[0].collate_fn([{key:sample[key] for key in non_feature_keys} for sample in samples])
    #     for key in feature_keys:
    #         # features are just kept one copy
    #         collated_item[key] = samples[0][key]
    #     return collated_item
    

def load_burn_data(burn_data_dir):
    df = pd.DataFrame()
    for file in glob.glob(str(burn_data_dir / "*.csv")):
        df_new = pd.read_csv(file, dtype={"GAGEID":str})
        df = pd.concat([df, df_new], axis=0, join='outer', ignore_index=False)
    aggregate = df.groupby("GAGEID")[["overlap_area_m2", "Ig_Date_str", "frac_low", "frac_moderate", "frac_unburned_low", "frac_inc_greenness", "frac_high"]].apply(lambda x: x.to_dict(orient="records"))
    return aggregate

    
    
    


    

        
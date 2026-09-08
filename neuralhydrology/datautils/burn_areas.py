import logging
import pickle
import sys
import os
from pathlib import Path
from typing import List, Dict
import ee

import numpy as np
import pandas as pd
from numba import njit
import geopandas as gpd
import json

def compute_burn_area(shape_file:Path, start_date, end_date):
    all_basins = gpd.read_file(shape_file)
    all_basins.set_index("GAGEID")
    
    ee_fc = ee.FeatureCollection(all_basins.__geo_interface__)
    ee_modis = ee.ImageCollection('MODIS/061/MOD09A1').filterDate(start_date, end_date)
    def extract_nbr(image):
        image = image.select(['sur_refl_b02', 'sur_refl_b07'])
        nbr_image = filtered_image.normalizedDifference(['sur_refl_b02', 'sur_refl_b07']).rename('NBR')
        feature_col = image.sampleRegions(
                    properties = ["GAGEID"],
                    collection=all_basins,
                    scale=1000,
                    geometries=False 
        )
        selectors = ["GAGEID", "NBR"]
        reducer = ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.first(), 
                            sharedInputs=True
                    ).group(groupField=0, groupName="group_id")
        result_dict = sampled_fc.reduceColumns(
                reducer=reducer, selectors=selectors
            )
        return feature_col
    
    nbr_collection = ee_modis.map(extract_nbr)
    return nbr_collection
        

    for year in years:
        year_dir = data_dir / "burn_area" / str(year)
        os.mkdirs(year_dir, exist_ok = True)
        os.open(year_dir / "burn_area.csv")


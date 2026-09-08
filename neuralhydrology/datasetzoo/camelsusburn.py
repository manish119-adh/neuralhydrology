from neuralhydrology.datasetzoo.camelsus import CamelsUS
class CamelsUSBurn(CamelsUS):

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsUSBurn, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)
        
    def _load_burn_data(self):
        pass

    def __getitem__(self, index):
        item = super(CamelsUSBurn, self).__getitem__(index)
        # Add burn item to the dictionary

    def __len__(self):
        return super(CamelsUSBurn, self).__len__(index)
        
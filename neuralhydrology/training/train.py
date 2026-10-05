from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config
from neuralhydrology.hyperparameters.hyperopt import optimize as hyperoptimize
from skopt.space import Integer, Categorical, Real

    


def start_training(cfg: Config):
    """Start model training.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    """
    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    if cfg.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")
    trainer.initialize_training()
    trainer.train_and_validate()





if __name__ == "__main__":
    from pathlib import Path
    import time
    from neuralhydrology.datasetzoo.camelsusburn import CamelsUSBurn
    from neuralhydrology.modelzoo.revnetmodified import ModifiedRevnet
    from neuralhydrology.evaluation.evaluate import start_evaluation
    import random
    from torch.utils.data.dataloader import DataLoader
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
        "train_burn_start_dates":{"01013500":"1986-12"},
        "train_burn_end_dates":{"01013500":"2010-01"},
        "month_emb_dim":4,
        "model":"revnet",
        "dataset":"camels_us_burn",
        "run_dir":"notebooks/burndata/runs"
        
        })
    cfg = Config(dict_config, dev_mode=True) # Update config
    trainer = BaseTrainer(cfg)
    trainer.initialize_training()
    dataset = trainer.loader.dataset
    batch_size, collate_fn, num_workers = trainer.loader.batch_size, trainer.loader.collate_fn, trainer.loader.num_workers
    del trainer.loader
    # Replace data loader with a persistent worker data loader
    trainer.loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True)
    
    trainer.train_and_validate()   
    start_evaluation(cfg=cfg, 
        run_dir=cfg.run_dir, 
        epoch=None, 
        period="test"
    )

from neuralhydrology.utils.config import Config
from skopt.utils import use_named_args
from skopt import gp_minimize
from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.earlystopper import EarlyStopper
from neuralhydrology.datasetzoo.splitdataset import basinwise_kfold_cross_validation
from torch.utils.data import DataLoader
from skopt.space import Integer, Categorical, Real
from functools import reduce
import random
import time
from tqdm import tqdm
import sys
import gc
import math
import logging

logger = logging.getLogger("Camels US")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from torch.cuda import memory_allocated as cuda_memory, memory_summary as cuda_memory_summary

class EarlyStopperWithModel(EarlyStopper):
    """
    Subclass of early stopper which does the same thing except check_early_stopping
    also takes an optional model parameter and if present also saves the model weights
    if this model is the best model so that the best model can be loaded as final model
    """

    def __init__(self, patience: int, min_delta: float):
        super(EarlyStopperWithModel, self).__init__(patience, min_delta)
        self.best_model = None

    

    def check_early_stopping(self, validation_loss: float, model=None) -> bool:
        min_validation_loss = self._min_validation_loss
        stop_early = super(EarlyStopperWithModel, self).check_early_stopping(validation_loss)
        if validation_loss < min_validation_loss - self._min_delta:
            # save the best model
            if self.best_model is not None:
                del self.best_model
                self.best_model = None
            if model is not None:
                self.best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return stop_early


class KFoldTrainer:
    """
     Trainer for K fold cross validation along with hyper parameter 
     optimization
    """
    def __init__(self, cfg: Config, parameter_space):
        self.cfg = cfg
        self.n_folds = cfg.as_dict()["n_folds"]
        self.n_initial_points = cfg.as_dict()["n_initial_points"]
        self.n_calls = cfg.as_dict()["n_calls"]
        self.dataset_stride = cfg.as_dict()["cross_validation_stride"]
        self.n_epochs = cfg.epochs
        self.trainer = BaseTrainer(cfg)
        self.trainer.initialize_training() # Initialize dataset and model for the trainer
        # Out of trainers main dataset, construct k fold cross validation dataset
        self.folds = basinwise_kfold_cross_validation(self.trainer.loader.dataset, "camesus_burn", self.n_folds, stride = self.dataset_stride)
        self.batch_size = self.trainer.loader.batch_size
        self.collate_fn = self.trainer.loader.collate_fn
        self.num_workers = self.trainer.loader.num_workers
        # Train and validation data loaders for each fold
        self.initial_model = {k: v.clone() for k, v in self.trainer.model.state_dict().items()}
        self.original_dataset = self.trainer.loader.dataset


    

    def get_fold_validation_loss(self, agg = lambda lst : sum(lst)/len(lst)):
        # train one model each for each fold in cross validation
        def get_validation_loss(validation_loader, fold, epoch):
            self.trainer.model.eval()
            pbar = tqdm(validation_loader, file=sys.stdout, disable=self.trainer._disable_pbar, total=len(validation_loader))
            pbar.set_description(f"validation fold {fold} epoch {epoch}")
            nan_count = 0
            n_processed = 0
            val_loss = 0
            for i, data in enumerate(pbar):
                batch_size = 0
                for key in data.keys():
                    if key.startswith('x_d'):
                        data[key] = {k: v.to(self.trainer.device) for k, v in data[key].items()}
                        batch_size = data[key][next(iter(data[key]))].shape[0] if not batch_size else batch_size
                    elif not key.startswith('date'):
                        data[key] = data[key].to(self.trainer.device)
                    
                data = self.trainer.model.pre_model_hook(data, is_train=True)
                predictions = self.trainer.model(data)
                if self.trainer.noise_sampler_y is not None:
                    for key in filter(lambda k: 'y' in k, data.keys()):
                        noise = self.trainer.noise_sampler_y.sample(data[key].shape)
                        # make sure we add near-zero noise to originally near-zero targets
                        data[key] += (data[key] + self._target_mean / self.trainer._target_std) * noise.to(self.trainer.device)
                loss, all_losses = self.trainer.loss_obj(predictions, data)
                loss = loss.detach()
                val_loss = (val_loss * n_processed + loss * batch_size)/(n_processed + batch_size)
                n_processed += batch_size
            return val_loss
  
        folds_loss = []
        valid_loader = None
        early_stopper = None
        for i in range(self.n_folds):
            del self.trainer.optimizer
            del self.trainer.loss_obj
            del self.trainer.loader
            del valid_loader
            gc.collect()
            # Restart the fresh model and reinitialize loss_obj and optimizer
            self.trainer.model.load_state_dict(self.initial_model)
            self.trainer.optimizer = self.trainer._get_optimizer()
            self.trainer.loss_obj = self.trainer._get_loss_obj().to(self.trainer.device)        
            self.trainer.model.train()
            self.trainer.loader = DataLoader(self.folds[i][0], 
                    batch_size=self.batch_size, 
                    collate_fn=self.collate_fn, 
                    num_workers=self.num_workers, 
                    persistent_workers=True)
            valid_loader = DataLoader(self.folds[i][1], 
                    batch_size=self.batch_size, 
                    collate_fn=self.collate_fn, 
                    num_workers=self.num_workers, 
                    persistent_workers=True)
            # Early stopping is always used. If not sset from configs, take default of 4
            patience = self.trainer._patience_early_stopping if self.trainer._patience_early_stopping is not None else 4
            del early_stopper 
            early_stopper = EarlyStopperWithModel(patience = patience, min_delta = 0.0001)
            for epoch in range(1, self.n_epochs + 1):
                if not self.trainer._dynamic_learning_rate:
                    if epoch in self.cfg.learning_rate.keys():
                        logger.log(logging.INFO, f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
                        for param_group in self.trainer.optimizer.param_groups:
                            param_group["lr"] = self.cfg.learning_rate[epoch]
                self.trainer._train_epoch(epoch)
                val_loss = get_validation_loss(valid_loader, i, epoch)  
                if early_stopper.check_early_stopping(val_loss, self.trainer.model):
                    break
                best_loss = val_loss if  math.isinf(early_stopper._min_validation_loss) else early_stopper._min_validation_loss              
            if early_stopper.best_model is not None:
                self.trainer.model.load_state_dict(early_stopper.best_model)      
            gc.collect()
            folds_loss.append(best_loss)
        return float(agg(folds_loss))
        


def optimize(run_function ,base_config:Config, parameter_space, dev_mode=True, random_state=None, verbose=True, n_initial_points=5, n_calls=25):
    """
    This takes an optimization function `run_function` which takes a single
    configuration parameter and returns a numerical value, base_config (base
    configuration) which consists of all configuration parameters not subject
    to hyperparameter optimization (if such parameters are present in
    base_config, they will be modified via updates), and parameter_space
    which is a list of named skopt dimensions (Real, Integer, Categorical)
    defining the configuration parameters subject to optimization. Uses
    Bayesian optimization (skopt.gp_minimize) to return the best config
    (one that minimizes the value returned by run_function).
    """
    sn = 0
    @use_named_args(parameter_space)
    def optimization_function(**kwargs):
        nonlocal sn
        sn += 1
        logger.log(logging.INFO, f" {sn:03d}: Hyper parameter optimization: Trying parameters {kwargs}")
        config = base_config.as_dict() | kwargs
        start = time.time_ns()
        val = run_function(Config(config, dev_mode=dev_mode))
        logger.info(f"Finished hyperparameter optimization in {(time.time_ns() - start)/1000000000:.3f}s")
        return val

    result = gp_minimize(
        func=optimization_function,
        dimensions=parameter_space,
        n_calls=n_calls,          # total evaluations (~ init_points + n_iter combined)
        n_initial_points=n_initial_points,  # random exploration steps before GP kicks in
        random_state=random_state,
        verbose=verbose
    )
    best_params = {dim.name: val for dim, val in zip(parameter_space, result.x)}
    return result.fun, Config(base_config.as_dict() | best_params, dev_mode=dev_mode)




if __name__ == "__main__":
    from pathlib import Path
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
        "cross_validation_stride":20, 
        "n_folds":5,
        "n_initial_points":5,
        "n_calls":100,
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
        "dataset":"camels_us_burn"
        
        })
    cfg = Config(dict_config, dev_mode=True) # Update config
    parameter_space =  [
        Integer(4, 50, name="daily_hidden_size", dtype=int),  
        Integer(4, 50, name="burn_hidden_size1", dtype=int), 
        Integer(2, 20, name="burn_hidden_size2", dtype=int),
        Integer(2, 25, name="burn_kernel_size1", dtype=int),
        Integer(2, 25, name="burn_kernel_size2", dtype=int),
        Integer(4, 100, name="hidden_size", dtype=int),
        Integer(1, 10, name="month_emb_dim", dtype=int)
    ]
    def optimization_fn(cfg):
        trainer = KFoldTrainer(cfg, parameter_space)
        return trainer.get_fold_validation_loss()
    best, best_config = optimize(optimization_fn, 
                cfg, 
                parameter_space, 
                n_initial_points=cfg.as_dict()["n_initial_points"], 
                n_calls=cfg.as_dict()["n_calls"]
            )
    dhh = 90
    run_dir = Path("runs/test_run_2408_165247")
    


    

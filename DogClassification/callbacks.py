import os
from typing import Any, Literal, get_args
import wandb
import numpy as np
import torch

from DogClassification.trainer import Trainer

_MODE = Literal["min", "max"]

class EarlyStopping:
    def __init__(
        self, 
        monitor: str = 'val_loss', 
        patience: int = 5, 
        delta: float = 0, 
        mode: _MODE = 'min',
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_scores = None

    def __call__(self, trainer: Trainer, **kwargs):
        if self.best_scores is None:
            self.best_scores = trainer.monitor_dict[self.monitor][-1]
        elif (self.mode == 'min' and trainer.monitor_dict[self.monitor][-1] < self.best_scores - self.delta) or \
             (self.mode == 'max' and trainer.monitor_dict[self.monitor][-1] > self.best_scores + self.delta):
            self.counter = 0
            self.best_scores = trainer.monitor_dict[self.monitor][-1]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True
                
class ModelCheckpoint:
    def __init__(
        self, 
        filepath: str, 
        monitor: str = 'val_loss', 
        mode: _MODE = 'min',
        save_top_k: int = 1
    ) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_scores = [None] * save_top_k
        self.best_score_paths = [None] * save_top_k
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def __call__(self, model: torch.nn.Module, trainer: Trainer, **kwargs):
        save_path = self.filepath.format(
            epoch=trainer.current_epoch, 
            val_loss=trainer.monitor_dict['val_loss'][-1],
            train_loss=trainer.monitor_dict['train_loss'][-1],
            val_acc=trainer.monitor_dict['val_acc'][-1],
            train_acc=trainer.monitor_dict['train_acc'][-1],
        )
        
        if any(score is None for score in self.best_scores):
            index_empty = self.best_scores.index(None)
            
            self.best_scores[index_empty] = trainer.monitor_dict[self.monitor][-1]
            self.best_score_paths[index_empty] = save_path
            
            trainer.save_checkpoint(model, save_path)
        elif (self.mode == 'min' and trainer.monitor_dict[self.monitor][-1] < max(self.best_scores)) or \
             (self.mode == 'max' and trainer.monitor_dict[self.monitor][-1] > min(self.best_scores)):
            index_replace = np.argmax(self.best_scores) if self.mode == 'min' else np.argmin(self.best_scores)
            old_path = self.best_score_paths[index_replace]
            
            self.best_scores[index_replace] = trainer.monitor_dict[self.monitor][-1]
            self.best_score_paths[index_replace] = save_path
            
            os.remove(old_path)
            trainer.save_checkpoint(model, save_path)
            
class Monitor:
    def __init__(self, model_name) -> None:
        wandb.init(project="CV-BTL-1", name=model_name)
        
    def __call__(self, trainer: Trainer, **kwargs) -> Any:
        wandb.log({
            k: v[-1]
            for k, v in trainer.monitor_dict.items()
        })
    
    def rename(self, model_name):
        wandb.finish()
        wandb.init(project="CV-BTL-1", name=model_name)
        
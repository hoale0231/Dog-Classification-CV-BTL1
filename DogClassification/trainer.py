from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from DogClassification.model import DogClassification


class Trainer:
    def __init__(
        self, 
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None,
        criterion: Module = nn.CrossEntropyLoss(), 
        epochs: int = 100,
        callbacks: List = [],
    ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.epochs = epochs
        self.should_stop = False
        self.current_epoch = -1
        self.monitor_dict = {
            "train_acc": [],
            "val_acc": [],
            "train_loss": [],
            "val_loss": [],
        }
        
    def fit(self, model: Module, train_dataloader: DataLoader, valid_dataloader: DataLoader):
        loaders = {'train': train_dataloader, 'val': valid_dataloader}
        
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            print(f'Using {torch.cuda.device_count()} GPUs')
            model.cuda()
        else:
            print('Using CPU')
        
        for self.current_epoch in range(self.current_epoch + 1, self.epochs):
            for mode in ['train', 'val']:
                print(f'[{mode}] Epoch {self.current_epoch+1}/{self.epochs}', end=' ')
                if mode == 'train':
                    model.train()
                if mode == 'val':
                    model.eval()
                
                epoch_loss = 0
                epoch_acc = 0
                samples = 0

                for inputs, targets in tqdm(loaders[mode], leave=False):
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    
                    self.optimizer.zero_grad()
                    output = model(inputs)
                    loss = self.criterion(output, targets)
                    
                    if mode == 'train':
                        loss.backward()
                        self.optimizer.step()
                    
                    if torch.cuda.is_available():
                        acc = accuracy_score(targets.data.cpu().numpy(), output.max(1)[1].cpu().numpy())
                    else:
                        acc = accuracy_score(targets.data, output.max(1)[1])

                    epoch_loss += loss.data.item() * inputs.shape[0]
                    epoch_acc += acc * inputs.shape[0]
                    samples += inputs.shape[0]
                    
                epoch_loss /= samples
                epoch_acc /= samples
                
                self.monitor_dict[f"{mode}_loss"].append(epoch_loss)
                self.monitor_dict[f"{mode}_acc"].append(epoch_acc)
                
                print(f'Loss: {epoch_loss:0.2f} Accuracy: {epoch_acc:0.2f}')
                
                if mode == 'val':
                    self.scheduler.step(epoch_loss)
                    
            for callback in self.callbacks:
                callback(trainer=self, model=model)
            
            if self.should_stop:
                break
                    
        return model
    
    def load_checkpoint(self, model: Module, checkpoint):
        try:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.monitor_dict = checkpoint['monitor_dict']
            self.current_epoch = checkpoint['epoch']
            if 'classes' in checkpoint:
                model.classes = checkpoint['classes']
        except Exception as e:
            return False
        return True
    
    def save_checkpoint(self, model: Module, save_path: str):
        if isinstance(model, DogClassification):
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': model.state_dict(),
                'classess': model.classes,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'monitor_dict': self.monitor_dict
                }, save_path)
        else:
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'monitor_dict': self.monitor_dict
                }, save_path)
          
    def test(self, model, test_dataloader, device = None):
        if device:
            model = model.to(device)
        elif torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        preds = []
        trues = []
        wrong_sample = []
                
        for i, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            if torch.cuda.is_available() and (device != 'cpu'):
                inputs = inputs.cuda()
                pred = model(inputs).data.cpu().numpy().copy()
            else:
                pred = model(inputs).data.numpy().copy()
                
            true = targets.numpy().copy()
            preds.append(pred)
            trues.append(true)
            
            for i, (t, p) in enumerate(zip(true, pred)):
                if t != p.argmax(0):
                    wrong_sample.append((inputs[i].cpu(), p, t))

        return np.concatenate(preds), np.concatenate(trues), wrong_sample


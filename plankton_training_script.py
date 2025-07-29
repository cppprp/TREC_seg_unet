#!/usr/bin/env python3
"""
Complete training script for plankton 3D segmentation with all optimizations.

Features:
- Plankton-specific FocalDiceLoss
- Multi-scale augmentation for size diversity
- Comprehensive biological evaluation metrics
- Weights & Biases integration
- Early stopping based on small plankton performance
- Mixed precision training
- Advanced learning rate scheduling
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import tifffile
import numpy as np
from tqdm import tqdm

# Import our plankton-optimized modules
from plankton_config import (
    PlanktonProjectConfig, 
    create_plankton_development_config, 
    create_plankton_production_config
)
from plankton_focal_loss import PlanktonFocalDiceLoss, dice_score_plankton
from plankton_metrics import PlanktonMetrics, evaluate_plankton_batch, get_key_plankton_metrics
from plankton_augmentations import PlanktonMixUp, PlanktonTestTimeAugmentation
from plankton_optimized_learning_tools import create_datasets, OptimizedDataset
import tools as tf

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class PlanktonTrainer:
    """
    Complete trainer class for plankton segmentation with all optimizations.
    """
    
    def __init__(self, config: PlanktonProjectConfig):
        self.config = config
        self.device = config.device
        self.logger = self._setup_logging()
        self.wandb_run = None
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = GradScaler() if config.training.use_mixed_precision else None
        
        # Evaluation
        self.metrics_calculator = PlanktonMetrics(
            voxel_size_um=config.data.voxel_size_um,
            min_object_size_um=config.data.min_plankton_size_um
        )
        
        # Augmentation
        self.mixup = PlanktonMixUp(
            alpha=config.augmentation.mixup_alpha,
            prob=config.augmentation.mixup_prob
        ) if config.augmentation.use_mixup else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            metric_name=config.training.early_stopping_metric
        ) if config.training.use_early_stopping else None
        
        # Training history
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': [],
            'learning_rates': [], 'epochs': []
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.config.paths['logs'] / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_model(self):
        """Initialize model architecture"""
        from torch_em.model import UNet3d
        
        self.model = UNet3d(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            initial_features=self.config.model.initial_features,
            final_activation=self.config.model.final_activation
        ).to(self.device)
        
        self.logger.info(f"Model created with {self._count_parameters()} parameters")
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        if self.config.training.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.training.learning_rate
            )
        elif self.config.training.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.training.learning_rate,
                weight_decay=1e-5
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
        
        # Scheduler
        if self.config.training.use_scheduler:
            if self.config.training.scheduler_type == "cosine_warm_restarts":
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer, 
                    T_0=self.config.training.warmup_epochs,
                    T_mult=2
                )
            elif self.config.training.scheduler_type == "reduce_on_plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',  # We want to maximize our metric
                    patience=self.config.training.scheduler_patience,
                    factor=0.5,
                    verbose=True
                )
        
        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        self.logger.info(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
    
    def setup_loss_function(self):
        """Setup plankton-optimized loss function"""
        self.loss_fn = self.config.get_loss_function().to(self.device)
        self.logger.info(f"Loss function: {type(self.loss_fn).__name__}")
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare training data"""
        self.logger.info("Loading training data...")
        
        # Load data from ML patches directory
        ml_patches_dir = self.config.paths['ml_patches']
        if not ml_patches_dir.exists():
            raise FileNotFoundError(f"ML patches directory not found: {ml_patches_dir}")
        
        images = []
        labels = []
        
        # Load data from subfolders
        subfolders = [f for f in ml_patches_dir.iterdir() if f.is_dir()]
        
        for folder in subfolders:
            self.logger.info(f"Processing folder: {folder}")
            
            # Find label files
            label_files = list(folder.glob("*.labels*"))
            
            for label_file in label_files:
                # Find corresponding image file
                image_file = label_file.with_suffix('').with_suffix('.tif')
                if not image_file.exists():
                    # Try without .labels in the name
                    image_name = label_file.name.replace('.labels', '')
                    image_file = folder / image_name
                
                if image_file.exists():
                    try:
                        # Load and preprocess
                        image = tf.normalise(tifffile.imread(str(image_file)))
                        label = tifffile.imread(str(label_file))
                        
                        images.append(image)
                        labels.append(label)
                        
                        self.logger.info(f"Loaded: {image_file.name} with shape {image.shape}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load {image_file}: {e}")
                else:
                    self.logger.warning(f"Image file not found for label: {label_file}")
        
        if not images:
            raise ValueError("No valid image-label pairs found")
        
        self.logger.info(f"Loaded {len(images)} image-label pairs")
        
        # Create datasets with plankton-specific settings
        train_dataset, val_dataset = create_datasets(images, labels, self.config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with all optimizations"""
        self.model.train()
        
        total_loss = 0
        total_metrics = {}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Apply MixUp if enabled
            if self.mixup and np.random.random() < self.config.augmentation.mixup_prob:
                images, labels = self.mixup(images, labels)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler:
                with autocast():
                    predictions = self.model(images)
                    if hasattr(self.loss_fn, 'forward') and len(self.loss_fn.forward.__code__.co_varnames) > 2:
                        loss, loss_components = self.loss_fn(predictions, labels)
                    else:
                        loss = self.loss_fn(predictions, labels)
                        loss_components = {'total_loss': loss.item()}
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard precision
                predictions = self.model(images)
                if hasattr(self.loss_fn, 'forward') and len(self.loss_fn.forward.__code__.co_varnames) > 2:
                    loss, loss_components = self.loss_fn(predictions, labels)
                else:
                    loss = self.loss_fn(predictions, labels)
                    loss_components = {'total_loss': loss.item()}
                
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = dice_score_plankton(predictions, labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                if 'dice' not in total_metrics:
                    total_metrics['dice'] = 0
                total_metrics['dice'] += dice.item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # Log to wandb
            if self.wandb_run and batch_idx % self.config.wandb.log_frequency == 0:
                log_dict = {
                    'train/batch_loss': loss.item(),
                    'train/batch_dice': dice.item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch
                }
                log_dict.update({f'train/{k}': v for k, v in loss_components.items()})
                self.wandb_run.log(log_dict)
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate with comprehensive plankton metrics"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        predictions = self.model(images)
                        if hasattr(self.loss_fn, 'forward') and len(self.loss_fn.forward.__code__.co_varnames) > 2:
                            loss, _ = self.loss_fn(predictions, labels)
                        else:
                            loss = self.loss_fn(predictions, labels)
                else:
                    predictions = self.model(images)
                    if hasattr(self.loss_fn, 'forward') and len(self.loss_fn.forward.__code__.co_varnames) > 2:
                        loss, _ = self.loss_fn(predictions, labels)
                    else:
                        loss = self.loss_fn(predictions, labels)
                
                total_loss += loss.item()
                
                # Collect predictions and targets for comprehensive evaluation
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                
                # Update progress bar
                dice = dice_score_plankton(predictions, labels)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}'
                })
        
        # Comprehensive plankton evaluation
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        comprehensive_metrics = evaluate_plankton_batch(
            all_predictions, all_targets, 
            voxel_size_um=self.config.data.voxel_size_um
        )
        
        # Add average loss
        comprehensive_metrics['loss'] = total_loss / len(val_loader)
        
        return comprehensive_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        if not self.config.training.save_best_only:
            checkpoint_path = self.config.paths['checkpoints'] / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.paths['best_model'] / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
            # Also save just the model state dict for easier loading
            model_path = self.config.paths['best_model'] / 'best_model_state.pth'
            torch.save(self.model.state_dict(), model_path)
            
            self.logger.info(f"New best model saved at epoch {epoch}")
            
            # Log model to wandb
            if self.wandb_run and self.config.wandb.save_model:
                wandb.save(str(best_path))
    
    def train(self):
        """Main training loop with all optimizations"""
        # Setup everything
        self.config.create_directories()
        self.config.print_config()
        self.config.save_config()
        
        # Initialize wandb if enabled
        if self.config.wandb.use_wandb and WANDB_AVAILABLE:
            self.wandb_run = self.config.setup_wandb()
        
        # Setup model, optimizer, loss
        self.setup_model()
        self.setup_optimizer_and_scheduler()
        self.setup_loss_function()
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Training variables
        best_metric_value = 0.0
        start_time = time.time()
        
        self.logger.info("Starting training...")
        
        try:
            for epoch in range(self.config.training.n_epochs):
                epoch_start_time = time.time()
                
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validate
                if epoch % self.config.training.validate_every == 0:
                    val_metrics = self.validate_epoch(val_loader, epoch)
                else:
                    val_metrics = {'loss': 0.0}  # Skip validation
                
                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        # Use the early stopping metric for scheduling
                        metric_for_scheduler = val_metrics.get(
                            self.config.training.early_stopping_metric, 
                            val_metrics.get('dice_overall', 0)
                        )
                        self.scheduler.step(metric_for_scheduler)
                    else:
                        self.scheduler.step()
                
                # Log comprehensive results
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start_time
                
                self.logger.info(f"Epoch {epoch+1}/{self.config.training.n_epochs}")
                self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics.get('dice', 0):.4f}")
                
                if epoch % self.config.training.validate_every == 0:
                    # Get key plankton metrics for logging
                    key_val_metrics = get_key_plankton_metrics(val_metrics)
                    
                    self.logger.info("  Validation Metrics:")
                    for metric, value in key_val_metrics.items():
                        self.logger.info(f"    {metric}: {value:.4f}")
                
                self.logger.info(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
                
                # Store training history
                self.training_history['train_loss'].append(train_metrics['loss'])
                self.training_history['val_loss'].append(val_metrics.get('loss', 0))
                self.training_history['train_metrics'].append(train_metrics)
                self.training_history['val_metrics'].append(val_metrics)
                self.training_history['learning_rates'].append(current_lr)
                self.training_history['epochs'].append(epoch)
                
                # Weights & Biases logging
                if self.wandb_run:
                    log_dict = {
                        'epoch': epoch,
                        'train/epoch_loss': train_metrics['loss'],
                        'train/epoch_dice': train_metrics.get('dice', 0),
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time
                    }
                    
                    # Add validation metrics
                    if epoch % self.config.training.validate_every == 0:
                        for metric, value in val_metrics.items():
                            log_dict[f'val/{metric}'] = value
                    
                    self.wandb_run.log(log_dict)
                
                # Check for best model
                current_metric_value = val_metrics.get(
                    self.config.training.early_stopping_metric,
                    val_metrics.get('dice_overall', 0)
                )
                
                is_best = current_metric_value > best_metric_value
                if is_best:
                    best_metric_value = current_metric_value
                
                # Save checkpoint
                if ((epoch + 1) % self.config.training.save_checkpoint_every == 0 or 
                    is_best or epoch == self.config.training.n_epochs - 1):
                    self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Early stopping check
                if self.early_stopping and epoch % self.config.training.validate_every == 0:
                    if self.early_stopping(current_metric_value, self.model):
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Save final model
            final_path = self.config.paths['final_model'] / 'final_model.pth'
            torch.save(self.model.state_dict(), final_path)
            
            # Save training history
            import json
            history_path = self.config.paths['logs'] / 'training_history.json'
            
            # Convert numpy types to native Python types for JSON serialization
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [
                        {k: float(v) if isinstance(v, (np.number, torch.Tensor)) else v 
                         for k, v in item.items()} if isinstance(item, dict) else float(item) if isinstance(item, (np.number, torch.Tensor)) else item
                        for item in value
                    ]
                else:
                    serializable_history[key] = value
            
            with open(history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            # Final logging
            total_time = time.time() - start_time
            self.logger.info(f"Training completed! Total time: {total_time/3600:.2f} hours")
            self.logger.info(f"Best {self.config.training.early_stopping_metric}: {best_metric_value:.4f}")
            
            if self.wandb_run:
                self.wandb_run.log({
                    'training_completed': True,
                    'total_training_time_hours': total_time / 3600,
                    f'best_{self.config.training.early_stopping_metric}': best_metric_value
                })
                self.wandb_run.finish()
    
    def _count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

class EarlyStopping:
    """Early stopping implementation"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, metric_name: str = "dice_small_meso"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
    
    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = model.state_dict().copy()
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                # Restore best model
                model.load_state_dict(self.best_model_state)
                return True
        else:
            self.best_score = current_score
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        
        return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Train plankton 3D segmentation model")
    parser.add_argument('--config', choices=['dev', 'prod'], default='dev',
                        help='Configuration preset to use')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory for the project')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment')
    parser.add_argument('--voxel_size_um', type=float, default=1.0,
                        help='Voxel size in micrometers')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--aggressive_aug', action='store_true',
                        help='Use aggressive augmentation')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config == 'dev':
        config = create_plankton_development_config(
            args.experiment_name or "plankton_dev"
        )
    else:
        base_dir = args.base_dir or os.getcwd()
        config = create_plankton_production_config(
            base_dir, 
            args.experiment_name or "plankton_prod",
            args.voxel_size_um
        )
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.n_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.aggressive_aug:
        config.augmentation.aggressive_augmentation = True
    if args.no_wandb:
        config.wandb.use_wandb = False
    
    # Create trainer and run
    trainer = PlanktonTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

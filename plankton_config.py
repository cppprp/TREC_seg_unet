import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

@dataclass
class PlanktonDataConfig:
    """Plankton-specific data configuration"""
    patch_shape: Tuple[int, int, int] = (128, 128, 128)
    halo: Tuple[int, int, int] = (64, 64, 64)
    block_shape: Tuple[int, int, int] = (128, 128, 128)
    train_val_split: float = 0.8
    bit_depth: int = 16
    label_bit_depth: int = 8
    normalization_percentile: float = 95.0
    min_object_size: int = 10000
    min_foreground_ratio: float = 0.01
    
    # PLANKTON-SPECIFIC SETTINGS
    voxel_size_um: float = 1.0          # Micrometers per voxel
    min_plankton_size_um: float = 15    # Minimum plankton size to consider
    max_plankton_size_um: float = 300   # Maximum expected plankton size
    
    # Size class definitions for evaluation
    size_classes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'nano': (0.2, 2),         # Nanoplankton  
        'micro': (2, 20),         # Microplankton
        'small_meso': (20, 50),   # Small mesoplankton (critical!)
        'large_meso': (50, 200),  # Large mesoplankton
        'macro': (200, 2000)      # Macroplankton
    })
    
    # Data loading settings
    exclusion_criteria: List[str] = field(default_factory=lambda: [
        'pre', '._', '.DS', 'overlaps', 'debris', 'bubble'
    ])

@dataclass
class PlanktonModelConfig:
    """Model configuration optimized for plankton"""
    in_channels: int = 1
    out_channels: int = 2
    initial_features: int = 32
    final_activation: str = "Sigmoid"
    
    # Architecture improvements for plankton
    use_attention_gates: bool = False      # Enable when ready for advanced features
    use_deep_supervision: bool = False     # Multi-scale loss
    dropout_rate: float = 0.1

@dataclass
class PlanktonTrainingConfig:
    """Training configuration optimized for plankton challenges"""
    batch_size: int = 1
    learning_rate: float = 1e-4
    n_epochs: int = 300
    optimizer: str = "AdamW"
    
    # Loss function settings
    loss_type: str = "focal_dice"          # "focal_dice", "topk_focal", "standard_dice"
    focal_alpha: float = 0.75              # Higher for extreme class imbalance
    focal_gamma: float = 3.0               # Higher to focus on hard examples
    dice_weight: float = 0.6               # Balance between focal and dice
    small_plankton_boost: float = 0.5      # Extra weight for small plankton
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine_warm_restarts"  # "cosine_warm_restarts", "reduce_on_plateau"
    warmup_epochs: int = 10
    scheduler_patience: int = 15
    
    # Training stability
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: str = "dice_small_meso"  # Focus on small plankton!
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    save_checkpoint_every: int = 25
    save_best_only: bool = False
    
    # Validation frequency
    validate_every: int = 1

@dataclass
class PlanktonAugmentationConfig:
    """Augmentation configuration for plankton diversity"""
    use_augmentation: bool = True
    aggressive_augmentation: bool = False
    
    # Geometric augmentation (critical for size diversity)
    scales: Tuple[float, float] = (0.4, 2.5)  # 6x range for 20-200μm diversity
    degrees: float = 20.0
    translation: float = 0.1
    elastic_deformation: bool = True
    elastic_displacement: float = 7.5
    
    # Intensity augmentation
    gamma_range: Tuple[float, float] = (-0.3, 0.3)
    noise_std: Tuple[float, float] = (0, 0.15)
    blur_std: Tuple[float, float] = (0, 1.5)
    
    # Microscopy-specific
    motion_artifacts: bool = True
    spike_noise: bool = True
    bias_field: bool = True
    
    # Advanced augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.3
    
    # Test-time augmentation
    use_tta: bool = True
    tta_scales: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])

@dataclass 
class WandBConfig:
    """Weights & Biases configuration"""
    use_wandb: bool = True
    project_name: str = "plankton-segmentation"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["3d-segmentation", "plankton", "unet"])
    notes: Optional[str] = None
    
    # Logging settings
    log_frequency: int = 10              # Log every N batches
    log_images: bool = True              # Log sample predictions
    log_histograms: bool = False         # Log parameter histograms (expensive)
    log_gradients: bool = False          # Log gradient norms (for debugging)
    
    # What to save
    save_code: bool = True
    save_config: bool = True
    save_model: bool = True

class PlanktonProjectConfig:
    """
    Main configuration class for plankton 3D segmentation project.
    
    Extends the base configuration with plankton-specific optimizations.
    """
    
    def __init__(self, base_dir: str = None, experiment_name: str = None):
        # Set base directory
        if base_dir is None:
            base_dir = os.getcwd()
        self.base_dir = Path(base_dir)
        
        # Experiment naming
        self.experiment_name = experiment_name or "plankton_experiment"
        
        # Initialize all configurations
        self.data = PlanktonDataConfig()
        self.model = PlanktonModelConfig()
        self.training = PlanktonTrainingConfig()
        self.augmentation = PlanktonAugmentationConfig()
        self.wandb = WandBConfig()
        
        # Setup paths
        self._setup_paths()
        
        # Device configuration
        self.device = self._get_device()
        
        # Set up experiment-specific wandb settings
        if self.wandb.run_name is None:
            self.wandb.run_name = self.experiment_name
    
    def _setup_paths(self):
        """Setup all project paths"""
        # Main directories
        self.paths = {
            'base': self.base_dir,
            'data': self.base_dir / 'data',
            'models': self.base_dir / 'models' / self.experiment_name,
            'output': self.base_dir / 'output' / self.experiment_name,
            'logs': self.base_dir / 'logs' / self.experiment_name,
            'temp': self.base_dir / 'temp',
        }
        
        # Data subdirectories
        data_dir = self.paths['data']
        self.paths.update({
            'raw_data': data_dir / 'raw',
            'processed_data': data_dir / 'processed',
            'ml_patches': data_dir / 'ml_patches',
            'roi_labels': data_dir / 'roi_labels',
            'roi_xray': data_dir / 'roi_xray',
        })
        
        # Output subdirectories
        output_dir = self.paths['output']
        self.paths.update({
            'predictions': output_dir / 'predictions',
            'foreground': output_dir / 'foreground',
            'boundaries': output_dir / 'boundaries',
            'cleaned': output_dir / 'cleaned',
            'visualizations': output_dir / 'visualizations',
            'evaluation': output_dir / 'evaluation',
        })
        
        # Model subdirectories
        model_dir = self.paths['models']
        self.paths.update({
            'checkpoints': model_dir / 'checkpoints',
            'best_model': model_dir / 'best',
            'final_model': model_dir / 'final',
        })
    
    def create_directories(self):
        """Create all necessary directories"""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        print(f"Created directories for experiment: {self.experiment_name}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            print("GPU is available")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Running with Mac silicon chip")
            return torch.device("mps")
        else:
            print("GPU is NOT available. Training will be slow!")
            return torch.device("cpu")
    
    def get_augmentation_transforms(self, mode: str = "training"):
        """Get plankton-optimized augmentation transforms"""
        from plankton_augmentations import create_plankton_augmentation_pipeline
        
        if not self.augmentation.use_augmentation and mode == "training":
            return None
        
        return create_plankton_augmentation_pipeline(
            mode=mode,
            voxel_size_um=self.data.voxel_size_um,
            aggressive=self.augmentation.aggressive_augmentation
        )
    
    def get_loss_function(self):
        """Get plankton-optimized loss function"""
        from plankton_focal_loss import create_plankton_loss
        
        return create_plankton_loss(
            loss_type=self.training.loss_type,
            alpha=self.training.focal_alpha,
            gamma=self.training.focal_gamma,
            dice_weight=self.training.dice_weight,
            small_plankton_boost=self.training.small_plankton_boost,
            voxel_size_um=self.data.voxel_size_um
        )
    
    def print_config(self):
        """Print current configuration"""
        print("=== Plankton Segmentation Configuration ===")
        print(f"Experiment: {self.experiment_name}")
        print(f"Base directory: {self.base_dir}")
        print(f"Device: {self.device}")
        print(f"Voxel size: {self.data.voxel_size_um}μm")
        print(f"Plankton size range: {self.data.min_plankton_size_um}-{self.data.max_plankton_size_um}μm")
        print(f"Patch shape: {self.data.patch_shape}")
        print(f"Batch size: {self.training.batch_size}")
        print(f"Learning rate: {self.training.learning_rate}")
        print(f"Loss type: {self.training.loss_type}")
        print(f"Focal parameters: α={self.training.focal_alpha}, γ={self.training.focal_gamma}")
        print(f"Augmentation: {'Aggressive' if self.augmentation.aggressive_augmentation else 'Standard'}")
        print(f"Scale range: {self.augmentation.scales}")
        print(f"Early stopping metric: {self.training.early_stopping_metric}")
        print(f"Weights & Biases: {self.wandb.use_wandb}")
        print("=" * 45)
    
    def save_config(self, filepath: str = None):
        """Save configuration to file"""
        import json
        
        if filepath is None:
            filepath = self.paths['logs'] / 'config.json'
        
        config_dict = {
            'experiment_name': self.experiment_name,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__,
            'wandb': self.wandb.__dict__,
            'paths': {k: str(v) for k, v in self.paths.items()},
            'device': str(self.device)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to: {filepath}")
    
    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        if not self.wandb.use_wandb:
            return None
        
        try:
            import wandb
            
            # Initialize wandb
            run = wandb.init(
                project=self.wandb.project_name,
                entity=self.wandb.entity,
                name=self.wandb.run_name,
                tags=self.wandb.tags,
                notes=self.wandb.notes,
                config={
                    'experiment_name': self.experiment_name,
                    'data': self.data.__dict__,
                    'model': self.model.__dict__,
                    'training': self.training.__dict__,
                    'augmentation': self.augmentation.__dict__,
                    'device': str(self.device)
                }
            )
            
            # Save code if requested
            if self.wandb.save_code:
                wandb.run.log_code(".")
            
            print(f"Weights & Biases initialized: {wandb.run.url}")
            return run
            
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            self.wandb.use_wandb = False
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.wandb.use_wandb = False
            return None

# Factory functions for different scenarios
def create_plankton_development_config(experiment_name: str = "dev_test") -> PlanktonProjectConfig:
    """Create configuration for development/testing"""
    config = PlanktonProjectConfig(experiment_name=experiment_name)
    
    # Development settings (faster iteration)
    config.training.n_epochs = 20
    config.training.batch_size = 1
    config.data.patch_shape = (64, 64, 64)  # Smaller for faster training
    config.augmentation.aggressive_augmentation = False
    config.wandb.use_wandb = True  # Still track development runs
    config.wandb.project_name = "plankton-dev"
    
    return config

def create_plankton_production_config(
    base_dir: str, 
    experiment_name: str = "production_run",
    voxel_size_um: float = 1.0
) -> PlanktonProjectConfig:
    """Create configuration for production training"""
    config = PlanktonProjectConfig(base_dir, experiment_name)
    
    # Production settings
    config.training.n_epochs = 300
    config.training.batch_size = 2 if torch.cuda.is_available() else 1
    config.data.patch_shape = (128, 128, 128)
    config.data.voxel_size_um = voxel_size_um
    config.augmentation.aggressive_augmentation = True
    config.wandb.use_wandb = True
    
    return config

def create_plankton_inference_config(
    model_path: str, 
    output_dir: str,
    voxel_size_um: float = 1.0
) -> PlanktonProjectConfig:
    """Create configuration for inference only"""
    config = PlanktonProjectConfig(experiment_name="inference")
    
    # Update paths
    config.paths['models'] = Path(model_path).parent
    config.paths['output'] = Path(output_dir)
    config.data.voxel_size_um = voxel_size_um
    
    # Disable training-specific features
    config.wandb.use_wandb = False
    config.augmentation.use_augmentation = False
    
    return config

# Example usage
if __name__ == "__main__":
    # Test configuration creation
    config = create_plankton_development_config("test_plankton_config")
    config.print_config()
    
    # Test path creation
    config.create_directories()
    
    # Test config saving
    config.save_config()
    
    print("\nConfiguration test completed successfully!")

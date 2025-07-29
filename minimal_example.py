#!/usr/bin/env python3
"""
Minimal example to get started with real plankton data.

This script shows exactly what you need to do to start training
on your real data with minimal setup.
"""

import os
from pathlib import Path

def check_requirements():
    """Check that all required files and packages are available"""
    print("🔍 Checking requirements...")
    
    required_files = [
        'plankton_config.py',
        'plankton_losses.py', 
        'plankton_metrics.py',
        'plankton_augmentations.py',
        'train_plankton.py',
        'plankton_inference.py',
        'plankton_postprocessing.py',
        'plankton_instance_segmentation.py',
        'optimized_learning_tools.py',
        'tools.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all core pipeline files are in your current directory.")
        return False
    
    # Check packages
    try:
        import torch
        import torchio
        import numpy as np
        import tifffile
        import sklearn
        print("✅ All required files and packages found!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Install with: pip install torch torchvision torch-em torchio scikit-image tifffile")
        return False

def setup_directory_structure(base_dir):
    """Create the expected directory structure"""
    print(f"📁 Setting up directory structure in: {base_dir}")
    
    dirs_to_create = [
        'data/ml_patches',
        'models', 
        'output',
        'logs'
    ]
    
    base_path = Path(base_dir)
    for dir_name in dirs_to_create:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_path}")
    
    print("✅ Directory structure ready!")
    return base_path

def check_data_organization(data_dir):
    """Check if data is properly organized"""
    print(f"🔬 Checking data organization in: {data_dir}")
    
    ml_patches_dir = Path(data_dir) / 'ml_patches'
    
    if not ml_patches_dir.exists():
        print(f"❌ Directory not found: {ml_patches_dir}")
        print("Please create this directory and add your training data")
        return False
    
    # Look for subfolders with data
    subfolders = [d for d in ml_patches_dir.iterdir() if d.is_dir()]
    
    if not subfolders:
        print(f"❌ No subfolders found in {ml_patches_dir}")
        print("Expected structure:")
        print("  data/ml_patches/")
        print("  ├── experiment_1/")
        print("  │   ├── volume_001.tif")
        print("  │   ├── volume_001.labels.tif")
        print("  │   └── ...")
        print("  └── experiment_2/")
        return False
    
    total_pairs = 0
    for folder in subfolders:
        print(f"   Checking folder: {folder.name}")
        
        # Find image files
        image_files = list(folder.glob("*.tif"))
        label_files = list(folder.glob("*.labels.tif"))
        
        # Check for matching pairs
        pairs = 0
        for img_file in image_files:
            if 'labels' not in img_file.name:  # Skip label files
                expected_label = img_file.with_suffix('.labels.tif')
                if expected_label.exists():
                    pairs += 1
                else:
                    # Try alternative naming
                    alt_label_name = img_file.name.replace('.tif', '.labels.tif')
                    alt_label = folder / alt_label_name
                    if alt_label.exists():
                        pairs += 1
        
        print(f"     Found {pairs} image-label pairs")
        total_pairs += pairs
    
    if total_pairs == 0:
        print("❌ No valid image-label pairs found!")
        print("Make sure your files follow the naming convention:")
        print("  image.tif → image.labels.tif")
        return False
    
    print(f"✅ Found {total_pairs} total image-label pairs across {len(subfolders)} folders")
    return True

def create_quick_config(base_dir, voxel_size_um=1.0, experiment_name="my_plankton"):
    """Create a basic configuration for getting started"""
    print("⚙️ Creating configuration...")
    
    config_code = f'''
# Quick start configuration for your plankton data
from plankton_config import create_plankton_development_config

# Create development config (faster for initial testing)
config = create_plankton_development_config("{experiment_name}")

# Set your project directory
config.paths['base'] = Path("{base_dir}")
config.paths['ml_patches'] = Path("{base_dir}") / "data" / "ml_patches"

# IMPORTANT: Set your actual voxel size!
config.data.voxel_size_um = {voxel_size_um}

# Adjust these based on your plankton size range
config.data.min_plankton_size_um = 20   # Smallest plankton to detect
config.data.max_plankton_size_um = 200  # Largest plankton in your data

# Training settings (start conservative)
config.training.batch_size = 1          # Start with 1
config.training.n_epochs = 50           # Shorter for initial test
config.training.learning_rate = 1e-4    # Good starting point

# Enable experiment tracking (optional)
config.wandb.use_wandb = True           # Set to False if you don't want tracking

print("Configuration created successfully!")
config.print_config()
'''
    
    # Save config to file
    config_file = Path("my_plankton_config.py")
    with open(config_file, 'w') as f:
        f.write(config_code)
    
    print(f"✅ Configuration saved to: {config_file}")
    print("You can modify this file to adjust settings for your data")

def create_training_script(base_dir, experiment_name="my_plankton"):
    """Create a simple training script"""
    print("🎯 Creating training script...")
    
    training_code = f'''#!/usr/bin/env python3
"""
Simple training script for your plankton data.
"""

from pathlib import Path
from plankton_config import create_plankton_development_config
from train_plankton import PlanktonTrainer

def main():
    print("🦠 Starting plankton training...")
    
    # Create configuration
    config = create_plankton_development_config("{experiment_name}")
    
    # Set paths
    config.paths['base'] = Path("{base_dir}")
    config.paths['ml_patches'] = Path("{base_dir}") / "data" / "ml_patches"
    
    # IMPORTANT: Update this with your actual voxel size!
    config.data.voxel_size_um = 1.0  # CHANGE THIS!
    
    # Training settings
    config.training.n_epochs = 50
    config.training.batch_size = 1
    
    # Print configuration
    config.print_config()
    
    # Create trainer and start training
    trainer = PlanktonTrainer(config)
    trainer.train()
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()
'''
    
    script_file = Path("train_my_plankton.py")
    with open(script_file, 'w') as f:
        f.write(training_code)
    
    print(f"✅ Training script saved to: {script_file}")
    print("Run with: python train_my_plankton.py")

def create_inference_script(base_dir):
    """Create a simple inference script"""
    print("🔍 Creating inference script...")
    
    inference_code = f'''#!/usr/bin/env python3
"""
Simple inference script for your trained plankton model.
"""

import sys
from pathlib import Path
from plankton_inference import PlanktonInferenceEngine
from plankton_config import create_plankton_inference_config

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer_my_plankton.py <model_path> <input_volume>")
        print("Example: python infer_my_plankton.py models/best_model.pth new_data.tif")
        return
    
    model_path = sys.argv[1]
    input_volume = sys.argv[2]
    
    print(f"🦠 Running plankton inference...")
    print(f"Model: {{model_path}}")
    print(f"Input: {{input_volume}}")
    
    # Create inference configuration
    config = create_plankton_inference_config(
        model_path, 
        "{base_dir}/results",
        voxel_size_um=1.0  # CHANGE THIS TO YOUR VOXEL SIZE!
    )
    
    # Create inference engine
    engine = PlanktonInferenceEngine(model_path, config)
    
    # Load input volume
    import tifffile
    volume = tifffile.imread(input_volume)
    
    # Run prediction with instance creation
    segmentation, metadata = engine.predict_volume(
        volume,
        use_tta=True,  # Use test-time augmentation for best results
        create_instances=True  # Create individual plankton instances
    )
    
    # Save results
    output_prefix = Path(input_volume).stem
    engine.save_results(segmentation, metadata, output_prefix)
    
    # Print summary
    if metadata['processing_type'] == 'instance_segmentation':
        print(f"🎉 Found {{metadata['object_analysis']['num_instances']}} individual plankton!")
        print(f"Size distribution: {{metadata['object_analysis']['size_distribution']}}")
    else:
        print(f"🎉 Found {{metadata['object_analysis']['num_objects']}} plankton objects!")
    
    print(f"Results saved to: {base_dir}/results/")

if __name__ == "__main__":
    main()
'''
    
    script_file = Path("infer_my_plankton.py")
    with open(script_file, 'w') as f:
        f.write(inference_code)
    
    print(f"✅ Inference script saved to: {script_file}")
    print("Usage: python infer_my_plankton.py models/best_model.pth new_volume.tif")

def main():
    """Main setup function"""
    print("🦠" + "="*50 + "🦠")
    print("🦠  PLANKTON PIPELINE - QUICK START SETUP  🦠") 
    print("🦠" + "="*50 + "🦠\n")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the requirements above before continuing.")
        return
    
    # Get user input
    print("\n" + "="*50)
    print("📋 SETUP CONFIGURATION")
    print("="*50)
    
    base_dir = input("Enter your project directory (e.g., /path/to/plankton_project): ").strip()
    if not base_dir:
        base_dir = "./plankton_project"
        print(f"Using default: {base_dir}")
    
    voxel_size = input("Enter your voxel size in micrometers (e.g., 0.5): ").strip()
    if not voxel_size:
        voxel_size = "1.0"
        print("Using default: 1.0 μm")
    
    try:
        voxel_size_um = float(voxel_size)
    except ValueError:
        print("Invalid voxel size, using 1.0 μm")
        voxel_size_um = 1.0
    
    experiment_name = input("Enter experiment name (e.g., my_plankton_v1): ").strip()
    if not experiment_name:
        experiment_name = "my_plankton"
        print(f"Using default: {experiment_name}")
    
    print(f"\n🚀 Setting up plankton pipeline...")
    print(f"   Project dir: {base_dir}")
    print(f"   Voxel size: {voxel_size_um} μm")
    print(f"   Experiment: {experiment_name}")
    
    # Setup directories
    base_path = setup_directory_structure(base_dir)
    
    # Check data (don't fail if not ready yet)
    print(f"\n📊 Data organization check:")
    data_ready = check_data_organization(base_path / 'data')
    
    # Create configuration and scripts
    create_quick_config(base_dir, voxel_size_um, experiment_name)
    create_training_script(base_dir, experiment_name)
    create_inference_script(base_dir)
    
    print(f"\n🎉 Setup complete!")
    print(f"="*50)
    
    if data_ready:
        print("✅ Your data is properly organized. You're ready to train!")
        print(f"\nNext steps:")
        print(f"1. Run: python train_my_plankton.py")
        print(f"2. Monitor training progress")
        print(f"3. Use trained model: python infer_my_plankton.py models/best_model.pth new_data.tif")
    else:
        print("⚠️  Data not ready yet. Please:")
        print(f"1. Add your data to: {base_path}/data/ml_patches/")
        print(f"2. Organize as: experiment_folder/image.tif + image.labels.tif")
        print(f"3. Then run: python train_my_plankton.py")
    
    print(f"\n📚 Files created:")
    print(f"   my_plankton_config.py    - Configuration settings")
    print(f"   train_my_plankton.py     - Training script")
    print(f"   infer_my_plankton.py     - Inference script")
    
    print(f"\n🔧 To customize:")
    print(f"   - Edit voxel_size_um in the scripts")
    print(f"   - Adjust plankton size ranges") 
    print(f"   - Modify training parameters")
    
    print(f"\nHappy plankton hunting! 🦠🔬")

if __name__ == "__main__":
    main()

"""
Quick Start Training Script - Research Optimized
Simplified version for easy execution
"""

import os
import sys
from pathlib import Path
from multiprocessing import freeze_support

def main():
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("="*80)
    print("Brain Tumor Detection - Research-Optimized Training")
    print("Target: 96%+ Accuracy (Research Benchmark)")
    print("="*80)
    print()

    # Step 1: Check if we can import required modules
    print("[Step 1/5] Checking dependencies...")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        import cv2
        import numpy as np
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Ultralytics: {ultralytics.__version__}")
        print(f"✓ OpenCV: {cv2.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Using device: {device}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease run: pip install -r SCRIPTS/setup/requirements_research_optimized.txt")
        sys.exit(1)
    print()

    # Step 2: Check dataset
    print("[Step 2/5] Checking dataset...")
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "DATA"
    train_images = data_dir / "images" / "train"
    val_images = data_dir / "images" / "val"

    if not train_images.exists():
        print(f"✗ Training images not found at: {train_images}")
        sys.exit(1)
    if not val_images.exists():
        print(f"✗ Validation images not found at: {val_images}")
        sys.exit(1)

    train_count = len(list(train_images.glob("*.jpg")))
    val_count = len(list(val_images.glob("*.jpg")))
    print(f"✓ Training images: {train_count}")
    print(f"✓ Validation images: {val_count}")
    print()

    # Step 3: Create/check data.yaml
    print("[Step 3/5] Setting up dataset configuration...")
    data_yaml = project_root / "SCRIPTS" / "data" / "dataset.yaml"
    if not data_yaml.exists():
        print(f"✗ Dataset YAML not found at: {data_yaml}")
        sys.exit(1)
    print(f"✓ Using dataset config: {data_yaml}")
    print()

    # Step 4: Initialize model
    print("[Step 4/5] Loading YOLOv11 model...")
    try:
        model = YOLO('yolo11n.pt')
        param_count = sum(p.numel() for p in model.model.parameters())
        print(f"✓ YOLOv11n loaded successfully")
        print(f"✓ Parameters: {param_count:,}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)
    print()

    # Step 5: Enhanced training configuration
    print("[Step 5/5] Starting training with ENHANCED configuration...")
    print()
    print("Enhanced Hyperparameters (v2):")
    print("-" * 50)
    print(f"  Optimizer:       AdamW (better for small datasets)")
    print(f"  Learning Rate:   0.0005 (5x higher for faster learning)")
    print(f"  Epochs:          150 (with early stopping)")
    print(f"  Batch Size:      16")
    print(f"  Patience:        30 (more tolerance)")
    print(f"  Box Loss:        10.0 (prioritize localization)")
    print(f"  Conf Threshold:  0.20 (catch more tumors)")
    print(f"  IoU Threshold:   0.60 (better for small objects)")
    print(f"  Cache:           RAM (faster training)")
    print(f"  Device:          {device}")
    print("-" * 50)
    print()
    print("NEW IMPROVEMENTS:")
    print("  ✓ AdamW optimizer (adaptive learning)")
    print("  ✓ Higher learning rate (faster convergence)")
    print("  ✓ Increased box loss weight (better localization)")
    print("  ✓ Lower confidence threshold (more sensitive)")
    print("  ✓ Copy-paste augmentation (for small tumors)")
    print("  ✓ RAM caching (2-3x faster)")
    print("-" * 50)
    print()

    # Auto-start training (no prompt)
    print("\n" + "="*80)
    print("TRAINING STARTED - This will take 2-4 hours")
    print("="*80)
    print()

    # Output directory
    output_dir = project_root / "OUTPUT"
    output_dir.mkdir(exist_ok=True)

    # Research-optimized configuration WITH IMPROVEMENTS
    config = {
        # Basic settings
        'data': str(data_yaml),
        'epochs': 150,       # INCREASED: More epochs for better convergence
        'batch': 16,
        'imgsz': 640,        # Keep at 640 (standard for YOLO)
        'device': device,
        'workers': 0,        # Set to 0 for Windows multiprocessing
        
        # Optimizer (IMPROVED)
        'optimizer': 'AdamW',   # CHANGED: AdamW often better than SGD for small datasets
        'lr0': 0.0005,          # INCREASED: 5x higher learning rate for faster convergence
        'lrf': 0.01,
        'momentum': 0.937,      # Default momentum for AdamW
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,   # INCREASED: More warmup for stability
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Data augmentation (ENHANCED FOR SMALL TUMORS)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,        # INCREASED: More rotation for better generalization
        'translate': 0.1,
        'scale': 0.7,           # INCREASED: More scale variation
        'flipud': 0.0,          # No vertical flip (anatomical consistency)
        'fliplr': 0.5,          # Horizontal flip OK (brain symmetry)
        'mosaic': 1.0,          # Critical for small tumor detection
        'mixup': 0.1,           # INCREASED: Small amount of mixup for regularization
        'copy_paste': 0.1,      # NEW: Copy-paste augmentation for small objects
        'erasing': 0.4,
        
        # Loss weights (OPTIMIZED FOR SMALL OBJECTS)
        'box': 10.0,            # INCREASED: Higher weight for bounding box accuracy
        'cls': 0.3,             # DECREASED: Less weight on classification (binary is easier)
        'dfl': 2.0,             # INCREASED: More focus on distribution focal loss
        
        # Training settings
        'patience': 30,         # INCREASED: More patience before early stopping
        'save': True,
        'save_period': 10,
        'plots': True,
        'val': True,
        'amp': True,            # Mixed precision
        'cache': 'ram',         # CHANGED: Cache in RAM for faster training
        'close_mosaic': 15,     # Disable mosaic in last 15 epochs for fine-tuning
        
        # Inference (OPTIMIZED)
        'iou': 0.6,             # DECREASED: Lower IoU threshold for small tumors
        'conf': 0.20,           # DECREASED: Lower confidence threshold to catch more tumors
        'max_det': 300,         # Allow more detections per image
        
        # Output
        'project': str(output_dir),
        'name': 'yolov11_v2_enhanced',
        'exist_ok': True,
        'verbose': True,
    }

    try:
        # Train model
        results = model.train(**config)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        
        # Validate model
        print("\nRunning validation...")
        metrics = model.val()
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS:")
        print("="*80)
        print(f"mAP@0.5:      {metrics.box.map50:.4f}  (Target: 0.993)")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}  (Target: 0.801)")
        print(f"Precision:    {metrics.box.mp:.4f}")
        print(f"Recall:       {metrics.box.mr:.4f}")
        
        f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-7)
        print(f"F1-Score:     {f1_score:.4f}  (Target: 0.990)")
        print("="*80)
        
        # Compare with research
        print("\nComparison with Research Benchmarks:")
        print("-" * 60)
        research_map50 = 0.993
        achieved_map50 = float(metrics.box.map50)
        gap = achieved_map50 - research_map50
        
        if achieved_map50 >= 0.95:
            status = "✓ EXCELLENT (Clinical-grade)"
        elif achieved_map50 >= 0.85:
            status = "✓ GOOD (Approaching clinical-grade)"
        elif achieved_map50 >= 0.70:
            status = "○ IMPROVED (Better than baseline)"
        else:
            status = "• BASELINE LEVEL"
        
        print(f"Achieved mAP@0.5: {achieved_map50:.4f}")
        print(f"Research Target:  {research_map50:.4f}")
        print(f"Gap:              {gap:+.4f}")
        print(f"Status:           {status}")
        print("-" * 60)
        
        print(f"\n✓ Model saved to: {output_dir / 'yolov11_research_optimized' / 'weights' / 'best.pt'}")
        print(f"✓ Results saved to: {output_dir / 'yolov11_research_optimized'}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("Training pipeline completed successfully!")
    print("="*80)

if __name__ == '__main__':
    freeze_support()
    main()

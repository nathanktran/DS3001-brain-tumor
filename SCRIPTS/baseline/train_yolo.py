"""
YOLO11n Training Script for Brain Tumor Detection
This script implements fine-tuning of a pretrained YOLO11n model on brain tumor MRI scans.
Includes data augmentation, transfer learning, and comprehensive training monitoring.
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime
import json
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorYOLOTrainer:
    """YOLO trainer for brain tumor detection with enhanced configurations"""
    
    def __init__(self, data_config_path: str, output_dir: str = "../OUTPUT"):
        """Initialize trainer with configuration paths"""
        self.data_config_path = data_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create training run directory
        self.run_name = f"brain_tumor_yolo11n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.training_results = None
        
    def setup_model(self, pretrained_weights: str = "yolo11n.pt"):
        """Initialize YOLO11n model with pretrained weights"""
        logger.info(f"Loading YOLO11n model with weights: {pretrained_weights}")
        
        try:
            self.model = YOLO(pretrained_weights)
            logger.info("Model loaded successfully")
            
            # Print model architecture summary
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def configure_training(self) -> dict:
        """Configure training parameters with data augmentation"""
        training_config = {
            # Basic training parameters
            'epochs': 100,
            'batch': 16,  # Adjust based on GPU memory
            'imgsz': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Data augmentation parameters
            'hsv_h': 0.015,  # HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,    # HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,    # HSV-Value augmentation (fraction)
            'degrees': 10.0,  # Image rotation (+/- deg)
            'translate': 0.1,  # Image translation (+/- fraction)
            'scale': 0.5,     # Image scale (+/- gain)
            'shear': 0.0,     # Image shear (+/- deg)
            'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
            'flipud': 0.0,    # Image flip up-down (probability)
            'fliplr': 0.5,    # Image flip left-right (probability)
            'mosaic': 1.0,    # Mosaic augmentation (probability)
            'mixup': 0.1,     # MixUp augmentation (probability)
            'copy_paste': 0.0,  # Copy-paste augmentation (probability)
            
            # Validation and saving
            'val': True,
            'save': True,
            'save_period': 10,  # Save checkpoint every N epochs
            'cache': False,  # Cache images for faster training (True/False/"ram")
            'rect': False,  # Rectangular training
            'cos_lr': False,  # Cosine learning rate scheduler
            'close_mosaic': 10,  # Disable mosaic augmentation for final epochs
            'resume': False,  # Resume training from last checkpoint
            'amp': True,  # Automatic Mixed Precision training
            'fraction': 1.0,  # Dataset fraction to train on (0-1)
            'profile': False,  # Profile ONNX and TensorRT speeds during training
            'freeze': None,  # Freeze layers: backbone=10, first3=0,1,2
            'multi_scale': False,  # Multi-scale training
            
            # Loss function weights
            'box': 7.5,      # Box loss gain
            'cls': 0.5,      # Class loss gain
            'dfl': 1.5,      # Distribution focal loss gain
            
            # Early stopping
            'patience': 50,   # Epochs to wait for no observable improvement
            
            # Validation settings
            'iou': 0.7,      # Intersection over Union (IoU) threshold for NMS
            'conf': 0.25,    # Object confidence threshold for detection
            
            # Project settings
            'project': str(self.output_dir),
            'name': self.run_name,
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
        }
        
        return training_config
    
    def train_model(self):
        """Train the YOLO model with configured parameters"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Get training configuration
        config = self.configure_training()
        
        logger.info("Starting YOLO11n training for brain tumor detection")
        logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
        
        # Save training configuration
        config_file = self.run_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        try:
            # Start training
            self.training_results = self.model.train(
                data=self.data_config_path,
                **config
            )
            
            logger.info("Training completed successfully!")
            
            # Save training results
            self._save_training_summary()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        return self.training_results
    
    def _save_training_summary(self):
        """Save comprehensive training summary"""
        if self.training_results is None:
            logger.warning("No training results to save")
            return
        
        # Get the latest run directory from ultralytics
        runs_dir = self.output_dir / self.run_name
        if runs_dir.exists():
            # Find the actual training run directory
            train_dirs = list(runs_dir.glob("train*"))
            if train_dirs:
                latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                
                # Copy important files to our run directory
                important_files = [
                    'weights/best.pt',
                    'weights/last.pt',
                    'results.csv',
                    'confusion_matrix.png',
                    'results.png',
                    'train_batch0.jpg',
                    'train_batch1.jpg',
                    'train_batch2.jpg',
                    'val_batch0_labels.jpg',
                    'val_batch0_pred.jpg'
                ]
                
                for file_name in important_files:
                    src_file = latest_train_dir / file_name
                    if src_file.exists():
                        dst_file = self.run_dir / file_name.replace('/', '_')
                        # Create directory if needed
                        dst_file.parent.mkdir(exist_ok=True)
                        try:
                            import shutil
                            shutil.copy2(src_file, dst_file)
                            logger.info(f"Copied {file_name} to results directory")
                        except Exception as e:
                            logger.warning(f"Could not copy {file_name}: {e}")
        
        # Create training summary
        summary = {
            'training_completed': True,
            'run_name': self.run_name,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'YOLO11n',
            'dataset': str(self.data_config_path),
            'device_used': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            summary['gpu_info'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(0)
            }
        
        # Save summary
        summary_file = self.run_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_file}")
    
    def analyze_training_results(self):
        """Analyze and visualize training results"""
        runs_dir = self.output_dir / self.run_name
        
        # Look for results.csv file
        results_files = list(runs_dir.rglob("results.csv"))
        
        if not results_files:
            logger.warning("No results.csv file found")
            return
        
        results_file = results_files[0]
        
        try:
            # Load training results
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # Remove any whitespace from column names
            
            # Create comprehensive training analysis
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'YOLO11n Training Results - {self.run_name}', fontsize=16, fontweight='bold')
            
            # 1. Loss curves
            if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
                axes[0, 0].plot(df.index, df['train/box_loss'], label='Train Box Loss', alpha=0.8)
                axes[0, 0].plot(df.index, df['val/box_loss'], label='Val Box Loss', alpha=0.8)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Box Loss')
                axes[0, 0].set_title('Bounding Box Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Class loss
            if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
                axes[0, 1].plot(df.index, df['train/cls_loss'], label='Train Class Loss', alpha=0.8)
                axes[0, 1].plot(df.index, df['val/cls_loss'], label='Val Class Loss', alpha=0.8)
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Class Loss')
                axes[0, 1].set_title('Classification Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. mAP metrics
            if 'metrics/mAP50' in df.columns and 'metrics/mAP50-95' in df.columns:
                axes[0, 2].plot(df.index, df['metrics/mAP50'], label='mAP@0.5', alpha=0.8)
                axes[0, 2].plot(df.index, df['metrics/mAP50-95'], label='mAP@0.5:0.95', alpha=0.8)
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('mAP')
                axes[0, 2].set_title('Mean Average Precision')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Precision and Recall
            if 'metrics/precision' in df.columns and 'metrics/recall' in df.columns:
                axes[1, 0].plot(df.index, df['metrics/precision'], label='Precision', alpha=0.8)
                axes[1, 0].plot(df.index, df['metrics/recall'], label='Recall', alpha=0.8)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_title('Precision and Recall')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df.index, df['lr/pg0'], label='Learning Rate', alpha=0.8, color='red')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Training summary table
            final_metrics = df.iloc[-1]
            summary_data = []
            
            metrics_to_show = [
                ('mAP@0.5', 'metrics/mAP50'),
                ('mAP@0.5:0.95', 'metrics/mAP50-95'),
                ('Precision', 'metrics/precision'),
                ('Recall', 'metrics/recall'),
                ('Box Loss (Val)', 'val/box_loss'),
                ('Class Loss (Val)', 'val/cls_loss')
            ]
            
            for metric_name, column_name in metrics_to_show:
                if column_name in final_metrics:
                    value = final_metrics[column_name]
                    summary_data.append([metric_name, f"{value:.4f}"])
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Final Value'])
                axes[1, 2].axis('tight')
                axes[1, 2].axis('off')
                table = axes[1, 2].table(cellText=summary_df.values, 
                                        colLabels=summary_df.columns,
                                        cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                axes[1, 2].set_title('Final Training Metrics')
            
            plt.tight_layout()
            
            # Save the analysis plot
            analysis_plot_path = self.run_dir / "training_analysis.png"
            plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training analysis plot saved to {analysis_plot_path}")
            
            plt.show()
            
            # Print summary
            print("\\n" + "="*60)
            print("TRAINING RESULTS SUMMARY")
            print("="*60)
            print(f"Model: YOLO11n")
            print(f"Total Epochs: {len(df)}")
            print(f"Best mAP@0.5: {df['metrics/mAP50'].max():.4f}" if 'metrics/mAP50' in df.columns else "mAP not available")
            print(f"Best mAP@0.5:0.95: {df['metrics/mAP50-95'].max():.4f}" if 'metrics/mAP50-95' in df.columns else "mAP50-95 not available")
            print(f"Final Precision: {final_metrics.get('metrics/precision', 'N/A'):.4f}" if isinstance(final_metrics.get('metrics/precision', 'N/A'), (int, float)) else "Precision not available")
            print(f"Final Recall: {final_metrics.get('metrics/recall', 'N/A'):.4f}" if isinstance(final_metrics.get('metrics/recall', 'N/A'), (int, float)) else "Recall not available")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Failed to analyze training results: {e}")
    
    def get_best_model_path(self) -> Path:
        """Get path to the best trained model"""
        runs_dir = self.output_dir / self.run_name
        best_weights = list(runs_dir.rglob("best.pt"))
        
        if best_weights:
            return best_weights[0]
        else:
            logger.warning("Best model weights not found")
            return None

def main():
    """Main training function"""
    # Configuration
    data_config = "dataset.yaml"
    output_dir = "../OUTPUT"
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("CUDA not available, using CPU (training will be much slower)")
    
    # Initialize trainer
    trainer = BrainTumorYOLOTrainer(data_config, output_dir)
    
    # Setup model
    trainer.setup_model("yolo11n.pt")
    
    # Train model
    results = trainer.train_model()
    
    # Analyze results
    trainer.analyze_training_results()
    
    # Get best model path
    best_model_path = trainer.get_best_model_path()
    if best_model_path:
        logger.info(f"Best trained model saved at: {best_model_path}")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()

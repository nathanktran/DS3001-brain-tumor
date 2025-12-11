"""
Comprehensive Training Script with Research-Based Improvements
Implements findings from Priyadharshini et al. (2025) to achieve 96%+ accuracy

Key Improvements:
1. Enhanced preprocessing (log transform, CLAHE, edge detection)
2. Research-optimized hyperparameters
3. 5-fold cross-validation
4. Advanced augmentation strategies
5. Proper evaluation metrics aligned with research
"""

import os
import sys
import torch
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.model_selection import KFold

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import custom modules
try:
    from yolov11_improved_preprocessing import EnhancedMRIPreprocessor
    from yolov11_research_config import YOLOv11ResearchConfig
except ImportError:
    print("Warning: Custom modules not found. Using basic configuration.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchOptimizedTrainer:
    """
    Complete training pipeline implementing research best practices
    Target: 96%+ accuracy as demonstrated in Priyadharshini et al. (2025)
    """
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "../OUTPUT",
                 use_preprocessing: bool = True,
                 use_5fold_cv: bool = False):
        """
        Initialize trainer with research-optimized settings
        
        Args:
            data_path: Path to dataset
            output_dir: Output directory for results
            use_preprocessing: Apply enhanced preprocessing pipeline
            use_5fold_cv: Use 5-fold cross-validation
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_preprocessing = use_preprocessing
        self.use_5fold_cv = use_5fold_cv
        
        # Initialize preprocessor if needed
        if self.use_preprocessing:
            self.preprocessor = EnhancedMRIPreprocessor(target_size=(640, 640))
        
        # Initialize config
        self.config_manager = YOLOv11ResearchConfig(
            dataset_path=str(self.data_path),
            output_dir=str(self.output_dir)
        )
        
        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"research_run_{self.run_timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Research-Optimized Trainer")
        logger.info(f"Output directory: {self.run_dir}")
        logger.info(f"Enhanced preprocessing: {self.use_preprocessing}")
        logger.info(f"5-fold CV: {self.use_5fold_cv}")
    
    def preprocess_dataset(self) -> Path:
        """
        Apply enhanced preprocessing to entire dataset
        
        Pipeline:
        1. Log transformation (intensity normalization)
        2. CLAHE (contrast enhancement)
        3. Edge-based ROI extraction
        4. Intensity normalization
        
        Returns:
            Path to preprocessed dataset
        """
        if not self.use_preprocessing:
            logger.info("Preprocessing disabled, using original images")
            return self.data_path
        
        logger.info("="*80)
        logger.info("STAGE 1: Enhanced Preprocessing (Research-Based)")
        logger.info("="*80)
        
        preprocessed_dir = self.output_dir / "preprocessed_data"
        preprocessed_dir.mkdir(exist_ok=True)
        
        # Process training and validation sets
        for split in ['train', 'val']:
            input_dir = self.data_path / 'images' / split
            output_dir = preprocessed_dir / 'images' / split
            
            if not input_dir.exists():
                logger.warning(f"Directory not found: {input_dir}")
                continue
            
            logger.info(f"Processing {split} set...")
            count = self.preprocessor.process_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                save_intermediate=False
            )
            logger.info(f"Processed {count} images in {split} set")
            
            # Copy labels
            labels_input = self.data_path / 'labels' / split
            labels_output = preprocessed_dir / 'labels' / split
            labels_output.mkdir(parents=True, exist_ok=True)
            
            if labels_input.exists():
                import shutil
                for label_file in labels_input.glob('*.txt'):
                    shutil.copy2(label_file, labels_output / label_file.name)
        
        # Create dataset YAML for preprocessed data
        yaml_path = self.config_manager.create_enhanced_dataset_yaml(
            train_path='images/train',
            val_path='images/val',
            class_names=['negative', 'positive']
        )
        
        # Copy YAML to preprocessed directory
        import shutil
        yaml_dest = preprocessed_dir / 'data.yaml'
        shutil.copy2(yaml_path, yaml_dest)
        
        logger.info(f"Preprocessing complete! Data saved to: {preprocessed_dir}")
        return preprocessed_dir
    
    def train_single_model(self, 
                          model_name: str = 'yolo11n.pt',
                          data_yaml: str = None,
                          custom_config: Dict = None) -> Tuple[YOLO, Dict]:
        """
        Train single YOLOv11 model with research-optimized config
        
        Args:
            model_name: Pretrained model to use
            data_yaml: Path to dataset YAML
            custom_config: Optional custom configuration overrides
            
        Returns:
            Tuple of (trained model, results dictionary)
        """
        logger.info("="*80)
        logger.info(f"STAGE 2: Training {model_name}")
        logger.info("="*80)
        
        # Load model
        model = YOLO(model_name)
        logger.info(f"Loaded {model_name}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        # Get research-optimized config
        config = self.config_manager.get_research_based_hyperparameters()
        
        # Apply custom overrides if provided
        if custom_config:
            config.update(custom_config)
        
        # Update project directory
        config['project'] = str(self.run_dir)
        config['name'] = f"{model_name.replace('.pt', '')}_research"
        
        # Set data path
        if data_yaml is None:
            data_yaml = str(self.data_path / 'data.yaml')
        
        # Save training config
        config_path = self.run_dir / f"{config['name']}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Training configuration saved to: {config_path}")
        
        # Print key hyperparameters
        logger.info("\nKey Hyperparameters (Research-Optimized):")
        logger.info(f"  Epochs: {config['epochs']}")
        logger.info(f"  Batch size: {config['batch']}")
        logger.info(f"  Learning rate: {config['lr0']}")
        logger.info(f"  Optimizer: {config['optimizer']}")
        logger.info(f"  Patience: {config['patience']}")
        logger.info(f"  Augmentation: HSV={config['hsv_h']}/{config['hsv_s']}/{config['hsv_v']}")
        
        # Start training
        logger.info(f"\nStarting training on: {data_yaml}")
        logger.info(f"Target: 96%+ accuracy (research benchmark)")
        logger.info("-"*80)
        
        try:
            results = model.train(data=data_yaml, **config)
            logger.info("✓ Training completed successfully!")
            
            # Validate model
            logger.info("\nValidating model...")
            metrics = model.val()
            
            # Extract and log metrics
            results_dict = {
                'model': model_name,
                'timestamp': self.run_timestamp,
                'config': config,
                'metrics': {
                    'mAP50': float(metrics.box.map50),
                    'mAP50_95': float(metrics.box.map),
                    'precision': float(metrics.box.mp),
                    'recall': float(metrics.box.mr),
                    'f1_score': 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / 
                               (float(metrics.box.mp) + float(metrics.box.mr) + 1e-7)
                }
            }
            
            logger.info("\n" + "="*80)
            logger.info("RESULTS:")
            logger.info("="*80)
            logger.info(f"mAP@0.5:     {results_dict['metrics']['mAP50']:.4f} (Target: 0.993)")
            logger.info(f"mAP@0.5:0.95: {results_dict['metrics']['mAP50_95']:.4f} (Target: 0.801)")
            logger.info(f"Precision:   {results_dict['metrics']['precision']:.4f}")
            logger.info(f"Recall:      {results_dict['metrics']['recall']:.4f}")
            logger.info(f"F1-Score:    {results_dict['metrics']['f1_score']:.4f} (Target: 0.990)")
            logger.info("="*80)
            
            # Compare with research benchmarks
            self._compare_with_research(results_dict['metrics'])
            
            # Save results
            results_path = self.run_dir / f"{config['name']}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"\nResults saved to: {results_path}")
            
            return model, results_dict
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def train_with_5fold_cv(self, model_name: str = 'yolo11n.pt') -> List[Dict]:
        """
        Train with 5-fold cross-validation for robust evaluation
        Research shows CV improves reliability by ~2-3%
        
        Args:
            model_name: Model to train
            
        Returns:
            List of results from each fold
        """
        logger.info("="*80)
        logger.info("5-FOLD CROSS-VALIDATION TRAINING")
        logger.info("="*80)
        
        all_results = []
        
        for fold in range(5):
            logger.info(f"\n{'='*80}")
            logger.info(f"FOLD {fold + 1}/5")
            logger.info(f"{'='*80}")
            
            # Get fold-specific config
            fold_config = self.config_manager.get_5fold_cv_config(fold=fold)
            fold_config['name'] = f"yolov11_fold{fold}"
            
            # Train model
            model, results = self.train_single_model(
                model_name=model_name,
                custom_config=fold_config
            )
            
            results['fold'] = fold
            all_results.append(results)
        
        # Aggregate results
        logger.info("\n" + "="*80)
        logger.info("5-FOLD CROSS-VALIDATION SUMMARY")
        logger.info("="*80)
        
        metrics_df = pd.DataFrame([r['metrics'] for r in all_results])
        
        logger.info("\nAverage Performance Across Folds:")
        logger.info(f"  mAP@0.5:      {metrics_df['mAP50'].mean():.4f} ± {metrics_df['mAP50'].std():.4f}")
        logger.info(f"  mAP@0.5:0.95: {metrics_df['mAP50_95'].mean():.4f} ± {metrics_df['mAP50_95'].std():.4f}")
        logger.info(f"  Precision:    {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
        logger.info(f"  Recall:       {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
        logger.info(f"  F1-Score:     {metrics_df['f1_score'].mean():.4f} ± {metrics_df['f1_score'].std():.4f}")
        
        # Save aggregated results
        cv_summary = {
            'num_folds': 5,
            'timestamp': self.run_timestamp,
            'mean_metrics': metrics_df.mean().to_dict(),
            'std_metrics': metrics_df.std().to_dict(),
            'all_folds': all_results
        }
        
        summary_path = self.run_dir / '5fold_cv_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        logger.info(f"\nCV summary saved to: {summary_path}")
        
        return all_results
    
    def compare_yolo_variants(self) -> Dict[str, Dict]:
        """
        Compare YOLOv9, YOLOv10, and YOLOv11 as done in research
        
        Returns:
            Dictionary with results from all three variants
        """
        logger.info("="*80)
        logger.info("COMPARING YOLO VARIANTS (Research Benchmark)")
        logger.info("="*80)
        
        configs = self.config_manager.get_comparison_configs()
        results = {}
        
        for variant, config in configs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {variant.upper()}")
            logger.info(f"{'='*80}")
            
            model, variant_results = self.train_single_model(
                model_name=config['model'],
                custom_config=config
            )
            
            results[variant] = variant_results
        
        # Create comparison table
        self._create_comparison_table(results)
        
        return results
    
    def _compare_with_research(self, metrics: Dict[str, float]):
        """Compare achieved metrics with research benchmarks"""
        
        research_benchmarks = {
            'mAP50': 0.993,
            'mAP50_95': 0.801,
            'precision': 0.992,
            'recall': 0.984,
            'f1_score': 0.990,
            'accuracy': 0.9622  # 96.22% on BraTS2020
        }
        
        logger.info("\nComparison with Research Benchmarks:")
        logger.info("-" * 60)
        logger.info(f"{'Metric':<15} {'Achieved':<12} {'Research':<12} {'Gap':<12}")
        logger.info("-" * 60)
        
        for metric, research_val in research_benchmarks.items():
            if metric in metrics:
                achieved = metrics[metric]
                gap = achieved - research_val
                gap_str = f"{gap:+.4f}"
                logger.info(f"{metric:<15} {achieved:<12.4f} {research_val:<12.4f} {gap_str:<12}")
        
        logger.info("-" * 60)
    
    def _create_comparison_table(self, results: Dict[str, Dict]):
        """Create comparison table for YOLO variants"""
        
        # Extract metrics
        comparison_data = []
        for variant, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': variant.upper(),
                'mAP@0.5': metrics['mAP50'],
                'mAP@0.5:0.95': metrics['mAP50_95'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = self.run_dir / 'yolo_variants_comparison.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info("\n" + "="*80)
        logger.info("YOLO VARIANTS COMPARISON")
        logger.info("="*80)
        logger.info(df.to_string(index=False))
        logger.info(f"\nComparison table saved to: {csv_path}")
        
        # Create visualization
        self._plot_comparison(df)
    
    def _plot_comparison(self, df: pd.DataFrame):
        """Create comparison visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('YOLO Variants Comparison - Research Benchmarks', 
                    fontsize=16, fontweight='bold')
        
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            bars = ax.bar(df['Model'], df[metric], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel(metric, fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            # Add research benchmark line
            if metric == 'mAP@0.5':
                ax.axhline(y=0.993, color='r', linestyle='--', label='Research Target')
            elif metric == 'F1-Score':
                ax.axhline(y=0.990, color='r', linestyle='--', label='Research Target')
            
            ax.legend()
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plot_path = self.run_dir / 'yolo_variants_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {plot_path}")
        plt.close()


def main():
    """Main execution function"""
    
    print("="*80)
    print("YOLOv11 Research-Optimized Training Pipeline")
    print("Target: 96%+ Accuracy (Priyadharshini et al., 2025)")
    print("="*80)
    
    # Configuration
    DATA_PATH = "../DATA"
    OUTPUT_DIR = "../OUTPUT"
    
    # Initialize trainer
    trainer = ResearchOptimizedTrainer(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        use_preprocessing=True,  # Enable enhanced preprocessing
        use_5fold_cv=False  # Set to True for 5-fold CV
    )
    
    # Stage 1: Preprocess dataset
    preprocessed_data_path = trainer.preprocess_dataset()
    
    # Stage 2: Train YOLOv11 with research-optimized config
    data_yaml = str(preprocessed_data_path / 'data.yaml')
    model, results = trainer.train_single_model(
        model_name='yolo11n.pt',
        data_yaml=data_yaml
    )
    
    # Optional: Compare with other YOLO variants
    # comparison_results = trainer.compare_yolo_variants()
    
    # Optional: 5-fold cross-validation
    # cv_results = trainer.train_with_5fold_cv('yolo11n.pt')
    
    print("\n" + "="*80)
    print("✓ Training pipeline completed successfully!")
    print(f"✓ Results saved to: {trainer.run_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Comprehensive Evaluation Script for Brain Tumor Detection YOLO Model
This script evaluates trained YOLO models using multiple metrics including
mAP, precision, recall, F1-score, and provides detailed analysis.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import json
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorEvaluator:
    """Comprehensive evaluator for brain tumor detection model"""
    
    def __init__(self, model_path: str, data_config_path: str, output_dir: str = "../OUTPUT"):
        """Initialize evaluator with model and data paths"""
        self.model_path = Path(model_path)
        self.data_config_path = data_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create evaluation directory
        self.eval_dir = self.output_dir / f"evaluation_{self.model_path.stem}"
        self.eval_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.results = None
        
    def load_model(self):
        """Load the trained YOLO model"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(str(self.model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_model(self, 
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.7,
                      imgsz: int = 640) -> Dict:
        """Evaluate model on validation dataset"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting model evaluation...")
        
        # Run validation
        results = self.model.val(
            data=self.data_config_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=True,
            project=str(self.output_dir),
            name=f"evaluation_{self.model_path.stem}"
        )
        
        self.results = results
        
        # Extract metrics
        evaluation_metrics = {
            'model_path': str(self.model_path),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'image_size': imgsz,
            'metrics': {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                           (float(results.box.mp) + float(results.box.mr)) 
                           if (float(results.box.mp) + float(results.box.mr)) > 0 else 0.0,
                'mAP_per_class': results.box.ap.tolist() if results.box.ap is not None else [],
                'precision_per_class': results.box.p.tolist() if results.box.p is not None else [],
                'recall_per_class': results.box.r.tolist() if results.box.r is not None else []
            }
        }
        
        # Save evaluation metrics
        metrics_file = self.eval_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(evaluation_metrics, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to {metrics_file}")
        
        return evaluation_metrics
    
    def detailed_analysis(self, val_images_path: str, val_labels_path: str) -> Dict:
        """Perform detailed analysis including per-image evaluation"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting detailed analysis...")
        
        val_images_path = Path(val_images_path)
        val_labels_path = Path(val_labels_path)
        
        image_files = list(val_images_path.glob("*.jpg"))
        
        detailed_results = {
            'total_images': len(image_files),
            'per_image_results': [],
            'summary_stats': {}
        }
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        all_confidences = []
        all_labels = []
        
        for img_file in image_files:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Get predictions
            results = self.model(str(img_file), verbose=False)
            
            # Load ground truth labels
            label_file = val_labels_path / f"{img_file.stem}.txt"
            
            ground_truth_boxes = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            ground_truth_boxes.append([class_id, x_center, y_center, width, height])
            
            # Process predictions
            predictions = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    # Convert to YOLO format (normalized)
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    h, w = image.shape[:2]
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    predictions.append([cls, x_center, y_center, width, height, conf])
            
            # Calculate metrics for this image
            has_ground_truth = len(ground_truth_boxes) > 0
            has_predictions = len(predictions) > 0
            
            if has_ground_truth and has_predictions:
                true_positives += 1
                all_labels.append(1)
                all_confidences.append(max([p[5] for p in predictions]))
            elif has_ground_truth and not has_predictions:
                false_negatives += 1
                all_labels.append(1)
                all_confidences.append(0.0)
            elif not has_ground_truth and has_predictions:
                false_positives += 1
                all_labels.append(0)
                all_confidences.append(max([p[5] for p in predictions]))
            else:
                true_negatives += 1
                all_labels.append(0)
                all_confidences.append(0.0)
            
            # Store per-image result
            image_result = {
                'image_name': img_file.name,
                'ground_truth_count': len(ground_truth_boxes),
                'prediction_count': len(predictions),
                'has_ground_truth': has_ground_truth,
                'has_predictions': has_predictions,
                'max_confidence': max([p[5] for p in predictions]) if predictions else 0.0
            }
            
            detailed_results['per_image_results'].append(image_result)
        
        # Calculate summary statistics
        total_images = len(image_files)
        accuracy = (true_positives + true_negatives) / total_images if total_images > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        detailed_results['summary_stats'] = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'sensitivity': recall,  # Same as recall
            'specificity': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        }
        
        # Calculate PR curve if we have both labels and confidences
        if all_labels and all_confidences:
            try:
                precision_curve, recall_curve, thresholds = precision_recall_curve(all_labels, all_confidences)
                ap_score = average_precision_score(all_labels, all_confidences)
                
                detailed_results['pr_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': thresholds.tolist(),
                    'average_precision': float(ap_score)
                }
            except Exception as e:
                logger.warning(f"Could not calculate PR curve: {e}")
        
        # Save detailed results
        results_file = self.eval_dir / "detailed_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Detailed analysis saved to {results_file}")
        
        return detailed_results
    
    def create_evaluation_report(self, 
                               evaluation_metrics: Dict, 
                               detailed_analysis: Dict = None):
        """Create comprehensive evaluation report with visualizations"""
        
        logger.info("Creating evaluation report...")
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive visualization
        fig_rows = 3 if detailed_analysis else 2
        fig, axes = plt.subplots(fig_rows, 3, figsize=(18, 6 * fig_rows))
        fig.suptitle(f'YOLO11n Brain Tumor Detection - Evaluation Report', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # 1. Main metrics bar plot
        metrics = evaluation_metrics['metrics']
        metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['mAP50'],
            metrics['mAP50_95'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
        bars = axes_flat[0].bar(metric_names, metric_values, color=colors, alpha=0.8)
        axes_flat[0].set_ylabel('Score')\n        axes_flat[0].set_title('Model Performance Metrics')
        axes_flat[0].set_ylim(0, 1)
        axes_flat[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes_flat[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Per-class mAP (if available)
        if metrics.get('mAP_per_class'):
            class_names = ['negative', 'positive']
            map_values = metrics['mAP_per_class']
            
            axes_flat[1].bar(class_names, map_values, alpha=0.8)
            axes_flat[1].set_ylabel('mAP@0.5')
            axes_flat[1].set_title('Per-Class mAP')
            axes_flat[1].grid(True, alpha=0.3)
            
            for i, value in enumerate(map_values):
                axes_flat[1].text(i, value + 0.01, f'{value:.3f}', 
                                ha='center', va='bottom', fontweight='bold')
        else:
            axes_flat[1].text(0.5, 0.5, 'Per-class mAP\\nNot Available', 
                            ha='center', va='center', transform=axes_flat[1].transAxes,
                            fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
            axes_flat[1].set_title('Per-Class mAP')
        
        # 3. Metrics summary table
        summary_data = [
            ['mAP@0.5', f"{metrics['mAP50']:.4f}"],
            ['mAP@0.5:0.95', f"{metrics['mAP50_95']:.4f}"],
            ['Precision', f"{metrics['precision']:.4f}"],
            ['Recall', f"{metrics['recall']:.4f}"],
            ['F1-Score', f"{metrics['f1_score']:.4f}"]
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        axes_flat[2].axis('tight')
        axes_flat[2].axis('off')
        table = axes_flat[2].table(cellText=summary_df.values, colLabels=summary_df.columns,
                                 cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes_flat[2].set_title('Evaluation Summary')
        
        if detailed_analysis:
            # 4. Confusion Matrix
            stats = detailed_analysis['summary_stats']
            cm_data = np.array([[stats['true_negatives'], stats['false_positives']],
                               [stats['false_negatives'], stats['true_positives']]])
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted No Tumor', 'Predicted Tumor'],
                       yticklabels=['Actual No Tumor', 'Actual Tumor'],
                       ax=axes_flat[3])
            axes_flat[3].set_title('Confusion Matrix')
            
            # 5. Classification metrics
            clf_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity']
            clf_values = [
                stats['accuracy'],
                stats['precision'],
                stats['recall'],
                stats['f1_score'],
                stats['sensitivity'],
                stats['specificity']
            ]
            
            bars = axes_flat[4].bar(clf_metrics, clf_values, alpha=0.8, color='skyblue')
            axes_flat[4].set_ylabel('Score')
            axes_flat[4].set_title('Classification Metrics')
            axes_flat[4].set_ylim(0, 1)
            axes_flat[4].grid(True, alpha=0.3)
            axes_flat[4].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, clf_values):
                axes_flat[4].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 6. PR Curve (if available)
            if 'pr_curve' in detailed_analysis:
                pr_data = detailed_analysis['pr_curve']
                axes_flat[5].plot(pr_data['recall'], pr_data['precision'], 'b-', linewidth=2,
                                label=f'AP = {pr_data["average_precision"]:.3f}')
                axes_flat[5].set_xlabel('Recall')
                axes_flat[5].set_ylabel('Precision')
                axes_flat[5].set_title('Precision-Recall Curve')
                axes_flat[5].grid(True, alpha=0.3)
                axes_flat[5].legend()
                axes_flat[5].set_xlim(0, 1)
                axes_flat[5].set_ylim(0, 1)
            else:
                axes_flat[5].text(0.5, 0.5, 'PR Curve\\nNot Available', 
                                ha='center', va='center', transform=axes_flat[5].transAxes,
                                fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
                axes_flat[5].set_title('Precision-Recall Curve')
        
        # Hide unused subplots
        for i in range(len(axes_flat)):
            if (not detailed_analysis and i >= 3) or (detailed_analysis and i >= 6):
                axes_flat[i].axis('off')
        
        plt.tight_layout()
        
        # Save the evaluation report
        report_path = self.eval_dir / "evaluation_report.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation report saved to {report_path}")
        
        plt.show()
        
        # Print detailed report
        print("\\n" + "="*80)
        print("BRAIN TUMOR DETECTION MODEL EVALUATION REPORT")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Confidence Threshold: {evaluation_metrics['conf_threshold']}")
        print(f"IoU Threshold: {evaluation_metrics['iou_threshold']}")
        print("\\n" + "-"*50)
        print("PERFORMANCE METRICS")
        print("-"*50)
        print(f"mAP@0.5:       {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95:  {metrics['mAP50_95']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1-Score:      {metrics['f1_score']:.4f}")
        
        if detailed_analysis:
            print("\\n" + "-"*50)
            print("DETAILED ANALYSIS")
            print("-"*50)
            stats = detailed_analysis['summary_stats']
            print(f"Total Images:      {detailed_analysis['total_images']}")
            print(f"True Positives:    {stats['true_positives']}")
            print(f"False Positives:   {stats['false_positives']}")
            print(f"False Negatives:   {stats['false_negatives']}")
            print(f"True Negatives:    {stats['true_negatives']}")
            print(f"Accuracy:          {stats['accuracy']:.4f}")
            print(f"Sensitivity:       {stats['sensitivity']:.4f}")
            print(f"Specificity:       {stats['specificity']:.4f}")
            
            if 'pr_curve' in detailed_analysis:
                print(f"Average Precision: {detailed_analysis['pr_curve']['average_precision']:.4f}")
        
        print("\\n" + "="*80)
        
        # Achievement check
        target_map = 0.85
        achieved_target = metrics['mAP50'] >= target_map
        
        print(f"\\n🎯 TARGET ACHIEVEMENT:")
        print(f"Target mAP@0.5: {target_map:.2f}")
        print(f"Achieved mAP@0.5: {metrics['mAP50']:.4f}")
        
        if achieved_target:
            print("✅ TARGET ACHIEVED! Model exceeds 85% mAP threshold.")
        else:
            difference = target_map - metrics['mAP50']
            print(f"❌ Target not achieved. Need {difference:.4f} more mAP points.")
            
        print("="*80)

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset config')
    parser.add_argument('--output', type=str, default='../OUTPUT', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    parser.add_argument('--detailed', action='store_true', help='Run detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BrainTumorEvaluator(args.model, args.data, args.output)
    
    # Load model
    evaluator.load_model()
    
    # Evaluate model
    evaluation_metrics = evaluator.evaluate_model(
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    detailed_analysis = None
    if args.detailed:
        # Run detailed analysis
        val_images_path = "../DATA/images/val"
        val_labels_path = "../DATA/labels/val"
        detailed_analysis = evaluator.detailed_analysis(val_images_path, val_labels_path)
    
    # Create evaluation report
    evaluator.create_evaluation_report(evaluation_metrics, detailed_analysis)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    # For direct execution without command line arguments
    # You can modify these paths as needed
    
    # Look for the best model from recent training
    output_dir = Path("../OUTPUT")
    model_files = list(output_dir.rglob("best.pt"))
    
    if model_files:
        # Use the most recent model
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"Found trained model: {model_path}")
        
        evaluator = BrainTumorEvaluator(str(model_path), "dataset.yaml")
        evaluator.load_model()
        
        # Evaluate model
        evaluation_metrics = evaluator.evaluate_model()
        
        # Run detailed analysis
        detailed_analysis = evaluator.detailed_analysis("../DATA/images/val", "../DATA/labels/val")
        
        # Create report
        evaluator.create_evaluation_report(evaluation_metrics, detailed_analysis)
        
    else:
        print("No trained model found. Please run training first or specify model path.")
        print("Usage: python evaluate_model.py --model path/to/model.pt")

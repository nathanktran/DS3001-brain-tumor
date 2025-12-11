#!/bin/bash

echo "=========================================="
echo "Enhanced YOLOv11 Brain Tumor Detection"
echo "With Attention Mechanisms & HKCIoU Loss"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Set environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Run enhanced analysis
echo "Starting enhanced YOLOv11 training with research improvements..."
cd /u/ddz2sb/Brain-Tumor-DS4002/SCRIPTS

python yolov11_enhanced_brain_tumor.py

echo "Enhanced analysis completed!"
echo "Check OUTPUT folder for enhanced results and model weights."

# Display final results if available
if [ -f "../OUTPUT/enhanced_results_summary.json" ]; then
    echo ""
    echo "Enhanced Results Summary:"
    echo "========================"
    python -c "
import json
with open('../OUTPUT/enhanced_results_summary.json', 'r') as f:
    results = json.load(f)
if 'evaluation_results' in results:
    eval_results = results['evaluation_results']
    print(f'Enhanced mAP@0.5: {eval_results.get(\"mAP50\", 0)*100:.1f}%')
    print(f'Enhanced mAP@0.5:0.95: {eval_results.get(\"mAP50_95\", 0)*100:.1f}%') 
    print(f'Enhanced Precision: {eval_results.get(\"precision\", 0)*100:.1f}%')
    print(f'Enhanced Recall: {eval_results.get(\"recall\", 0)*100:.1f}%')
    print(f'Enhanced F1 Score: {eval_results.get(\"f1_score\", 0)*100:.1f}%')
    print(f'Improvement over baseline: +{eval_results.get(\"improvement_over_baseline\", 0):.1f}%')
    print('')
    print('Implemented Improvements:')
    for improvement in results.get('improvements_implemented', []):
        print(f'  ✓ {improvement}')
"
fi

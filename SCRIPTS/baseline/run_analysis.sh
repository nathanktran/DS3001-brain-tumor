#!/bin/bash

echo "==============================================="
echo "YOLOv11 Brain Tumor Detection Analysis Runner"
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "yolov11_brain_tumor_analysis.py" ]; then
    echo "Error: Please run this script from the SCRIPTS directory"
    exit 1
fi

# Check if dataset exists
if [ ! -d "../DATA/images/train" ]; then
    echo "Error: Dataset not found at ../DATA/images/train"
    echo "Please ensure your dataset is in the correct location"
    exit 1
fi

# Check Python and required packages
echo "Checking system requirements..."

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyTorch not found. Installing required packages..."
    pip install torch torchvision ultralytics opencv-python scikit-learn seaborn pandas matplotlib
fi

python3 -c "from ultralytics import YOLO; print('Ultralytics YOLO: Available')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Ultralytics YOLO..."
    pip install ultralytics
fi

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Create output directory if it doesn't exist
mkdir -p ../OUTPUT

echo ""
echo "Starting YOLOv11 brain tumor detection analysis..."
echo "This may take 1-3 hours depending on your hardware."
echo "Results will be saved to ../OUTPUT/"
echo ""

# Run the analysis with output logging
python3 yolov11_brain_tumor_analysis.py 2>&1 | tee ../OUTPUT/training_log.txt

echo ""
echo "Analysis complete! Check ../OUTPUT/ for results and visualizations."

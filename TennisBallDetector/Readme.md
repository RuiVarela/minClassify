# TennisBallDetector
A tennis ball predictor

## Setup
```bash
# setup python
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib timm tqdm pandas

# export needs
pip3 install onnx onnxruntime

# unpack dataset
mkdir source
cd source
# download dataset
# https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut
unzip Dataset.zip 

# run a training session
python main.py 

# visualize dataset keypoints
python main.py --plot_inference_results
```

# Credits
- https://github.com/yastrebksv/TrackNet
# FaceKeypoints
Facial keypoints trainer

This was based on a kaggle dataset
https://www.kaggle.com/competitions/facial-keypoints-detection/

## Setup
```bash
# setup python
python3 -m venv .venv
source .venv/bin/activate
pip install tensorboard torch torchvision torchinfo tqdm matplotlib pandas

# export needs
pip3 install onnx onnxruntime

# Dowload Dataset
mkdir ds
cd ds
# https://www.kaggle.com/competitions/facial-keypoints-detection/
unzip facial-keypoints-detection.zip
unzip test.zip
unzip training.zip
```

# Credits
- [dataset](https://www.kaggle.com/code/ofirflaysher/facial-keypoints-detection)

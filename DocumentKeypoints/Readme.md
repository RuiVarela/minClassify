# DocumentKeypoints
A document keypoint finder

## Setup
```bash
# setup python
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib timm tqdm

# export needs
pip3 install onnx onnxruntime

# unpack dataset
mkdir source
cd source
# download dataset
tar -zxvf testDataset.tar.gz 


python main.py --generate_source_images
python main.py --generate_dataset

# visualize dataset keypoints
python main.py --plot_dataset_tagging

# run a training session
python main.py 

# visualize dataset keypoints
python main.py --plot_inference_results
```

# Credits
- http://smartdoc.univ-lr.fr/smartdoc-2015-challenge-1/
- https://github.com/khurramjaved96/Recursive-CNNs
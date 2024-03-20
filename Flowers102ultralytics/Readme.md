# classify Flowers102
Train a simple classifier for Flowers102 dataset

## Setup
```bash
# setup python
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics scipy

# unpack dataset
mkdir source
cd source
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz
cd ..
python unpack.py

# train
yolo classify train data=flowers model=yolov8n-cls.pt imgsz=256 epochs=20

# validate
yolo classify val data=flowers model=runs/classify/train/weights/best.pt imgsz=256 split=test
yolo classify val data=flowers model=runs/classify/train/weights/best.pt imgsz=256 split=val
yolo classify val data=flowers model=runs/classify/train/weights/best.pt imgsz=256 split=train

# export
yolo classify export model=runs/classify/train/weights/best.pt format=onnx imgsz=256 dynamic=true simplify=true
```
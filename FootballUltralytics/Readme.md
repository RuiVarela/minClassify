
# Football Object detector
Train a objects detector on football content

## Setup
```bash
# setup python
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics 

# unpack dataset
mkdir source
cd source
# download roboflow dataset https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
unzip football-players-detection.v8i.yolov8.zip
# update data.yaml to use full paths
cd ..

# train
yolo task=detect mode=train model=yolov8s.pt data=source/data.yaml epochs=25 imgsz=800 plots=True

# validate
yolo classify val data=flowers model=runs/classify/train/weights/best.pt imgsz=256 split=test
yolo classify val data=flowers model=runs/classify/train/weights/best.pt imgsz=256 split=val
yolo classify val data=flowers model=runs/classify/train/weights/best.pt imgsz=256 split=train

yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=source/data.yaml imgsz=800 split=test
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=source/data.yaml imgsz=800 split=val
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=source/data.yaml imgsz=800 split=train

# export
yolo detect export model=runs/detect/train/weights/best.pt format=onnx imgsz=800 dynamic=true simplify=true
```


# Credits
- https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
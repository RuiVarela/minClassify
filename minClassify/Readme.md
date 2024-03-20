# minClassify
A minimal classifier training environment

Includes the skeleton code common on classification tasks:
- Training loop
- Resume training
- Load weights
- Export to onnx
- Training report graph
- Training report cvs

# How to run a new test
1. Implement a data loading function
2. Implement a model creation function
3. Run

# Development
```bash
python3 -m venv .venv
source .venv/bin/activate

pip3 install torch torchvision torchinfo tqdm matplotlib 

# flowers dataset needs
pip3 install scipy

# export needs
pip3 install onnx onnxruntime

python python main.py
```

# Credits
- [cnn for fashion](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST)
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
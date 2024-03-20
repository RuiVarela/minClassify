# classifyFlowers102
Train a simple classifier for Flowers102 dataset

## Setup
```bash
# setup python
python3 -m venv .venv
source .venv/bin/activate

pip install super-gradients tensorboard


tensorboard --host 0.0.0.0  --logdir checkpoints/cifar_100/RUN_20240304_120852_849126/
# http://192.168.1.98:6006/

```
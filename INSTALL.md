# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/swz30/Restormer.git
cd Restormer
```

2. Make conda environment
```
conda create -n restormer python=3.7
conda activate restormer
```

3. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy tensorboard pyyaml requests scipy tb-nightly yapf lpips xlsxwriter
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```
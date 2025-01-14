# ALS_pretraining


## Installation
The code has been tested with Ubuntu 24.04.1, python 3.10, CUDA 12.1.1

We use conda to create a virtual environment. 
```
conda create -n als python=3.10
conda activate als
```

Install pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install pcdet 
```
conda install -c conda-forge gcc=10 gxx_linux-64=10
python setup.py develop
```

Install other libraries
```
pip3 install spconv-cu120
pip3 install transformers 
pip3 install kornia 
pip3 install matplotlib  
pip3 install opencv-python
pip3 install torch_scatter
pip3 install scikit-learn
pip3 install pyransac3d
pip3 install laspy
pip3 install psutil
pip3 install lazrs
pip3 install gpustat
```

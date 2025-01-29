# Advancing ALS Applications with Large-Scale Pre-training: Dataset Development and Downstream Assessment

## Abstract
The pre-training and fine-tuning paradigm has revolutionized satellite remote sensing applications. However, this approach remains largely underexplored for airborne laser scanning (ALS), an important technology for applications such as forest management and urban planning. In this study, we address this gap by constructing a large-scale ALS point cloud dataset and evaluating its impact on downstream applications. Our dataset comprises ALS point clouds collected across the contiguous United States, provided by the United States Geological Survey's 3D Elevation Program. To ensure efficient data collection while capturing diverse land cover and terrain types, we introduce a geospatial sampling method that selects point cloud tiles based on land cover maps and digital elevation models. As a baseline self-supervised learning model, we adopt BEV-MAE, a state-of-the-art masked autoencoder for 3D outdoor point clouds, and pre-train it on the constructed dataset. The pre-trained models are subsequently fine-tuned for downstream tasks, including tree species classification, terrain scene recognition, and point cloud semantic segmentation. Our results show that the pre-trained models significantly outperform their scratch counterparts across all downstream tasks, demonstrating the transferability of the representations learned from the proposed dataset. Furthermore, we observe that scaling the dataset using our geospatial sampling method consistently enhances performance, whereas pre-training on datasets constructed with random sampling fails to achieve similar improvements. These findings highlight the utility of the constructed dataset and the effectiveness of our sampling strategy in the pre-training and fine-tuning paradigm.

## Paper
You can download the technical report from [arXiv](https://arxiv.org/abs/2501.05095).

## Installation
The code has been tested with python 3.10, pytorch 2.3.1, and CUDA 12.1.1

We use conda to create a virtual environment. 
```
conda create -n als python=3.10
conda activate als
```

Install pytorch:
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install pcdet 
```
conda install -c conda-forge gcc=10 gxx_linux-64=10
pip3 install ninja
python setup.py develop
```

Install other libraries
```
pip3 install spconv-cu120
pip3 install transformers 
pip3 install kornia 
pip3 install matplotlib  
pip3 install opencv-python
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip3 install scikit-learn
pip3 install pyransac3d
pip3 install laspy
pip3 install psutil
pip3 install lazrs
pip3 install gpustat
pip3 install gdown
pip3 install numpy==1.26.4
```
## Dataset preparation
Please see the readme in each dataset folder. For example, for DALES dataset see data/dales/

## Train and test

```
cd tools
```

Modify the JOB_NAME and TASK_ID of the run_train_eval.sh
JOB_NAME corresponds to the name of config file in tools/cfgs/${dataset}_model/

TASK_ID is an ID assigned to the experiment.

Then, the following script will run the train and test. 
```
sh run_train_eval.sh
```

## Pre-trained models
coming soon ... 

## Acknowledement 
This repo is based on 
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [BEV-MAE](https://github.com/VDIGPKU/BEV-MAE). 

## Citation
If you find our work useful in your research, please consider citing:
```
@misc{xiu2025advancingalsapplicationslargescale,
      title={Advancing ALS Applications with Large-Scale Pre-training: Dataset Development and Downstream Assessment}, 
      author={Haoyi Xiu and Xin Liu and Taehoon Kim and Kyoung-Sook Kim},
      year={2025},
      eprint={2501.05095},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.05095}, 
}
```

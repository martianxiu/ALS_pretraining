# Advancing ALS Applications with Large-Scale Pre-training: Dataset Development and Downstream Assessment

## Abstract
The pre-training and fine-tuning paradigm has revolutionized satellite remote sensing applications. However, this approach remains largely underexplored for airborne laser scanning (ALS), an important technology for applications such as forest management and urban planning. In this study, we address this gap by constructing a large-scale ALS point cloud dataset and evaluating its impact on downstream applications. Our dataset comprises ALS point clouds collected across the contiguous United States, provided by the United States Geological Survey's 3D Elevation Program. To ensure efficient data collection while capturing diverse land cover and terrain types, we introduce a geospatial sampling method that selects point cloud tiles based on land cover maps and digital elevation models. As a baseline self-supervised learning model, we adopt BEV-MAE, a state-of-the-art masked autoencoder for 3D outdoor point clouds, and pre-train it on the constructed dataset. The pre-trained models are subsequently fine-tuned for downstream tasks, including tree species classification, terrain scene recognition, and point cloud semantic segmentation. Our results show that the pre-trained models significantly outperform their scratch counterparts across all downstream tasks, demonstrating the transferability of the representations learned from the proposed dataset. Furthermore, we observe that scaling the dataset using our geospatial sampling method consistently enhances performance, whereas pre-training on datasets constructed with random sampling fails to achieve similar improvements. These findings highlight the utility of the constructed dataset and the effectiveness of our sampling strategy in the pre-training and fine-tuning paradigm.

## Paper
The technical report is available on [arXiv](https://arxiv.org/abs/2501.05095).

## Installation
This code has been tested with Python 3.10, PyTorch 2.3.1, and CUDA 12.1.1.

### Create and activate the environment:
```sh
conda create -n als python=3.10
conda activate als
```

### Install PyTorch:
```sh
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Install OpenPCDet:
```sh
conda install -c conda-forge gcc=10 gxx_linux-64=10
pip3 install ninja
python setup.py develop
```

### Install additional dependencies:
```sh
pip3 install spconv-cu120 transformers kornia matplotlib opencv-python
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip3 install scikit-learn pyransac3d laspy psutil lazrs gpustat gdown numpy==1.26.4
```

## Dataset Preparation
Refer to the README file in each dataset folder for specific instructions. For example, the DALES dataset details can be found in `data/dales/`.

## Training and Testing
Navigate to the `tools` directory:
```sh
cd tools
```

Modify the `JOB_NAME` and `TASK_ID` in `run_train_eval.sh`:
- `JOB_NAME` should match the name of the configuration file in `tools/cfgs/${dataset}_model/`.
- `TASK_ID` is an identifier for the experiment.

Then, run the following script to start training and testing:
```sh
sh run_train_eval.sh
```

## Pre-trained Models
Coming soon...

## Geospatial Data sampling and Dataset Development
The geospatial data sampling and dataset development part of code is provided in [this repo](https://github.com/martianxiu/usgs_data_analysis/tree/main).

## Acknowledgment
This repository is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [BEV-MAE](https://github.com/VDIGPKU/BEV-MAE).

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

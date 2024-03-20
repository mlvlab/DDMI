# DDMI: Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations
**[Project Page](https://dogyunpark.github.io/ddmi) |
[Paper](https://arxiv.org/abs/2401.12517)**

Dogyun Park,
Sihyeon Kim,
Sojin Lee,
Hyunwoo J. Kim†.

This repository is an official implementation of the ICLR 2024 paper DDMI (Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations).

<div align="center">
  <img src="asset/mainresult.png" width="800px" />
</div>

## Overall Framework
We propose a latent diffusion model that generates hierarchically decomposed positional embeddings of Implicit neural representations, enabling high-quality generation on various data domains.
<div align="center">
  <img src="asset/main.png" width="800px" />
</div>

## Note
Data preparation, training codes for occupancy, NeRF, and video, and checkpoints for the pre-trained models will be released soon after the code refactoring.

## Setup
To install requirements, run:
```bash
git clone https://github.com/mlvlab/DDMI.git
cd DDMI
conda create -f requirements.yaml
conda activate ddmi
```
(RECOMMENDED, linux) Install [PyTorch 2.2.0 with CUDA 11.8](https://pytorch.org/get-started/locally/) for [xformers](https://github.com/facebookresearch/xformers/edit/main/README.md), recommended for memory-efficient computation. Also, install pytorch compatible [torch-scatter](https://data.pyg.org/whl/torch-2.2.0%2Bcu118.html) version for 3D.

## Data Preparation
### Image
We have utilized two datasets for 2D image experiments: [AFHQ-V2](https://github.com/clovaai/stargan-v2) and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans). We have used `dog` and `cat` categories in AFHQ-V2 dataset. You may change the location of the dataset by changing `data_dir` of config files in `configs/`, and specify `test_data_dir` to measure r-FID during training. Each dataset should be structured as below:

```
Data
|-- folder
    |-- image1.png
    |-- image2.png
    |-- ...
```

### Video
We have used dataloader from [PVDM](https://github.com/sihyun-yu/PVDM) and [SkyTimelapse](https://github.com/weixiong-ur/mdgan) dataset. You may change the location of the dataset by changing `data_dir` of config files in `configs/`, and specify `test_data_dir` to measure r-FVD during training. Dataset should be structured as below:
```
Data
|-- train
    |-- video1
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- video2
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- ...
|-- val
    |-- video1
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- ...
```

### 3D


## Training
To train other signal domains, you may change the `domain` of config files in `configs/`, e.g., `image`, `occupancy`, `nerf`, or `video`. Currently, different network is trained for different signal domain. By default, the model's checkpoint will be stored in `./results`. If training D2C-VAE in the first-stage is unstable, i.e., NAN value, try increasing `sn_reg_weight_decay` or `sn_reg_weight_decay_init` of config files to increase the weight of spectral regularization.
### First-stage training (D2C-VAE)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp d2c-vae --configs configs/d2c-vae/img.yaml
```

### Second-stage training (LDM)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp ldm --configs configs/ldm/img.yaml
```

## Generation
You can generate a signal from the pre-trained model in `./results` by changing the `mode` of config files to `eval` from `train`, then run:
```bash
python main.py --exp ldm --configs configs/ldm/img.yaml
```
For arbitrary-resolution 2D image generation with consistent content, you only have to change `test_resolution`  of config files with a fixed seed.


## Acknowledgement
This repo is built upon [ADM](https://github.com/openai/guided-diffusion), [latent-diffusion](https://github.com/CompVis/latent-diffusion), and [PVDM](https://github.com/sihyun-yu/PVDM).

## Citation
```bibtex
@article{park2024ddmi,
  title={DDMI: Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations},
  author={Park, Dogyun and Kim, Sihyeon and Lee, Sojin and Kim, Hyunwoo J},
  journal={arXiv preprint arXiv:2401.12517},
  year={2024}
}
```



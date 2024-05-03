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

# Overall Framework
We propose a latent diffusion model that generates hierarchically decomposed positional embeddings of Implicit neural representations, enabling high-quality generation on various data domains.
<div align="center">
  <img src="asset/main.png" width="800px" />
</div>

# Setup
To install requirements, run:
```bash
git clone https://github.com/mlvlab/DDMI.git
cd DDMI
conda create -n ddmi python==3.8
conda activate ddmi
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```bash
pip install accelerate omegaconf einops pyspng natsort av ema-pytorch timm ninja gdown scipy
```
(RECOMMENDED, linux) Install [PyTorch 2.2.0 with CUDA 11.8](https://pytorch.org/get-started/locally/) for [xformers](https://github.com/facebookresearch/xformers/edit/main/README.md), recommended for memory-efficient computation. Also, install pytorch compatible [torch-scatter](https://data.pyg.org/whl/torch-2.2.0%2Bcu118.html) version for 3D.

# Data Preparation
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
We have used [ShapeNet dataset v1](https://shapenet.org/) and dataloader following [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks#preprocessed-data). You may change the location of the dataset by changing `data_dir` of config files in `configs/`.


### NeRF
We have used srn-cars dataset following [pixel-NeRF](https://github.com/arielbenitah/pixel-nerf) or you may download the dataset from [here](https://ujchmura-my.sharepoint.com/personal/przemyslaw_spurek_uj_edu_pl/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fprzemyslaw%5Fspurek%5Fuj%5Fedu%5Fpl%2FDocuments%2Fds%2Ezip&parent=%2Fpersonal%2Fprzemyslaw%5Fspurek%5Fuj%5Fedu%5Fpl%2FDocuments&ga=1). You may change the location of the dataset by changing `data_dir` of config files in `configs/`. Dataset should be structured as below:
```
Data
|-- cars
    |-- sampled
        |-- car00000.npz
        |-- car00001.npz
        |-- ...
```

# Training
To train other signal domains, you may change the `domain` of config files in `configs/`, e.g., `image`, `occupancy`, `nerf`, or `video`. Currently, different network is trained for different signal domain. By default, the model's checkpoint will be stored in `./results`. If training D2C-VAE in the first stage is unstable, i.e., NAN value, try increasing `sn_reg_weight_decay` or `sn_reg_weight_decay_init` of config files to increase the weight of spectral regularization.
## First-stage training (Discrete to Continuous space VAE)
D2C-VAE aims to learn the latent space that generates PEs between discrete data and continuous function, i.e., point clouds to occupancy function, pixel image to continuous RGB image.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp d2c-vae --configs configs/d2c-vae/img.yaml
```

## Second-stage training (Latent diffusion model)
After training D2C-VAE, we learn the latent diffusion model on the latent space of D2C-VAE.
Since latent variable is represented as a set of 2D planes, we use 2D convolution UNet model for LDM across different modalities.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp ldm --configs configs/ldm/img.yaml
```

# Evaluation
In our paper, we have utilized several evaluation metrics for assessing generation quality: FID for image, MMD and COV for 3D shape, and FVD for video evaluation.
You can change the total number sampling steps (NFE) by changing the `sampling_timesteps` in config file.
### Image
To evaluate FID of trained 2D image model, run the following script by changing the `mode` of config files to `eval` from `train`:
```bash
python main.py --exp ldm --configs configs/ldm/img.yaml
```

### Video
To evaluate FVD of trained video model, run the following script by changing the `mode` of config files to `eval` from `train`:
```bash
python main.py --exp ldm --configs configs/ldm/video.yaml
```

### 3D occupancy
You first need to generate an occupancy function and process it to make point clouds.
First, run the following script by changing the `mode` of config files to `eval` from `train`. The generated 3D shapes will be saved in the `eval` folder, located in the directory specified in config `save_pth`.
```bash
python main.py --exp ldm --configs configs/ldm/occupancy.yaml
```
Then, run the following script to sample 2048 point clouds from the mesh.
```bash
python eval_3d/meshtopc.py --pth [location of mesh files] --save_pth [save location of point clouds]
```
Finally, run the following script to measure MMD and COV between ground truth point clouds and generated point clouds.
```bash
python eval_3d/compute_metrics_3d.py --gt_pth [location of ground truth point clouds] --save_pth [location of generated point clouds]
```

# Generation
You can generate a signal from the pre-trained model in `./results` by changing the `mode` of config files to `gen` from `train`, then run:
```bash
python main.py --exp ldm --configs configs/ldm/img.yaml
```
For arbitrary-resolution 2D image generation with consistent content, you only have to change `test_resolution`  of config files with a fixed seed.


# Acknowledgement
This repo is built upon [ADM](https://github.com/openai/guided-diffusion), [latent-diffusion](https://github.com/CompVis/latent-diffusion), and [PVDM](https://github.com/sihyun-yu/PVDM).

# Citation
```bibtex
@article{park2024ddmi,
  title={DDMI: Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations},
  author={Park, Dogyun and Kim, Sihyeon and Lee, Sojin and Kim, Hyunwoo J},
  journal={arXiv preprint arXiv:2401.12517},
  year={2024}
}
```



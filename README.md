# DDMI: Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations
**[Project Page](https://dogyunpark.github.io/ddmi) |
[Paper](https://arxiv.org/abs/2401.12517)**

Dogyun Park,
Sihyeon Kim,
Sojin Lee,
Hyunwoo J. Kim†.

This repository is an official implementation of the ICLR 2024 paper DDMI (Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations).

<div align="center">
  <img src="asset/main.png" width="900px" />
</div>

## Setup
To install requirements, run:
```
git clone https://github.com/mlvlab/DDMI.git
cd DDMI
conda create -f requirements.yaml
conda activate ddmi
```

## Training
### First stage training
Change config file to train other domains.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp ldm --configs confi
gs/d2c-vae/img.yaml
```

### Second stage training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp ldm --configs confi
gs/d2c-vae/img.yaml
```


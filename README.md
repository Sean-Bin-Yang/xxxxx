## xxxxx xxxx

This code is the official PyTorch implementation of our paper: xxxx

### Introduction

**xxxx** is a simple yet effective representation learning framework for trajectory data, equipped with closed-loop theoretical guarantees through a transcription refinement mechanism. It integrates road network–aware generative reconstruction with feedback-driven contrastive learning, enabling the model to capture both fine-grained local movement semantics and global spatio-temporal dependencies—without relying on manually designed augmentation views.

<div align=center>
<img src="Framework.png"/>
</div>

### Quickstart

> [!IMPORTANT]
> This project is fully tested under Python 3.10. It is recommended that you set the Python version to 3.10.

#### 1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.10 and PyTorch 2.0.2+cu118), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

#### 2. Data Preparation

We conduct experiments on four trajectory datasets and their corresponding road networks: Porto, Beijing, Xi’an, and Chengdu. For reproducibility, the [Porto](https://drive.google.com/file/d/1UsqNyAk-nJWj4s5qIJk1hdlEfWVde1Yt/view?usp=sharing) dataset is provided.

To obtain path trajectories, please refer to the map-matching method [FMM](https://github.com/cyang-kth/fmm). For computing trajectory similarity, you may use [traj-dist](https://github.com/bguillouet/traj-dist). To generate detour trajectories, please refer to [JCLRNT](https://github.com/mzy94/JCLRNT). 

For example:

- `./data/porto/rn/...` contains the road network data.
- `./data/porto/traj/...` contains the raw trajectory data.
- `./data/porto/...` contains the preprocessed data used for training. 

After performing map matching with [FMM](https://github.com/cyang-kth/fmm), you can run ./utils/imp_aware_masking.py to get importantance-aware mask.

### Pre-Train

You can pre-train **xxx** through the following commands：

```shell
# Porto
python pretining.py --exp_id <set_exp_id> --dataset porto --device 0 --lr 1e-6 --batch_size 32 --epochs 10 --g_depths 0 --g_heads_per_layer [8,16,1] --g_dim_per_layer [16,16,128] --g_dropout 0.1 --enc_embed_dim 128 --enc_ffn_dim 512 --enc_depths 2 --enc_num_heads 8 --enc_emb_dropout 0.1 --enc_tfm_dropout 0.1 --dec_embed_dim 128 --dec_ffn_dim 512 --dec_depths 2 --dec_num_heads 8 --dec_emb_dropout 0.1 --dec_tfm_dropout 0.1
```

 ## Fine-tune

The pretrained model can be further fine-tuned for various downstream tasks. By adapting the model to task-specific data, you can improve performance on applications such as trajectory similarity estimation, prediction, or classification.



# UniAD
Official PyTorch Implementation of [A Unified Model for Multi-class Anomaly Detection](https://arxiv.org/abs/2206.03687), Accepted by NeurIPS 2022.

![Image text](docs/setting.jpg)
![Image text](docs/res_mvtec.jpg)

## 1. Quick Start

### 1.1 MVTec-AD

- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move some to `./data/MVTec-AD/`. The MVTec-AD dataset directory should be as follows. 

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
        |-- json_vis_decoder
        |-- train.json
        |-- test.json
```

- **cd the experiment directory** by running `cd ./experiments/MVTec-AD/`. 

- **Train or test** by running: 

    (1) For slurm group:  `sh train.sh #NUM_GPUS #PARTITION` or `sh test.sh #NUM_GPUS #PARTITION`.

    (2) For torch.distributed.launch:  `sh train_torch.sh #NUM_GPUS #GPU_IDS` or `sh test_torch.sh #NUM_GPUS #GPU_IDS`, *e.g.*, train with GPUs 1,3,4,6 (4 GPUs in total): `sh train_torch.sh 4 1,3,4,6`.

    **Note**: During test, please *set config.saver.load_path* to load the checkpoints. 

- **Results and checkpoints**. 

| Platform | GPU | Detection AUROC | Localization AUROC | Checkpoints | Note |
| ------ | ------ | ------ | ------ | ------ | ------ | 
| slurm group | 8 GPUs (NVIDIA Tesla V100 16GB)|  96.7 | 96.8 | [here](https://drive.google.com/file/d/1q03ysv_5VJATlDN-A-c9zvcTuyEeaQHG/view?usp=sharing) | ***A unified model for all categories*** |
| torch.distributed.launch | 1 GPU (NVIDIA GeForce GTX 1080 Ti 11 GB)|  97.0 | 97.6 | [here](https://drive.google.com/file/d/1v282ZlibC-b0H9sjLUlOSCFNzEv-TIuh/view?usp=sharing) | ***A unified model for all categories*** |


### 1.2 CIFAR-10

- **Create the CIFAR-10 dataset directory**. Download the CIFAR-10 dataset from [here](http://www.cs.toronto.edu/~kriz/cifar.html). Unzip the file and move some to `./data/CIFAR-10/`. The CIFAR-10 dataset directory should be as follows. 

```
|-- data
    |-- CIFAR-10
        |-- cifar-10-batches-py
```

- **cd the experiment directory** by running `cd ./experiments/CIFAR-10/01234/`. Here we take class 0,1,2,3,4 as normal samples, and other settings are similar.

- **Train or test** by running: 

    (1) For slurm group:  `sh train.sh #NUM_GPUS #PARTITION` or `sh test.sh #NUM_GPUS #PARTITION`.

    (2) For torch.distributed.launch:  `sh train_torch.sh #NUM_GPUS #GPU_IDS` or `sh test_torch.sh #NUM_GPUS #GPU_IDS`.

    **Note**: During test, please *set config.saver.load_path* to load the checkpoints. 

- **Results and checkpoints**. Training on 8 GPUs (NVIDIA Tesla V100 16GB) results in following performance.

| Normal Samples | {01234} | {56789} | {02468} | {13579} | Mean |
| ------ | ------ | ------ | ------ | ------ | ------ |
| AUROC | 84.4 | 79.6 | 93.0 | 89.1 | 86.5 |


## 2. Visualize Reconstructed Features

We **highly recommend** to visualize reconstructed features, since this could directly prove that our UniAD *reconstructs anomalies to their corresponding normal samples*. 

### 2.1 Train Decoders for Visualization

- **cd the experiment directory** by running `cd ./experiments/train_vis_decoder/`. 

- **Train** by running: 

    (1) For slurm group:  `sh train.sh #NUM_GPUS #PARTITION`.

    (2) For torch.distributed.launch: `sh train_torch.sh #NUM_GPUS #GPU_IDS #CLASS_NAME`.

    **Note**: for torch.distributed.launch, you should *train one vis_decoder for a specific class for one time*. 

### 2.2 Visualize Reconstructed Features

- **cd the experiment directory** by running `cd ./experiments/vis_recon/`. 

- **Visualize** by running (only support 1 GPU): 

    (1) For slurm group:  `sh vis_recon.sh #PARTITION`.

    (2) For torch.distributed.launch:  `sh vis_recon_torch.sh #CLASS_NAME`.

    **Note**: for torch.distributed.launch, you should *visualize a specific class for one time*. 

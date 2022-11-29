# Calibration-Sample-Prioritization
This repository contains code to our paper [Can Calibration Improve Sample Prioritization?](https://openreview.net/forum?id=LnygZu8WJk) presented at NeurIPS 2022 – Has It Trained Yet? (HITY) workshop.

### Abstract
Calibration can reduce overconfident predictions of deep neural networks, but can *calibration also accelerate training?* In this paper, we show that it can when used to prioritize some examples for performing subset selection. We study the effect of popular calibration techniques in selecting better subsets of samples during training (also called sample prioritization) and observe that calibration can improve the quality of subsets, reduce the number of examples per epoch (by at least 70%), and can thereby speed up the overall training process. We further study the effect of using calibrated pre-trained models coupled with calibration during training to guide sample prioritization, which again seems to improve the quality of samples selected.

### Execution
Use ```train_calib_prioritization.py``` to start training.
```sh
python train_calib_prioritization.py --nn_arch resnet_34 --dataset cifar10 --lr 0.01 --scheduler_type cosine --epochs 200 --warmup_epochs 10 --batch_size 32 --calibration mixup --mixup_alpha 0.15 --num_subset 4500 --importance_criterion entropy
```
This command starts training a ```resnet-34``` with ```mixup``` calibration on a 10% subset of CIFAR-10 training data with ```max entropy``` as a sample prioritization criterion for 200 epochs (with 10 initial warm-up epochs using all training data). You can also use a target model to guide training using the ```--target_arch``` and ```--use_target```.

## Citation
```
@inproceedings{tata_hity22,
 title={Can Calibration Improve Sample Prioritization?},
 author={Ganesh Tata and Gautham Krishna Gudur and Gopinath Chennupati and Mohammad Emtiyaz Khan},
 booktitle={Has it Trained Yet? NeurIPS 2022 Workshop},
 year={2022},
 url={https://openreview.net/forum?id=LnygZu8WJk}
}
```

# Re-implementation of ConvNets on ILSVRC 2012 classification task with PyTorch

Contact email: imdchan@yahoo.com

## Introduction

Here are re-implementations of Convolutional Networks on ILSVRC 2012 with PyTorch.

Evaluation metrics are the top-1 and top-5 error rates.

## Requirements

- A single TITAN RTX (24G memory) is used.

- Python 3.7+

- PyTorch 1.0+

## Usage

1. Clone this repository

        git clone https://github.com/longrootchen/alexnet-imagenet-pytorch.git

2. Train a model, taking alexnet as an example

        python -u train.py --work-dir ./experiments/alexnet

3. Evaluate a model on the validation set, taking alexnet as an example

        python -u eval.py --work-dir ./experiments/alexnet --ckpt-name last_checkpoint.pth --test-root ./datasets/val
        
## Results

The single AlexNet model converges at 60-th epoch and achieving a top-1 error rate of 41.20% and a top-5 error rate of 18.20% on the validation set.

| Error Rate (%) | Top-1 origin | Top-5 origin | Top-1 re-implementation | Top-5 re-implementation |
| ----- | ----- | ----- | ----- | ----- |
| AlexNet [1] | 40.7%  | 18.2% | 41.20% | 18.20% |

Here are visualizations for training loss and error rates (dark blue for train, light blue for val; red for train, pink for val) for AlexNet.

![Training loss](https://github.com/longrootchen/alexnet-imagenet-pytorch/blob/master/images/training_loss.png)

![Top-1 error](https://github.com/longrootchen/alexnet-imagenet-pytorch/blob/master/images/top1_error.png)

![Top-5 error](https://github.com/longrootchen/alexnet-imagenet-pytorch/blob/master/images/top5_error.png)

## References

[1] Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In NIPS, 2012.

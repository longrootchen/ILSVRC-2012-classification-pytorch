# Introduction

Here is a re-implementation of AlexNet model on ILSVRC 2012 with PyTorch. 

I got the top-1 and top-5 error rates comparable to the original paper.

# Requirements
## Software

Requirements for [PyTorch](https://pytorch.org/)

## Hardware

I used a single 2080Ti (12G momery) GPU.

# Usage

1. Clone this repository

        git clone https://github.com/longrootchen/alexnet-imagenet-pytorch.git

2. Train

        python train.py
        
# Results

|  | Top-1 Error | Top-5 Error |
| ----- | ----- | ----- |
| original paper | 40.7% | 18.2% |
| re-implementation | 41.2% | 18.2% |

The single AlexNet model converges at 60-th epoch and achieving a top-1 error rate of 41.20% and a top-5 error rate of 18.2% on the validation set.

# References

[1] Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In NIPS, 2012.

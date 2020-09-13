# Introduction

Here is a re-implementation of AlexNet model on ILSVRC 2012 with PyTorch. 

I got the top-1 and top-5 error rates comparable to the original paper.

Contact email: imdchan@yahoo.com

# Requirements

Requirements for [PyTorch](https://pytorch.org/)

# Usage

1. Clone this repository

        git clone https://github.com/longrootchen/alexnet-imagenet-pytorch.git

2. Train

        python train.py
        
# Results

The single AlexNet model converges at 60-th epoch and achieving a top-1 error rate of 41.20% and a top-5 error rate of 18.20% on the validation set.

|  | Top-1 Error | Top-5 Error |
| ----- | ----- | ----- |
| original paper | 40.7% | 18.2% |
| re-implementation | 41.20% | 18.20% |

Here are visualizations for training loss and error rates (dark blue for train, light blue for val; red for train, pink for val).

![Training loss](https://github.com/longrootchen/alexnet-imagenet-pytorch/blob/master/images/training_loss.png)

![Top-1 error](https://github.com/longrootchen/alexnet-imagenet-pytorch/blob/master/images/top1_error.png)

![Top-5 error](https://github.com/longrootchen/alexnet-imagenet-pytorch/blob/master/images/top5_error.png)

# References

[1] Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In NIPS, 2012.

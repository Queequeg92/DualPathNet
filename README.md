Dual Path Networks on cifar-10 and fashion-mnist datasets.
=============
----------

We construct a dual path network with WideResNet28-10 as the backbone network. The growth rate of densenet structure in the three convolutional stages are 16, 32 and 64, respectively. For details of the dual path network and wide resnet, please refer to [1] and [2]. We call this model DualPathNet28-10, which has 47.75M parameters (WideResNet28-10 has 37.5M parameters).  
The implementation details are as in [2]. **We did not fine-tune the hyperparameters. You might get better results after fine-tuning.**


## Results on cifar-10:
Accuracy: 96.35 vs 96.10(WideResNet28-10)  
![image](/doc/p1.png)

## Results on fashion-mnist:
Accuracy: 95.73  
![image](/doc/p2.png)

## To-Do:
More dual path networks.  
**Welcome to make contributions!**

## Pre-requisites:
pytorch http://pytorch.org/  
tensorboard https://www.tensorflow.org/get_started/summaries_and_tensorboard  
tensorboard-pytorch https://github.com/lanpa/tensorboard-pytorch

## How to Run:
```shell
# cd to the /scripts folder.
cd /path-to-this-repository/scripts  
# run the shells.
sh dualpath28-10.sh
```
## Acknowledgments:
Code in fashion_mnist.py are based on https://github.com/kefth/fashion-mnist/blob/master/fashionmnist.py  
All rights belongs to the original author.
## References:
[1] Chen, Yunpeng, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, and
    Jiashi Feng. "Dual Path Networks." arXiv preprint arXiv:1707.01629   (2017).  
[2] Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016).  

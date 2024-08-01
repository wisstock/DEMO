# SeNet
*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=ySg2z8-fZRA&t=357s).*

# Usage
### SE-ResNet
```python
import torch
from senet_pytorch import SEResNet

config_name = 50 # 101 and 150 are also available
r = 16 # reduction ratio
se_resnet50 = SEResNet(50, in_channels=3, out_channels=1000, r=r)

image = torch.rand(1, 3, 224, 224)
outputs = se_resnet50(image) # [1, n_classes]
```
### SE-ResNeXt
```python
import torch
from senet_pytorch import SEResNeXt

config_name = 50 # 101 and 150 are also available
r = 16 # reduction ratio
C = 32 # cardinality
se_resnext50 = SEResNeXt(50, in_channels=3, out_channels=1000, C=C, r=r)

image = torch.rand(1, 3, 224, 224)
outputs = se_resnext50(image) # [1, n_classes]
```
SeNet was introducted in 2017 paper [Squeeze-and-Excitation Networks
](https://arxiv.org/pdf/1709.01507.pdf). In this work, they focus on the channel
relationship and propose a novel architectural unit, which they term the “Squeeze-and-Excitation” block:
#### 1) __SE Block__ :

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/SeNet/Squeeze-and-Excitation.png"
>
</p>

Which consists of two operations:
-  __Squeeze__ - global average pooling that creates single value for each channel

        Squeeze = nn.AdaptiveAvgPool2d((1,1))
    
- __Excitation__ - two fully-connected layers followed by `ReLU` and `Sigmoid` respectively. Their task is to find the underlying connections between the channels and point out only relevant to the input. Reduction ratio `r` is controlling the computational cost of SEBlock. 
     
      Excitation = nn.Sequential(
          nn.Linear(in_channels, in_channels//r)
          nn.ReLU()
          nn.Linear(in_channels//r, in_channels)
          nn.Sigmoid()
          )
         
      
#### 2) __(SE)ResNet-module__ :
<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/SeNet/residual-block.png"
>
</p>
      
      
# Architecture

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNet/architecture.png"
>
</p>




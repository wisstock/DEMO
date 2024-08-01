# ResNet
*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=wOuaGvxbtZo&t=261s).*

ResNet model was introduced in 2015 in a ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper. The model achieved __first place__ in __ImageNet localization and detection competition__ and also in __Coco detection and localization__. The key takeaways from the article are:
- The resiudal connections allow to deeper networks without degredetion of accuracy 
- Residual connections imporoves flow of gradient

# Usage
```python
import torch
from resnet_pytorch import ResNet

config_name = 50 # 101 and 150 are also available
resnet50 = ResNet(config_name)

image = torch.rand(1, 3, 224, 224)
outputs = resnet50(image) # [1, n_classes]
```

# Architecture
The implementation contains __ResNet50__ configuration with __bottleneck building block__.

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/ResNet/bottleneck.png"
>
</p>
    
![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/ResNet/architectures.png)


# TO-DO
- [x] add different configurations
- [x] convert it to .py file and add usage




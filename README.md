# retinanet

## REFERENCE
- paper : https://arxiv.org/pdf/1708.02002.pdf
- paper review : 
  - https://deep-learning-study.tistory.com/504
  - https://csm-kr.tistory.com/5

- What is Anchor Box?
  - https://yhu0409.tistory.com/2 
- Why Focal Loss?
  - https://gaussian37.github.io/dl-concept-focal_loss/ 
- What is 1 stage detector?
  - https://chacha95.github.io/2020-02-26-Object-Detection3/ 
- [https://developers.arcgis.com/python/guide/how-retinanet-works/](https://developers.arcgis.com/python/guide/how-retinanet-works/)
- [https://herbwood.tistory.com/19](https://herbwood.tistory.com/19)

## Introduction

> RetinaNet is one of the best one-stage object detection models that has proven to work well with dense and small scale objects. For this reason, it has become a popular object detection model to be used with aerial and satellite imagery.
> 

> one-stage detector는 region proposal 과정이 없어 전체 이미지를 빽빽하게 순회하면서 sampling하는 dense sampling 방법을 수행하기 때문에 two-stage detector에 비해 훨씬 더 많은 후보 영역을 생성합니다. 다시 말해 **class imbalance 문제가 two-stage detector보다 더 심각합니다.** 기존의 sampling heuristic 방법을 적용해도 여전히 배경으로 쉽게 분류된 sample이 압도적으로 많기 때문에 학습이 비효율적으로 진행됩니다.

RetinaNet 논문에서는 학습 시 training imbalance가 주된 문제로 보고, 이러한 문제를 해결하여 one-stage detector에서 적용할 수 있는 새로운 loss function을 제시합니다.
> 

## **RetinaNet architecture**

There are four major components of a RetinaNet model architecture (Figure 3):

> **a) Bottom-up Pathway** - The backbone network (e.g. ResNet) which calculates the feature maps at different scales, irrespective of the input image size or the backbone.

> **b) Top-down pathway and Lateral connections** - The top down pathway upsamples the spatially coarser feature maps from higher pyramid levels, and the lateral connections merge the top-down layers and the bottom-up layers with the same spatial size.

> **c) Classification subnetwork** - It predicts the probability of an object being present at each spatial location for each anchor box and object class.

> **d) Regression subnetwork** - It's regresses the offset for the bounding boxes from the anchor boxes for each ground-truth object.
> 
> 
> ![Figure 3. RetinaNet model architecture](https://developers.arcgis.com/assets/img/python-graphics/retinanet.png)
> 
> Figure 3. RetinaNet model architecture
> 
> 본 논문에서는 **Focal loss**라는 새로운 loss function을 제시합니다. Focal loss는 cross entropy loss에 class에 따라 변하는 동적인 scaling factor를 추가한 형태를 가집니다. 이러한 loss function을 통해 학습 시 easy example의 기여도를 자동적으로 down-weight하며, hard example에 대해서 가중치를 높혀 학습을 집중시킬 수 있습니다. Focal loss의 효과를 실험하기 위해 논문에서는 one-stage detector인 **RetinaNet**을 설계합니다. 해당 네트워크는 ResNet-101-FPN을 backbone network로 가지며 anchor boxes를 적용하여 기존의 two-stage detector에 비해 높은 성능을 보여줍니다.
> 

## **Focal Loss**

> **Focal loss**는 one-stage detector 모델에서 foreground와 background class 사이에 발생하는 극단적인 class imbalance(가령 1:1000)문제를 해결하는데 사용됩니다. Focal loss는 이진 분류에서 사용되는 Cross Entropy(이하 CE) loss function으로부터 비롯됩니다.
> 

### **CE loss**

> $CE(p,y)=\LARGE\{ ^{−log(p),\ \ {if} \ y=1} _{−log(1−p),\ \ otherwise}$
> 
> 
> 
> $p_t=\LARGE\{^{p,\ \ if \  y=11}_{−p,\ \ otherwise}$
> 
> CE loss의 문제는 모든 sample에 대한 예측 결과를 동등하게 가중치를 둔다는 점입니다. 이로 인해 어떠한 sample이 쉽게 분류될 수 있음에도 불구하고 작지 않은 loss를 유발하게 됩니다. 많은 수의 easy example의 loss가 더해지면 보기 드문 class를 압도해버려 학습이 제대로 이뤄지지 않습니다.
> 

### **Balanced Cross Entropy**

> 이러한 문제를 해결하기 위해 가중치 파라미터인 α∈[0,1]α∈[0,1]를 곱해준 **Balanced Cross Entropy**가 등장합니다.
> 
> 
> $CE(pt)=−αlog(pt)$
> 
> $y=1$ 일 때 $\alpha$를 곱해주고, $y=-1$ 일 때 $1-\alpha$ 를 곱해줍니다.
> 하지만 Balanced CE는 positive/negative sample 사이의 균형을 잡아주지만, easy/hard sample에 대해서는 균형을 잡지 못합니다. 논문에서는 Balanced Cross Entropy를 baseline으로 삼고 실험을 진행합니다.
> 

### **Focal Loss**

> $\LARGE FL(pt)=\{^{−(1−p_t)^γlog(p_t),\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  if \ y=1}_{−(1−(1−p_t))^γlog(1−p_t),\ \ otherwise}$
> 
> 
> **Focal loss**는 easy example을 down-weight하여 hard negative sample에 집중하여 학습하는 loss function입니다. **Focal loss는 modulating factor $(1−p_t)^γ$ 와 tunable focusing parameter $γ$ 를 CE에 추가한 형태를 가집니다.**
>

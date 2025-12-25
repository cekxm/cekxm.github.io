---
layout: post
title: "Multiple Object Tracking"
date: 2024-12-30 00:01:52 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---

## 算法

### SORT

见伟佳PPT

### Deep-sort

见伟佳PPT。

补充：比较有趣的一点，这边说要使用 matching cascade，原因如下。但是到了 strong-sort 又换为 vanilla matching，因为 tracker 质量提高了，用 matching cascade 反而不好。

使用 matching cascade 的原因

> 当物体被遮挡的时间较长时，随后的卡尔曼滤波器预测会增加与物体位置相关的不确定性。因此，状态空间中的概率质量会扩散，观测概率会变得不那么集中。直观地看，关联度量应该通过增加测量与跟踪之间的距离来考虑这种概率质量的扩散。令人感到反直觉的是，当两个跟踪对象争夺同一个检测时，马氏距离会偏向更大的不确定性，因为它会有效地减少任何检测到投影轨迹均值的标准差距离。这是一种不希望的行为，因为它可能会导致跟踪分裂和不稳定的跟踪。

考虑一个观测和两个轨迹A,B的欧式距离相等，A 的方差大。则马氏距离会偏向A。

### Strong-Sort

见伟佳PPT。

<img class="img-fluid" src="/images/2024-12-30-Multiple-Object-Tracking/image-20230731104348179.png" alt="image-20230731104348179" style="zoom: 50%;" />

补充：

ECC 相机运动补偿。当卡尔曼滤波器预测t+1时刻的状态  $$s_{t+1}$$ 时，如果相机运动了，则 t+1 时刻的图像坐标变化了，所以要将 $$s_{t+1}$$ 进行相应的变换。ECC 用于估计变换 warp，然后再用这个变换去做补偿。

<img class="img-fluid" src="/images/2024-12-30-Multiple-Object-Tracking/image-20230731203836510.png" alt="image-20230731203836510" style="zoom:50%;" />

一般卡尔曼滤波器的状态包含 BB 及BB的导数。**注意 warp 只作用在 BB 上。**

ECC 算法：

> ECC (Enhanced Correlation Coefficient) 算法是一种用于图像处理的算法，主要用于图像配准。图像配准是一种过程，它确定了一对图像间的几何变换，使得图像间的对应点匹配。这在许多计算机视觉和图像处理应用中都非常重要，例如运动跟踪、手术导航、立体视觉和目标识别等。
>
> ECC 算法是基于互相关的配准方法，它寻找最大化两个图像的相关性的变换。不过，与基础的互相关方法不同，ECC 算法使用一种增强的相关性度量，该度量对光照变化和其他常见的图像变形有更强的稳健性。
>
> ECC 算法的主要步骤包括:
>
> 1. 初始化一个估计的几何变换（通常为恒等变换）。
> 2. 使用当前的估计变换，将一个图像映射到另一个图像空间。
> 3. 计算映射后的图像和目标图像的增强相关性。
> 4. 通过最大化增强相关性来更新估计的变换。
> 5. 重复步骤 2-4，直到变换收敛，或者达到预定的迭代次数。
>
> ECC 算法的一个关键优点是，它可以适应各种不同的几何变换模型，包括刚性变换（旋转和平移）、仿射变换和更复杂的非线性变换。这使得 ECC 算法在处理复杂的相机运动和场景变形时，比许多其他图像配准方法更具有优势。

### OC-SORT 

见伟佳PPT

1. Observation-centric Re-Update (ORU)

2. Observation-Centric Momentum (OCM)，它用在正常的关联中

   <img class="img-fluid" src="/images/2024-12-30-Multiple-Object-Tracking/image-20230731211451900.png" alt="image-20230731211451900" style="zoom:50%;" />

   <img class="img-fluid" src="/images/2024-12-30-Multiple-Object-Tracking/image-20230731211547711.png" alt="image-20230731211547711" style="zoom:50%;" />

   将目标的运动方向加入了代价矩阵的计算，即第二项，为两个方向夹角（track方向和intention方向）。注意图中的点都是观测，所以为观测为中心

3. Observation-Centric Recovery (OCR)，它是正常关联后的第二次关联。如果有未被关联的轨迹，和未被关联的观测。则触发OCR。尝试将未被关联轨迹的上一个观测和当前帧的观测做关联，这可以解决短时间的遮挡问题。

### ByteTrack

见伟佳PPT

## 代码

1. [BoxMOT: pluggable SOTA tracking modules for object detectors](https://github.com/mikel-brostrom/yolo_trackin)

## 数据集

1. WX
2. JMU
3. [LSOTB-TIR: A Large-Scale High-Diversity Thermal Infrared Object Tracking Benchmark](https://github.com/QiaoLiuHit/LSOTB-TIR), 2020
4. [PTB-TIR: A Thermal Infrared Pedestrian Tracking Benchmark (TMM19)](https://github.com/QiaoLiuHit/PTB-TIR_Evaluation_toolkit)


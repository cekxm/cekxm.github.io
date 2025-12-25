---
layout: post
title: "One Stage Detection"
date: 2024-12-30 00:13:30 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---

## One stage detection

### UnitBox: An Advanced Object Detection Network

一种新的IoU Loss

![image-20231209154250114](/images/2024-12-30-one-stage-detection/image-20231209154250114.png){: .img-fluid}

首先有个表示一个bbox的方法：给定一个点，一个bbox表示为 xt,xb,xl,xr

然后给定一个点，有两个bbox，有上面的IoU loss

<img class="img-fluid" src="/images/2024-12-30-one-stage-detection/image-20231210110333912.png" alt="image-20231210110333912" style="zoom: 50%;" />

### Focal Loss for Dense Object Detection

![image-20231209154644100](/images/2024-12-30-one-stage-detection/image-20231209154644100.png){: .img-fluid}

提出了一种叫 focal loss，The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training.

负样本太多了，很多负样本的二分类是简单的。但是CE对这些简单样本也会有一个不小的loss。所以即使二分类器对简单样本分类正确，它们的总loss也太大了。

focal loss 力图降低简单样本的loss，让二分类器更关注被分类错误的样本。

它不仅关注正负样本的不均衡，**更关注简单样本和困难样本的不均衡。**

### FCOS: Fully Convolutional One-Stage Object Detection

![image-20231210100557495](/images/2024-12-30-one-stage-detection/image-20231210100557495.png){: .img-fluid}

Different from anchor-based detectors, which consider the location on the input image as the center of (multiple) anchor boxes and regress the target bounding box with these anchor boxes as references, we directly regress the target bounding box at the location. In other words, our detector directly views locations as training samples instead of anchor boxes in anchor-based detectors.

anchor-based detectors 中，training samples 是 anchor boxes，在某一个点的推理结果是以该点为中心的anchor box。

而 fcos中，training samples就是location，推理结果直接是一个bbox。

正样本的判定：如果Pi上的位置映射回原图像，落在ground-truth中，则是正样本。If a location falls into
multiple bounding boxes, it is considered as an ambiguous sample. We simply choose the bounding box with minimal area as its regression target. In the next section, we will show that with multi-level prediction, the number of ambiguous samples can be reduced significantly and thus they hardly affect the detection performance.

为什么fcos有效，作者提出了一种解释：It is worth noting that FCOS can leverage as many foreground
samples as possible to train the regressor. It is different from anchor-based detectors, which only consider the anchor boxes with a highly enough IOU with ground-truth boxes as positive samples. We argue that it may be one of the reasons that FCOS outperforms its anchor-based counterparts.

#### Loss function

<img class="img-fluid" src="/images/2024-12-30-one-stage-detection/image-20231210102210584.png" alt="image-20231210102210584" style="zoom: 80%;" />

1. 训练时，训练C个二分类器，而不是一个多类的分类器，因为 focal loss就是二分类的
2. 这边还没加入 center-ness
3. Lcls 为focal loss
4. Lreg 为 novel Iou loss
5. 为什么要除于 Npos，论文里面没说。可能的原因是为了自适应目标框大小，

#### Multilevel Prediction with FPN for FCOS

见上面的图

为了减少ambiguous samples，并非任意的点都算是正样本。为每一层设定了一个回归的最大距离 mi，超过 mi 或小于 $$m_{i-1}$$ ，都不算是正样本。

> 小于 $$m_{i-1}$$，意味着在这一层，框太小了，应该在更低的层进行检测。

#### Center-ness

> We observed that it is due to a lot of low-quality predicted bounding boxes produced by locations far away from the center of an object.

如果一个点远离bbox中心，通常这个点的预测质量不好。所以引入了center-ness.

在每一点除了预测类别，bbox，还需要预测centerness （with binary cross entropy loss）

center-ness 如何用？

> When testing, the center-ness predicted by the network is multiplied with the classification score
> thus can down-weight the low-quality bounding boxes predicted by a location far from the center of an object.
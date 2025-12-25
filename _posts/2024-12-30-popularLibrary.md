---
layout: post
title: "Popularlibrary"
date: 2024-12-30 00:14:24 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---

## 深度学习资料

### 资源库

#### github

#### [paperwithcode](https://paperswithcode.com/)

#### [Huggingface](https://huggingface.co/)

### 综合型库

#### [OpenMMLab](https://github.com/open-mmlab)

openmmlab 很多工具开箱即可用。

OpenMMLab 为香港中文大学-商汤科技联合实验室 MMLab 开源的算法平台，不到两年时间，已经包含众多 SOTA 计算机视觉算法。

OpenMMLab 在Github上不是一个单独项目，除了大家所熟知的 Github 上万 star 目标检测库 MMDetection，还有其他方向的代码库和数据集，非常值得从事计算机视觉研发的朋友关注。

近期 OpenMMLab 进行了密集更新，新增了多个库，官方称涉及超过 10 个研究方向，开放超过 100 种算法和 600 种预训练模型，目前Github总星标超过 1.7 万。是CV方向系统性较强、社区活跃的开源平台。

这些库大部分都基于深度学习 PyTorch 框架，算法紧跟前沿，方便易用，文档较为丰富，无论对于研究还是工程开发的朋友都很值得了解。

[包含很多库](https://zhuanlan.zhihu.com/p/159562429)

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab Generative Model toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMlab deep learning model deployment toolset.

#### [timm](https://github.com/rwightman/pytorch-image-models)

- 以分类为任务，包含多种分类模型
- 图像预处理
- optimizer
- scheduler

#### [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)

指标，但能作为 Loss 吗？不知道，还得再探究一下。

#### [pytorch-accelerated](https://github.com/Chris-hughes10/pytorch-accelerated)

- 多GPU，多设备训练
- 训练流程的简化

#### [transformers](https://huggingface.co/docs/transformers/index)

Transformers provides APIs to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you time from training a model from scratch. The models can be used across different modalities such as:

- 📝 Text: text classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages.
- 🖼️ Images: image classification, object detection, and segmentation.
- 🗣️ Audio: speech recognition and audio classification.
- 🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

可以看到图像方面是图像分类、物体检测、分割

### 训练框架和加速

#### [fastai](https://github.com/fastai/fastai)

这个是训练的框架库，有点类似 pyotrch-accelerated.  可能更通用

 fastai includes:

- A new type dispatch system for Python along with a semantic type hierarchy for tensors
- A GPU-optimized computer vision library which can be extended in pure Python
- An optimizer which refactors out the common functionality of modern optimizers into two basic pieces, allowing optimization algorithms to be implemented in 4–5 lines of code
- A novel 2-way callback system that can access any part of the data, model, or optimizer and change it at any point during training
- A new data block API
- And much more...

#### [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

Lightning forces the following structure to your code which makes it reusable and shareable:

- Research code (the LightningModule).
- Engineering code (you delete, and is handled by the Trainer).
- Non-essential research code (logging, etc... this goes in Callbacks).
- Data (use PyTorch DataLoaders or organize them into a LightningDataModule).

Once you do this, you can train on multiple-GPUs, TPUs, CPUs, IPUs, HPUs and even in 16-bit precision without changing your code!

### Metric Learning

#### [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

##### Miner, loss, reducer, regularizer 的关系

![high_level_loss_function_overview](/images/2024-12-30-popularLibrary/high_level_loss_function_overview.png){: .img-fluid}

The miner finds positive and negative pairs that it thinks are particularly difficult. 

Loss functions can be customized using distances, reducers, and regularizers. In the diagram below, a miner finds the indices of hard pairs within a batch. These are used to index into the distance matrix, computed by the distance object. For this diagram, the loss function is pair-based, so it computes a loss per pair. In addition, a regularizer has been supplied, so a regularization loss is computed for each embedding in the batch. The per-pair and per-element losses are passed to the reducer, which (in this diagram) only keeps losses with a high value. The averages are computed for the high-valued pair and element losses, and are then added together to obtain the final loss.

[Distances](https://kevinmusgrave.github.io/pytorch-metric-learning/distances)

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**CosineSimilarity**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#cosinesimilarity) |                                                              |
| [**DotProductSimilarity**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#dotproductsimilarity) |                                                              |
| [**LpDistance**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#lpdistance) |                                                              |
| [**SNRDistance**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#snrdistance) | [Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) |

[Losses](https://kevinmusgrave.github.io/pytorch-metric-learning/losses)

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**AngularLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss) | [Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf) |
| [**ArcFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss) | [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf) |
| [**CentroidTripletLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#centroidtripletloss) | [On the Unreasonable Effectiveness of Centroids in Image Retrieval](https://arxiv.org/pdf/2104.13643.pdf) |
| [**CircleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss) | [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf) |
| [**ContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss) | [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) |
| [**CosFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#cosfaceloss) | - [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf) - [Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf) |
| [**FastAPLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#fastaploss) | [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf) |
| [**GeneralizedLiftedStructureLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss) | [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf) |
| [**IntraPairVarianceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#intrapairvarianceloss) | [Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf) |
| [**LargeMarginSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#largemarginsoftmaxloss) | [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf) |
| [**LiftedStructreLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#liftedstructureloss) | [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf) |
| [**MarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#marginloss) | [Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf) |
| [**MultiSimilarityLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss) | [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) |
| [**NCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ncaloss) | [Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf) |
| [**NormalizedSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss) | - [NormFace: L2 Hypersphere Embedding for Face Verification](https://arxiv.org/pdf/1704.06369.pdf) - [Classification is a Strong Baseline for DeepMetric Learning](https://arxiv.org/pdf/1811.12649.pdf) |
| [**NPairsLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#npairsloss) | [Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf) |
| [**NTXentLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss) | - [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf) - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf) - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) |
| [**ProxyAnchorLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyanchorloss) | [Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/pdf/2003.13911.pdf) |
| [**ProxyNCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyncaloss) | [No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf) |
| [**SignalToNoiseRatioContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#signaltonoiseratiocontrastiveloss) | [Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) |
| [**SoftTripleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#softtripleloss) | [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf) |
| [**SphereFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#spherefaceloss) | [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf) |
| [**SubCenterArcFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#subcenterarcfaceloss) | [Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf) |
| [**SupConLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss) | [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) |
| [**TripletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss) | [Distance Metric Learning for Large Margin Nearest Neighbor Classification](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf) |
| [**TupletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tupletmarginloss) | [Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf) |
| [**VICRegLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#vicregloss) | [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf) |

[Miners](https://kevinmusgrave.github.io/pytorch-metric-learning/miners)

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**AngularMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#angularminer) |                                                              |
| [**BatchEasyHardMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batcheasyhardminer) | [Improved Embeddings with Easy Positive Triplet Mining](http://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf) |
| [**BatchHardMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batchhardminer) | [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf) |
| [**DistanceWeightedMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer) | [Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf) |
| [**EmbeddingsAlreadyPackagedAsTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#embeddingsalreadypackagedastriplets) |                                                              |
| [**HDCMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#hdcminer) | [Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf) |
| [**MaximumLossMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#maximumlossminer) |                                                              |
| [**MultiSimilarityMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#multisimilarityminer) | [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) |
| [**PairMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#pairmarginminer) |                                                              |
| [**TripletMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#tripletmarginminer) | [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf) |
| [**UniformHistogramMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#uniformhistogramminer) |                                                              |

[Reducers](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers)

| Name                                                         | Reference Papers |
| ------------------------------------------------------------ | ---------------- |
| [**AvgNonZeroReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#avgnonzeroreducer) |                  |
| [**ClassWeightedReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#classweightedreducer) |                  |
| [**DivisorReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#divisorreducer) |                  |
| [**DoNothingReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#donothingreducer) |                  |
| [**MeanReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#meanreducer) |                  |
| [**PerAnchorReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#peranchorreducer) |                  |
| [**ThresholdReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#thresholdreducer) |                  |

[Regularizers](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers)

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**CenterInvariantRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#centerinvariantregularizer) | [Deep Face Recognition with Center Invariant Loss](http://www1.ece.neu.edu/~yuewu/files/2017/twu024.pdf) |
| [**LpRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#lpregularizer) |                                                              |
| [**RegularFaceRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#regularfaceregularizer) | [RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf) |
| [**SparseCentersRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#sparsecentersregularizer) | [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf) |
| [**ZeroMeanRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#zeromeanregularizer) | [Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) |

[Samplers](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers)

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**MPerClassSampler**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#mperclasssampler) |                                                              |
| [**HierarchicalSampler**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#hierarchicalsampler) | [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf) |
| [**TuplesToWeightsSampler**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#tuplestoweightssampler) |                                                              |
| [**FixedSetOfTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#fixedsetoftriplets) |                                                              |

[Trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers)

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**MetricLossOnly**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#metriclossonly) |                                                              |
| [**TrainWithClassifier**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#trainwithclassifier) |                                                              |
| [**CascadedEmbeddings**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#cascadedembeddings) | [Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf) |
| [**DeepAdversarialMetricLearning**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#deepadversarialmetriclearning) | [Deep Adversarial Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf) |
| [**UnsupervisedEmbeddingsUsingAugmentations**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#unsupervisedembeddingsusingaugmentations) |                                                              |
| [**TwoStreamMetricLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#twostreammetricloss) |                                                              |

[Testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers)

| Name                                                         | Reference Papers |
| ------------------------------------------------------------ | ---------------- |
| [**GlobalEmbeddingSpaceTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globalembeddingspacetester) |                  |
| [**WithSameParentLabelTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#withsameparentlabeltester) |                  |
| [**GlobalTwoStreamEmbeddingSpaceTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globaltwostreamembeddingspacetester) |                  |

Utils

| Name                                                         | Reference Papers |
| ------------------------------------------------------------ | ---------------- |
| [**AccuracyCalculator**](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation) |                  |
| [**HookContainer**](https://kevinmusgrave.github.io/pytorch-metric-learning/logging_presets) |                  |
| [**InferenceModel**](https://kevinmusgrave.github.io/pytorch-metric-learning/inference_models) |                  |
| [**TorchInitWrapper**](https://kevinmusgrave.github.io/pytorch-metric-learning/common_functions/#torchinitwrapper) |                  |
| [**DistributedLossWrapper**](https://kevinmusgrave.github.io/pytorch-metric-learning/distributed/#distributedlosswrapper) |                  |
| [**DistributedMinerWrapper**](https://kevinmusgrave.github.io/pytorch-metric-learning/distributed/#distributedminerwrapper) |                  |

Base Classes, Mixins, and Wrappers

| Name                                                         | Reference Papers                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**CrossBatchMemory**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#crossbatchmemory) | [Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf) |
| [**GenericPairLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#genericpairloss) |                                                              |
| [**MultipleLosses**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multiplelosses) |                                                              |
| [**MultipleReducers**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#multiplereducers) |                                                              |
| **EmbeddingRegularizerMixin**                                |                                                              |
| **WeightMixin**                                              |                                                              |
| [**WeightRegularizerMixin**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#weightregularizermixin) |                                                              |
| [**BaseDistance**](https://kevinmusgrave.github.io/pytorch-metric-learning/distance/#basedistance) |                                                              |
| [**BaseMetricLossFunction**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#basemetriclossfunction) |                                                              |
| [**BaseMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#baseminer) |                                                              |
| [**BaseTupleMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#basetupleminer) |                                                              |
| [**BaseSubsetBatchMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#basesubsetbatchminer) |                                                              |
| [**BaseReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#basereducer) |                                                              |
| [**BaseRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#baseregularizer) |                                                              |
| [**BaseTrainer**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#basetrainer) |                                                              |
| [**BaseTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#basetester) |                                                              |

### **S**elf-**S**upervised **L**earning and Contrastive learning

#### [VISSL](https://github.com/facebookresearch/vissl)

VISSL is a computer **VI**sion library for state-of-the-art **S**elf-**S**upervised **L**earning research with [PyTorch](https://pytorch.org/). VISSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations. Key features include:

- **Reproducible implementation of SOTA in Self-Supervision**: All existing SOTA in Self-Supervision are implemented - [SwAV](https://arxiv.org/abs/2006.09882), [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo(v2)](https://arxiv.org/abs/1911.05722), [PIRL](https://arxiv.org/abs/1912.01991), [NPID](https://arxiv.org/pdf/1805.01978.pdf), [NPID++](https://arxiv.org/abs/1912.01991), [DeepClusterV2](https://arxiv.org/abs/2006.09882), [ClusterFit](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_ClusterFit_Improving_Generalization_of_Visual_Representations_CVPR_2020_paper.pdf), [RotNet](https://arxiv.org/abs/1803.07728), [Jigsaw](https://arxiv.org/abs/1603.09246). Also supports supervised trainings.
- **Benchmark suite**: Variety of benchmarks tasks including [linear image classification (places205, imagenet1k, voc07, food, CLEVR, dsprites, UCF101, stanford cars and many more)](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification), [full finetuning](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/fulltune), [semi-supervised benchmark](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/semi_supervised), [nearest neighbor benchmark](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/nearest_neighbor), [object detection (Pascal VOC and COCO)](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/object_detection).
- **Ease of Usability**: easy to use using yaml configuration system based on [Hydra](https://github.com/facebookresearch/hydra).
- **Modular**: Easy to design new tasks and reuse the existing components from other tasks (objective functions, model trunk and heads, data transforms, etc.). The modular components are simple *drop-in replacements* in yaml config files.
- **Scalability**: Easy to train model on 1-gpu, multi-gpu and multi-node. Several components for large scale trainings provided as simple config file plugs: [Activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html), [ZeRO](https://arxiv.org/abs/1910.02054), [FP16](https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use), [LARC](https://arxiv.org/abs/1708.03888), Stateful data sampler, data class to handle invalid images, large model backbones like [RegNets](https://arxiv.org/abs/2003.13678), etc.
- **Model Zoo**: Over *60 pre-trained self-supervised model* weights.

### 数据增强

#### timm

#### [Albumentations](https://github.com/albumentations-team/albumentations)

主要是颜色、对比度、模糊等方面的

Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

### 图像变换模块

#### [kornia](https://github.com/kornia/kornia)

这个和数据增强不同，它是可以放在模型中的。

Inspired by existing packages, this library is composed by a subset of packages containing operators that can be inserted within neural networks to train models to perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.

At a granular level, Kornia is a library that consists of the following components:

| **Component**                                                | **Description**                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [kornia](https://kornia.readthedocs.io/en/latest/index.html) | a Differentiable Computer Vision library, with strong GPU support |
| [kornia.augmentation](https://kornia.readthedocs.io/en/latest/augmentation.html) | a module to perform data augmentation in the GPU             |
| [kornia.color](https://kornia.readthedocs.io/en/latest/color.html) | a set of routines to perform color space conversions         |
| [kornia.contrib](https://kornia.readthedocs.io/en/latest/contrib.html) | a compilation of user contrib and experimental operators     |
| [kornia.enhance](https://kornia.readthedocs.io/en/latest/enhance.html) | a module to perform normalization and intensity transformation |
| [kornia.feature](https://kornia.readthedocs.io/en/latest/feature.html) | a module to perform feature detection                        |
| [kornia.filters](https://kornia.readthedocs.io/en/latest/filters.html) | a module to perform image filtering and edge detection       |
| [kornia.geometry](https://kornia.readthedocs.io/en/latest/geometry.html) | a geometric computer vision library to perform image transformations, 3D linear algebra and conversions using different camera models |
| [kornia.losses](https://kornia.readthedocs.io/en/latest/losses.html) | a stack of loss functions to solve different vision tasks    |
| [kornia.morphology](https://kornia.readthedocs.io/en/latest/morphology.html) | a module to perform morphological operations                 |
| [kornia.utils](https://kornia.readthedocs.io/en/latest/utils.html) | image to tensor utilities and metrics for vision problems    |

### 物体检测

#### [Detectron2](https://github.com/facebookresearch/detectron2)

Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.

#### [EfficientDet](https://github.com/rwightman/efficientdet-pytorch)

EfficientDet: Scalable and Efficient Object Detection CVPR2020 的一个实现

作者是 timm 的作者

### 语义分割

#### [Segementation Models](https://github.com/qubvel/segmentation_models.pytorch#architectures)

The main features of this library are:

- High level API (just two lines to create a neural network)
- 9 models architectures for binary and multi class segmentation (including legendary Unet)
- 113 available encoders (and 400+ encoders from [timm](https://github.com/rwightman/pytorch-image-models))
- All encoders have pre-trained weights for faster and better convergence
- Popular metrics and losses for training routines
- Architectures 可以搭配不同的 Encoders

##### Architectures

- Unet [[paper](https://arxiv.org/abs/1505.04597)] [[docs](https://smp.readthedocs.io/en/latest/models.html#unet)]
- Unet++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id2)]
- MAnet [[paper](https://ieeexplore.ieee.org/abstract/document/9201310)] [[docs](https://smp.readthedocs.io/en/latest/models.html#manet)]
- Linknet [[paper](https://arxiv.org/abs/1707.03718)] [[docs](https://smp.readthedocs.io/en/latest/models.html#linknet)]
- FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#fpn)]
- PSPNet [[paper](https://arxiv.org/abs/1612.01105)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pspnet)]
- PAN [[paper](https://arxiv.org/abs/1805.10180)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pan)]
- DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)] [[docs](https://smp.readthedocs.io/en/latest/models.html#deeplabv3)]
- DeepLabV3+ [[paper](https://arxiv.org/abs/1802.02611)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id9)]

##### Encoders

这边他说的 Encoders 是指主干模块。

MMDetection

### Video Object Segmentation (VOS)

[YouTube-VOS](https://youtube-vos.org/)

### 跟踪

- **Pytracking**
- **Pysot**
- mmtracking
- **Trackit**

**Pytracking**文件比较难懂，但是容易follow；

**Pysot** 集成Siamese系列算法，该工具包半年未更新，未必一定要在此基础展开工作，**但是如果学习siamese代码，必须要看此代码并学习；**

**Trackit** 集成了TensorRT,可对模型加速**。**

#### [pysot](https://github.com/STVIR/pysot)

The goal of PySOT is to provide a high-quality, high-performance codebase for visual tracking *research*. It is designed to be flexible in order to support rapid implementation and evaluation of novel research. PySOT includes implementations of the following visual tracking algorithms:

- [SiamMask](https://arxiv.org/abs/1812.05050)
- [SiamRPN++](https://arxiv.org/abs/1812.11703)
- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- [SiamFC](https://arxiv.org/abs/1606.09549)

using the following backbone network architectures:

- [ResNet{18, 34, 50}](https://arxiv.org/abs/1512.03385)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](https://github.com/STVIR/pysot#references) below.

Evaluation toolkit can support the following datasets:

📎 [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 📎 [VOT16/18/19](http://votchallenge.net/) 📎 [VOT18-LT](http://votchallenge.net/vot2018/index.html) 📎 [LaSOT](https://arxiv.org/pdf/1809.07845.pdf) 📎 [UAV123](https://arxiv.org/pdf/1804.00518.pdf)

##### [教程](https://blog.csdn.net/weixin_42495721/article/details/110732592?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163695830416780255266434%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163695830416780255266434&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-110732592.first_rank_v2_pc_rank_v29&utm_term=pysot&spm=1018.2226.3001.4187)

#### [pytracking](https://github.com/visionml/pytracking)

pytracking 是由苏黎世联邦理工学院 Computer Vision Lab 出品。包含该实验在跟踪方面的持续的工作。同时他也是一个跟踪的框架库。

比如 [Stark](https://github.com/researchmm/Stark) 的实现就用到了这个库。

#### [mmtracking](https://github.com/open-mmlab/mmtracking)

Supported methods of video object detection:

-  [DFF](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/dff) (CVPR 2017)
-  [FGFA](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/fgfa) (ICCV 2017)
-  [SELSA](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/selsa) (ICCV 2019)
-  [Temporal RoI Align](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/temporal_roi_align) (AAAI 2021)

Supported methods of multi object tracking:

-  [SORT/DeepSORT](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/deepsort) (ICIP 2016/2017)
-  [Tracktor](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/tracktor) (ICCV 2019)
-  [QDTrack](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/qdtrack) (CVPR 2021)
-  [ByteTrack](https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/bytetrack) (arXiv 2021)

Supported methods of single object tracking:

-  [SiameseRPN++](https://github.com/open-mmlab/mmtracking/blob/master/configs/sot/siamese_rpn) (CVPR 2019)
-  [STARK](https://github.com/open-mmlab/mmtracking/blob/master/configs/sot/stark) (ICCV 2021)

Supported methods of video instance segmentation:

-  [MaskTrack R-CNN](https://github.com/open-mmlab/mmtracking/blob/master/configs/vis/masktrack_rcnn) (ICCV 2019)

##### Stark



### 知识蒸馏

#### [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)

***torchdistill*** (formerly *kdkit*) offers various state-of-the-art knowledge distillation methods and enables you to design (new) experiments simply by editing a declarative yaml config file instead of Python code. Even when you need to extract intermediate representations in teacher/student models, you will **NOT** need to reimplement the models, that often change the interface of the forward, but instead specify the module path(s) in the yaml file. Refer to [this paper](https://github.com/yoshitomo-matsubara/torchdistill#citation) for more details.

In addition to knowledge distillation, this framework helps you design and perform general deep learning experiments (**WITHOUT coding**) for reproducible deep learning studies. i.e., it enables you to train models without teachers simply by excluding teacher entries from a declarative yaml config file. You can find such examples below and in [configs/sample/](https://github.com/yoshitomo-matsubara/torchdistill/tree/main/configs/sample/).

#### [repdistiller](https://github.com/HobbitLong/RepDistiller)

**(1) covers the implementation of the following ICLR 2020 paper:**

"Contrastive Representation Distillation" (CRD). [Paper](http://arxiv.org/abs/1910.10699), [Project Page](http://hobbitlong.github.io/CRD/).

![image-20220418124936487](/images/2024-12-30-popularLibrary/image-20220418124936487.png){: .img-fluid}



**(2) benchmarks 12 state-of-the-art knowledge distillation methods in PyTorch, including:**

(KD) - Distilling the Knowledge in a Neural Network
(FitNet) - Fitnets: hints for thin deep nets
(AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
(SP) - Similarity-Preserving Knowledge Distillation
(CC) - Correlation Congruence for Knowledge Distillation
(VID) - Variational Information Distillation for Knowledge Transfer
(RKD) - Relational Knowledge Distillation
(PKT) - Probabilistic Knowledge Transfer for deep representation learning
(AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
(FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer
(FSP) - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
(NST) - Like what you like: knowledge distill via neuron selectivity transfer

### 自蒸馏 Self Knowledge Distillation

[Self-Knowledge-Distillation-Lib](https://github.com/winycg/Self-Knowledge-Distillation-Lib)

Comparison of Self-KD methods on ResNet-50

| Method        | Venue     | Accuracy(%) |
| ------------- | --------- | ----------- |
| Cross-entropy | -         | 77.79       |
| DDGSD [1]     | AAAI-2019 | 81.73       |
| DKS [2]       | CVPR-2019 | 80.75       |
| SAD [3]       | ICCV-2019 | 78.33       |
| BYOT [4]      | ICCV-2019 | 79.76       |
| Tf-KD-reg [5] | CVPR-2020 | 79.84       |
| CS-KD [6]     | CVPR-2020 | 79.99       |
| FRSKD [7]     | CVPR-2021 | 80.51       |

Reference

[1] DDGSD: Data-Distortion Guided Self-Distillation for Deep Neural Networks. AAAI-2019

[2] DKS: Deeply-supervised Knowledge Synergy. CVPR-2019.

[3] SAD: Learning Lightweight Lane Detection CNNs by Self Attention Distillation. ICCV-2019.

[4] BYOT: Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. ICCV-2019

[5] Tf-KD-reg: Revisiting Knowledge Distillation via Label Smoothing Regularization. CVPR-2020.

[6] CS-KD: Regularizing Class-wise Predictions via Self-knowledge Distillation. CVPR-2020.

[7] FRSKD: Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation. CVPR-2021.

Comparison of advanced regularization methods Self-KD methods on ResNet-50

| Method              | Venue               | Accuracy |
| ------------------- | ------------------- | -------- |
| Cross-entropy       | -                   | 77.79    |
| Label Smoothing [1] | CVPR-2016           | 80.33    |
| Virtual Softmax [2] | NeurIPS-2018        | 79.68    |
| Focal Loss [3]      | ICCV-2017           | 79.31    |
| Maximum Entropy [4] | ICLR Workshops 2017 | 78.11    |
| Cutout [5]          | ArXiv.2017          | 80.42    |
| Random Erase [6]    | AAAI-2020           | 80.64    |
| Mixup [7]           | ICLR-2018           | 81.39    |
| CutMix [8]          | ICCV-2019           | 82.47    |
| AutoAugment [9]     | CVPR-2019           | 81.41    |

[1] Label Smoothing: Rethinking the inception architecture for computer vision. CVPR-2016.

[2] Virtual Softmax: Virtual class enhanced discriminative embedding learning. NeurIPS-2018.

[3] Focal Loss: Focal loss for dense object detection. ICCV-2017.

[4] Maximum Entropy: Regularizing neural networks by penalizing confident output distributions. ICLR Workshops 2017.

[5] Cutout: Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552

[6] Random Erase: Random erasing data augmentation. AAAI-2020.

[7] Mixup: mixup: Beyond empirical risk minimization. ICLR-2018.

[8] CutMix: Cutmix: Regularization strategy to train strong classifiers with localizable features. ICCV-2019.

[9] Autoaugment: Autoaugment: Learning augmentation strategies from data. CVPR-2019.


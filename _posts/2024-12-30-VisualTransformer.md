---
layout: post
title: "Visualtransformer"
date: 2024-12-30 00:19:31 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---


## Self-attention

### single head

![img](/images/2024-12-30-VisualTransformer/v2-b081f7cbc5ecd2471567426e696bde15_720w.jpg){: .img-fluid}

**query, key, value**

![img](/images/2024-12-30-VisualTransformer/v2-6cc342a83d25ac76b767b5bbf27d9d6e_720w.jpg){: .img-fluid}

**query 和所有的 key（包含自己的）相关**

### multi-head

![img](/images/2024-12-30-VisualTransformer/v2-688516477ad57f01a4abe5fd1a36e510_720w.jpg){: .img-fluid}

**可以看到多头情况下，不同套的头是独立求信息$$b^{i,head}$$**的，当然多头的信息接下来会融合在一起，求一个总信息$$b$$。

### Positional Encoding

![img](/images/2024-12-30-VisualTransformer/v2-b8886621fc841085300f5bb21de26f0e_720w.jpg){: .img-fluid}

**这个与位置编码乘起来的矩阵** $$W^P$$ 是手工设计的。

## ViT

这个工作是谷歌的，一般ViT狭义指这个工作。

![img](/images/2024-12-30-VisualTransformer/v2-7439a17c2e9aa981c95d783a93cb8729_720w.jpg){: .img-fluid}

代码读了，没什么问题。

## Visual Transformers

![img](/images/2024-12-30-VisualTransformer/v2-2fb2c139742de4d4cde6391c560ef62d_720w.jpg){: .img-fluid}

不再将图像划分成patch，而是通过 spatial attention的机制从图像中提取 Tokens。

根据应用模型的后端会分为两种：

1. 分类或其他和语义有关的，不需要 projector，直接从 CLS token 通过 MLP 进行分类。
2. 分割或需要 pixel to pixel 的，需要 projector。

![Filter-based Tokenizer](/images/2024-12-30-VisualTransformer/v2-618f6d18ece8f83bf7b697ecc3f9da66_720w.jpg){: .img-fluid}
$$
T={\underbrace{\rm{softmax}_{HW}(XW_A)}_{A\in R^{HW\times L}}}^T X=A^TX
$$
共有L个tokens，每个token相当于一个语义。token 长度为 CT（图上原为C，但参考代码里为CT。token的长度不一定要和feature map 一样。）

token 是通过 spatial attention 获得的。所以共有L个SA，每个SA是$$H\times W$$的。

每一个SA，是 X 通过 $$1\times1$$卷积，然后在 HW 上 softmax 得到的。

## DETR

### 整体结构

![img](/images/2024-12-30-VisualTransformer/v2-3d43474df51c545ad6bafc19b3c8ccc3_720w.jpg){: .img-fluid}

1. backbone 提取图像特征，然后进行序列化。序列化是以 $$H\times W$$（有下采样）的顺序作为序列化，也就是有这么多个tokens。
2. 同样 positional encoding 的尺寸也是 $$H\times W$$的。
3. encoder 好像没怎么改动。
4. decoder 引入了 object query，它的作用类似于 positional encoding。object query 是学习得到的。
5. decoder 的tokens 个数是由object query 的序列个数所决定的，和encoder token个数不一样。object query 是用来是查询物体的 slot。也就是说，有 $$N$$ 个 slots，就可以检测到 $$N$$ 个物体。经过 decoder，$$N$$ 个token 分别经过 FFN，得到 $$N$$ 个预测。

[facebook 关于 object query 的回答](https://github.com/facebookresearch/detr/issues/178)

> You can think of the object queries as slots that the model can use to make its predictions, and it turns out experimentally that it will tend to reuse a given slot to predict objects in a given area of the image. Note that we use the terminology "position embedding" that is borrowed from the NLP literature, but in the decoder there is nothing that is "positional" since everything is a set hence permutation equivariant.
>
> We did some analysis, but we didn't see any clear specialization for the classes wrt object queries.

可以看到，object query 和图像区域有关，而和物体类别没有清晰的关系。

### bipartite matching loss

一张图像的物体个数一般远小于 $$N$$，在GT之后补 $$\varnothing$$ ，凑够  $$N$$ 个。则N个GT和N个query之间， 使用匈牙利算法需找一个最佳匹配（no replacement）。令第i个GT $$y_i=(c_i,b_i)$$，它的匹配为第 $$\sigma(i)$$ 个预测 $$\hat{y}_{\sigma(i)}$$，则所有GT的最佳匹配为
$$
\hat{\sigma}=\arg \min_{\sigma \in }\sum_i^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})
$$
其中，
$$
\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})=-1_{c_i\ne \varnothing}\hat{p}_{\sigma (i)}(c_i)+1_{c_i\ne \varnothing}\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})
$$
若求得的最佳匹配为 $$\hat{\sigma }$$，接下来需要计算Loss，以进行梯度传播。这个Loss和上面定义稍有不同，
$$
\mathcal{L}_{Hungarian}(y_i, \hat{y})=\sum_{i=1}^N \left[ -\log \hat{p}_{\hat{\sigma} (i)}(c_i)+1_{c_i\ne \varnothing}\mathcal{L}_{box}(b_i,\hat{b}_{\hat{\sigma}(i)}) \right]
$$
即空集也是要考虑的。另外实际中，

> In practice, we down-weight the log-probability term when ci = ∅ by a factor 10 to account for class imbalance.

### 思考

Loss 很灵活，可以先进行最优化匹配，然后根据匹配后的结构进行Loss计算。

## IPT

**Pre-Trained Image Processing Transformer**

![img](/images/2024-12-30-VisualTransformer/v2-c31e196c02fc6d83325921b7dd9c7716_1440w.jpg){: .img-fluid}



以Transformer为核心，配上不同的 Heads 和 Tails ，以完成相对应的底层视觉任务。这个工作所提出的IPT就是一个基于Transformer的，适用于3种Low-Level Computer Vision Task的预训练模型。要使模型同时与这3种任务兼容，就需要3个 Heads 和 Tails，加上一个共享的 Body 。

1. 有 $$N_t$$ 个任务，需要 $$N_t$$ 个 heads 和 tails。在训练时，body 是公用的，heads 和 tails 是各自任务训练的。
2. heads 没什么特殊的。
3. tail 关键有个 task-specific embeddings $$E_t^i\in R^{P^2 \times C},i\in\{1,...,N_t\}$$，这些是学习得到的，每一个任务有一个，会加到解码器的输入上。应该是所有patch加上去的这个 $$E_t$$ 是一样的。由于训练的时候，以及推断的时候，假设知道了要执行的任务，所以知道要加上哪一个 $$E_t$$。

解码过程：

<img class="img-fluid" src="/images/2024-12-30-VisualTransformer/image-20210924093242385.png" alt="image-20210924093242385" style="zoom:67%;" />

---
layout: post
title: "Diffusion"
date: 2025-12-25 21:42:24 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left # or right
typora-root-url: ../
---

## 参考资料

### 论文和博客

[1] 2022 Understanding Diffusion Models A Unified Perspective.pdf

[2] What are Diffusion Models Lilian Weng.pdf

[3] [blog - Flow With What You Know](https://drscotthawley.github.io/blog/posts/FlowModels.html)

### toy example

- [GitHub - varun-ml/diffusion-models-tutorial: Experiment with diffusion models that you can run on your local jupyter instances](https://github.com/varun-ml/diffusion-models-tutorial?tab=readme-ov-file)
- Classifier-Free-Guidance (CFG) https://github.com/dome272/Diffusion-Models-pytorch
- Classifier-Free-Guidance (CFG) forked from 上面的链接，增加了一些功能和一个博客
  - https://github.com/tcapelle/Diffusion-Models-pytorch
  - https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1
- [GitHub - dome272/Diffusion-Models-pytorch: Pytorch implementation of Diffusion Models (https://arxiv.org/pdf/2006.11239.pdf)](https://github.com/dome272/Diffusion-Models-pytorch)
- [GitHub - lucidrains/denoising-diffusion-pytorch: Implementation of Denoising Diffusion Probabilistic Model in Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file)



## Variational Diffusion Models

### Markovian Hierarchical Variational Autoencoder

按照 [1] 的思路，首先介绍 Markovian Hierarchical Variational Autoencoder (MHVA)

![image-20250826145311374](/images/2025-12-25-diffusion/image-20250826145311374.png){: .img-fluid}

从左往右是编码，从右往左是解码。MHVA 规定，解码时，满足马尔可夫性，也就是生成 $$z_t$$  只依赖于 $$z_{t+1}$$。

编码也是马尔可夫的，这个应该是默认的。因此有
$$
p(x,z_{1:T})=p(z_T)p_\theta (x\mid z_1 )\prod_{t=2}^Tp_\theta (z_{t-1}\mid z_t) \tag{1}
$$

$$
q_\phi (z_{1:T}\mid x)=q_\phi(z_1\mid x)\prod_{t=2}^T q_\phi (z_{t}\mid z_{t-1}) \tag{2}
$$

### Variational Diffusion Models 可以认为是MHVA再满足三个约束

1. 隐藏变量和数据 x 维度一样
2. 编码器的结构不是学习得到的，而是设定为高斯
3. 设置编码器的高斯参数随时间而变化，使得T时刻 $$z_T$$ 为标准高斯 $$N(0,I)$$。

![image-20250826151937355](/images/2025-12-25-diffusion/image-20250826151937355.png){: .img-fluid}

由于 z 和 x 维度一样，从这儿开始，如上图所示，统一用 x 表示。

接下来，就是给出满足上述三个约束的 $$q(x_{t}\mid x_{t-1})$$ 的参数了。然后推导得到约束3。这个推导参考文献[2]比较详细。这儿不再赘述。
$$
q(x_{t}\mid x_{t-1})=N(x_t\mid \sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I )
$$
在推导中，还有参数 $$\beta_t$$, $$\bar{\alpha}_t$$。这几个参数可以相互转化。

上式用**参数化**的技巧，就是
$$
x_{t}=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_{t-1} , \epsilon_{t-1}\sim N(0,I)
$$
注意，参数 $$\alpha_t$$ (or $$\beta_t$$ or 其他变体) 的设置，可以按照预先规定的某种函数（scheduler），或者也可以学习得到。

其次，经过推导，还有一个重要的性质
$$
x_{t}=\sqrt{\bar{\alpha_t}}x_{0}+\sqrt{1-\bar{\alpha_t}}\epsilon , \epsilon\sim N(0,I)
$$
也就是说，按编码的流程要采样得到 $$x_{t}$$，可以根据上式一步采样，不需要先采样 $$x_{1},x_{2}...x_{t-1}$$。这一点在训练中是有用的。

### 似然最大化，ELBO

接下来，就是要通过最大似然来学习解码器 $$p_\theta (x_{t-1}\mid x_t)$$ 了。具体来说，有满足某种分布的$$x$$数据集（这个 $$x$$ 就是上面的 $$x_0$$），要最大化似然
$$
\log p(x)=\log \int p(x_{0:T})dx_{1:T}
$$
> 这边有个疑惑，就是在后续推导中，有用到 $$q(x_{1:T}\mid x_0)$$。如果先进行编码，再进行解码，岂不是有两个不同的 $$x_1,x_2,...x_{T-1}$$。正确理解应该是，$$x_{1:T}$$ 是被 marginalized  out 的，所以考虑了所有可能，但每一种可能下，都只有一个值，不会有两个值。不管是p还是q，都是概率模型，所以任意值下都是可计算概率的。可以认为，先进行编码，得到了$$x_{1:T}$$，然后计算在当前模型参数下$$p_\theta (x_{t-1}\mid x_t)$$的似然，及其梯度。

$$p_\theta (x_{t-1}\mid x_t)$$可以设为任意某种分布，只要知道$$z_{t-1},z_t$$（由编码生成）就可以计算概率$$p_\theta (x_{t-1}\mid x_t)$$。但是根据大段的推导，$$p_\theta (x_{t-1}\mid x_t)$$ 应尽量趋近后验概率 $$q(x_{t-1}\mid x_t,x_0)$$。

 推导参见参考文献。

$$\log p(x)=\log \int p(x_{0:T})dx_{1:T}>ELBO$$，ELBO由三项组成

![image-20250826210812707](/images/2025-12-25-diffusion/image-20250826210812707.png){: .img-fluid}

其中，

1. 第一项类似于VAE中的ELBO，可以通过蒙特卡洛估计来近似和优化。但在我看的代码中，这一项并没有使用。
2. 第二项没有需训练的参数，也是等于0.
3. 第三项可以推导出，$$p_\theta (x_{t-1}\mid x_t)$$ 应尽量趋近后验概率 $$q(x_{t-1}\mid x_t,x_0)$$。

关于第一项，在代码实现中没有被使用，AI是这样回答的。

> ![image-20250826211846887](/images/2025-12-25-diffusion/image-20250826211846887.png){: .img-fluid}

现在进一步分析上述第三项。可以推导得到，$$q(x_{t-1}\mid x_t,x_0)$$ 是一个高斯分布。因此 $$p_\theta (x_{t-1}\mid x_t)$$ 也应该是高斯分布，方差和$$q(x_{t-1}\mid x_t,x_0)$$的方差相同（因为方差和$$x_0$$无关，所以可以直接设为相等），均值要尽量接近<img class="img-fluid" src="/images/2025-12-25-diffusion/image-20250826222640680.png" alt="image-20250826222640680" style="zoom:50%;" />

可以看出 $$\mu_q$$ 和 $$x_0$$ 有关，而$$p_\theta (x_{t-1}\mid x_t)$$ 并没有$$x_0$$的信息，所以可以训练一个神经网络 $$\hat{x}_\theta (x_t,t)$$，从$$x_t$$来预测 $$x_0$$

<img class="img-fluid" src="/images/2025-12-25-diffusion/image-20250826223028627.png" alt="image-20250826223028627" style="zoom:50%;" />

由于两个高斯的KL距离是可以直接计算，再经过推导，最大化ELBO就等价于，在每个timestep，神经网络的输出 $$\hat{x}_\theta (x_t,t)$$要和$$x_0$$越符合，越好。即最优解为

<img class="img-fluid" src="/images/2025-12-25-diffusion/image-20250826223326616.png" alt="image-20250826223326616" style="zoom: 50%;" />

上式中的系数，也可以推导等于

<img class="img-fluid" src="/images/2025-12-25-diffusion/image-20250826223502799.png" alt="image-20250826223502799" style="zoom:50%;" />

### 训练过程

训练采用sgd，对于每一个$$x_0$$，只随机取一个$$t$$，采样得到 $$x_t$$（不用迭代，直接采样），然后计算上式，上式即为 loss。

### 推理过程

推理过程则需要执行完整的解码过程。从一个随机向量 $$x_T$$出发，通过神经网络，计算得到 $$\hat{x}_\theta (x_t,t)$$，然后由上面 ，计算得到均值 $$\mu_\theta(x_t,t)$$；计算后验方差（$$p_\theta$$ 的方差和后验概率 $$q(x_{t-1}\mid x_t,x_0)$$ 的方差相等）；采样得到 $$x_{T-1}$$，如此迭代，最后得到 $$x_0$$。

### 简化的Loss

见[2] 中的一些补充

Empirically, Ho et al. (2020) 提出不带上式中系数的Loss

<img class="img-fluid" src="/images/2025-12-25-diffusion/image-20250827094521508.png" alt="image-20250827094521508" style="zoom:50%;" />

其中，$$\epsilon_t$$ 是编码过程采样得到 $$x_t$$ 的标准高斯噪声，$$\epsilon_\theta$$ 是学习的神经网络。大概理解是这样：目标是要让解码的 $$x_{t-1}$$ 分布均值接近编码的 $$x_{t-1}$$ 分布均值，编码的 $$x_{t-1}$$ 分布均值和 $$x_{t}$$ 之间有关系，除了乘性系数外，差值为 $$\epsilon_t$$ 。解码的 $$x_{t-1}$$ 分布均值经过推导式子和编码时大概一样，区别是需要预测一个$$\epsilon_\theta$$。见下图 Algorithm2 Sampling step 4.

![image-20250827095354468](/images/2025-12-25-diffusion/image-20250827095354468.png){: .img-fluid}

#### 训练和推理

![image-20250827143550828](/images/2025-12-25-diffusion/image-20250827143550828.png){: .img-fluid}

### 学习扩散噪声参数

上述，扩散噪声参数是按照scheduler确定的，在实现中，也可以联合学习得到。

## noise-conditioned score networks (NCSN)

注意：本小节引用了AI的回答，其中 $$p$$ 泛指概率，而不是特制反向采样中的概率分布。

### 分数 $$s_\theta(x_t, t)$$

所谓“分数”，指的是概率密度函数的对数关于数据的梯度，即：
$$
s(x) = \nabla_x \log p(x)
$$
其中 $$ p(x) $$ 是数据的概率密度函数，分数 $$ s(x) $$ 描述了数据在概率密度上的局部变化方向和幅度。通过训练一个模型 $$ s_\theta(x) $$ 来逼近真实的分数 $$ \nabla_x \log p(x) $$，可以间接学习数据的概率分布，而无需直接估计 $$ p(x) $$ 本身。

在扩散模型中，分数可以有两种相关定义，具体取决于上下文：

- **边缘分布的分数**：$$ \nabla_{x_t} \log p(x_t) $$，其中 $$ p(x_t) $$ 是时间步 $$ t $$ 的边缘分布，即： $$p(x_t) = \int p(x_t \mid  x_0) p_{\text{data}}(x_0) \, dx_0$$ 这个分数描述了 $$ x_t $$ 在整个数据分布上的概率密度梯度。然而，$$ p(x_t) $$ 通常难以直接计算，因为它涉及对所有可能的 $$ x_0 $$ 积分。
- **条件分布的分数**：$$ \nabla_{x_t} \log p(x_t \mid  x_0) $$，其中 $$ p(x_t \mid  x_0) $$ 是前向过程中给定原始数据 $$ x_0 $$ 的条件分布。根据扩散模型的前向过程： $$p(x_t \mid  x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$ 这个分数的计算更直接，因为它是高斯分布的梯度： $$\nabla_{x_t} \log p(x_t \mid  x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t}$$

在实际的扩散模型训练中（如 DDPM），通常使用 **去噪分数匹配（Denoising Score Matching, DSM）**，训练分数模型 $$ s_\theta(x_t, t) $$ 来逼近 **条件分数** $$ \nabla_{x_t} \log p(x_t \mid  x_0) $$，而不是边缘分数 $$ \nabla_{x_t} \log p(x_t) $$。这是因为条件分数的表达式已知且易于计算，而边缘分数需要复杂的积分。

### 分数如何用在采样中

大体思路如下：根据扩散模型的推导，在反向过程 $$ p_\theta(x_{t-1} \mid  x_t) $$ 的均值需要用到 $$ x_0 $$ 的估计（下面第2点），而$$ x_0 $$ 的估计和分数$$s_\theta(x_t, t)$$有关系（下面第1点）。

1. **分数的可计算性**：

前向过程定义了 $$ p(x_t \mid  x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I) $$，其分数可以通过高斯分布的性质直接计算： $$\nabla_{x_t} \log p(x_t \mid  x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t}$$ 这个分数是已知的，因此可以作为训练目标。

训练好的分数模型 $$ s_\theta(x_t, t) $$ 可以用来估计 $$ x_0 $$（原始数据）的值：
$$
s_\theta(x_t, t) \approx  -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t}
$$
重排后：
$$
x_0 \approx \frac{x_t + (1 - \bar{\alpha}_t) s_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

2. **与反向过程的联系**：

反向过程 $$ p_\theta(x_{t-1} \mid  x_t) $$ 的均值需要用到 $$ x_0 $$ 的估计。分数模型 $$ s_\theta(x_t, t)$$ 提供了从 $$ x_t $$ 估计 $$ x_0 $$ 的方法。

均值可以通过高斯条件分布公式得到：
$$
\mu_q(x_t, x_0) = \sqrt{\alpha_t} x_{t-1} + \frac{(1 - \alpha_t) \sqrt{\bar{\alpha}_{t-1}} x_0}{1 - \bar{\alpha}_t}
$$
将 $$s_\theta(x_t, t)$$ 代入反向过程的均值表达式，DDPM 推导出简化的均值形式：
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} s_\theta(x_t, t) \right)
$$
这个表达式直接用 $$ x_t $$ 和 $$ s_\theta(x_t, t) $$ 表示均值。

以上公示由AI生成，用作思路理解。

### 和上面简化的Loss中 $$\epsilon$$ 的关系

$$\epsilon$$ 是 $$x_t$$ 和它分布均值之间的差。即
$$
\epsilon =x_t -  \sqrt{\bar{\alpha}_t} x_0
$$
$$\epsilon_\theta(x_t,t)$$ 是它的估计。

因此有
$$
s_\theta(x_t, t) \approx  -\frac{\epsilon_\theta(x_t,t)}{1 - \bar{\alpha}_t}
$$


## Classifier Guided Diffusion

分类器引导扩散中，有一个条件 $$y$$，使用了条件分数 $$ \nabla_{x_t} \log p(x_t \mid  y) $$ 来调整无条件扩散模型的采样过程，符合分数模型的核心思想。

**分类器 $$p(y \mid  x_t)$$ 和无条件扩散模型 $$s_\theta(x_t, t)$$ 都是预训练的**，然后通过调整反向过程的均值。分类器 $$ p_\phi(y \mid  x_t, t) $$ 是预训练的，在带噪声的图像 $$ x_t \sim q(x_t \mid  x_0) $$ 上训练，以预测类别 $$ y $$。这确保分类器梯度在不同噪声水平下可靠，为条件分数提供准确的引导。

具体来说，论文通过以下方式从分数模型角度解释分类器引导扩散：

- 它将无条件分数 $$ \nabla_{x_t} \log p(x_t) $$（由**预训练**的扩散模型提供）与分类器的梯度  $$ \nabla_{x_t} \log p(y \mid  x_t) $$ 结合，构造条件分数 $$ \nabla_{x_t} \log p(x_t \mid  y) $$。这一步通过贝叶斯准则可以推导。
  $$
  \nabla_{x_t} \log p(x_t \mid  y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y \mid  x_t)
  $$

- 反向过程的均值通过条件分数调整，确保采样生成符合特定类别 $$ y $$ 的样本。

替换为条件分数后，得到：

$$ \tilde{\mu}_t(x_t, y) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \nabla_{x_t} \log p(x_t \mid  y) \right) $$
$$
\tilde{\mu}_t(x_t, y) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \left( s_\theta(x_t, t) + s \cdot \nabla_{x_t} \log p_\phi(y \mid  x_t) \right) \right)
$$
这里的 $$ s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t) $$，而 $$ \nabla_{x_t} \log p_\phi(y \mid  x_t) $$ 是分类器梯度，整体形成条件分数 $$ \nabla_{x_t} \log p(x_t \mid  y) $$。

> “To perform conditional sampling, we train a classifier $$ p_\phi(y \mid  x_t) $$ on noisy images to predict the class label $$ y $$, and use its gradient $$ \nabla_{x_t} \log p_\phi(y \mid  x_t) $$ to guide the diffusion process. Specifically, we modify the reverse process mean as follows:
> $$
> \tilde{\mu}_t(x_t, y) = \mu_t(x_t) + s \cdot \sigma_t^2 \cdot \nabla_{x_t} \log p_\phi(y \mid  x_t)
> $$
> where $$ \mu_t(x_t) $$ is the original mean of the reverse process, $$ \sigma_t^2 $$ is the variance, and $$ s $$ is a scale factor controlling the strength of the classifier guidance.”

这边分类器梯度前面有正负不一致。先不管。

## Classifier-Free Guidance

**论文中的符号和其他文献不同，$$\lambda $$ 是时间 $$t$$，但是采样时 $$\lambda$$ 从小到大。$$z_\lambda$$ 对应 $$x_t$$。**

CFG 的核心是联合训练一个支持条件和无条件的扩散模型，然后在采样时通过线性组合分数函数来实现引导。论文将扩散模型表述在连续时间框架下，$$ \lambda $$ 被定义为前向过程的噪声水平，$$\lambda=\log \alpha_\lambda^2/\sigma_\lambda^2$$ 即 log SNR。其中前向过程为 $$ q(z_\lambda \mid  x) = \mathcal{N}(\alpha_\lambda x, \sigma_\lambda^2 I) $$，$$ \alpha_\lambda^2 = 1 / (1 + e^{-\lambda}) $$，$$ \sigma_\lambda^2 = 1 - \alpha_\lambda^2 $$。

### 训练过程（Algorithm 1 in the paper）：

1. 使用单个神经网络参数化分数估计器 $$ \epsilon_\theta(z_\lambda, c) $$（条件模型）和 $$ \epsilon_\theta(z_\lambda) $$（无条件模型）。

2.  在训练时，以概率 $$ p_{\text{uncond}} $$（超参数，通常为 0.1-0.2）随机将条件信息 $$ c $$ 设置为无条件标识符 $$ \emptyset $$，从而联合训练两个模型。
   训练目标是去噪分数匹配（Denoising Score Matching）：$$ \mathbb{E}_{\epsilon, \lambda} [ \\mid  \epsilon_\theta(z_\lambda) - \epsilon \\mid _2^2 ] $$，其中 $$ z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon $$，$$ \epsilon \sim \mathcal{N}(0, I) $$。

3. 论文描述：“We jointly train the unconditional and conditional models simply by randomly setting $$ c $$ to the unconditional class identifier $$ \emptyset $$ with some probability $$ p_{\text{uncond}} $$, set as a hyperparameter.”

   ![image-20250930110741671](/images/2025-12-25-diffusion/image-20250930110741671.png){: .img-fluid}

### 采样过程（Algorithm 2 in the paper）：

1. 从纯噪声 $$ z_{\lambda_T} \sim \mathcal{N}(0, I) $$ 开始，逐步去噪。
2. 在每个时间步 $$ t $$，计算引导分数：$$ \tilde{\epsilon}_\theta(z_\lambda, c) = (1 + w) \epsilon_\theta(z_\lambda, c) - w \epsilon_\theta(z_\lambda) $$，其中 $$ w $$ 是引导强度（guidance strength，通常 $$ w > 0 $$）。
3. 使用这个引导分数来更新样本，例如在 DDPM 采样中替换原分数函数。
   论文解释：“Form the classifier-free guided score at log SNR $$ \lambda_t $$: $$ \tilde{\epsilon}_t = (1 + w) \epsilon_\theta(z_\lambda, c) - w \epsilon_\theta(z_\lambda) $$.”

直观上，$$ w = 0 $$ 时退化为标准条件采样；$$ w > 0 $$ 时，增强条件分数并减弱无条件分数，从而提高样本质量（更符合条件 $$ c $$），但降低多样性。

![image-20250930110925539](/images/2025-12-25-diffusion/image-20250930110925539.png){: .img-fluid}

该算法中，step 3的解释如下，见[2]

![image-20250930111119169](/images/2025-12-25-diffusion/image-20250930111119169.png){: .img-fluid}

step 4 中，$$\tilde{x}_t$$ 是当前时间步的去噪数据估计，基于引导分数 $$\tilde{\epsilon}_t$$。它表示在条件 $$ c $$ 引导下，模型预测的原始数据 $$ x $$。

step 5中，计算均值 $$\tilde{\mu}_t$$:
$$\tilde{\mu}_t = \alpha_{\lambda_{t+1}} \tilde{x}_t + \sigma_{\lambda_{t+1}} \frac{\alpha_{\lambda_t} \tilde{x}_t - z_t}{\sigma_{\lambda_t}}$$

意义：$$\tilde{\mu}_t$$ 是条件反向分布 $$ p(z_{\lambda_{t+1}} \mid  z_{\lambda_t}, c) = \mathcal{N}(z_{\lambda_{t+1}}; \tilde{\mu}_t, \sigma_{\lambda_{t+1}}^2 I) $$ 的均值，引导采样朝符合条件 $$ c $$ 的方向移动。
作为 $$ z_t $$ 和 $$\tilde{x}_t$$ 的函数：公式明确显示 $$\tilde{\mu}_t$$ 依赖于当前样本 $$ z_t $$ 和引导预测器 $$\tilde{x}_t$$。其中：

$$ \alpha_{\lambda_{t+1}} \tilde{x}_t $$: 缩放后的去噪估计，代表信号部分。
$$ \sigma_{\lambda_{t+1}} \frac{\alpha_{\lambda_t} \tilde{x}_t - z_t}{\sigma_{\lambda_t}} $$: 噪声调整项，通过 $$\tilde{x}_t$$ 和 $$ z_t $$ 的差值引入引导分数的影响。

### 简化采样

实际编程中，没有按algorithm2这么复杂。还是按 DDPM 简化Loss的采样过程，不过把 $$\epsilon$$ 替换成引导的 $$\tilde{\epsilon}$$ 。

## DDIM

应该可以肯定的是，DDIM和DDPM的训练过程是一样的，采样过程不一样。

但在理论上，forward process（本论文称为 inference）不一样。DDIM引入了非马尔可夫forward process。

### NON-MARKOVIAN FORWARD PROCESSES

前向过程是一个随机过程，用联合概率密度分布表示。由于在DDPM中，训练过程是在每一个 $$t$$，让神经网络根据 $$x_t$$ 去预测分数$$ \nabla_{x_t} \log p(x_t \mid  x_0) $$ （等价于预测 $$x_0$$，或 $$\epsilon(x_t,t)$$）。这个训练过程只和 $$q(x_t\mid x_0)$$ 有关，和联合概率密度分布没有关系，实际上有无数联合概率密度分布，可以获得这个边际分布。DDPM定义了其中的具有马尔可夫性的一种。

DDIM论文直接设计另一种联合概率分布。给定一个实数向量 $$\sigma \in R^T_{\ge 0}$$，定义联合概率密度分布：

![image-20251003144728914](/images/2025-12-25-diffusion/image-20251003144728914.png){: .img-fluid}

论文证明，上面的这个联合概率分布，满足
$$
q_\sigma(x_t\mid x_0)=N(\sqrt{\alpha_t}x_0,(1-\alpha_t)I)
$$
也就是和DDPM中相同。所以训练过程和 DDPM 一样。

这个联合概率分布的设计中，直接给出了后向概率分布 $$q_\sigma(x_{t-1}\mid x_t,x_0)$$，这样在采样时，就可以直接按照这个概率分布进行采样。

> The magnitude of $$\sigma$$ controls the how stochastic the forward process is; when $$\sigma=0$$, we reach an extreme case where as long as we observe x0 and xt for some t, then $$x_{t-1}$$ become known and fixed.

在上面的讨论中，并没有给出潜在的前向过程，也即是如何从 $$x_0$$ 逐步推理得到 $$x_T$$。论文说这个过程可以由贝叶斯公式推导得到 $$q_\sigma(x_{t}\mid x_{t-1},x_0)$$，它是一个高斯分布。由于有依赖 $$x_0$$，它不是一个马尔可夫过程。熟悉DDPM的话，可以知道，在训练时，实际上并不需要去执行这个前向过程。

### SAMPLING FROM GENERALIZED GENERATIVE PROCESSES

前面说过，生成过程可以直接借助 $$q_\sigma(x_{t-1}\mid x_t,x_0)$$

> Intuitively, given a noisy observation $$x_t$$, we first make a prediction of the corresponding $$x_0$$, and then use it to obtain a sample $$x_{t-1}$$ through the reverse conditional distribution $$q_\sigma(x_{t-1}\mid x_t,x_0)$$, which we have defined.

如果我们采用simple loss预测 $$\epsilon$$，则

![image-20251003153043856](/images/2025-12-25-diffusion/image-20251003153043856.png){: .img-fluid}

上文中，$$f_\theta^t$$ 就是对 $$x_0$$ 的预测。把式（9）中$$f_\theta^t$$代入式(10)，可得

![image-20251003153420286](/images/2025-12-25-diffusion/image-20251003153420286.png){: .img-fluid}

上式就是采样过程。其中$$\epsilon_\theta^{(t)}$$ 是神经网络的输出。某一个特定的 $$\alpha$$，则退回成DDPM。

而 $$\alpha_t=0$$，即为 DDIM。

### 加速采样过程

上述讨论已经定义了DDIM，但还没涉及到加速采样。加速采样，也有类似的理论，但是训练过程不变，因此也要满足 marginal 要和 DDPM 一样。

假设我们已经选择了时间的子集 $$\tau$$，$$\tau_S=T$$；定义联合概率密度

![image-20251003154717877](/images/2025-12-25-diffusion/image-20251003154717877.png){: .img-fluid}

也就是说， $$\tau$$ 内的时间满足上面的非马尔可夫inference过程，其他时间是一个星形的概率图结构。由于marginal 要和 DDPM 一样，所以训练不变。

在生成时，只用到了 $$\tau$$ 内的时间，其他时间就不管。


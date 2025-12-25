---
layout: post
title: "Montecarlo"
date: 2024-12-30 23:43:33 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left # or right
typora-root-url: ../
---

---
title: Monte Carlo
date: 2022-10-24 10:00:00
math: true
tags:
---

## 蒙特卡罗及其相关算法

参考 [Machine Learning](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA)

### 蒙特卡罗

Monte Carlo 用来近似随机变量的函数 $$f(x)$$ 的期望，做法是用样本的平均近似统计的平均。
$$
E[f(x)],x\sim p(x)\leftarrow \hat{\mu}_n = \frac{1}{n}\sum f(x_i), \textrm{if }x_1,\cdots,x_n\sim p(x)\quad iid
$$
这边假设了

1. 能够从 $$p(x)$$ 方便地采样
2. 能够计算 $$f(x)$$

> Monte Carlo 可以用来近似定积分，只要把定积分看成是均匀分布下积分子的期望

蒙特卡洛近似有以下性质

1. 无偏的，$$E[\hat{\mu}_n]=E[f(x)]$$ 
2. 一致的，$$\hat{\mu}_n \rightarrow E[f(x)],\textrm{ as }n\rightarrow \infty$$
3. 标准差以 $$1/\sqrt{n}$$ 速率收敛。 $$\sigma^2[\hat{\mu}_n]\rightarrow 0,\textrm{ as }n\rightarrow 0$$

$$
\sigma^2[\hat{\mu}_n]=\frac{1}{n}\sigma^2[f(x)]
$$

4. $$MSE = Bias^2 + Variance=\frac{1}{n}\sigma^2[f(x)]$$，当 $$n\rightarrow 0$$ 时也收敛到0

> Estimator 的一些性质：
>
> 定义：
>
> 1. 待估计的变量 $$\theta $$，采用 $$\hat{\theta}(x)$$ 进行估计
> 2. Bias: $$ B(\hat{\theta})=E(\hat{\theta})-\theta $$
> 3. Variance: $$ \sigma^2(\hat{\theta})=E[(\hat{\theta}-E(\theta))^2] $$，注意这儿是估计子和其期望之差的平方
> 4. MSE: $$MSE(\hat{\theta})=E[(\hat{\theta}-\theta)^2]$$，注意这儿是估计子和真值之差的平方
> 5. $$MSE=Bias^2+Var$$
>
> MSE 和方差的关系，
>
>  ![mse_variance](/images/2024-12-30-MonteCarlo/mse_variance.png){: .img-fluid}

### Importance Sampling，重要性采样

名字虽然是采样，但它还是一种估计子。

一般情况下，无法从 $$p(x)$$ 进行采样。假设可以从 $$q(x)$$ 进行采样，$$s.t.\quad q=0 \Rightarrow p=0$$ （p 非零的地方，q 必须非零，也就是必须能采样到）
$$
E[f(x)]=\frac{1}{n}\sum f(x_i)\frac{p(x_i)}{q(x_i)}=\frac{1}{n}\sum f(x_i)w(x_i)
$$
其中 $$w(x)$$  为 importance weight. 

通过选取 q，能取得比蒙特卡罗更低的 variance。（最佳的 q 应该是  $$f(x)p(x)$$）

<img class="img-fluid" src="/images/2024-12-30-MonteCarlo/importancesampling.png" alt="importancesampling" style="zoom: 50%;" />

> 从上图可知，当 $$q(x)=f(x)p(x)$$ 时，$$\sigma^2(I_n)=0 $$
>
> ![image-20221025145830590](/images/2024-12-30-MonteCarlo/image-20221025145830590.png){: .img-fluid}



#### Importance sampling without normalization

![iswon](/images/2024-12-30-MonteCarlo/iswon-16666810018881.png){: .img-fluid}

上图中，p(x),q(x) 没有归一化，不能计算，但是假设可以从 $$q(x)$$进行采样。

则 IS 中多了一个未知量 $$z_q/z_p$$，幸运的是，这个未知量，也可以用 IS 来估计

![iswon2](/images/2024-12-30-MonteCarlo/iswon2.png){: .img-fluid}

### 采样

不管哪种蒙特卡罗，都需要采样。

#### Smirnov transform (inverse transform sampling)

这个就是数字图像处理里面学过的。

#### Rejection sampling

目标：从一个复杂的 pdf $$f$$ 采样。注意 $$f$$ 可能除归一化系数可知

![rej](/images/2024-12-30-MonteCarlo/rej.png){: .img-fluid}

### Markov Chain Monte Carlo

直观理解，由于维度灾难的原因，当要从高维度空间 pdf $$p$$ 采样时，即使我们有一个看起来很近似的 pdf $$q$$，可能效果也不好。MCMC 构建一个马尔可夫链，每一次只移动一小步，保证尽量呆在高概率的区域。由于马尔可夫链的各态历经性，$$n\rightarrow \infty$$时会覆盖 $$p$$ 的非零区域。

![1](/images/2024-12-30-MonteCarlo/1.png){: .img-fluid}

#### 马尔可夫链的性质

##### 各态历经性

![2](/images/2024-12-30-MonteCarlo/2.png){: .img-fluid}

就是说

1. 当MC满足一系列条件时，$$n\rightarrow \infty$$ 时，样本平均即为期望
2. 如果 MC 非周期，则 $$n\rightarrow \infty$$ 时，MC 采样就是根据平稳分布进行采样

##### Time-homogeneous

转移概率矩阵 $$T$$ 和时间无关

##### 平稳

$$
\pi T=\pi
$$

##### irreducible

##### aperiodic

##### Detailed balance (a.k.a. Reversibility)

![1](/images/2024-12-30-MonteCarlo/1-16667972601191.png){: .img-fluid}

#### Metropolis 算法

它是早期的 MCMC 算法

假设和目标如下。注意 $$\pi$$ 可以是归一化系数不可知

![1](/images/2024-12-30-MonteCarlo/1-16667974132692.png){: .img-fluid}

算法

![1](/images/2024-12-30-MonteCarlo/1-16667977542883.png){: .img-fluid}

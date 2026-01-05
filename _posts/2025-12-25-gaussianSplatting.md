---
layout: post
title: "NeRF and Gaussian Splatting"
date: 2025-12-25 20:57:46 +0800
categories: []
description: Gaussian Splatting, NeRF, Alpha-blending, Point-Based Rendering, Jacobian 
tags: cv, math
thumbnail: 
toc:
  sidebar: left # or right
typora-root-url: ../
---

参考资料：

[1] https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/

[2] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, "3D Gaussian splatting for real-time radiance field rendering," *ACM Transactions on Graphics (TOG)*, vol. 42, no. 4, pp. 1-14, 2023.

代码：

[1] [GitHub - yzslab/gaussian-splatting-lightning: A 3D Gaussian Splatting framework with various derived algorithms and an interactive web viewer](https://github.com/yzslab/gaussian-splatting-lightning)

[2] [GitHub - nerfstudio-project/gsplat: CUDA accelerated rasterization of gaussian splatting](https://github.com/nerfstudio-project/gsplat): gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings.

## Point-Based Rendering，Radiance Fields，Gaussian Splatting

### 1. 技术演进关系：从“连续场”到“离散点”，到“球”

- **NeRF (2020):** 采用的是**体渲染 (Volume Rendering)**。它像是在计算光线穿过一片“连续雾气”时的颜色积累。虽然效果好，但每渲染一个像素都要采样数百个点并运行神经网络，速度极慢。
- **Point-based Rendering (早期):** 传统的基于点的渲染通常只处理离散的像素点。只用离散点有空洞、aliasing 等问题。关于高质量**基于点的渲染（Point-based Rendering）**的开创性工作通过‘**喷溅（Splatting）**’具有大于像素范围的点基元（如圆盘、椭圆盘、椭球体或面元 Surfel）解决了这些问题。
- **3D Gaussian Splatting (2023):** 它借鉴了 NeRF 的**辐射场（Radiance Field）**概念，但抛弃了昂贵的体渲染采样。它改用一种名为 **Point-based Alpha-blending** 的方案，将场景表示为数百万个“3D 高斯椭球体”。

------

### 2. 核心数学联系：Alpha-blending 是共同语言

无论是 NeRF 还是 3DGS，最终合成像素颜色的数学公式几乎是一样的。对于光线上的 $$N$$ 个点，其颜色 $$C$$ 的计算公式通常表示为：

$$C = \sum_{i=1}^{N} T_i \alpha_i c_i$$

其中：

- $$c_i$$ 是第 $$i$$ 个点的颜色。
- $$\alpha_i$$ 是该点的不透明度（Opacity）。
- $$T_i$$ 是累积透射率（前面所有点遮挡后剩下的光强）。

## Alpha

在计算机图形学（包括 NeRF 和 3D Gaussian Splatting）中，$$\alpha$$ 通常被定义为**不透明度（Opacity）**。

虽然我们常说 $$\alpha$$ 代表“透明度”，但从数学定义和计算逻辑上来看，它衡量的是物体“有多实”或者说“遮挡光线的能力”。

### 1. 核心定义

- **$$\alpha = 1$$：** 完全**不透明**（Opaque）。光线完全无法穿过，背景完全被遮挡。
- **$$\alpha = 0$$：** 完全**透明**（Transparent）。光线无阻碍穿过，物体完全不可见。
- **中间值：** 半透明。例如 $$\alpha = 0.5$$ 表示该点吸收/贡献了 50% 的光，剩下的 50% 穿过去看背景。

> 数学公式中的直观体现：
>
> 在混合公式 $$C = \alpha \cdot C_{src} + (1 - \alpha) \cdot C_{bg}$$ 中：
>
> - $$\alpha$$ 是**源颜色**（物体本身）的权重。
> - $$(1 - \alpha)$$ 才是**背景颜色**透过来的比例（即真正的透明度 $$Transparency$$）。

------

### 2. 在 3D Gaussian Splatting (3DGS) 中的含义

在 3DGS 中，每个高斯球都有一个学习到的 $$\alpha$$ 值。这个值在渲染时起到了两个关键作用：

1. **贡献颜色：** 它决定了这个高斯点对最终像素颜色贡献了多少。
2. **遮挡（Occlusion）：** 它决定了光线在穿过这个点后，还有多少能量能够到达后面的点。
   - 如果前面的点 $$\alpha=1$$，光线就被“吸干”了，后面的点即使再亮也看不见。这种累积的效果在论文中被称为**累积透射率（Accumulated Transmittance）**。

### 3. 名词辨析

虽然在口语中经常混用，但在查阅文献或编写代码时请记住：

- **$$\alpha$$ (Alpha):** 不透明度 (Opacity)。
- **$$1 - \alpha$$:** 透明度 (Transparency / Transmittance)。

**有趣的事实：** 在 3DGS 的训练过程中，如果一个高斯点的 $$\alpha$$ 变得非常小（比如接近 0），算法会认为这个点“没什么用”并将其**销毁（Pruning）**，从而优化显存。

## NeRF

**NeRF**（Neural Radiance Fields，神经辐射场）是 2020 年由 Google Research 团队提出的一种革命性技术，它彻底改变了计算机视觉中**三维重建**和**新视角合成（New View Synthesis）**的方法。

简单来说，NeRF 的核心思想是：**用一个神经网络来存储和表示整个三维场景。**

------

### 1. 核心工作原理

NeRF 将三维场景建模为一个连续的**体积函数**。它的输入和输出非常简洁：

- **输入：** 一个五维向量，包括空间坐标 $$(x, y, z)$$ 和观察视角 $$(\theta, \phi)$$。
- **输出：** 该点处的**颜色** (RGB) 和**体积密度**（Density，即该点有多“实”，决定了光线通过的阻力）。

### 2. 主要渲染流程

1. **光线投射 (Ray Casting)：** 从摄像机发射出一条穿过像素的光线。
2. **采样 (Sampling)：** 在这条光线上选取成百上千个采样点。
3. **神经网络查询：** 将这些点的坐标输入多层感知机（MLP）模型，预测颜色和密度。
4. **体渲染 (Volume Rendering)：** 将光线上所有点的预测值加权融合，计算出最终像素的颜色。

在 NeRF（神经辐射场）中，虽然最终合成图像的步骤也常被通俗地称为 **Alpha-blending**，但其严谨的数学基础是基于物理的**体渲染（Volume Rendering）**模型。

NeRF 使用神经网络预测**体积密度（Volume Density）$$\sigma$$**，并通过数值积分将其转换为我们看到的像素颜色。

------

### 1. 核心物理公式（连续形式）可以不看

当一条光线 $$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$$ 穿过场景时，其最终合成的颜色 $$C(\mathbf{r})$$ 由下式决定：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

其中各分量的含义如下：

- **$$\sigma(\mathbf{r}(t))$$ (Density)：** 体积密度。可以理解为光线穿过该点时被阻挡的概率。$$\sigma$$ 越大，该点越“实”。

- **$$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$$：** 该点在观察方向 $$\mathbf{d}$$ 下发出的颜色。

- $$T(t)$$ (Transmittance)： 累积透射率。表示光线从起点到 $$t$$ 处没被挡住的概率。

  

  $$T(t) = \exp\left( -\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds \right)$$

------

### 2. 离散化与 Alpha-blending 的联系

由于计算机无法直接计算连续积分，NeRF 将光线划分为 $$N$$ 个小区间，并在每个区间内采样。离散化后的求和公式与传统的 Alpha-blending 逻辑高度一致：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \mathbf{c}_i$$

在这个公式中，**$$\alpha_i$$（不透明度）** 是由 **$$\sigma_i$$（密度）** 计算得来的：

$$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$$

> **关键变量解释：**
>
> - **$$\delta_i$$：** 相邻采样点之间的距离。
> - **$$\alpha_i$$：** 这一段区间的“局部不透明度”。如果密度 $$\sigma_i$$ 很大，或者步长 $$\delta_i$$ 很大，那么这一段就越不透明（$$\alpha_i$$ 趋近于 1）。
> - **$$T_i$$：** 到达第 $$i$$ 个点时的剩余光强。公式为 $$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$。

------

### 3. $$\sigma$$ (Density) 与 $$\alpha$$ (Alpha) 的本质区别

虽然它们都描述“物体有多实”，但在 NeRF 中有着重要的物理区别：

| **维度**     | **体积密度 σ (Density)**                   | **不透明度 α (Alpha)**                                   |
| ------------ | ------------------------------------------ | -------------------------------------------------------- |
| **定义域**   | $$[0, +\infty)$$，可以是任意正实数。         | $$[0, 1]$$，标准化的概率范围。                             |
| **物理含义** | 单位长度内物质的衰减系数，**与步长无关**。 | 某一特定厚度区域的遮挡率，**受采样距离 $$\delta$$ 影响**。 |
| **网络输出** | NeRF 神经网络直接预测的值。                | 通过公式计算出的中间变量。                               |

直观理解：

$$\sigma$$ 就像是“雾的浓度”。雾可以非常浓（$$\sigma$$ 很大），但如果你只走了一毫米（$$\delta$$ 极小），这层雾对你视线的遮挡（$$\alpha$$）依然很小；只有浓度和路程的乘积足够大时，这段空间才变得不透明。

## Point-based alpha-blending

### 1. 核心公式

对于屏幕上的某一个像素，其最终颜色 $$C$$ 是通过将覆盖该像素的所有 3D 高斯投影点（按深度由近到远排序）进行叠加得到的：

$$C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

为了方便理解，我们通常将其拆解为三个物理量：

- **$$c_i$$ (Color)：** 第 $$i$$ 个高斯点在该像素位置的颜色。
- **$$\alpha_i$$ (Opacity)：** 第 $$i$$ 个高斯点在该像素位置的**有效不透明度**。
- **$$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$ (Transmittance)：** **累积透射率**。它表示光线从相机出发，穿过前面 $$i-1$$ 个点后，还剩下多少能量能到达第 $$i$$ 个点。

------

### 2. $$\alpha_i$$ 的计算：3D 到 2D 的关键

在 Point-based 渲染中，$$\alpha_i$$ 并不是一个固定值，而是由高斯函数的空间分布决定的。对于第 $$i$$ 个高斯点：

$$\alpha_i = \text{Opacity}_i \cdot \exp \left( -\frac{1}{2} (x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i) \right)$$

- **$$\text{Opacity}_i$$：** 该点在训练中学习到的基础不透明度。
- **指数部分：** 描述了 2D 高斯分布。这意味着当你观察这个点时，中心最实，越往边缘越透明。

### 对比

我们可以清晰地看到，NeRF和Point-based alpha-blending的**成像模型（Image Formation Model）是相同的**。然而，它们的**渲染算法却大相径庭**。

**NeRF** 是一种连续表示法，它隐式地表示空间是空闲还是被占据；为了找到公式 (2) 中的采样点，需要进行昂贵的随机采样，这随之带来了噪声和巨大的计算开销。

相比之下，**点（Points）** 是一种非结构化的离散表示法，它足够灵活，允许几何体的创建、销毁和位移。

## 3D Gaussian Splatting 

3D Gaussian Splatting (3DGS) 的反向传播（back propagation, BP）是其训练核心，用于优化高斯点云的参数（如位置 $$\mu$$, 协方差 $$\Sigma$$, 不透明度 $$\alpha$$, 球谐系数 SH）。与 NeRF 的 MLP 优化不同，3DGS 使用**显式参数**（点云属性），通过可微分渲染（differentiable rendering）计算梯度，结合 PyTorch 的 autograd 实现 BP。以下我详细解释 3DGS 的反向传播机制，结合 **yzslab/gaussian-splatting-lightning** 仓库的代码（基于你的使用场景），从数学到实现，尽量图文并茂（文字描述模拟图形），并回答如何在代码中观察 BP。

---

### 1. **反向传播的总体流程**
3DGS 的训练目标是通过多视图图像（e.g., lego 数据集的 PNG + transforms.json）监督，优化高斯参数，最小化渲染图像与真实图像的损失（如 L1 + SSIM）。BP 过程涉及：

1. **前向传播**：从高斯参数 → 投影 → alpha 混合 → 渲染图像。
2. **损失计算**：渲染图像与 GT（ground truth）图像比较。
3. **反向传播**：通过 autograd 计算梯度，更新高斯参数。
4. **密度控制**：非 BP 部分，定期调整点云（克隆/分裂/剔除）。

**图示（文字模拟）**：
```
[高斯参数: μ, Σ, α, SH] → [投影: 2D means/scales] → [alpha 混合: RGB] → [Loss: L1+SSIM]
    ↑ (梯度流回)            ↑ (梯度流回)             ↑ (梯度流回)         ← [GT 图像]
```

---

### 2. **数学原理：可微分渲染与梯度计算**
3DGS 的渲染管道是可微分的，梯度通过链式法则从损失函数流回高斯参数。以下拆解关键步骤：

##### **2.1 前向传播（渲染）**
- **输入**：N 个高斯，参数为：
  - $$\mu_i \in \mathbb{R}^3$$：中心位置。
  - $$\Sigma_i \in \mathbb{R}^{3 \times 3}$$：协方差矩阵（分解为缩放 $$S_i$$ 和旋转 $$R_i$$）。
  - $$\alpha_i \in [0,1]$$：不透明度（sigmoid 输出，实际上在计算中 $$\sigma (\alpha_i)$$ 是不透明度）。
  - SH 系数：视图相关颜色 $$\mathbf{c}_i(\mathbf{d})$$（球谐函数）。
- **投影（Splatting）**：
  - 每个高斯投影到 2D 图像平面：
    $$
    \mu_{i,2D} = P \cdot V \cdot \mu_i
    $$
    $$
    \Sigma_{i,2D} = J \cdot W \cdot \Sigma_i \cdot W^T \cdot J^T
    $$
    - $$P$$: 相机投影矩阵（内参）。
    - $$V$$: 视图矩阵（外参）。
    - $$J$$: 投影的雅可比矩阵（透视效应）。
    - $$W$$: 视图旋转部分。
  - 2D 高斯权重：$$w_i(u,v) = \alpha_i \cdot \exp(-\frac{1}{2} (\mathbf{p} - \mu_{i,2D})^T \Sigma_{i,2D}^{-1} (\mathbf{p} - \mu_{i,2D}))$$。$$w_i$$ 相当于 alpha-blending中的alpha.
- **Alpha 混合**：
  - 像素颜色：
    $$
    C(\mathbf{p}) = \sum_{i=1}^N \mathbf{c}_i \cdot w_i \cdot T_i, \quad T_i = \prod_{j=1}^{i-1} (1 - w_j)
    $$
    - $$T_i$$: 透射率，模拟光线衰减。
    - 按深度排序（front-to-back）。

##### **2.2 损失函数**
- 比较渲染图像 $$C(\mathbf{p})$$ 与 GT 图像 $$\hat{C}(\mathbf{p})$$：
  $$
  \mathcal{L} = \lambda_1 \cdot \|\hat{C} - C\|_1 + \lambda_2 \cdot (1 - \text{SSIM}(\hat{C}, C))
  $$
  - 典型：$$\lambda_1 = 1, \lambda_2 = 0.2$$（见 `configs/blender.yaml`）。
  
- 损失是标量，驱动梯度计算。

### 投影中 Jacobian的推导

**Jacobian 矩阵 $$J$$** 描述了从 3D 相机坐标 $$(X, Y, Z)$$ 到 2D 图像坐标 $$(x, y)$$ 的透视投影变换的局部线性化过程。

------

#### 1. 透视投影函数

给定相机的焦距 $$f_x, f_y$$ 和光心坐标 $$c_x, c_y$$，3D 空间点 $$P = (X, Y, Z)^T$$ 投影到屏幕上的公式为：

$$x = f_x \frac{X}{Z} + c_x$$

$$y = f_y \frac{Y}{Z} + c_y$$

#### 2. Jacobian 矩阵 $$J$$ 的偏导推导

Jacobian 矩阵是一个 $$2 \times 3$$ 的矩阵，包含了投影函数对 $$X, Y, Z$$ 的一阶偏导数：

$$J = \begin{pmatrix} \frac{\partial x}{\partial X} & \frac{\partial x}{\partial Y} & \frac{\partial x}{\partial Z} \\ \frac{\partial y}{\partial X} & \frac{\partial y}{\partial Y} & \frac{\partial y}{\partial Z} \end{pmatrix}$$

我们可以逐项计算：

- **对于 $$x$$ 坐标：**
  - $$\frac{\partial x}{\partial X} = \frac{f_x}{Z}$$
  - $$\frac{\partial x}{\partial Y} = 0$$
  - $$\frac{\partial x}{\partial Z} = -\frac{f_x X}{Z^2}$$
- **对于 $$y$$ 坐标：**
  - $$\frac{\partial y}{\partial X} = 0$$
  - $$\frac{\partial y}{\partial Y} = \frac{f_y}{Z}$$
  - $$\frac{\partial y}{\partial Z} = -\frac{f_y Y}{Z^2}$$

#### 3. 最终的 Jacobian 矩阵形式

将上述结果组合，得到最终在高斯均值点 $$(X, Y, Z)$$ 处评估的 Jacobian 矩阵：

$$J = \begin{pmatrix} \frac{f_x}{Z} & 0 & -\frac{f_x X}{Z^2} \\ 0 & \frac{f_y}{Z} & -\frac{f_y Y}{Z^2} \end{pmatrix}$$

------

#### 4. 在协方差投影中的应用

在 3D Gaussian Splatting 的渲染管线中，这个矩阵被用来将 3D 相机空间的协方差矩阵 $$\Sigma_{view}$$ 投影到 2D 图像空间 $$\Sigma'$$：

$$\Sigma' = J \Sigma_{view} J^T$$

**物理意义说明：**

- **$$1/Z$$ 项：** 体现了“近大远小”的缩放效果。
- **$$X/Z^2$$ 和 $$Y/Z^2$$ 项：** 体现了当 3D 点偏离光轴（即 $$X$$ 或 $$Y$$ 较大）时，透视投影引起的几何拉伸（各向异性）。这就是为什么屏幕边缘的高斯点往往看起来比中心的更“扁”的原因。

### **SH (Spherical Harmonics，球谐函数)**

在 3D Gaussian Splatting (3DGS) 和 NeRF 中，**View-dependent Colors（视点相关颜色）** 是实现真实感的关键。它能表现出金属的高光、镜面反射以及随着观察角度变化而产生的颜色闪烁。

为了在离散的点云表示中实现这一效果，3DGS 采用了 **SH (Spherical Harmonics，球谐函数)**。

------

#### 1. 为什么需要视点相关颜色？

如果一个物体的颜色是固定的（即 **Lambertian 漫反射表面**），那么无论你从哪个角度看，它都是一样的颜色。但现实世界中：

- **金属、玻璃：** 某些角度有强光。
- **各向异性材质：** 颜色随视角偏转而变化。

传统的点只能存储一个 RGB 值，而 3DGS 为每个高斯点存储了一组 **SH 系数**，使其颜色成为一个关于**观察方向**的函数。

------

#### 2. 什么是球谐函数 (Spherical Harmonics)?

**球谐函数**可以类比为“球面上的傅里叶级数”。

- **傅里叶级数**是用一堆正弦/余弦波来近似任何**周期信号（一维）**。
- **球谐函数**是用一组定义在球面上的一组基函数 $$Y_{lm}(\theta, \phi)$$ 来近似任何**球面函数（二维方向）**。

在 3DGS 中，每个高斯点不仅仅有一个颜色，而是拥有多层“频率系数”。当你改变视角时，算法会根据当前视角方向 $$(\theta, \phi)$$，对这些系数进行加权求和，计算出该视角下的实时 RGB 颜色。

------

#### 3. 数学表达

对于一个高斯点，其在视角方向 $$d$$ 下的颜色 $$C$$ 计算公式为：

$$C(d) = \sum_{l=0}^{k} \sum_{m=-l}^{l} c_{lm} Y_{lm}(d)$$

其中：

- **$$c_{lm}$$：** 这是存储在每个高斯点里的 **SH 系数**（模型要学习的参数）。
- **$$Y_{lm}(d)$$：** 这是预定义的球谐基函数。
- **$$k$$：** SH 的阶数（Degree）。

------

#### 4. SH 阶数的影响

3DGS 通常使用 **3 阶 SH (Degree 3)**：

- **0 阶 (Degree 0):** 只有 1 个系数。代表基础底色（漫反射），无论怎么看颜色都一样。
- **1-3 阶:** 随着阶数增加，模型可以表现出更复杂、更锐利的高光和环境反射效果。

存储代价：

每个高斯点如果使用 3 阶 SH，需要为 R、G、B 三个通道各存储 $$(3+1)^2 = 16$$ 个系数，总共 48 个系数。这就是为什么 3DGS 的模型文件（.ply）通常比 NeRF 的权重文件大的原因。

![Figure 9: A process of obtaining a view-dependant color (red component) of a point with _ℓmax = 2 and 9 learned coefficients. A sigmoid function maps the value into the [0, 1] interval. Oftentimes, clipping is used instead [Source: Image by the author]](/images/2025-12-25-gaussianSplatting/1bKNS_UyAOGcQvew-b-pciQ.png){: .img-fluid data-zoomable=""}

## 投影和alpha混合的可微性

---

是的，在 **3D Gaussian Splatting (3DGS)** 中，**投影（Projection/Splatting）** 和 **Alpha 混合（Alpha Blending）** 两个步骤都是**可微分的**（differentiable）。这是 3DGS 能够通过反向传播（back propagation）优化高斯参数（如位置 $$\mu$$、协方差 $$\Sigma$$、不透明度 $$\alpha$$、球谐系数 SH）的关键原因。以下详细解释两者的可微性，结合数学推导、代码实现（基于 **yzslab/gaussian-splatting-lightning** 仓库），并以图文并茂的方式说明（文字模拟图形），回答你的问题。

---

### 1. **投影（Projection/Splatting）的可微性**
**投影**步骤将 3D 高斯点云投影到 2D 图像平面，生成每个高斯的 2D 均值 $$\mu_{2D}$$、协方差 $$\Sigma_{2D}$$ 和权重 $$w_i$$，用于后续渲染。这个过程是可微的，因为所有变换（矩阵运算、指数函数等）都支持梯度计算。

#### **1.1 数学原理**
- **输入**：
  - 3D 高斯参数：$$\mu_i \in \mathbb{R}^3$$（中心位置）、$$\Sigma_i \in \mathbb{R}^{3 \times 3}$$（协方差，分解为旋转 $$R_i$$ 和缩放 $$S_i$$）、$$\alpha_i \in [0,1]$$（不透明度）。
  - 相机参数：视图矩阵 $$V$$（外参，4x4）、投影矩阵 $$P$$（内参，透视投影）。
- **投影公式**：
  1. **均值投影**：
     $$
     \mu_{i,2D} = P \cdot V \cdot \mu_i
     $$
     - $$V \cdot \mu_i$$：将 3D 位置从世界坐标系变换到相机坐标系。
     - $$P$$：透视投影（e.g., $$P = \begin{bmatrix} f_x/Z & 0 & c_x \\ 0 & f_y/Z & c_y \\ 0 & 0 & 1 \end{bmatrix}$$），输出 2D 像素坐标。
     - **可微性**：矩阵乘法是线性操作，透视除法 ($$/Z$$) 是可微的（雅可比矩阵 $$J$$ 提供导数）。
  2. **协方差投影**：
     $$
     \Sigma_{i,2D} = J \cdot W \cdot \Sigma_i \cdot W^T \cdot J^T
     $$
     - $$W$$：视图矩阵 $$V$$ 的 3x3 旋转部分。
     - $$J$$：投影雅可比矩阵，考虑透视畸变：
       $$
       J = \begin{bmatrix}
       \frac{f_x}{Z} & 0 & -\frac{f_x X}{Z^2} \\
       0 & \frac{f_y}{Z} & -\frac{f_y Y}{Z^2}
       \end{bmatrix}
       $$
     - $$\Sigma_i = R_i S_i S_i^T R_i^T$$：3D 协方差分解。
     - **可微性**：矩阵乘法、转置和 $$J$$ 的计算（涉及除法）都可微，梯度通过链式法则流回 $$\Sigma_i$$（即 $$S_i, R_i$$）。
  3. **高斯权重**：
     $$
     w_i(u,v) = \alpha_i \cdot \exp\left(-\frac{1}{2} (\mathbf{p} - \mu_{i,2D})^T \Sigma_{i,2D}^{-1} (\mathbf{p} - \mu_{i,2D})\right)
     $$
     - $$\alpha_i$$：sigmoid 输出，可微。
     - 指数函数和矩阵逆（$$\Sigma_{i,2D}^{-1}$$）可微（指数导数为自身，矩阵逆导数基于线性代数）。
- **梯度流**：
  - 损失 $$\mathcal{L}$$ 对像素颜色 $$C(\mathbf{p})$$ 的梯度 $$\frac{\partial \mathcal{L}}{\partial C}$$ 流回 $$w_i$$，再通过 $$w_i$$ 流回 $$\mu_{i,2D}, \Sigma_{i,2D}, \alpha_i$$，最终到 $$\mu_i, \Sigma_i, \alpha_i$$。

#### **1.2 代码实现（yzslab）**
- **文件**：`src/render/gs_renderer.py`，调用 `gsplat.rasterize_gaussians()`。
- **关键代码**（简化）：
  ```python
  def project_gaussians(means, scales, rotations, camera):
      means3D = means  # [N, 3]
      view_matrix = camera.view  # [4, 4]
      proj_matrix = camera.proj  # [3, 4]
      means2D = proj_matrix @ (view_matrix @ means3D)  # 矩阵乘法
      J = compute_jacobian(means3D, camera.focal)  # 雅可比
      cov3D = scales_to_cov(scales, rotations)  # [N, 3, 3]
      cov2D = J @ (view_matrix[:3,:3] @ cov3D @ view_matrix[:3,:3].T) @ J.T
      return means2D, cov2D
  ```
- **可微性**：
  - `torch.matmul`（矩阵乘法）可微。
  - `compute_jacobian` 计算 $$J$$（涉及除法 `/Z`），PyTorch autograd 自动处理。
  - `gsplat.rasterize_gaussians` 是 CUDA 实现的，可微分内核，基于 C++ 和 PyBind11（见 gsplat 源码）。
- **调试**：在 `viewer.py` 或 `render.py` 加断点，打印 `means2D.grad` 检查梯度。

#### **1.3 图示（文字模拟）**
```
3D 高斯 [μ, Σ, α] → [矩阵乘法: V, P] → [2D 均值 μ_2D] → [雅可比 J 变换] → [2D 协方差 Σ_2D]
                                                    ↓
                                                  [高斯权重 w_i: exp(-x^T Σ_2D^-1 x)]
梯度流: Loss ← w_i ← μ_2D, Σ_2D ← μ, Σ, α
```

---

### 2. **Alpha 混合（Alpha Blending）的可微性**
**Alpha 混合**将投影后的 2D 高斯按深度排序，累积颜色形成像素值。这个过程也是可微的，因为涉及的排序、乘法和累积操作支持梯度计算。

#### **2.1 数学原理**
- **输入**：
  - 每个高斯的 2D 参数：$$\mu_{i,2D}, \Sigma_{i,2D}, \alpha_i, \mathbf{c}_i$$（颜色，SH 插值）。
  - 像素坐标 $$\mathbf{p} = (u,v)$$。
- **混合公式**：
  $$
  C(\mathbf{p}) = \sum_{i=1}^N \mathbf{c}_i \cdot w_i \cdot T_i, \quad T_i = \prod_{j=1}^{i-1} (1 - w_j)
  $$
  - $$w_i = \alpha_i \cdot \exp(-\frac{1}{2} (\mathbf{p} - \mu_{i,2D})^T \Sigma_{i,2D}^{-1} (\mathbf{p} - \mu_{i,2D})$$：高斯权重。
  - $$T_i$$：透射率，模拟光线穿过前 i-1 个高斯的衰减。
  - 按深度 $$z_i$$（相机坐标 Z）排序（front-to-back）。
- **可微性**：
  - **颜色**：$$\frac{\partial C}{\partial \mathbf{c}_i} = w_i \cdot T_i$$，线性操作，可微。
  - **权重**：$$w_i$$ 依赖 $$\alpha_i, \mu_{i,2D}, \Sigma_{i,2D}$$，已证明可微（见投影）。
  - **透射率**：$$T_i = \prod_{j=1}^{i-1} (1 - w_j)$$，乘法链可微，导数：
    $$
    \frac{\partial T_i}{\partial w_k} = -T_i \cdot \frac{1}{1 - w_k} \quad (k < i)
    $$
  - **排序**：深度排序在渲染中是离散操作（不可微），但 3DGS 用 **tile-based rasterization**（分块光栅化），梯度只流经数值计算（不涉及排序本身），通过 CUDA 实现（如 gsplat 的 radix sort）。

#### **2.2 代码实现（yzslab）**
- **文件**：`src/render/gs_renderer.py`，调用 `gsplat.rasterize_gaussians()`。
- **关键代码**（简化）：
  ```python
  import gsplat
  def render(gaussians, camera):
      means2D, cov2D, opacities = project_gaussians(gaussians.means, gaussians.scales, camera)
      rgb = gsplat.rasterize_gaussians(
          means2D, cov2D, opacities, gaussians.colors, camera  # 排序+混合
      )
      return {"rgb": rgb}
  ```
- **可微性**：
  - `gsplat.rasterize_gaussians` 是 C++/CUDA 实现，内部计算 alpha 混合的梯度。
  - 排序在 tile 内（16x16 像素块），梯度流经 $$w_i, T_i$$，不依赖排序顺序。
- **调试**：在 `training_step()` 打印 `rgb.grad`：
  ```python
  rendered = self.renderer.render(gaussians, batch["camera"])
  loss = self.compute_loss(rendered["rgb"], batch["image"])
  loss.backward()
  print(rendered["rgb"].grad)  # 像素梯度
  ```

#### **2.3 图示（文字模拟）**
```
[高斯: c_i, w_i] → [排序: z_i] → [T_i = Π(1-w_j)] → [C = Σ(c_i * w_i * T_i)] → Loss
梯度流: Loss ← C ← c_i, w_i ← α_i, μ_i, Σ_i
```



### 排序

#### 1. **排序的频率：多久进行一次？**

在 3DGS 中，**排序**是光栅化（rasterization）的一部分，具体用于按深度（z 值）对高斯点进行排序（front-to-back），以正确执行 alpha 混合。排序的频率直接取决于**渲染的调用频率**，因为每次渲染都需要对高斯点进行排序。

#### **1.1 排序的时机**
- **渲染触发**：
  - 排序发生在每次**前向传播**（forward pass）中，即每次调用 `render()` 函数时（包括训练、验证和渲染阶段）。
  - 在训练中，`render()` 由 `training_step()` 调用，每处理一个批次（batch）数据（一张或多张 lego 图像）就渲染一次，因此排序也执行一次。
  - **频率**：**每训练一步（step）**，即每个 batch 都会进行一次排序。
- **训练频率**：
  - 你的训练日志显示 300 个 epoch，30,000 步（`epoch=300-step=30000.ckpt`），每 epoch 100 个 batch（lego 数据集 `train/` 有 100 张图像，`configs/blender.yaml` 默认 batch_size=1）。
  - **排序总次数**：30,000 次（每 step 一次）。
  - **时间开销**：排序由 CUDA 加速（gsplat 使用 radix sort），单次排序 ~0.1-1 毫秒（RTX 3090，~100k 高斯），占渲染时间 <10%。
- **验证/渲染阶段**：
  - 验证（validation）：每 `val_interval`（默认 1000 步，`configs/blender.yaml`）渲染验证图像，触发排序。
  - 手动渲染：运行 `python render.py --ckpt_path outputs/lego_test/checkpoints/epoch=300-step=30000.ckpt`，每次生成一张图像或视频帧都排序一次（e.g., 100 帧视频 → 100 次排序）。
- **Viewer**：运行 `python viewer.py --ply_path outputs/lego_test/checkpoints/epoch=300-step=30000-xyz_rgb.ply`，交互式查看时，每帧实时渲染，排序随帧率（如 60 FPS → 每秒 60 次）。

#### **1.2 代码中的排序**
- **文件**：`src/render/gs_renderer.py`，调用 `gsplat.rasterize_gaussians()`。
- **关键代码**（简化）：
  ```python
  import gsplat
  def render(gaussians, camera):
      means2D, cov2D, opacities = project_gaussians(gaussians.means, gaussians.scales, camera)
      rgb = gsplat.rasterize_gaussians(
          means2D, cov2D, opacities, gaussians.colors, camera, block_size=16
      )  # 内部排序
      return {"rgb": rgb}
  ```
- **排序位置**：
  - `gsplat.rasterize_gaussians`（gsplat 库，`gsplat/_torch_impl.py` 和 `cuda/rasterize.cu`）在每个 tile（16x16 像素块）内按深度 $$z_i$$（从 3D 均值 $$\mu_i$$ 投影得到）排序。
  - 代码伪逻辑（CUDA）：
    ```c
    // cuda/rasterize.cu
    for each tile:
        compute_depths(means3D, camera);  // z_i = (V * μ_i).z
        indices = radix_sort(depths);     // 按 z_i 排序
        accumulate_colors(indices, weights, colors); // alpha 混合
    ```
- **频率**：每个 batch 的 `training_step()` 调用 `render()`，触发一次排序（~100 次/epoch）。

#### **1.3 频率总结**
- **训练**：每 step（batch）排序一次，30,000 步 → 30,000 次排序。
- **验证**：每 1000 步（可调，`val_interval`），~30 次。
- **渲染/Viewer**：每帧一次（视频 100 帧 → 100 次，viewer 60 FPS → 60 次/秒）。
- **配置调整**：`configs/blender.yaml` 的 `batch_size` 或 `val_interval` 影响频率，但排序本身不可跳过（alpha 混合依赖正确顺序）。

---

#### 2. **排序的可微性：排序是可微的吗？**
**简答**：排序（sorting）是**不可微的**（non-differentiable），因为它是一个离散操作（改变高斯索引顺序），无法定义连续的导数。但在 3DGS 中，排序不影响反向传播的可微性，因为梯度流绕过了排序步骤，只依赖数值计算（权重 $$w_i$$ 和透射率 $$T_i$$）。这与投影和 alpha 混合的可微性不同。


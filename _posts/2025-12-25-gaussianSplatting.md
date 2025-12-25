---
layout: post
title: "Gaussiansplatting"
date: 2025-12-25 20:57:46 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left # or right
typora-root-url: ../
---

参考资料：

[1] https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362/

[2] grok 

代码：

[1] [GitHub - yzslab/gaussian-splatting-lightning: A 3D Gaussian Splatting framework with various derived algorithms and an interactive web viewer](https://github.com/yzslab/gaussian-splatting-lightning)

[2] [GitHub - nerfstudio-project/gsplat: CUDA accelerated rasterization of gaussian splatting](https://github.com/nerfstudio-project/gsplat): gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings.

# 3D Gaussian Splatting 

本文是grok 关于gaussian splatting 反向传播的回答，但是第一部分描述了前向过程。即

从高斯参数 → 投影 → alpha 混合 → 渲染图像

3D Gaussian Splatting (3DGS) 的反向传播（back propagation, BP）是其训练核心，用于优化高斯点云的参数（如位置 $$\mu$$, 协方差 $$\Sigma$$, 不透明度 $$\alpha$$, 球谐系数 SH）。与 NeRF 的 MLP 优化不同，3DGS 使用**显式参数**（点云属性），通过可微分渲染（differentiable rendering）计算梯度，结合 PyTorch 的 autograd 实现 BP。以下我详细解释 3DGS 的反向传播机制，结合 **yzslab/gaussian-splatting-lightning** 仓库的代码（基于你的使用场景），从数学到实现，尽量图文并茂（文字描述模拟图形），并回答如何在代码中观察 BP。

---

#### 1. **反向传播的总体流程**
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

#### 2. **数学原理：可微分渲染与梯度计算**
3DGS 的渲染管道是可微分的，梯度通过链式法则从损失函数流回高斯参数。以下拆解关键步骤：

##### **2.1 前向传播（渲染）**
- **输入**：N 个高斯，参数为：
  - $$\mu_i \in \mathbb{R}^3$$：中心位置。
  - $$\Sigma_i \in \mathbb{R}^{3 \times 3}$$：协方差矩阵（分解为缩放 $$S_i$$ 和旋转 $$R_i$$）。
  - $$\alpha_i \in [0,1]$$：不透明度（sigmoid 输出）。
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
  - 2D 高斯权重：$$w_i(u,v) = \alpha_i \cdot \exp(-\frac{1}{2} (\mathbf{p} - \mu_{i,2D})^T \Sigma_{i,2D}^{-1} (\mathbf{p} - \mu_{i,2D}))$$。
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



## 排序

### 1. **排序的频率：多久进行一次？**

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

### 2. **排序的可微性：排序是可微的吗？**
**简答**：排序（sorting）是**不可微的**（non-differentiable），因为它是一个离散操作（改变高斯索引顺序），无法定义连续的导数。但在 3DGS 中，排序不影响反向传播的可微性，因为梯度流绕过了排序步骤，只依赖数值计算（权重 $$w_i$$ 和透射率 $$T_i$$）。这与投影和 alpha 混合的可微性不同。


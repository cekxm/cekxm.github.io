---
layout: post
title: "Pytorch Note"
date: 2024-12-30 00:19:02 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---


## Tensor

tensors 和 numpy ndarrays 类似，额外的好处是可以放入GPU，用于加速计算。ndarrays 不能放入gpu。

tenor <--> numpy
b = a.numpy()
a = torch.from_numpy(b)
可以进行互换，注意互换行为类似于建立新的view，存储是同一份，修改一个，会改变另一个。

运算可以新生成一个 tensor，也可以 in-place。in-place 运算的名称要带一个后缀 _，比如 x.t_()

x.view((d1,d2,...)) 函数用于 reshape，注意不同的view所对应的存储是同一份，修改其中一个view，会改变所有view.

cuda tensor
x=x.to(device) 函数可以在cpu和gpu间进行转移
to 支持改变 dtype

tensor 的attrib
.requires_grad 是否需要梯度。只能改变 leaf node 的这个值
.grad 记录梯度。我测试只能保存 leaf node 的梯度？
.grad_fn 记录生成这个 tensor 的函数

### dtype

- torch.float32 / torch.float
- torch.float64 / torch.double
- torch.float16 / torch.half
- torch.uint8
- torch.int8
- torch.int16 / torch.short
- torch.int32 / torch.int
- torch.int64 / torch.long
- torch.bool

### 基本运算

可以使用以下两种风格

- Tensor.op，比如 b = a.flattern()，inplace 操作后缀 _
- torch.op，比如 b= torch.flattern(a)

### view()

改变tensor维数或维数的尺寸，不改变内存
**view 并不是任何时候都能用，还要看tensor 是不是内存连续的！**
见下面的例子。
view(-1) 类似于 flatten

### contiguous

```python
>>> a = torch.arange(12).view((3,4))
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> b = a.t()
>>> b
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
>>> b.is_contiguous()
False
>>> b.view(-1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

b 无法转为一维的，因内存不连续。所以实际上，不少 tensor操作会首先执行 contiguous()

### flatten

b 无法转为一维的，因内存不连续。所以实际上，不少 tensor操作会首先执行 contiguous()，

```python
>>> b
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
>>> b.is_contiguous()
False
>>> c = b.flatten()
>>> c
tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
>>> b[0,0]=-1
>>> c
tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
>>> b
tensor([[-1,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
```

### reshape

有了 view 为什么还要shape呢？原因同上，当内存不连续时，实际上进行了 deepcopy。

### squeeze/unsqueeze

squeeze 把所有为size==1的维度去掉。

torch.unsqueeze(input, dim, out=None)
在第 dim （从1开始算）个维度后增加size=1的1维。dim=0，即为 prepend 维度1. e.g.
(3,5) -> (1,3,5)

## Parameter

Parameter 是 Tensor 的子类，用于做 module 的参数。
Parameter的用处：

- 在 Module 的 init 中声明。当它在 Module 中被声明为属性时，会自动加入到 module 的参数列表，也就是会在 Module.parameters iterator 中出现。如果在 init 中声明 Tensor 则没有这个效果。
- 默认requires_grad 属性为 True

```python
class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        ...
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
```

- module 的参数要用 Parameter
- 有时候我们希望在module中记录一些中间状态，但不希望注册，则用 Tensor

parameter.data 就是 tensor
**注意，parameter.data.requires_grad** 可能为False，不影响 parameter 的更新。

```python
conv2d = torch.nn.Conv2d(1, 20, 5)
print(conv2d.weight.requires_grad)
print(conv2d.weight.data.requires_grad)

```

### 参数注册和顺序

在 `Module.__init__` 函数中声明层，会注册相应的参数。在 Module.parameters() 返回的参数迭代器中，按照 init 函数中声明的顺序来排序。改变声明顺序不会对网络的功能造成问题，只是要注意参数的顺序，可以打印参数的名称来查看顺序。

named_parameters() 是一个迭代器，需要类似函数调用

```python
for name,para in model.named_parameters():
    print(name, para)
```

### 参数初始化

参数初始化：
可使用两种方法

1. nn.init
2. tensor.xxx_ 函数
   para.data.normal_(mean, stddev)

## autograd

- autograd 机制深入到 tensor 这一级，创建一个 tensor，它就有属性 grad, requires_grad。
- Torch 对 tensor 的运算自动支持 autograd机制，不需要其他操作
- 调用 loss.backward() 会按照有向图的反方向自动计算梯度
- autograd 机制是低层的，所以用户Module代码可以没有 backward 函数

该机制是深入到 tensor 这一层
例： 
x ->y -> out
out.backward() 是计算 out 对以其为顶点的树的结点的梯度，比如对 x 的梯度
若
out -> loss
假设 loss 对 out 的梯度为 do
out.backward(do) 是计算 loss 对 x 的梯度

梯度的计算用 jacobian 
y = f(x)
z = g(y)
令 y 对 x 的Jacobian 为 J

**missing**

### 解除记录梯度的方法

**tensor**解除记录梯度

1. x.requires_grad_(False)
2. x.detach()
3. 全局解除和全局恢复，一般和 with 搭配使用，有以下几个函数

- torch.no_grad()
- torch.enable_grad()
- torch.set_grad_enabled(True/False)
  这三个和 torch.is_grad_enabled() 有关，它不是为每一个parameters设置，而是上层的一种统一机制。其中 no_grad, enable_grad
  这两个是一对的。但一般在搭配 with 使用时，可以只使用 no_grad()，因为 with 机制的 exit 函数会恢复原来的设置。

```python
with torch.no_grad():  #当评估已经训练好的网络时，就不用跟踪梯度了
    pass
```

### 梯度清零

梯度计算是累积的，每一个 batch 迭代要清零梯度。
optimizer.zero_grad() 
**注意，梯度清零是在 batch 级，而不是在 epoch 级。**

### autograd.Function

当需要一种新的**运算**，这种运算torch不提供的时候，就需要扩展 autograd。办法是通过继承 autograd.Function。Function 里面必须有 forward, backward 函数。通常 forward里面会使用torch的一些运算，但在反向传播时直接调用 backward函数，而不再根据forward中的运算进行反向传播计算。

如下代码自定义了一个运算，注意它的backward和forward的运算没有关系，而是把梯度强性设置为 0.8。

```python
class SumFunction(Function):

    @staticmethod
    def forward(ctx, input, bias=None):
        ctx.save_for_backward(input, bias)
        output = input.sum()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.ones(input.shape)*0.8
        if bias is not None and ctx.needs_input_grad[1]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_bias
```

### 使用 autograd.Function

```python
sum2 = SumFunction.apply
y = sum2(x)
```

## torch.nn

torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

params = list(net.parameters())
params[0] 是Parameter 类对象
params[0].data 是 tensor

## nn.Module

以下都属于 Module。

- CNN 中的层，比如 Conv2d
- Sequential，是把许多层串接在一起
- model或net，一般网络就是一个 nn.Module
  自定义 Module，必须要有 forward 函数，不需要 backward 函数，因为不管是 pytorch提供的运算还是自定义的运算（autograd.Function）,这部分已经处理好了。

```python
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """        In the constructor we instantiate two nn.Linear modules and assign them as        member variables.        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """        In the forward function we accept a Tensor of input data and we must return        a Tensor of output data. We can use Modules defined in the constructor as        well as arbitrary operators on Tensors.        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
```

## nn.functional

**nn.functional 提供的函数和 nn.Module提供的类功能是一样的**

比如

- nn.functional.conv1d
- nn.Conv1d
  注意大小写。让我们看下它们的实现代码。

```python
class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
```

```python
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    if input is not None and input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
               _single(0), groups, torch.backends.cudnn.benchmark,
               torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)
```

可见，Conv1d 调用了 conv1d，而 conv1d 调用了 ConvNd。

**nn.functional 和 nn.Module 都可以用于定义CNN网络，也都支持 autograd 机制，区别是**

使用 nn.functional，必须自己定义层的参数(weight, bias 等)并进行维护。而使用nn.Module，这些参数有类的机制进行定义和维护

### 建议

- 对于有参数的层，使用 nn.Module
- 对于无参数的层，建议使用 nn.Module，因为 pytorch 基本上都提供了响应的 Module。当然使用 nn.functional 也是可以的
- nn.functional 是更灵活的，在有些只需要运算，而不需要保持参数的情况下，使用 nn.functional

## 参数保存和读取

### `state_dict`

- nn.Module 有`state_dict`，保存参数，key 是 `__init__` 声明的layer名称+参数名，比如 conv1.weight，value 是 参数 tensor
- optimizier 也有`state_dict`，保存它自己的状态，以及超参数

### 三个函数

- torch.save: 保存 dict
- torch.load: 读取 dict
- torch.nn.Module.load_state_dict: 加载参数

### 标准用法

#### train and test

```python
torch.save(model.state_dict(), PATH)
```

For test/deploy

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

#### checkpoint

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```

Load checkpoint:

```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()# - or -model.train()
```

## Dataset

### Imagefolder

文件夹结构

```
- data
    - class 1
        - img1.jpg
        - img2.jpg
    - class 2
        - ...
data = torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
```

执行后，Dataset getitem 返回 (sample, target)

### 自定义 Dataset

自定义Dataset是 torch.utils.data.Dataset 的子类，必须实现 `__getitem__` 和 `__len__` 方法。

`__getitem__` 方法使得 dataset 可以用索引 [n]。但它的输出是什么格式，是可以自己定义的。
比如 ImageFolder Dataset，返回 tuple (tensor, label)

而 [这个 tutorial ](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html">https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)中，getitem 返回
{'image': image, 'landmarks': landmarks}

### Transform

transform 是可以被当作函数调用的类。

transform 一般作为生成时 Dataset 时的参数，并且在 getitem 中使用。所以

- 若使用pytorch提供的Dataset，transform 的接口和输出必须和它们的 getitem 兼容。
- 使用 torchvision.transforms 中 PIL 相关的变换，注意它的接口是 PIL image，输出也是 PIL image。
- 如果是自定义 Dataset，则可灵活处理。

#### torchvision.transforms

提供了以下几种：

- on PIL Image
- on torch.*Tensor
- conversion
- Lambda and Functional

和图像相关的就是第一种了。

```
from PIL import Image
path = './data/animals/cats/cats_00001.jpg'
with open(path,'rb') as f:
    img = Image.open(f) #这是一个 lazy 操作，还没读取内容
    img = img.convert('RGB')

from torchvision import transforms
transform = transforms.RandomResizedCrop(200)
img2 = transform(img)
```

- 注意，PIL 读取后像素值在[0,255]

##### ToTensor

该变换把 PIL image 或 numpy.ndarray (H x W x C) in the range [0, 255] 变换成 torch.FloatTensor of shape (C x H x W) in the range **[0.0, 1.0]**。

#### 自定义 transform

transform 是可以被当作函数调用的类，为 object 的子类，需实现
`__call__` 方法和`__init__` 方法。

对于图像，出了PIL，还常用 skimage 读取。

```
from skimage import io
img = io.imread('./data/animals/cats/cats_00001.jpg')
```

- 注意，io 读取后像素值在[0,255]

#### transform 串接

torchvision.transforms.Compose([...])



### torch.utils.data.Dataloader

支持两种 dataset

- map style，即Dataset子类，实现len和getitem的
- iterable style，
   IterableDataset 子类，实现 `__iter__()` 方法。,

#### 参数

- batch_size
- shuffle 每一次batch（含第一次batch），重新排序。默认是 False。此项和 sampler 不共同使用。
- sampler 从 data 中获取sample的方法

#### sampler

- torch.utils.data.SubsetRandomSampler(indices)，从给定的 indices 中获取。可用于 train/validation 的划分，比如这个 [gist](https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb">https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb)

#### automatic batching

- It always prepends a new dimension as the batch dimension.It automatically converts
- NumPy arrays and Python numerical values into PyTorch Tensors.
- **It preserves the data structure**, e.g., if each sample is a dictionary, it outputs a dictionary with the same set of keys but batched Tensors as values (or lists if the values can not be converted into Tensors). Same for list s, tuple s, namedtuple s, etc.
  即 dictionary 还是 dictionary，tuple 还是 tuple.

#### `collate_fn`

## Optimizer

- torch.optim.Adadelta
- torch.optim.Adagrad
- torch.optim.Adam
- torch.optim.AdamW
- torch.optim.SGD
- ...

## pytorch 训练代码参考

注意：

- 训练时，每个batch会更新一次参数，最终的train loss是在epoch这一级的，是所有batch的平均结果，因此实际上是不同的参数的平均结果。
- 而验证时，一次epoch后，用最后的参数进行验证，loss 对应的参数是一致的

```python
# 模型实例
model_ft = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
data_transforms = transforms.Compose([...]}

# 数据部分，以下是训练用
# 若有validation，则还需要响应的 data 和 loader
image_datasets = datasets.ImageFolder(os.path.join(data_dir,x),data_transforms)
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle = True, num_workers=4)

dataloaders['val']

val_acc_history = []
best_acc = 0.0

for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    
    # 这个函数不是进行训练，而是设置模型的mode
    # 部分层在 train 和 val 时计算是不一样的，比如 dropout, bn
    # 通过设置 mode，告诉这些层要如何计算
    model.train()
    
    # 这两个参数是epoch 级别统计 loss 的 
    running_loss = 0.0
    running_corrects = 0
    
    # 每个epoch先 train，需要梯度
    # 因为后面eval时设置了 no_grad
    # 这里得纠正回来
    with torch.enable_grad():
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 注意，zero_grad() 是在 batch 级别的
            optimizer_ft.zero_grad()
            
            output = model_ft.forward(inputs)
            loss = criterion(output, labels)
            
            _, preds = torch.max(output, 1) #为了求 train loss
            loss.backward()
            optimizer_ft.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(dataloaders['train'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
            
    # 这个函数不是进行训练，而是设置模型的mode
    model_ft.eval()
    
    running_loss = 0.0
    running_corrects = 0
       
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model_ft.forward(inputs)
            loss = criterion(output, labels)
            
            _, preds = torch.max(output, 1)
            
            # validation 就没有 backward 和 step 了
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('eval', epoch_loss, epoch_acc))
    
    # 比较 evaluation 的 acc，保存最大的那个
    if epoch_acc > best_acc:
       best_acc = epoch_acc
       best_model_wts = copy.deepcopy(model_ft.state_dict())
    val_acc_history.append(epoch_acc)
        
print('Best val Acc: {:4f}'.format(best_acc))
```

## misc

### 进度条

```python
for i in tqdm(range(10000)):
    time.sleep(0.001)
```


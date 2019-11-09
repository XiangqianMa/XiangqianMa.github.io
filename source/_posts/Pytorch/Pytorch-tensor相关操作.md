---
title: Pytorch-tensor相关操作
date: 2019-07-19 16:15:31
summary: tensor操作中常用的几项操作。
categories:
- 深度学习
- 深度学习框架
- Pytorch
tags:
- 深度学习
- 深度学习框架
mathjax: true
img: /images/24.jpg
thumbnail: /images/24.jpg
---

# Tensor及使用

## tensor.requires_grad
在创建张量后，如果未进行特殊指定，默认不对该张量进行梯度计算。需要注意的是，只有当一个张量的所有输入都不需要进行梯度计算时，该张量才不需要进行梯度计算。
在网络模型中，其中间参数是默认进行求导的，因而网络的输出也默认是需要求导的。
在写代码的过程中，**不要**把网络的输入和 Ground Truth 的 `requires_grad` 设置为 True。虽然这样设置不会影响反向传播，但是需要额外计算网络的输入和 Ground Truth 的导数，增大了计算量和内存占用不说，这些计算出来的导数结果也没啥用。因为我们只需要神经网络中的参数的导数，用来更新网络，其余的导数都不需要。
通过使用该方法，可以将模型的部分参数设置为不需要进行求导。这一做法常被用于迁移学习中，对模型的一部分参数进行冻结，只更新其余的参数，官方例子如下。
```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# 用一个新的 fc 层来取代之前的全连接层
# 因为新构建的 fc 层的参数默认 requires_grad=True
model.fc = nn.Linear(512, 100)

# 只更新 fc 层的参数
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# 通过这样，我们就冻结了 resnet 前边的所有层，
# 在训练过程中只更新最后的 fc 层中的参数。
```

## torch.no_grad()
虽然可以使用第一条所介绍的`requires_grad`方法对单个张量进行梯度设置，但一个模型中存在大量的张量，逐个进行张量的设置会非常麻烦。这时就使用这一方法，对张量进行批量管理。
在我们对已训练完成的模型进行评估时，出来使用`model.eval()`将模型设置为评估模式之外，如果为了节省内存、显存，可以使用`torch.no_grad()`将模型中的所有参数设置为不进行梯度计算的模式。
```python
x = torch.randn(3, requires_grad = True)
print(x.requires_grad)
# True
print((x ** 2).requires_grad)
# True

with torch.no_grad():
    print((x ** 2).requires_grad)
    # False

print((x ** 2).requires_grad)
# True
```
## tensor.data
在0.4版本之后，`Variable`被取消，统一使用`tensor`代替，`.data`本来是被用来从`Variable`中获取`tensor`的，现在被用来从`tensor`中获取一个具有同样的数据和不进行梯度计算的版本，两者共享内存空间，也就是说修改两者中的任意一个都会导致另一个值被修改。
## tensor.detach()
Pytorch的自动求导系统不会跟踪`tensor.data`的变化，因而可能会导致求导结果出错。而`torch.detach()`会被自动求导系统追踪。如下例所示：

```python
a = torch.tensor([7., 0, 0], requires_grad=True)
b = a + 2
print(b)
# tensor([9., 2., 2.], grad_fn=<AddBackward0>)

loss = torch.mean(b * b)

b_ = b.detach()
b_.zero_()
print(b)
# tensor([0., 0., 0.], grad_fn=<AddBackward0>)
# 储存空间共享，修改 b_ , b 的值也变了

loss.backward()
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

在上述例子中，pytorch的自动求导系统进行了报错，出错的原因在于，b的值在进行反向传播之前被修改，这一修改会导致求导出错。`.detach()`之后的修改会被追踪，但当我们使用`.data`时，pytorch不会进行报错。
```python
a = torch.tensor([7., 0, 0], requires_grad=True)
b = a + 2
print(b)
# tensor([9., 2., 2.], grad_fn=<AddBackward0>)

loss = torch.mean(b * b)

b_ = b.data
b_.zero_()
print(b)
# tensor([0., 0., 0.], grad_fn=<AddBackward0>)

loss.backward()

print(a.grad)
# tensor([0., 0., 0.])

# 其实正确的结果应该是：
# tensor([6.0000, 1.3333, 1.3333])
```

## 设备的切换
在0.4之前，一般使用`.cuda()`方法将张量或模型移动至GPU中，在需要进行设备的切换时，这一做法显得比较麻烦。0.4之后增加了`.to(device)`操作，这样，只需要在程序的开始指定所使用的设备即可。

```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

a = torch.rand([3,3]).to(device)
# 干其他的活
b = torch.rand([3,3]).to(device)
# 干其他的活
c = torch.rand([3,3]).to(device)
```
#### GPU Tensor -> Numpy
在我们想把 GPU tensor 转换成 Numpy 变量的时候，需要先将 tensor 转换到 CPU 中去，因为 Numpy 是 CPU-only 的。其次，如果 tensor 需要求导的话，还需要加一步 detach（防止求导出错），再转成 Numpy 。

```python
x  = torch.rand([3,3], device='cuda')
x_ = x.cpu().numpy()

y  = torch.rand([3,3], requires_grad=True, device='cuda').
y_ = y.cpu().detach().numpy()
# y_ = y.detach().cpu().numpy() 也可以
# 二者好像差别不大？我们来比比时间：
start_t = time.time()
for i in range(10000):
    y_ = y.cpu().detach().numpy()
print(time.time() - start_t)
# 1.1049120426177979

start_t = time.time()
for i in range(10000):
    y_ = y.detach().cpu().numpy()
print(time.time() - start_t)
# 1.115112543106079
# 时间差别不是很大，当然，这个速度差别可能和电脑配置
# （比如 GPU 很贵，CPU 却很烂）有关。
```

## tensor.item()
该方法只适用于tensor中包含单个值的情况，使用该方法会直接得到该tensor中所包含的值。当tensor中包含多个元素时，可以使用`tensor.tolist()`。

*********

<center>

![](https://pic2.zhimg.com/80/v2-7a79e86e8006918808455318cf425d61_hd.jpg)

</center>
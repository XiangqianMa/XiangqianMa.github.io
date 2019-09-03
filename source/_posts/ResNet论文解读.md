---
title: ResNet论文解读
date: 2019-04-30 11:05:17
summary: 经典神经网络结构ResNet论文解读
categories:
- 深度学习
- 经典网络结构
tags: 
- 论文阅读
- 深度学习
mathjax: true
---
## Deep Residual Learning for Image Recogition

---

## 作者

**Kaiming He**    Xiangyu Zhang       Shaoqing Ren    Jian Sun



**何凯明大佬**

<center>

![](http://kaiminghe.com/img/me.jpg)

</center>


## 内容

这篇论文主要用于解决网络层数加深时,模型的训练问题.



### 深度网络的退化问题

如下图所示,为历年ISLVRC竞赛中取得冠军的各个网络结构,观察该图片可知,随着网络层数的增加,模型的复杂度不断提升,进而可以提取更为丰富的特征,因而也得到了更好的结果.

<center>
{% asset_img ISLVRC冠军结构.jpg ISLVRC冠军结构 %}
</center>



但事实上,模型的表达能力和模型复杂度并不是成正比关系的,在论文中,作者指出,随着深度的增加,模型出现了退化问题(Degradation problem),如下图所示.

<center>

{% asset_img 模型退化问题.jpg 模型退化问题 %}

</center>



网络深度增加时，网络准确度出现饱和，甚至出现下降.这一问题并不是由过拟合所导致的,因为在图中,56层网络的训练误差同样很大.

在深层网络存在着梯度消失或者爆炸的问题，这使得深度学习模型很难训练。但是现在已经存在一些技术手段,如BatchNorm来缓解这个问题。因此，出现深度网络的退化问题是非常令人诧异的。



### 残差学习

深度网络的退化问题至少说明深度网络不容易训练。但是我们考虑这样一个事实：现在你有一个浅层网络，你想通过向上堆积新层来建立深层网络，一个极端情况是这些增加的层什么也不学习，仅仅复制浅层网络的特征，即这样新层是恒等映射（Identity mapping）。在这种情况下，深层网络应该至少和浅层网络性能一样，不应该出现退化现象。

为了解决这一问题,在本文中作者提出了残差学习的思想.对于一个堆积层结构（几层堆积而成）当输入为$x$时其学习到的特征记为$H(x)$，现在我们希望其可以学习到残差$F(x)=H(x)-x$，这样其实原始的学习特征是$F(x)+x$。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。残差学习的结构如下图所示。这有点类似与电路中的“短路”，所以是一种短路连接（shortcut connection）。

<center>

{% asset_img 残差模块.jpg 残差模块 %}

</center>



从数学角度解释残差学习更为容易的原因.

将残差单元表示为:

<center>

$y_l=h(x_l)+F(x_l, W_l)$

$x_{l+1}=f(y_l)$

</center>

其中,$h(x_l)$表示恒等映射,$F(x_l, W_l)$表示残差映射,$f$为ReLU激活函数,那么从浅层$l$到深层$L$的特征为:

<center>

$x_L=x_l+\sum_{i=l}^{L-1}F(x_i, W_i)$

</center>

使用链式求导法则可以得到损失相对于第$x_l$层的梯度为:

<center>

$\frac{\partial loss}{\partial x_l} = \frac{\partial loss}{partial x_L} \cdot \frac{\partial x_L}{\partial x_l} =  \frac{\partial loss}{\partial x_L} \cdot (1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1}F(x_i, W_i))$

</center>



由上式我们可以发现,小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易(上面的推导并不是严格的证明)。



### ResNet的网络结构

ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如图所示。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：**当feature map大小降低一半时，feature map的数量增加一倍**，这保持了网络层的复杂度。从图中可以看到，ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习，其中虚线表示feature map数量发生了改变。图中展示的34-layer的ResNet，还可以构建更深的网络如表所示。从表中可以看到，对于18-layer和34-layer的ResNet，其进行的两层间的残差学习，当网络更深时，其进行的是三层间的残差学习，三层卷积核分别是1x1，3x3和1x1，一个值得注意的是隐含层的feature map数量是比较小的，并且是输出feature map数量的1/4。

<center>
{% asset_img resnet结构.jpg resnet结构 %}
ResNet网络结构

</center>



<center>
{% asset_img 不同的resnet结构.jpg 不同的resnet结构 %}
不同的ResNet网络结构

</center>



#### 残差单元

ResNet使用两种残差单元，如图所示。左图对应的是浅层网络，而右图对应的是深层网络。对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：（1）采用zero-padding增加维度，此时一般要先做一个downsample，可以采用strde=2的pooling，这样不会增加参数；（2）采用新的映射（projection shortcut），一般采用1x1的卷积(可以改变维度)，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用projection shortcut。

<center>
{% asset_img 残差单元.jpg 残差单元 %}
残差单元示意图(左侧为浅层网络使用的残差单元,右侧为深层网络)

</center>



## 结果



如图所示,左侧为不使用残差模块的普通深度网络,右侧为ResNet.

<center>
{% asset_img 结果对比.jpg 结果对比 %}
结果示意图

</center>



从图中我们可以看出,在左侧,未使用残差模块的网络明显出现了退化现象,而右侧则无此问题.



## 一种更为优秀的残差模块

采用前置激活可以提升残差模块的性能,如图所示:



<center>
{% asset_img 残差模块改进.jpg 残差模块改进 %}

</center>




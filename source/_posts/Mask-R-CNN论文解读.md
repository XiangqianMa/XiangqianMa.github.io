---
title: Mask R-CNN论文解读
date: 2019-05-05 14:56:28
summary: Mask R-CNN论文解读
categories:
- 深度学习
- 实例分割
tags: 
- 论文阅读
- 深度学习
---
# Mask R-CNN

## 作者
何凯明 Geeorgia Gkioxari Piotr Dollar Ross Girshick
## 内容
在本文中，作者提出了一种用于目标实例分割的方法。该方法在检测目标的同时针对每一个目标实例产生一个高质量的分割蒙板。Mask R-CNN通过在Faster R-CNN现有的用于目标检测的分支的基础上添加用于目标mask预测的分支实现。

## 目标检测、目标实例分割、语义分割
如下图所示：
<center>
![](https://img-blog.csdn.net/20171121232307984?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhbWVudGluZ3Rhbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

</center>

* 图片分类
仅需要识别出一张图片中存在哪几类目标即可。
* 目标检测
需要给出图片中目标的类别和具体位置。
* 语义分割
对图片中的目标进行像素级分割，但只需要区分不同类别目标即可，统一类别的目标不需要区分。
* 实例分割
对图片中的目标进行像素级分割，但需要区分不同的实例，同一类别的不同个体同样需要进行区分。

## Mask R-CNN:解决实例分割问题
**R-CNN的网络结构：**
<center>
{% asset_img 5d9736d2-0b1a-43fa-912a-60be077b0387.jpg%}
</center>

**Fast R-CNN的网络结构：**
<center>
{% asset_img b65ae552-1ac7-478c-91d8-82ae6e23d285.jpg%}

</center>

**Faster R-CNN的网络结构**
<center>
{% asset_img b19b660d-8e82-48cf-817d-1429475f15be.jpg%}

</center>

**Mask R-CNN的总体框架如图所示：**
<center>
{% asset_img 27847427-802c-4928-8626-e535429eae7a.jpg%}

</center>
<center>
{% asset_img d1f5ad89-4152-4cf0-b22b-6b307397dee6.jpg%}

</center>

作者在Faster R-CNN原有的用于预测目标bounding boxes的子网络的基础上，添加用于预测mask的分支，使用FCN(全卷积网络)对每一个RoI分别进行预测。Faster R-CNN并不是为网络输入与输出之间的像素级的匹配而设计的，这一问题主要是由RoIPool层的空间量化操作所导致的。为了解决这一问题，作者提出了一种简单的、无需量化的层，即RoIAlign，该层的引入极大地保证了空间位置地准确性。RoIAlign层对最终的检测结果有着极大的影响。
除此之外，作者发现很有必要对mask预测和类别预测进行解耦和。针对每一类分别预测一层mask，类别之间不存在竞争关系，将类别预测任务交给RoI的分类分支。

### RoIAlign
给定特征图如下所示：
<center>
{% asset_img ee623a5f-c1f5-4a2a-98f7-4c94fa379a3a.jpg%}

</center>

传统的RoIPool层针对每一个RoI分别产生一层小的特征图。RoIPool的步骤如下：
1. 首先对RoI的浮点坐标、大小参数进行量化，将其对应到特征图中。
<center>
{% asset_img bf8dcd4c-4274-44c7-ab9d-0ed1ebea64fa.jpg%}

</center>

2. 接着经过量化的RoI将被划分为格子，针对每一个格子内部进行池化操作，进而得到固定大小的特征图。
<center>
{% asset_img 82c99a99-edf8-4c09-9e5c-b7f9ea190722.jpg%}

</center>

在上述操作中，共存在两步取整操作，一个是将RoI对应至特征图时，一个是对量化后的RoI进行划分。这两步取整量化操作会导致原始RoI与抽取出的特征图在空间位置上不匹配。这一问题不会对目标的分类造成大的影响，但会对mask预测造成极大的负面影响。

为了解决这一问题，作者提出了RoIAlign层，RoIAlign去除了量化取整操作，使得抽取的特征图与输入图片有着精确的位置对应。对于RoI中的每一个格子，使用双线性插值法计算其对应的值，双线性插值法需要的原始值来自于格子四角位置上的值。如下图所示：
<center>
{% asset_img 86330610-af9b-4205-9736-3240e2bcadb2.jpg%}

</center>
整体步骤如下：
1. 使用浮点运算，将RoI对应至特征图的相应位置
<center>
{% asset_img 2e9be7aa-3b8d-4bc1-ab91-c7a24239e866.jpg%}

</center>

2. 将每一个格子划分为四个小格子
<center>
{% asset_img 6424cdaa-67a6-4774-8d71-662bbba20f6c.jpg%}

</center>

3. 使用双线性插值法计算每一个格子的值，取四角的值为原始值
<center>
{% asset_img 4ab90a74-ed8e-49ee-93d6-08ebfcc44f9e.png%}

</center>

4. 对每一个格子进行池化操作，得到最终结果
<center>
{% asset_img 6cad9bd2-e36e-4823-8430-af71c149bd85.png%}

</center>

### 损失函数
损失函数由类别损失、边框回归损失、mask损失三部分构成，与其他方法不同，计算mask损失时，对预测的各个类别的mask分别使用sigmoid函数进行激活（而不是使用softmax函数对所有类别的mask进行激活），接着使用二维交叉熵计算损失。

### 实现细节
用于预测mask的子网络的结构如图所示：

<center>
{% asset_img e0da9a7e-36b8-468e-8860-db023a099a91.jpg%}

</center>

左侧使用resnet-c4作为前面的卷积网络，将rpn生成的roi映射到C4的输出，并进行roi pooling，最后进行分叉预测三个目标。右侧即使用Faster R-CNN加FPN的结构。

### 对比实验
与FCIS+++的对比，如下图所示：
<center>
{% asset_img 170fbc97-7e4f-44bd-bf89-dbeae157b64a.jpg%}

</center>
在FCIS++的预测中会在目标重合位置出现一条直线，而Mask R-CNN的预测结果则没有。

**消融实验**
结果如图所示：
<center>
{% asset_img 3095a77a-a0f9-4b7f-9482-42fde3b002fd.jpg%}

</center>
作者分别给出了不同backbone、多任务和独立任务、使用RoIAligh和不使用、使用FPN进行结果预测和使用全连接层的对比结果。

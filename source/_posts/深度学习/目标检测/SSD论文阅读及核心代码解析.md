---
title: SSD论文阅读及核心代码解析
date: 2019-04-30 16:50:20
summary: SSD论文、代码解读
categories:
- 深度学习
- 目标检测
tags:
- 深度学习
- 目标检测
mathjax: true
cover: true
---

## SSD:Single Shot MultiBox Detector

---
## 作者
**Wei Liu**

## 内容
在本文中,作者提出了一种使用单个神经网络进行目标检测的框架,该框架的特点如下:
1. 将网络的bounding boxes的输出空间划分为default boxes的集合,这些boxes具有不同的尺度和比例.对于被选中用于目标预测的feature maps,网络会针对该特征图中的每一个位置预测多个default boxes,这些default boxes又称为anchors.
对于每一个default box,网络给出其存在目标的分数(每一个类别分别预测一个),同时给出在default box的基础上进行形状调整的参数.

2. 为了应对目标的尺度变化,SSD在来自于网络的多个具有不同分辨率的feature maps上进行预测.其中,靠近网络前侧,具有较高分辨率的feature map适用于进行小目标的检测,而后侧的feature maps适用于大目标的检测.

作者将本文的贡献总结如下:
1. SSD的核心是:在事先设定的固定大小,比例的default boxes的基础上,使用较小的卷积核预测目标隶属于某一类的分数以及box的偏差.
2. 为了获得较高的检测准确率,在具有不同大小的feature maps上产生具有不同尺度的预测,且这些预测又可以具有不同的长宽比.
3. 网络可以进行end-to-end训练,同时取得不错的准确率.

## SSD网络结构
## 模型
SSD方法基于前馈神经网络,该网络产生固定数目的bounding boxes集合,同时给出这些boxes中存在某一类别的目标的分数,在这些bounding boxes的基础上,使用NMS算法求出最后的结果.网络的结构如下:
<center>
{% asset_img SSD网络结构.png SSD网络结构 %}
</center>

由上图可以发现,总共选取了6层具有不同大小的feature maps用于目标检测,与R-CNN等方法使用全连接层给出预测结果不同,SSD使用较小的卷积计算得到预测结果.对于每一层feature map,使用一个卷积核计算得到各个default boxes属于某一类的分数,使用另一个卷积核得到在各个default boxes位置的基础上的偏差.

### 训练
#### 匹配策略
在进行训练的时候,需要决定哪一个default boxes与ground truth相关联,只有被选中的default boxes才会参与训练.所采取的策略为:**首先将每一个ground truth和与其覆盖率最大的default boxes进行匹配,接着将与ground truth的jaccard重合率高于设定阈值的default boxes视为匹配**,这样,在第一步保证每一个ground truth都会有匹配的default boxes的基础上,第二步使得多个default boxes可以匹配同一个ground truth.这一设定简化了学习问题,使得网络可以对多个重合的default boxes给出较高的预测分数,而不是仅仅选择具有最高重合率的一个.
匹配代码如下:
```python
# ./layers/box_utils.py
def point_form(boxes):
    # 将(cx, cy, w, h) 形式的box坐标转换成 (xmin, ymin, xmax, ymax) 形式
    return torch.cat( (boxes[:2] - boxes[2:]/2), # xmin, ymin
                    (boxes[:2] + boxes[2:]/2), 1) # xmax, ymax


def intersect(box_a, box_b):
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]) box_a 与 box_b 两个集合中任意两个 box 的交集, 其中res[i][j]代表box_a中第i个box与box_b中第j个box的交集.(非对称矩阵)
    # 思路: 先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    # 两个box的交集可以看成是一个新的box, 该box的左上角坐标是box_a和box_b左上角坐标的较大值, 右下角坐标是box_a和box_b的右下角坐标的较小值
    A = box_a.size(0)
    B = box_b.size(0)
    # box_a 左上角/右下角坐标 expand以后, 维度会变成(A,B,2), 其中, 具体可看 expand 的相关原理. box_b也是同理, 这样做是为了得到a中某个box与b中某个box的左上角(min_xy)的较大者(max)
    # unsqueeze 为增加维度的数量, expand 为扩展维度的大小
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A,B,2),
                        box_b[:, :2].unsqueeze(0).expand(A,B,2)) # 在box_a的 A 和 2 之间增加一个维度, 并将维度扩展到 B. box_b 同理
    # 求右下角(max_xy)的较小者(min)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A,B,2),
                        box_b[:, 2:].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((max_xy, min_xy), min=0) # 右下角减去左上角, 如果为负值, 说明没有交集, 置为0
    return inter[:, :, 0] * inter[:, :, 0] # 高×宽, 返回交集的面积, shape 刚好为 [A, B]


def jaccard(box_a, box_b):
    # A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]), 代表了 box_a 和 box_b 两个集合中任意两个 box之间的交并比
    inter = intersect(box_a, box_b) # 求任意两个box的交集面积, shape为[A, B], 即[num_obj, num_priors]
    area_a = ((box_a[:,2]-box_a[:,0]) * (box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:,2]-box_b[:,0]) * (box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter) # [A,B], 这里会将A中的元素复制B次
    union = area_a + area_b - inter
    return inter / union # [A, B], 返回任意两个box之间的交并比, res[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.

def encode(matched, priors, variances):
    # 对边框坐标进行编码, 需要宽度方差和高度方差两个参数, 具体公式可以参见原文公式(2)
    # matched: [num_priors,4] 存储的是与priorbox匹配的gtbox的坐标. 形式为(xmin, ymin, xmax, ymax)
    # priors: [num_priors, 4] 存储的是priorbox的坐标. 形式为(cx, cy, w, h)
    # return : encoded boxes: [num_priors, 4]
    g_cxy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] # 用互相匹配的gtbox的中心坐标减去priorbox的中心坐标, 获得中心坐标的偏移量
    g_cxy /= (variances[0]*priors[:, 2:]) # 令中心坐标分别除以 d_i^w 和 d_i^h, 正如原文公式所示
    #variances[0]为0.1, 令其分别乘以w和h, 得到d_i^w 和 d_i^h
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:] # 令互相匹配的gtbox的宽高除以priorbox的宽高.
    g_wh = torch.log(g_wh) / variances[1] # 这里这个variances[1]=0.2 不太懂是为什么.
    return torch.cat([g_cxy, g_wh], 1) # 将编码后的中心坐标和宽高``连接起来, 返回 [num_priors, 4]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # threshold: (float) 确定是否匹配的交并比阈值
    # truths: (tensor: [num_obj, 4]) 存储真实 box 的边框坐标
    # priors: (tensor: [num_priors, 4], 即[8732, 4]), 存储推荐框的坐标, 注意, 此时的框是 default box, 而不是 SSD 网络预测出来的框的坐标, 预测的结果存储在 loc_data中, 其 shape 为[num_obj, 8732, 4].
    # variances: cfg['variance'], [0.1, 0.2], 用于将坐标转换成方便训练的形式(参考RCNN系列对边框坐标的处理)
    # labels: (tensor: [num_obj]), 代表了每个真实 box 对应的类别的编号
    # loc_t: (tensor: [batches, 8732, 4]),
    # conf_t: (tensor: [batches, 8732]),
    # idx: batches 中图片的序号, 标识当前正在处理的 image 在 batches 中的序号
    overlaps = jaccard(truths, point_form(priors)) # [A, B], 返回任意两个box之间的交并比, overlaps[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.

    # 二部图匹配(Bipartite Matching)
    # [num_objs,1], 得到对于每个 gt box 来说的匹配度最高的 prior box, 前者存储交并比, 后者存储prior box在num_priors中的位置
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True) # keepdim=True, 因此shape为[num_objs,1]
    # [1, num_priors], 即[1,8732], 同理, 得到对于每个 prior box 来说的匹配度最高的 gt box
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_prior_idx.squeeze_(1) # 上面特意保留了维度(keepdim=True), 这里又都把维度 squeeze/reduce 了, 实际上只需用默认的 keepdim=False 就可以自动 squeeze/reduce 维度.
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 维度压缩后变为[num_priors], best_prior_idx 维度为[num_objs],
    # 该语句会将与gt box匹配度最好的prior box 的交并比置为 2, 确保其最大, 以免防止某些 gtbox 没有匹配的 priorbox.

    # 假想一种极端情况, 所有的priorbox与某个gtbox(标记为G)的交并比为1, 而其他gtbox分别有一个交并比
    # 最高的priorbox, 但是肯定小于1(因为其他的gtbox与G的交并比肯定小于1), 这样一来, 就会使得所有
    # 的priorbox都与G匹配, 为了防止这种情况, 我们将那些对gtbox来说, 具有最高交并比的priorbox,
    # 强制进行互相匹配, 即令best_truth_idx[best_prior_idx[j]] = j, 详细见下面的for循环

    # 注意!!: 因为 gt box 的数量要远远少于 prior box 的数量, 因此, 同一个 gt box 会与多个 prior box 匹配.
    for j in range(best_prior_idx.size(0)): # range:0~num_obj-1
        best_truth_idx[best_prior_idx[j]] = j
        # best_prior_idx[j] 代表与box_a的第j个box交并比最高的 prior box 的下标, 将与该 gtbox
        # 匹配度最好的 prior box 的下标改为j, 由此,完成了该 gtbox 与第j个 prior box 的匹配.
        # 这里的循环只会进行num_obj次, 剩余的匹配为 best_truth_idx 中原本的值.
        # 这里处理的情况是, priorbox中第i个box与gtbox中第k个box的交并比最高,
        # 即 best_truth_idx[i]= k
        # 但是对于best_prior_idx[k]来说, 它却与priorbox的第l个box有着最高的交并比,
        # 即best_prior_idx[k]=l
        # 而对于gtbox的另一个边框gtbox[j]来说, 它与priorbox[i]的交并比最大,
        # 即但是对于best_prior_idx[j] = i.
        # 那么, 此时, 我们就应该将best_truth_idx[i]= k 修改成 best_truth_idx[i]= j.
        # 即令 priorbox[i] 与 gtbox[j]对应.
        # 这样做的原因: 防止某个gtbox没有匹配的 prior box.
    mathes = truths[best_truth_idx]
    # truths 的shape 为[num_objs, 4], 而best_truth_idx是一个指示下标的列表, 列表长度为 8732,
    # 列表中的下标范围为0~num_objs-1, 代表的是与每个priorbox匹配的gtbox的下标
    # 上面的表达式会返回一个shape为 [num_priors, 4], 即 [8732, 4] 的tensor, 代表的就是与每个priorbox匹配的gtbox的坐标值.
    conf = labels[best_truth_idx]+1 # 与上面的语句道理差不多, 这里得到的是每个prior box匹配的类别编号, shape 为[8732]
    conf[best_truth_overlap < threshold] = 0 # 将与gtbox的交并比小于阈值的置为0 , 即认为是非物体框
    loc = encode(matches, priors, variances) # 返回编码后的中心坐标和宽高.
    loc_t[idx] = loc # 设置第idx张图片的gt编码坐标信息
    conf_t[idx] = conf # 设置第idx张图片的编号信息.(大于0即为物体编号, 认为有物体, 小于0认为是背景)
```

**代码流程为**:
1. 计算出每一个ground truth与每一个default boxes的jaccard overlap;
2. 挑出与每一个ground truth最匹配(重复度最高)的default boxes;
3. 挑出与每一个default boxes最匹配的ground truth;
4. 注意,最终的匹配结果要保证在每一个ground truth都有与之匹配的default boxes的基础上,可以存在多个default boxes匹配同一个ground truth,这就是
    ```
   for j in range(best_prior_idx.size(0)):
      best_truth_idx[best_prior_idx[j]] = j
   ```
    这一for循环完成的功能.
5. 将重复率低于阈值的标记为背景目标.

#### 损失函数
模型的整体损失函数如下:
<center>
$L ( x , c , l , g ) = \frac { 1 } { N } \left( L _ { c o n f } ( x , c ) + \alpha L _ { l o c } ( x , l , g ) \right)$
</center>

其中N表示匹配的default boxes的数目,其中定位误差的计算方式如下:
<center>
$L _ { l o c } ( x , l , g ) = \sum _ { i \in P o s } ^ { N } \sum _ { m \in \{ c x , c y , w , h \} } x _ { i j } ^ { k } \operatorname { smooth } _ { \mathrm { LI } } \left( l _ { i } ^ { m } - \hat { g } _ { j } ^ { m } \right)$

$\hat { g } _ { j } ^ { c x } = \left( g _ { j } ^ { c x } - d _ { i } ^ { c x } \right) / d _ { i } ^ { w } \quad \hat { g } _ { j } ^ { c y } = \left( g _ { j } ^ { c y } - d _ { i } ^ { c y } \right) / d _ { i } ^ { h }$

$\hat { g } _ { j } ^ { w } = \log \left( \frac { g _ { j } ^ { w } } { d _ { i } ^ { w } } \right) \quad \hat { g } _ { j } ^ { h } = \log \left( \frac { g _ { j } ^ { h } } { d _ { i } ^ { h } } \right)$
</center>

与Faster R-CNN类似,预测相对于default boxes的中心坐标的偏差,以及其宽和高.其中$x_ij^p$表示,将第i个default boxes与类别p的第j个ground truth进行匹配,这里,要将ground truth转换为相对于default boxes的偏移量.
置信度损失如下:
<center>
$L _ { c o n f } ( x , c ) = - \sum _ { i \in P o s } ^ { N } x _ { i j } ^ { p } \log \left( \hat { c } _ { i } ^ { p } \right) - \sum _ { i \in N e g } \log \left( \hat { c } _ { i } ^ { 0 } \right) \quad  where  \quad \hat { c } _ { i } ^ { p } = \frac { \exp \left( c _ { i } ^ { p } \right) } { \sum _ { p } \exp \left( c _ { i } ^ { p } \right) }$
</center>

其中$\alpha$设置为1.
代码如下:
```python
# layers/modules/multibox_loss.py

class MultiBoxLoss(nn.Module):
    # 计算目标:
    # 输出那些与真实框的iou大于一定阈值的框的下标.
    # 根据与真实框的偏移量输出localization目标
    # 用难样例挖掘算法去除大量负样本(默认正负样本比例为1:3)
    # 目标损失:
    # L(x,c,l,g) = (Lconf(x,c) + αLloc(x,l,g)) / N
    # 参数:
    # c: 类别置信度(class confidences)
    # l: 预测的框(predicted boxes)
    # g: 真实框(ground truth boxes)
    # N: 匹配到的框的数量(number of matched default boxes)

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes= num_classes # 列表数
        self.threshold = overlap_thresh # 交并比阈值, 0.5
        self.background_label = bkg_label # 背景标签, 0
        self.use_prior_for_matching = prior_for_matching # True 没卵用
        self.do_neg_mining = neg_mining # True, 没卵用
        self.negpos_ratio = neg_pos # 负样本和正样本的比例, 3:1
        self.neg_overlap = neg_overlap # 0.5 判定负样本的阈值.
        self.encode_target = encode_target # False 没卵用
        self.variance = cfg["variance"]

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        # loc_data: [batch_size, 8732, 4]
        # conf_data: [batch_size, 8732, 21]
        # priors: [8732, 4] default box 对于任意的图片, 都是相同的, 因此无需带有 batch 维度
        num = loc_data.size(0) # num = batch_size
        priors = priors[:loc_data.size(1), :] # loc_data.size(1) = 8732, 因此 priors 维持不变
        num_priors = (priors.size(0)) # num_priors = 8732
        num_classes = self.num_classes # num_classes = 21 (默认为voc数据集)

        # 将priors(default boxes)和ground truth boxes匹配
        loc_t = torch.Tensor(num, num_priors, 4) # shape:[batch_size, 8732, 4]
        conf_t = torch.LongTensor(num, num_priors) # shape:[batch_size, 8732]
        for idx in range(num):
            # targets是列表, 列表的长度为batch_size, 列表中每个元素为一个 tensor,
            # 其 shape 为 [num_objs, 5], 其中 num_objs 为当前图片中物体的数量, 第二维前4个元素为边框坐标, 最后一个元素为类别编号(1~20)
            truths = targets[idx][:, :-1].data # [num_objs, 4]
            labels = targets[idx][:, -1].data # [num_objs] 使用的是 -1, 而不是 -1:, 因此, 返回的维度变少了
            defaults = priors.data # [8732, 4]
            # from ..box_utils import match
            # 关键函数, 实现候选框与真实框之间的匹配, 注意是候选框而不是预测结果框! 这个函数实现较为复杂, 会在后面着重讲解
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx) # 注意! 要清楚 Python 中的参数传递机制, 此处在函数内部会改变 loc_t, conf_t 的值, 关于 match 的详细讲解可以看后面的代码解析
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 用Variable封装loc_t, 新版本的 PyTorch 无需这么做, 只需要将 requires_grad 属性设置为 True 就行了
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 # 筛选出 >0 的box下标(大部分都是=0的)
        num_pos = pos.sum(dim=1, keepdim=True) # 求和, 取得满足条件的box的数量, [batch_size, num_gt_threshold]

        # 位置(localization)损失函数, 使用 Smooth L1 函数求损失
        # loc_data:[batch, num_priors, 4]
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, 4], 复制下标成坐标格式, 以便获取坐标值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)# 获取预测结果值
        loc_t = loc_t[pos_idx].view(-1, 4) # 获取gt值
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # 计算损失

        # 计算最大的置信度, 以进行难负样本挖掘
        # conf_data: [batch, num_priors, num_classes]
        # batch_conf: [batch, num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes) # reshape

        # conf_t: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
        loss_c[pos.view(-1, 1)] = 0 # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
        # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
        loss_c = loss_c.view(num, -1) # reshape
        # 进行降序排序, 并获取到排序的下标
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标
        _, idx_rank = loss_idx.sort(1)
        # num_pos: [batch, 1], 统计每个样本中的obj个数
        num_pos = pos.long().sum(1, keepdim=True)
        # 根据obj的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 获取到负样本的下标
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算包括正样本和负样本的置信度损失
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg: [batch, num_priors]
        # neg_idx: [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 按照pos_idx和neg_idx筛选目标数据
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 计算二者的交叉熵
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # 将损失函数归一化后返回
        N = num_pos.data.sum()
        loss_l = loss_l / N
        loss_c = loss_c / N
        return loss_l, loss_c
```

#### 确定default boxes的尺度以及比率

实验表明使用低层的feature maps可以提高语义分割质量,因为底层的feature maps包含输入目标的细节信息.
来自于一个网络不同层的feature maps具有不同的感受野,在SSD网络中,default boxes不需要与每一层的真实的感受野严格匹配.作者将default boxes的大小设计为与特定尺度的目标相关联.假设在模型中使用了m层特征图.每一层特征图的default boxes的尺度计算如下:
<center>
$s _ { k } = s _ { \min } + \frac { s _ { \max } - s _ { \min } } { m - 1 } ( k - 1 ) , \quad k \in [ 1 , m ]$
</center>

其中$s_{min}$为$0.2$,$s_{max}$为$0.9$,分别代表最低、最高层的尺度.
对于,每一层中的default又引入了五种不同的比率,即:
<center>
${ 1,2,3 , \frac { 1 } { 2 } , \frac { 1 } { 3 }}$
</center>

依据比率，得出宽的计算公式为:
<center>
$w _ { k } ^ { a } = s _ { k } \sqrt { a _ { r } }$
</center>

高的计算公式为:
<center>
$h _ { k } ^ { a } = s _ { k } / \sqrt { a _ { r } }$
</center>

对于比率1,又额外定义了一个尺度,计算如下:
<center>
$s _ { k } ^ { \prime } = \sqrt { S _ { k } S _ { k + 1 } }$
</center>

这样,每一层特征图的每一个位置上便有六个不同比率的default boxes,将每一个位置上的6个default boxes的中心坐标设置为:
<center>
$\left( \frac { i + 0.5 } { \left| f _ { k } \right| } , \frac { j + 0.5 } { \left| f _ { k } \right| } \right)$
</center>

其中 $\left| f _ { k } \right|$表示第k个特征图的大小，$i , j \in \left[ 0 , \left| f _ { k } \right|\right.]$,对应特征图上所有可能的位置点.

实际中,不同的数据集适用于不同的尺度以及比例,若数据集中包含有更多的小目标,则需要设计更多的小尺度default boxes,相应的,若包含有更多的大目标,则需要设计更多的大尺度default boxes.
实现这一功能的代码如下:借鉴自:[代码出处 ](https://hellozhaozheng.github.io/z_post/PyTorch-SSD/#MultiBox, "代码出处")
```python
# `layers/functions/prior_box.py`

class PriorBox(object):
    # 所谓priorbox实际上就是网格中每一个cell推荐的box
    def __init__(self, cfg):
        # 在SSD的init中, cfg=(coco, voc)[num_classes=21]
        # coco, voc的相关配置都来自于data/cfg.py 文件
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): # 存放的是feature map的尺寸:38,19,10,5,3,1
            # from itertools import product as product
            for i, j in product(range(f), repeat=2):
                # 这里实际上可以用最普通的for循环嵌套来代替, 主要目的是产生anchor的坐标(i,j)

                f_k = self.image_size / self.steps[k] # steps=[8,16,32,64,100,300]. f_k大约为feature map的尺寸
                # 求得center的坐标, 浮点类型. 实际上, 这里也可以直接使用整数类型的 `f`, 计算上没太大差别
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该零cy与i对应.

                # aspect_ratios 为1时对应的box
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                # 综上, 每个卷积特征图谱上每个像素点最终产生的 box 数量要么为4, 要么为6, 根据不同情况可自行修改.
        output = torch.Tensor(mean).view(-1,4)
        if self.clip:
            output.clamp_(max=1, min=0) # clamp_ 是clamp的原地执行版本
        return output # 输出default box坐标(可以理解为anchor box)
```
#### Hard negative mining

这一策略主要用于解决正负样本数目不均衡的问题,在进行边框匹配之后,大多数的default boxes都是负样本.这一结果会导致样本不平衡.因而在训练时,没有使用所有的负样本,而是首先依据每一个default box的置信度损失进行排序,选出最高的几个,使得正负样本的比例为1:3.使用选出的样本进行训练.

## 实验结果
|模型|mAP|FPS|
|---|---|---|
|SSD 300|74.3%|59|
|SSD 512|76.9%|-|
|Faster R-CNN|73.2%|7|
|YOLOV1|63.4%|45|
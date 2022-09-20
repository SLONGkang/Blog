---
title: Digit Recognizer By CNN (Conv Natural Network)
date: 2022-09-20 02:50:24
tags: 
	- CNN 
	- 神经网络
	- 机器学习
	- 残差网络
cover: https://pic3.zhimg.com/v2-8e7a76f466037e77f3c232799a56b630_1440w.jpg?source=172ae18b
description: 在MINST数据集上应用残差卷积网络
copyright_author: KongFuKang-功夫康
copyright_author_href: https://kongfukang.com/
copyright_url: https://kongfukang.com/4000/2022/09/18/MCMC/
copyright_info: 本博客所有文章除特别声明外，均采用<a href = 'https://creativecommons.org/licenses/by-nc-sa/4.0/' >CC BY-NC-SA 4.0</a> 许可协议。转载请注明来自 KongFuKang-功夫康！

---
# CNN for this Question
[http://t.csdn.cn/krfpj](http://t.csdn.cn/krfpj)
更多信息参考keras官网

[主页 - Keras 中文文档](https://keras.io/zh/)

CNN for this Question:

1.介绍

5层**Sequential**网络（Functional、Model Subclassing），使用tensorflow.keras构建。

2.数据预处理

1）查看数据可知训练数据和测试数据的数据模式；

train.csv：[label, pixel**x**[**x** 0:783]], **x** = i * 28 + j,说明pixel**x**是第i行第j列的像素，整个图像为28*28像素大小。

test.csv: [pixel**x**[**x** 0:783]], 同上，只是缺少了label列。

2）数据异常值检查处理和标准化

空值、异常值检测处理。

data.isnull().any().describe()

本实例中，若有空值或者异常值，可填充为0，但是实验提供的数据没有空数据。

标准化：将数据从0~255标准化至0.0~1.0，可以减少色彩差别引起的影响，同时加快卷积网络的拟合

3）数据塑性

目前的像素数据为784*1的一维数据，可reshape为28*28*1的二维数据，大小为28*28，通道数为1（MNIST因为是灰度图像，只有一个通道，若为RGB则每个像素信息需要3通道）。

将label信息转换为独热编码（eg:1→[0,1,0,0,0,0,0,0,0,0]; 2→[0,0,1,0,0,0,0,0,0,0]）to_categorical()

将训练数据随机分为十份，其中一份作为验证集，其余九份作为训练集。train_test_split()

3.构建CNN

1)使用卷积神经网络，网络结构为Input→[[Conv2D →relu]*2 →MaxPool2D →Dropout]*2 →Flatten →Dense →Dropout →Out.

![在这里插入图片描述](https://img-blog.csdnimg.cn/225d7a7750c34b3495be20b3cf7e8e6e.png)


【为什么这样设计网络结构】

[深度学习项目_3 网络模型的构建( tensorflow)](https://blog.csdn.net/chumingqian/article/details/123972605)

2)优化器

可选用传统梯度优化：BGD、**SGD**、MBGD

动量momentum

AdaGrad算法

**RMSprop算法**

Adam算法

各个算法的介绍：

[优化器(Optimizer)（SGD、Momentum、AdaGrad、RMSProp、Adam）_CityD的博客-CSDN博客_优化器](https://blog.csdn.net/tcn760/article/details/123965374)

3)回调函数

本实例中我使用ReduceLROnPlateau（自适应调整学习率）

ReduceLROnPlateau是基于验证集误差测量实现动态学习率缩减，当发现loss不再降低或者acc不再提高之后，降低学习率。

`torch**.**optim**.**lr_scheduler**.**ReduceLROnPlateau(optimizer,mode**=**'min',factor,patience**=**10,
verbose**=**False,threshold**=**0.0001,threshold_mode**=**'rel',cooldown**=**0,min_lr**=**0,eps**=**1e-08)`

- optimizer：表示网络优化器。
- mode(str)：有两种模式分别为'min'和'max'。min表示当loss不再下降的时候，学习率将减小；max表示当loss不再上升的时候，学习率将减小。默认值为'min'。
- factor：表示学习率每次降低多少，new_lr=old_lr*factor。
- patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率。
- verbose(bool)-如果为True，则为每次更新向stdout输出一条消息。默认值：False。
- threshold(float)-测量新最佳值的阈值，仅关注重大变化。默认值：1e-4。
- cooldown：减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
- min_lr：学习率的下限。
- eps：适用于lr的最小衰减。如果新旧lr之间的差异小于eps，则忽略更新。默认值：1e-8。

更多keras的回调函数：

[机器学习笔记 - Keras中的回调函数Callback使用教程_坐望云起的博客-CSDN博客_keras 回调函数](https://blog.csdn.net/bashendixie5/article/details/124207898)

4）数据增强

为了避免过度拟合问题，我们需要人工扩展手写数字数据集。我们可以使您现有的数据集更大。这个想法是通过小的转换来改变训练数据，以再现当有人在写数字时发生的变化。
例如，数字不居中比例不相同（一些人用大/小数字书写）图像旋转。。。
以改变阵列表示的方式改变训练数据，同时保持标签不变的方法称为数据增强技术。人们使用的一些常用增强功能有灰度、水平翻转、垂直翻转、随机裁剪、颜色抖动、平移、旋转等等。
通过对我们的训练数据应用这些转换，我们可以轻松地将训练示例的数量增加一倍或三倍，并创建一个非常健壮的模型。

使用`ImageDataGenerator`

5）训练模型（fit）

训练过程中可以看出优化器会影响最终的结果精度，epoch的大小会影响学习的速度和快慢
	![在这里插入图片描述](https://img-blog.csdnimg.cn/6b8d891367044027a8de56405573e8e2.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/bb730c311dd0400e8f0f15f2935bdd51.png)




---

4.评估模型

数据曲线：

1》CNN, RMSprop, epoch=30

![在这里插入图片描述](https://img-blog.csdnimg.cn/5d1e6016764d4d0c8f6af985735241b5.png)


2》CNN, SGD, epoch=30

![](https://img-blog.csdnimg.cn/579ab4566b194214b2d57a292332c17a.png)


5.预测结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/9475d72459f64f589ff67f8349c5ad25.png)


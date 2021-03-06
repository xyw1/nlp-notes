# 1 注意力机制介绍🌟🌟

学习目标：*本节主要介绍注意力机制的核心思想。*

相关知识点：

*注意力机制*

## PART1 浅谈注意力

注意力(attention)是人类学习中必不可少的要素，比如我们去阅读一个文章，或者试着去理解一本书中作者想表达的意思，我们通常在阅读过程中会把注意力放在比较重要的环节上，而不是去把每个细节都会一一记住。人的记忆是有限的，抓重点的学习习惯往往会得到事半功倍的效果。

那既然注意力这么重要，我们有没有办法把它用在AI应用中呢? 这就是注意力机制(attention mechanism)! 在过去几年取得了飞速的发展，而且已经成为很多应用的标配。把注意力机制放到机器上，其实就是让机器学习选择性地去学习，同时知道如何把注意力放在更重要的事情上，比如对于一段文字来讲，理解其含义可能只需要把重点放在几个核心的单词上。

## PART2 注意力机制类别

<img src="/Users/yunwanxu/Library/Application Support/typora-user-images/image-20210814201314062.png" alt="image-20210814201314062" style="zoom:33%;" />

注意力机制在不同应用下的使用也大同小异。对于图片来讲，注意力需要放到某一个区域上; 对于文本来讲，注意力需要放在某几个单词上; 另外，这里所讲的自注意力机制跟传统的注意力机制有所不一样，能够更有效地解决梯度，并行化的问题。

# 2 计算机视觉中的注意力机制🌟🌟🌟

学习目标：*本节主要讲解如何把注意力机制用在看图说话任务中。*

相关知识点：

*看图说话*

## PART1 Image Captioning

看图说话(image captioning)是指，根据给定的图片生成一段文本描述，这个描述就是对于图片的理解。 实际上，这个问题可以理解为把一个图片转换成文本，图片理解这块可采用CNN模型，文本生成模块可采用LSTM模块。



## PART2 这种架构的问题

## PART3 加入注意力机制

# 3 序列模型中的注意力机制🌟🌟🌟🌟

## PART1 Seq2Seq的一些问题

## PART2 Seq2Seq加入注意力机制

# 4 自注意力机制与Transformer🌟🌟🌟🌟

## PART1 自注意力机制介绍

## PART2 自注意力机制细节

## PART3 位置编码

# 5 代码实战：Transformer代码解读与实战🌟🌟🌟🌟

## PART1 Transformer代码解读与实战
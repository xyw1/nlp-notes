# 1 文本生成任务🌟🌟

## PART1 常见的文本生成任务

### 文本生成

文本生成是seq2seq模型最重要的应用场景

- Machine Translation

- Summarization
  - 长文本->短文本

- Creative Writing
  - 机器自动生成report
  - 机器读一个比较长的年报，输出核心观点，写成报告

- Image Captioning
  - 读图 生成图片描述

### Machine Translation

- 传统方法：

分析中文语法（语法树）---（语言学家专家知识）-->转换成英文语法树

- 现在的方法

端到端:

最常见的就是seq2seq

### Creative Writing

- 产品描述->营销文案
- 研报->提取重要信息生成规范报告（常见应用场景，特别是金融行业）
- 学生学习数据->report

### Image Captioning

给定图片生成描述



# 2 Seq2Seq详解🌟🌟🌟

## PART1 设计模型的核心思想

### 如何设计可以生成文本的模型

Input Text --> Text Understading --> Text Generation

所以，这里的核心有两点，分别是实现**文本理解**和**生成文本**，还有就是如何把这两部分拼在一起。为了能做到这一点，我们可以先回想一下之前讲过的内容。为了解决**文本理解的问题，回想一下我们当初如何用LSTM来做文本分类的**; 为了**解决文本生成问题，回想一下当初如何用训练好的RNN来做生成文本的**;如果能把这两点理解清楚，最后需要考虑的是如何**把这两个模块拼接在一起**。

### Recap: LSTM/RNN用于文本分类

用模型最后一个时刻的输出，当作对整个文本的理解

可以在模型最后套一个SVM

### Recap: LSTM/RNN用于语言模型

假设已经有了一个语言模型，每次根据上一时刻的输出决定此刻的输出，最后输出 <end>

### 思路：把两者合在一起

文本分类的最后一个单元的输出是后一个单元的输入，作为二者的纽带。

## PART2 Seq2Seq模型

### seq2seq模型

c=h4

![image-20210722210640206](/Users/yunwanxu/Library/Application Support/typora-user-images/image-20210722210640206.png)

### seq2seq细节

第一部分是encoder

c是纽带

第二部分是decoder

Seq2Seq模型非常重要，面试中也大概率会问到，它的细节务必要掌握。另外，在上面所讲解的Seq2Seq模型是最经典的版本，在这个基础上，我们可以加入注意力机制等技术来提升它的表达能力。另外，Seq2Seq模型的生成部分其实至关重要，因为它关系到生成出来的质量。一般在实际工作中，我们都会对它最终的输出做一些控制，确保生成我们想要的结果。

# 3 Decoding与Beam Search🌟🌟🌟🌟

## PART1 Greedy Decoding

### Inference/Decoding-Greedy Decoding

上面模型的问题在于，输出序列的时候，前一步关注概率最大的单词，输出的结果可能有问题，会影响到后面的结果（这种方法也叫greedy decoding）

简单来说，Greedy Decoding就是每次选择概率值最大的对应的单词。但这样做的缺点是，局部最优并不等于全局最好的，而且一旦选错了，后续生成的内容很可能也是错误的，具有错误的累加效果。对于此问题，更好的解决方法是每次考虑更多的可能性。

## PART2 暴力搜索

### 暴力搜索（Exhaustive search）

V种可能，序列长度为M，总共V^M种可能

暴力搜索显然不太现实，因为时间复杂度太高，但至少是有价值的。碰到任何比较难的问题，我们首先可以试着想出最简单的方法，这种方法往往是暴力搜索，再基于暴力搜索思考如何一步步改进，有可能最终找到更好的解法。在这个过程中，暴力搜索方法通常也会给到很好的启发。

折衷：beam search

## PART3 Beam Search

Greedy search考虑top1,暴力搜索考虑所有，Beam search考虑top k, beam size=k



时间复杂度 需要复习
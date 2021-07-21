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

### Recap: LSTM/RNN用于文本分类-1

### Recap: LSTM/RNN用于文本分类-2

### 思路：把两者合在一起

## PART2 Seq2Seq模型

# 3 Decoding与Beam Search🌟🌟🌟🌟

## PART1 Greedy Decoding

## PART2 暴力搜索

## PART3 Beam Search
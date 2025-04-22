# 背景   
1. 在计算机视觉领域：   
- 在计算机视觉领域的发展早期，研究人员发现，可以先在一个大型的数据集（比如 ImageNet）上训练一个卷积神经网络（CNN）模型。训练完成后，这个模型就能学到很多通用的图像特征（比如边缘、形状、纹理等），这些特征对各种视觉任务都是有用的。

- 然后，我们就可以把这个已经训练好的模型迁移到其他计算机视觉任务上（比如图像分类、物体检测、图像分割等），通过微调（fine-tuning）或者作为特征提取器来用，从而提升这些任务的性能，即让它们表现得更好、识别更准。

- 这就是“迁移学习（Transfer Learning）”的一个经典做法，也说明了CNN模型具有良好的通用性。
2. 在自然语言处理领域：
- 但是在 BERT 出现之前，NLP 领域并没有一个像 CNN + ImageNet 那样的统一框架——也就是说，没有一种预训练的深度模型可以很好地迁移到各种 NLP 任务（如问答、情感分析、命名实体识别等）中。

- 因为缺少统一的预训练模型，NLP 的研究者或者工程师们往往要为每个具体任务单独设计神经网络结构并从头训练，这既费时又不容易取得很好的效果。

- BERT 可以在大规模语料上预训练，然后迁移到不同的 NLP 任务中，取得很好的效果，像“CV 里的 CNN+ImageNet”那样，在 NLP 里实现了“预训练 + 微调”的范式。

- BERT 不仅让模型训练过程更简单（不用从零设计架构），还能让模型效果更好。

**BERT**也是站在巨人的肩膀上的。

# 摘要

**BERT**，它的全称是 Bidirectional Encoder Representations from Transformers，即“来自 Transformer 的双向编码器表示”。

与当时的一些语言表示模型（如 Peters 的 ELMo 和 Radford 的 GPT）不同，BERT 的设计目标是在所有网络层中同时考虑句子的左侧上下文和右侧上下文，来预训练深层的双向语言表示。

这里来解释一下为什么不同：

**ELMo（Embeddings from Language Models，2018）**：
- 特点：双向 LSTM
- 原理： ELMo 使用一个前向（left-to-right）LSTM 和一个后向（right-to-left）LSTM，分别读取句子。
- 问题： 虽然它是“前向 + 后向”，但这两个方向是分开训练的，不是“同时考虑左右”。
- 层级处理： 它的双向性不是在每一层都共同作用，而是最后将两个方向的输出拼接起来。

所以，ELMo 是“浅层双向”而不是 BERT 那种“深层双向”。

**GPT（Generative Pre-trained Transformer，2018）**：
- 特点：单向 Transformer（只看左边）
- 原理： GPT 使用标准的 Transformer decoder 架构，在训练时只看每个词的左边上下文（即过去的词）。
- 限制： 模型在预测一个词时，不能看到右边，否则就不是“自回归语言建模”了。

所以 GPT 是“严格单向”，只有“从左到右”的信息流。

**BERT（Bidirectional Encoder Representations from Transformers，2018）**：
- 特点：真正的“深层双向”Transformer
- 原理： BERT 使用 Transformer encoder，训练时通过 Masked Language Modeling（MLM），随机遮掉句中的部分词，然后让模型根据上下文去预测它。
- 优势： 因为每个被遮住的词周围的词都可以来自左边和右边，所以 BERT 在每一层都同时考虑左、右上下文。
> 这里我对这个优势的理解是：它能够在模型的每一层中，同时利用一个词的左边和右边的信息（上下文）来理解这个词的含义。

BERT 是“jointly conditioning on both left and right context in all layers”，这是它和 ELMo/GPT 最大的区别。



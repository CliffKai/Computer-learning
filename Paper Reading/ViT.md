# 背景    

1. 在计算机视觉领域：

- 在ViT出现之前，卷积神经网络（CNN）长期以来一直是计算机视觉领域的主导方法。  
  像AlexNet、VGG、ResNet等一系列模型在大型数据集（如ImageNet）上取得了巨大成功，推动了视觉任务（如图像分类、目标检测、图像分割等）的进步。

- CNN 之所以成功，很大程度上是因为其**局部感受野**、**权值共享**和**平移不变性**等设计，使得模型能有效提取局部特征（如边缘、纹理、形状等），同时又能保持计算效率。

- 当时研究人员普遍认为：视觉任务需要强烈的归纳偏置（inductive bias），比如局部相关性和平移等变性，因此CNN天然适合视觉。

> 局部感受野（Local Receptive Field）：在卷积神经网络（CNN）中，一个神经元不会感知整张输入图片，而只关注输入数据中的一个小区域，这个小区域就叫做局部感受野。每个卷积核（滤波器）扫描输入时，只处理一部分局部信息。                
权值共享（Weight Sharing）：在CNN中，同一个卷积核在整张图片上滑动并应用到不同位置，意味着不同位置用的是同一组参数（权值），这就叫做权值共享。                 
平移不变性（Translation Invariance）：当输入图像中的物体发生小幅度平移时，CNN提取到的特征仍然保持稳定，最终的识别结果不会大幅变化。              
平移不变性（Translation Invariance）：指的是当输入图像中的物体发生小幅度平移时，CNN提取到的特征仍然保持稳定，最终的识别结果不会大幅变化。             
 归纳偏置（Inductive Bias）：归纳偏置是指模型在学习数据之前，人为引入的假设或限制，目的是帮助模型更高效、更快地学习任务。也可以理解为：在设计模型时，我们预设了某些世界的规律。（CNN的局部感受野、权值共享就是针对图像数据特性的归纳偏置。）                    


2. Transformer 在自然语言处理领域的发展：

- 与此同时，在自然语言处理（NLP）领域，Transformer架构（由Vaswani等人在2017年提出）因其出色的建模长距离依赖关系的能力，在诸如机器翻译、文本生成、问答系统等任务中迅速取代了传统的RNN、LSTM等序列模型，成为主流。

- Transformer 的成功启发了视觉领域的研究者们思考：**能否用Transformer代替CNN，直接处理图像任务？**

3. ViT 出现前的探索和挑战：

- 尽管Transformer在NLP中表现优异，但直接将Transformer应用到视觉任务中存在困难，主要问题是：
  - 图像不像文本那样是天然的序列数据，需要设计一种方式把2D图像转成序列输入（况且将图片转换为1D序列会有各种各样的问题，正文中我们会一一进行讲解）。
  - Transformer缺少像CNN那样的局部归纳偏置，因此在小规模数据集上训练效果不好，很容易过拟合。
  - Transformer对数据量非常敏感，需要海量数据才能充分发挥其能力。

- 在ViT提出之前，有一些工作（比如"Image Transformer"、"Stand-Alone Self-Attention"）探索了在局部窗口内使用自注意力（self-attention）机制，但尚未完全取代CNN。

4. ViT 的突破：

- ViT由Google Research的团队提出（论文："An Image is Worth 16x16 Words"，2020年），首次成功地证明了**纯Transformer结构在图像分类任务上可以媲美甚至超越CNN**，特别是在超大规模数据集（如JFT-300M）预训练后。

- 核心做法：
  - 把输入图像切分成固定大小的小块（patches），每个小块展平成一个向量。
  - 将这些向量序列化后，送入标准的Transformer编码器进行处理。
  - 不引入CNN特有的归纳偏置，仅靠大数据和强大的建模能力学习视觉特征。

- ViT也遵循了类似BERT和CV领域CNN+ImageNet的“预训练+微调（fine-tuning）”范式：  
  在超大规模数据集上进行预训练，再在下游任务（如图像分类）上进行微调，取得了极佳效果。

# 摘要    

1. 在背景上：                
Transformer在自然语言处理（NLP）领域已经成为了标准（比如BERT、GPT都用Transformer），但在计算机视觉（CV）领域，Transformer的应用还比较少。            
计算机视觉领域当前主要还是依赖卷积神经网络（CNNs），如ResNet、EfficientNet等。    

2. 在使用上：           
目前在视觉任务中，注意力机制（Attention）通常是：与CNN结合使用（例如SE-Net，引入Squeeze-and-Excitation模块）、局部替代CNN某些部分（例如用Self-Attention代替卷积核，但整体仍是CNN结构）      
在已有的一些使用中，注意力机制一般是辅助CNN，而不是彻底取代CNN。           

3. 本文工作：
不需要依赖CNN结构，只要直接把图片切成小块（patches），然后把这些小块的序列送进纯Transformer模型，就能很好地完成图像分类任务。     
- pure Transformer：纯Transformer结构，没有CNN。      
- image patches：将一张图像切成小块（比如16×16像素的小块）。    

4. 效果：
当ViT在大量数据上预训练（比如用JFT-300M这种超大数据集），然后迁移到其他中小规模的数据集（如ImageNet、CIFAR-100、VTAB）进行微调时，ViT在这些任务上达到了非常优秀的效果，可以超过当前最先进的CNN模型（比如EfficientNet）。
而且，训练所需的计算资源还比CNN更少（这个只需要看看即可，对Google来说多少计算资源可能都算少）。 
> 这里凸显出了很重要的一点，无论是BERT还是ViT，当前这种模型的主流方向都是大规模数据集上预训练，然后迁移到较小的特定任务数据集上学习，进行微调。

# 引言      

1. 引言上来首先进行背景介绍：
- **Transformer模型在自然语言处理（NLP）中的成功**  
  - 自从**Vaswani et al., 2017**提出Transformer之后，Transformer成为了NLP领域的主流架构。
  - 典型做法是：**先在大规模语料上预训练**，再在小任务上**微调**（fine-tune）。
  - Transformer因为**计算效率高、可扩展性好**，使得可以训练拥有**上百亿参数**的大模型（如Brown et al. - GPT3:Language Models are Few-Shot Learners, 2020；Lepikhin et al., 2020 - GShard: Scaling Giant Models with Conditional Computation）。
  - **非常重要的一点是**：随着模型和数据规模不断增大，目前性能还没有饱和的迹象。


2. 转向计算机视觉（CV）领域的现状

- **计算机视觉（CV）领域仍以卷积神经网络（CNN）为主流**  
  - 像**LeCun et al., 1989**和**Krizhevsky et al., 2012**这样的经典工作确立了CNN在CV领域的统治地位。
  - **受到NLP成功启发**，有些工作开始尝试：
    - 将CNN与**self-attention机制结合**（如Wang et al., 2018；Carion et al., 2020）
    - 甚至**完全用self-attention取代卷积**（如Ramachandran et al., 2019；Wang et al., 2020a）
  - 但即便如此，这些方法目前还**没有在硬件上实现有效扩展**，并且在大规模视觉任务中，像**ResNet这种传统CNN架构仍然是主流**（Mahajan et al., 2018；Xie et al., 2020；Kolesnikov et al., 2020）。


3. 本文提出的研究动机

- **想法**：既然Transformer在NLP领域扩展得这么好，那能不能**直接拿Transformer用在图像上**，而且尽可能**少做改动**？
- **具体做法**：
  - 把图像**切成小块（patches）**
  - 将这些小块**线性嵌入（linear embedding）**成向量序列
  - 把它们喂给一个**标准的Transformer**（就像NLP里把单词序列喂给Transformer一样）
- **训练方式**：在图像分类任务上用**有监督学习**训练。

---

### 第四段：初步发现——中小数据集上效果一般

- **如果只在像ImageNet这种中型数据集上训练**，而且**没有用很强的正则化**，这种Transformer模型的效果：
  - **比同规模的ResNet差几个百分点**。
- **原因推测**：
  - Transformer相比CNN，**缺少某些归纳偏置（inductive bias）**，比如：
    - **平移等变性**（translation equivariance）
    - **局部性**（locality）
  - 因此Transformer在数据量不足时**泛化能力差一些**。

---

### 第五段：主要发现——大数据集训练后效果很好

- **如果用非常大的数据集（1400万到3亿张图）来训练**，情况就不同了：
  - 大数据量带来了**更强的归纳偏置学习能力**。
  - **Vision Transformer (ViT)** 取得了非常优秀的结果：
    - 在ImageNet-21k、JFT-300M预训练后，再在小数据集微调，效果极好。
    - 在多个图像识别任务上，**ViT可以达到或超过最先进水平**：
      - ImageNet上88.55%
      - ImageNet-Real上90.72%
      - CIFAR-100上94.55%
      - VTAB（19个任务合集）上77.63%

---

### 总结一下：
这段文字整体传达的**核心信息**是：

- Transformer在NLP里取得了巨大成功。
- 把标准Transformer稍作修改应用到图像上（Vision Transformer, ViT）是可行的。
- 但在小数据集上效果一般，因为Transformer缺少CNN那种先天的“归纳偏置”。
- 只要**用足够大的数据集预训练**，ViT可以非常强大，甚至超过现有最好的卷积网络。

---

要不要我顺便也帮你把这里提到的一些重要名词（比如**归纳偏置**、**Transformer机制在图像上应用的挑战**）做进一步深入解释？🌟
要的话告诉我！

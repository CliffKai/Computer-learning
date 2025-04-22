URBANWORLD: AN URBAN WORLD MODEL FOR 3D CITY GENERATION 

# 摘要 

Urban World集成了四个阶段: 
1. 从公开访问的OSM[^1]数据生成3D布局 
2. 使用城市多模态大语言模型(Urban MLLM)进行城市场景规划与设计 
3. 利用3D diffusion技术进行渲染 
4. MLLM辅助场景细化 

# 1.引言 
Urban World Models定义(未实现): 
1. 逼真且可交互 
2. 可定制可控 
3. 支持具身智能体学习 

开放平台[Omniverse](https://www.nvidia.com/en-us/omniverse/):当前已有的比较成熟的商业平台,可用于创建和模拟详细3D世界环境.

**UrbanWorld**定义:能够根据文本和OpenStreetMap2(OSM)提示自动创建可控的3D城市环境的生成式城市世界模型 
> 说明一下,我并没有找到OpenStreetMap2相关的数据和资料 

UrbanWorld框架包含四个关键模块:
1. 基于OSM数据,利用Blender3[^2]自动生成3D城市布局并进行详细的资产处理 
2. UrbanWorld采用经过微调的特定与城市的多模态大语言模型(Urban MLLM)根据用户指令有效地规划和设计城市场景,,生成详细的城市元素文本描述 
> 这一段我不太理解什么意思,贴一下原文:
> > Then, UrbanWorld adopts a fine-tuned urbanspecific multimodal large language model (called Urban MLLM) to effectively plan and design urban scenes following user instructions, generating detailed textual descriptions of urban elements. 
3. UrbanWorld集成了一个基于纹理扩散和细化地3D资源渲染器,可以通过文本和视觉条件灵活控制 
4. 为了进一步优化视觉外观,UrbanWorld利用 Urban MLLM 对三维城市场景进行仔细检查,提出详细的改进建议,并激活额外的渲染迭代
> 按照作者的说法:该框架充分利用了扩散模型的可控生成能力和大型语言模型的推理和规划能力,有助于实现高保真城市环境渲染和卓越的生成灵活性 

# 2.相关工作
## 2.1 三维城市场景生成
目前有三类基于深度学习的自动化方式来生成三维城市环境: 
1. NeRF-based methods:基于NeRF的方法 
2. diffusion-based methods:基于扩散的方法 
3. professional software script-based methods:基于专业软件脚本的方法 
## 2.2 三维世界模拟器
当前关于具身化和交互式城市环境的三维世界模拟工作较少

# 3.方法
构建"urban world model"当前有三大挑战:
1. 高效的实体环境构建 
2. 专业的城市场景规划 
3. 高质量纹理生成 
下面对该论文的四个关键模块工作进行说明 
## 3.1 基于OSM的城市布局生成 
UrbanWorld主要基于OSM生成3D城市布局,将OSM中包含的城市资源分离成独立的对象以便进行后续的逐元素渲染.在此过程中UrbanWorld还会记录对象的中心位置,以便进一步重组资产时使其与真实的城市布局相匹配. 
**OSM**数据包含大量信息,主要资源包括道路,建筑物,植被,森林,水体和其他基础设施要素的地理位置和属性.
## 3.2 大模型赋能的城市场景设计
UrbanWorld集成了一个先进的基于大量城市街景图像数据训练的城市大型语言模型(MLLM)来创建创建定制化的城市环境.
> 大模型来源:首先在全球范围内收集城市街景图像，并使用 GPT-4 为其标注相应的文本描述，然后进行人工检查并过滤了低质量数据。使用收集的数据集中约10万对图像-文本对,微调了一个开源MLLM,LLaVA-1.5. 

作用:用户通过UrbanWorld,使用文本指令（例如，大学里的教学区域）和选择的OSM布局图像作为输入,调用城市MLLM,返回关于每个资产的视觉外观和材料的多种详细描述.生成的资产描述将用作控制后续渲染过程的条件.
## 3.3 可控扩散型城市资产纹理渲染
定义了四个主要类别：建筑物、道路和路径、森林和植被、水体
逐元素渲染原则以确保渲染质量
为了加快渲染过程，合并一些城市元素类型
使用一种可控的基于扩散的方法实现渲染,该方法包含两个阶段:UV纹理生成和纹理细化
## 3.4 MLLM 辅助城市场景细化
城市资产渲染后,UrbanWorld 会根据从真实OSM数据中提取的位置信息自动重新组织资产,有效恢复原始城市布局

# 4.实验
## 4.1 实现细节
UrbanWorld 集成了三种关键技术:
1. Blender 作为专业的 3D 建模软件，基于扩散的渲染以及 MLLM 赋能的场景设计和细化.使用 Linux 系统上的 Blender-3.2.2 和兼容的 Blosm 插件来处理 OSM 数据转换.
2. 在基于扩散的渲染方面,使用 Stable Diffusion-1.5 作为基本的扩散骨干,并结合 ControlNet-Depth 生成多视角图像
3. 在UV纹理细化阶段使用ControlNet-inpainting 作为扩散控制器,并在真实感增强阶段使用ContrlNet-tile 

[^1]:[OSM](https://www.openstreetmap.org/):开放街道图（OpenStreetMap，简称OSM）是一个网上地图协作计划，其目标是创造一个内容自由且能让所有人编辑的世界地图，其OSM数据开源，可以自由下载使用 
[^2]:[Blender](https://www.blender.org/):Blender是一款自由和开源的三维计算机图形软件,它可以用于创建和渲染三维模型,动画,游戏,视觉效果等 
### KolektorSDD

- **下载链接**:
  - [KSDD](https://www.vicos.si/resources/kolektorsdd/)
 
  - KSDD 数据集：[Baidu Pan](https://pan.baidu.com/share/init?surl=HSzHC1ltHvt1hSJh_IY4Jg (password：1zlb))
 
- 代码仓库：[Github - vicoslab/mixed-segdec-net-comind2021](https://github.com/vicoslab/mixed-segdec-net-comind2021)
 
- 论文链接：[Arxiv - Mixed supervision for surface-defect detection: from weakly to fully supervised learning](https://arxiv.org/pdf/2104.06064v3)
 
- 介绍：
  - 包含**399张图片**，其中**52张有可见缺陷**,**347张无缺陷**
  - 包含**50 个实物（有缺陷的换向器）**，每个项目 8 个表面
  - 图像分辨率：宽度：**500 px**，高度：**1240-1270 px**，对于训练和评估，图像的大小应调整为 **512 x 1408 像素**
  - 所有图像在受控的工业环境中以真实案例的形式拍摄的
  - [paper-with-code - KSDD](https://paperswithcode.com/dataset/kolektorsdd）
   
### KolektorSDD2

- **下载链接**:
  - [KSDD 2](https://www.vicos.si/Downloads/KolektorSDD2)
    
  - KSDD2 数据集：[Kolektor Surface-Defect Dataset 2](https://go.vicos.si/kolektorsdd2)

- 论文链接：[Arxiv - SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](https://arxiv.org/pdf/2408.03143v2)

- 介绍：
  - 包括**356 张有明显缺陷的图像**,**2979 张无任何缺陷的图像**
  - 图像尺寸约为 **230 x 630 像素**
  - 包含 **246 张正样本图像**和 **2085 张负样本图像的训练集**
  - 包含 **110 张正样本图像**和 **894 张负样本图像的测试集**
  - 几种不同类型的缺陷（划痕、小斑点、表面缺陷等）
  - [paper-with-code - KSDD2](https://paperswithcode.com/dataset/kolektorsdd2）

### ELPV dataset

- **下载链接**:
  - [elpv dataset](https://github.com/zae-bayern/elpv-dataset)

- 论文链接：[Arxiv - Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts](https://arxiv.org/pdf/2403.06495v3)

- 代码仓库：
  - [mala-lab/inctrl](https://github.com/mala-lab/inctrl)
  - [mala-lab/winclip](https://github.com/mala-lab/winclip)

- 介绍：
  - 包含**2,624张功能和缺陷太阳能电池的8位灰度图像样本**
  - 具有从44个不同的太阳能模块中提取的不同程度的退化
  - 所有图像的大小和透视图均已标准化，且在提取太阳能电池之前，消除了由用于捕获EL图像的相机镜头引起的任何失真
  -  图像分辨率：**300 x 300 像素**
  -  [paper-with-code - ELPV](https://paperswithcode.com/dataset/elpv）

### MVTec AD

- **下载链接**:
  - [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
    
  - MVTec AD数据集:(https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz)
 
- 论文链接：[Arxiv - SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](https://arxiv.org/pdf/2408.03143v2)
 
- 代码仓库：[blaz-r/supersimplenet](https://github.com/blaz-r/supersimplenet)
 
- 介绍：
  - 包含**5354张高分辨彩色图像**，来自不同领域的5种纹理和10种物体
  - 15个类别中涵盖了不同类型的规则纹理(地毯，格子)和随机纹理(皮革，瓷砖，木材)。除外还有：瓶子、金属螺母，电缆等。
  - 包含用于训练的正常（即不包含缺陷）的图像，以及用于测试的异常图像。
  - 异常样本图像包含多种缺陷，缺陷是手工生成的。
  - 图像分辨率为**700x700-1024x1024像素**
  - 该数据集并给出了ground truth
  - **4.9GB**
  - [paper-with-code - mvtecad](https://paperswithcode.com/dataset/mvtecad)
 
### BTAD

- **下载链接**
  - [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)

- 论文链接：[UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection](https://pangdatangtt.github.io/static/pdfs/UniNet__arXix_.pdf)

- 代码仓库：[pangdatangtt/UniNet](https://github.com/pangdatangtt/UniNet)

- 介绍：
  - 包含**2830 张图像**，其中包含3种工业产品
  - 真实世界的工业异常数据集
   
### DeepPCB

- **下载链接**:
  - [DeepPCB](https://github.com/tangsanli5201/DeepPCB)
 
- 论文链接：[Arxiv - Online PCB Defect Detector On A New PCB Defect Dataset](https://arxiv.org/pdf/1902.06197v1.pdf)

- 代码仓库：[tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)

- 介绍：
  - 包含 **1,500 个图像对**，每个图像对由无缺陷的模板图像和对齐的测试图像组成，并带有注释
  - 其中包括 6 种最常见的 PCB 缺陷类型的位置：开路、短路、鼠咬、毛刺、针孔和杂散铜
  - 所有图像均由分辨率约为**每毫米 48 像素**的线性扫描 CCD 获得
  - 图像分辨率：模板和测试图像的原始大小约为 **16k x 16k 像素**，裁剪后的子图像为 **640 x 640 像素**
  - [paper-with-code - deeppcb](https://paperswithcode.com/dataset/deeppcb）
 
  
### RSDDs

- **下载链接**
  - [RSDDs](http://icn.bjtu.edu.cn/Visint/resources/RSDDs.aspx)
  - RSDDs 数据集：[Baidu Pan](https://pan.baidu.com/share/init?surl=svsnqL0r1kasVDNjppkEwg (password：nanr))

- 论文链接：[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045790622004992)

- 介绍：
  - 包含两种类型的数据集：
  -- 第一种是从快车道捕获的I型RSDDs数据集，其中包含67个具有挑战性的图像
  -- 第二种是从普通/重型运输轨道捕获的II型RSDDs数据集，其中包含128个具有挑战性的图像
  - 两个数据集的每幅图像至少包含一个缺陷，并且背景复杂且噪声很大。
  - RSDDs数据集中的这些缺陷已由一些专业的人类观察员在轨道表面检查领域进行了标记



### KolektorSDD

- **下载链接**:
  - [KSDD](https://www.vicos.si/resources/kolektorsdd/)
 
  - KSDD 数据集：[Baidu Pan](https://pan.baidu.com/share/init?surl=HSzHC1ltHvt1hSJh_IY4Jg (password：1zlb))
 
- 论文链接：[Segmentation-Based Deep-Learning Approach for Surface-Defect Detection](https://arxiv.org/pdf/1903.08536v3.pdf)

 
- 介绍：
  - 包含**399张图片**，其中**52张有可见缺陷**,**347张无缺陷**
  - 包含**50 个实物（有缺陷的换向器）**，每个项目 8 个表面
  - 图像分辨率：宽度：**500 px**，高度：**1240-1270 px**，对于训练和评估，图像的大小应调整为 **512 x 1408 像素**
  - 所有图像在受控的工业环境中以真实案例的形式拍摄的
  - [papers-with-code - KSDD](https://paperswithcode.com/dataset/kolektorsdd）
   
### KolektorSDD2

- **下载链接**:
  - [KSDD 2](https://www.vicos.si/Downloads/KolektorSDD2)
    
  - KSDD2 数据集：[Kolektor Surface-Defect Dataset 2](https://go.vicos.si/kolektorsdd2)

- 论文链接：[Arxiv - Mixed supervision for surface-defect detection: from weakly to fully supervised learning](https://arxiv.org/pdf/2104.06064v3)

- 代码仓库：[Github - vicoslab/mixed-segdec-net-comind2021](https://github.com/vicoslab/mixed-segdec-net-comind2021)

- 介绍：
  - 包括**356 张有明显缺陷的图像**,**2979 张无任何缺陷的图像**
  - 图像尺寸约为 **230 x 630 像素**
  - 包含 **246 张正样本图像**和 **2085 张负样本图像的训练集**
  - 包含 **110 张正样本图像**和 **894 张负样本图像的测试集**
  - 几种不同类型的缺陷（划痕、小斑点、表面缺陷等）
  - [papers-with-code - KSDD2](https://paperswithcode.com/dataset/kolektorsdd2）

### ELPV dataset

- **下载链接**:
  - [elpv dataset](https://github.com/zae-bayern/elpv-dataset)

- 论文链接：[Arxiv - Automatic Classification of Defective Photovoltaic Module Cells in Electroluminescence Images](https://arxiv.org/pdf/1807.02894v3.pdf)

- 代码仓库：
  - [Github - zae-bayern/elpv-dataset](https://github.com/zae-bayern/elpv-dataset)

- 介绍：
  - 包含**2,624张功能和缺陷太阳能电池的8位灰度图像样本**
  - 具有从44个不同的太阳能模块中提取的不同程度的退化
  - 所有图像的大小和透视图均已标准化，且在提取太阳能电池之前，消除了由用于捕获EL图像的相机镜头引起的任何失真
  - 图像分辨率：**300 x 300 像素**
  - [papers-with-code - ELPV](https://paperswithcode.com/dataset/elpv）

### MVTec AD

CVPR 2019  

- **下载链接**:
  - [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
    
  - MVTec AD数据集:(https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz)
 
- 论文链接：[CVPR - MVTecAD—AComprehensiveReal-World Dataset for Unsupervised Anomaly Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)
 
- 代码仓库：[Github - RizwanAliQau/tasad](https://github.com/RizwanAliQau/tasad)
 
- 介绍：
  - 包含**5354张高分辨彩色图像**，来自不同领域的5种纹理和10种物体
  - 15个类别中涵盖了不同类型的规则纹理(地毯，格子)和随机纹理(皮革，瓷砖，木材)。除外还有：瓶子、金属螺母，电缆等。
  - 包含用于训练的正常（即不包含缺陷）的图像，以及用于测试的异常图像。
  - 异常样本图像包含多种缺陷，缺陷是手工生成的。
  - 图像分辨率为**700x700-1024x1024像素**
  - 该数据集并给出了ground truth
  - **4.9GB**
  - [papers-with-code - mvtecad](https://paperswithcode.com/dataset/mvtecad)
 
### DAGM 2007

- **下载链接**
  - [DAGM 2007](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html)
  - DAGM 2007数据集：(https://zenodo.org/records/12750201)
 
- 论文链接：[Arxiv - Mixed supervision for surface-defect detection: from weakly to fully supervised learning](https://arxiv.org/pdf/2104.06064v3.pdf)

- 代码仓库：[Github - azzaelnaggar/data](https://github.com/azzaelnaggar/data)
 
- 介绍
  - 包括**10个数据集**，其中**前6个被称为开发数据集**，用于算法开发，**后4个数据集被称为竞赛数据集**，可用于评估性能
  - 这些数据是人工生成的，但与现实世界的问题类似
  - 每个开发（竞赛）数据集包含 **1000（2000）张“无缺陷”图像**和 **150（300）张“有缺陷”图像**，
  - 每个数据集由不同的纹理模型和缺陷模型生成，“无缺陷”图像显示背景纹理没有缺陷，“缺陷”图像在背景纹理上恰好有一个标记的缺陷。
  - 所有数据集已被随机分成大小相等的训练和测试子数据集
  - 弱标签以椭圆形粗略标示缺陷区域，值 0 和 255 分别表示背景和缺陷区域
  - 图像以**灰度 8 位 PNG 格式**保存
  - [papers-with-code - dagm2007](https://paperswithcode.com/dataset/dagm2007)
 
### VisA

- **下载链接**
  - [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)
 
- 论文链接：[Arxiv - SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation](https://arxiv.org/pdf/2207.14315v1.pdf)

- 代码仓库：[Github - open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
 
- 介绍
  - 包括** 10,821 张图像**，其中包含** 9,621 个正常样本**和** 1,200 个异常样本**
  - 包含** 12 个子集**，对应于 12 个不同的对象，四个子集是不同类型的印刷电路板（PCB），其结构相对复杂，包含晶体管、电容器、芯片等
  - 异常图像包含各种缺陷，包括划痕、凹痕、色点或裂纹等表面缺陷，以及错位或缺失部件等结构缺陷
  - [papers-with-code](https://paperswithcode.com/dataset/visa)
 
### BTAD

- **下载链接**
  - [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)

- 论文链接：[Arxiv - VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization](https://arxiv.org/pdf/2104.10036v1.pdf)

- 代码仓库：[openvinotoolkit/anomalib](https://github.com/openvinotoolkit/anomalib)

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
  - [papers-with-code - deeppcb](https://paperswithcode.com/dataset/deeppcb）
 
  
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

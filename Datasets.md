# Paired

### LOL-v1

- **下载链接**: 

  -  [Kaggle - LOL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset)

  -  LOw Light 配对数据集 (LOL): [Google Drive](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB)  ||  [Baidu Pan - Code:acp3](https://pan.baidu.com/s/1ABMrDjBTeHIJGlOFIeP1IQ)

    Synthetic Image Pairs from Raw Images: [Google Drive](https://drive.google.com/open?id=1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F)  ||  [Baidu Pan](https://pan.baidu.com/s/1drsMAkRMlwd9vObAM_9Iog)

    Testing Images: [Google Drive](https://drive.google.com/open?id=1OvHuzPBZRBMDWV5AKI-TtIxPCYY8EW70)  ||  [Baidu Pan](https://pan.baidu.com/s/1G2qg3oS12MmP8_dFlVRRug)

- 项目主页: [BMVC2018 Deep Retinex Decomposition](https://daooshee.github.io/BMVC2018website/) 

- 代码仓库: [Github - weichen582/RetinexNet](https://github.com/weichen582/RetinexNet)

- 论文链接: [Arixv - Deep Retinex Decomposition for Low-Light Enhancement](https://arxiv.org/abs/1808.04560)

- 介绍: 
  - 包含**485对低光/正常光图像**用于训练，**15对用于测试**。
  - 每对数据由低光输入图像和对应的正常曝光参考图像组成。
  - 图像分辨率为**400×600或600×400像素**，主要为室内场景，且包含显著噪声。
  - 所有图像通过真实拍摄获得，覆盖典型的低光噪声场景。
  - 330MB
  - [paper-with-code - LOL](https://paperswithcode.com/dataset/lol)



### LOL-v2

- **下载链接**:  [kaggle - LOL-v2-Dataset](https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset)
- 代码仓库:  [Github - flyywh/SGM-Low-Light](https://github.com/flyywh/SGM-Low-Light) 
- 论文链接: [Band Representation-Based Semi-Supervised Low-Light Image Enhancement: Bridging the Gap Between Signal Fidelity and Perceptual Quality](http://39.96.165.147/Pub%20Files/2021/ywh_tip21_2.pdf)
-  介绍:  
  - LOL-v2-real: 
    * 包含**689对训练数据**和**100对测试数据**。
    * 图像通过调整相机的ISO感光度和曝光时间在真实场景中拍摄，涵盖更多样化的室内外环境。
    * 分辨率统一调整为**600×400像素**（PNG格式）。
    * 相比LOL-v1，场景多样性更高，增强了算法泛化能力的验证
    * [paper-with-code - LOL-v2-real](https://paperswithcode.com/dataset/lol-v2)
  - LOL-v2-synthetic: 
    - 包含**900对训练数据**和**100对测试数据**。
    - 在RAISE中的1000张正常光图像, 计算YCbCr色彩空间中的Y通道(亮度)直方图, 在Adobe Lightroom中合成低光图像。
    - 专注于合成数据的可控性，可用于研究低光增强模型在模拟噪声下的性能
    - [paper-with-code - LOLv2-synthetic](https://paperswithcode.com/dataset/lolv2-synthetic)



### LOL-Blur

- **下载链接**: [Google Dirve - LOLBlur Dataset](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX)
- 代码仓库: [Github - sczhou/LEDNet](https://github.com/sczhou/LEDNet)
- 项目主页: [LEDNet](https://shangchenzhou.com/projects/LEDNet/)
- 论文链接: [Arxiv - LEDNet: Joint Low-light Enhancement and Deblurring in the Dark](https://arxiv.org/abs/2202.03373)
- 介绍: 
  - 合成数据集
  - 包含**10200对训练数据**，**1,800对测试数据**。



### SID

CVPR 2018

- **下载链接**:  [Baidu Drive](https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA) 
  - 下载所有部件后，需要 通过运行以下命令将它们组合在一起：`cat SonyPart* > Sony.zip`和`cat FujiPart* > Fuji.zip`
  -  不建议下RAW, 直接下其他人处理好的: [Baidu Disk - code: gplv](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w)  ||  [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)  (来自[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer))
- 代码仓库: [GitHub - cchen156/Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark)
- 项目主页: [SID](https://cchen156.github.io/SID.html)
- 作者主页: [https://cchen156.github.io/](https://cchen156.github.io/)
- 论文链接: https://cchen156.github.io/paper/18CVPR_SID.pdf
- 介绍
  - 数据集包含5094个原始的短曝光图像，每个图像都有相应的长期曝光参考图像。(也就是**5094对低光/正常光图像对**) 
  - **训练/验证/测试比例为7:1:2**。
  - 使用两个相机捕获图像：Sony α7SII 和Fujifilm X-T2。
  - 分辨率分别为索尼的**4,240 × 2,832**和富士的6,000 × 4,000。
  - 数据集包含室内和室外图像，室外场景照度为0.2lux至5lux，室内场景照度为0.03lux至0.3lux。
  - [paper-with-code - SID](https://paperswithcode.com/dataset/sid)



### SIMD

ICCV 2019

- **下载链接**:  [Part1](https://storage.googleapis.com/isl-datasets/DRV/short1.zip), [Part2](https://storage.googleapis.com/isl-datasets/DRV/short2.zip), [Part3](https://storage.googleapis.com/isl-datasets/DRV/short3.zip), [Part4](https://storage.googleapis.com/isl-datasets/DRV/short4.zip), [Part5](https://storage.googleapis.com/isl-datasets/DRV/short5.zip) and [long](https://storage.googleapis.com/isl-datasets/DRV/long.zip)   
  - 不建议下RAW, 直接下其他人处理好的 [Baidu Disk - Code: btux](https://pan.baidu.com/share/init?surl=Qol_4GsIjGDR8UT9IRZbBQ) ||   [Google Drive](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG) (来自[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer))
- 代码仓库: [GitHub - cchen156/Seeing-Motion-in-the-Dark](https://github.com/cchen156/Seeing-Motion-in-the-Dark)
- 作者主页: https://cchen156.github.io/
- 论文链接:  https://cchen156.github.io/paper/19ICCV_DRV.pdf
- 介绍
  - 包含**22,220张真实低光图像**
  - 分辨率为**3672 × 5496**
  - 随机分为3组：**训练集（64%）、验证集（12%）和测试集（24%）**。
  - 一些场景包括各种照明设置，包括具有不同色温、照度水平和位置的光源。



### Sony-Total-Dark

- **下载链接**: [Baidu Pan](https://pan.baidu.com/s/1mpbwVscbAfQJtkrrzBzJng?pwd=yixu)   (来自[HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet))
- 介绍
  - **2697 对图像对**
  - 是**SID的子集**(Sony部分), 并作出了相应的改进: 将原始格式图像转换为没有伽玛校正的SRGB图像，使得图像变得非常黑, 使任务更具挑战性
  - [paper-with-code - Sony-Total-Dark](https://paperswithcode.com/dataset/sony-total-dark)



### SICE

- **下载链接**: [Google Drive - SICE Dataset_Part1](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view)  [Google Drive - SICE Dataset_Part2](https://drive.google.com/file/d/16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN/view)
- 代码仓库:  [Github - csjcai/SICE](https://github.com/csjcai/SICE)
- 论文链接: [IEEE - Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images](https://www4.comp.polyu.edu.hk/~cslzhang/paper/SICE.pdf)

- 介绍: 
  - 包括室内和室外场景中的 589 个序列，总共包含 4,413 张多曝光图像，因此每个序列都有 3 到 18 张不同曝光级别的低对比度图像。
  - 使用七种消费级相机收集图像序列，包括索尼 α7RII、索尼 NEX-5N、佳能 EOS-5D Mark II、佳能 EOS-750D、尼康 D810、尼康 D7100 和 iPhone 6s，大多数图像的分辨率在 3000×2000 到 6000×4000 之间
  - 训练/验证/测试比例为7:1:2。



### Adobe FiveK

- **下载链接**: [官方](https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar)  ||  [Hugging Face](https://huggingface.co/datasets/logasja/mit-adobe-fivek)  ||  [Kaggle](https://www.kaggle.com/datasets/weipengzhang/adobe-fivek) 
  - Tip: 可以下载其他人处理好的([Baidu - code:cyh2](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2)  ||  [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view)), DNG格式太大了
- 代码仓库:  [Github - yuukicammy/mit-adobe-fivek-dataset](https://github.com/yuukicammy/mit-adobe-fivek-dataset)
- 项目主页: [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/)
- 介绍: 
  - 包含**5,000张真实和合成图像**
  - 涵盖各种光照条件和不同分辨率，包括直接从相机拍摄的原始图像以及由5位专业摄影师制作的编辑版本。
  - 80%用于训练, 20%用于测试。
    - [paper-with-code - MIT-Adobe FiveK](https://paperswithcode.com/dataset/mit-adobe-fivek)



### SDSD

- **下载链接**: [Baidu pan - Code: zcrb](https://pan.baidu.com/s/1CSNP_mAJQy1ZcHf5kXSrFQ)  ||  [Google Drive](https://drive.google.com/drive/folders/1-fQGjzNcyVcBjo_3Us0yM5jDu0CKXXrV?usp=sharing)
- 代码仓库: [Github - dvlab-research/SDSD](https://github.com/dvlab-research/SDSD)
- 介绍: 
  - 包含低光和常光视频的动态视频对的形式
  - 由室内子集和室外子集两部分组成
    - 室内子集有 70 个视频对
    - 室外子集有 80 个视频对。
  - `indoor_np`：用于训练的室内子集数据，所有视频帧都保存为.npy文件，分辨率为512 x 960，以便快速训练。
  - `outdoor_np`：用于训练的室外子集数据，所有视频帧都保存为.npy文件，分辨率为512 x 960，以便快速训练。
  - `indoor_png`：室内子集中的原始视频数据。所有帧都保存为.png文件，分辨率为1080 x 1920。
  - `outdoor_png`：室外子集中的原始视频数据。所有帧都保存为.png文件，分辨率为1080 x 1920。



### VE-LOL

- **下载链接**: 

  - VE-LOL-H: [[Dropbox\]](https://www.dropbox.com/s/yxod21zouvrqhpk/VE-LOL-H.zip?dl=0)  ||   [[Baiduyun - Code: 7c8i\]](https://pan.baidu.com/s/12UTjDNOsALUyMzm0rbpQ8Q)

  - VE-LOL-L: [[Dropbox\]](https://www.dropbox.com/s/vfft7a8d370gnh7/VE-LOL-L.zip?dl=0)  ||  [[Baiduyun - Code: a2x3\]](https://pan.baidu.com/s/1JqPho8k9Q3G_BmpEdtxyBQ) 

  - VE-LOL-L Results: [[Dropbox\]](https://www.dropbox.com/s/308dxl4yikc3t8k/VE-LOL-L-Results.zip?dl=0)  ||   [[Baiduyun - Code: 9okw\]](https://pan.baidu.com/s/1Q07WG8w0IkBAawYfHkYiHQ)
- 项目主页: [Benchmarking Low-Light Image Enhancement and Beyond](https://flyywh.github.io/IJCV2021LowLight_VELOL/)
- 论文链接: [Benchmarking Low-Light Image Enhancement and Beyond](https://flyywh.github.io/IJCV2021LowLight_VELOL/attached_files/ijcv21.pdf)
- 介绍:  

  - 包含**13,440张真实和合成低光照图像**及其不同分辨率的图像对。
  - 具有多样化的场景，例如自然风光、街道景观、建筑、人脸等。
  - **配对**部分VE-LOL-L包含2,100对用于训练和400对用于测试
  - **未配对**部分VE-LOL-H则包含6,940张用于训练和4,000张用于测试的图像。
  - VE-LOL-H部分还包含用于**高级物体检测任务的目标检测标签**。(来自于[ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset))



### MCR

- **下载链接**:  [Google Drive](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view?usp=share_link)  ||  [Baidu Netdisk - Code: 22cv](https://pan.baidu.com/s/1xOnVFJ7TMHUexw0-SFfMZA) 
- 代码仓库: [Github - TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark](https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark)
- 论文地址: [Arxiv - Abandoning the Bayer-Filter to See in the Dark](https://arxiv.org/abs/2203.04042)
- 介绍:  
  - 包含3,984张真实和合成的短曝光和长曝光图像 
  - 共有 498 个不同的场景，每个场景都有 1 个相应的 RGB 和单色GT以及 8 个不同的曝光颜色 Raw 输入。(8 次曝光单色图像可在 [Google Drive](https://drive.google.com/file/d/1mQml2a8U7HecRvCldxiwvjK4ZexUB7LU/view?usp=sharing)  ||  [Badui Netdisk - Code: 22cv](https://pan.baidu.com/s/1xvVKivKyjdgKomPI0F5pdQ) 获得)
  - 高分辨率（1,024×1,280像素）
  - 训练集:  3600对, 测试集:  384对
  - 图像采集自室内固定位置以及室内/室外滑动平台条件



### LSRW

- 下载链接: [BaiduNetdisk - Code: wmrr](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)
- 代码仓库: [Github - JianghaiSCU/R2RNet](https://github.com/JianghaiSCU/R2RNet)
- 论文地址: [Arxiv - R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network](https://arxiv.org/abs/2106.14501)
- 介绍: 
  - 包含5,650张真实低光配对图像
  - 分辨率各异
  - 场景包括室内和室外
  - 5,600对图像用于训练，剩余50对用于测试
  - *存在部分图像未对齐, 作者认为对低光增强任务不重要*



### LL-Gaussian

- 项目地址: [LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis](https://sunhao242.github.io/LL-Gaussian_web.github.io/)
- 论文地址: [Arxiv - LL-Gaussian: Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting for Novel View Synthesis](https://arxiv.org/abs/2504.10331)

- 代码与数据集暂未公开



# Unpaired

### ExDark

低照度条件下的**目标检测**

- **下载链接** : [Google Drive - ExDark](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view)

- 代码仓库:  [GitHub - cs-chan/Exclusively-Dark-Image-Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

- 论文链接: [Getting to know low-light images with the Exclusively Dark dataset](http://cs-chan.com/doc/cviu.pdf)

- 介绍: 

  - 涵盖了从**极低光照环境到黄昏时段（即10种不同条件）的图像**，**并带有图像类别和物体级别的标注**。

  - 类别: 

    自行车-652张图像 

    船-679张图像 

    瓶-547张图像  

    巴士-527张图像 

    汽车-638张图像 

    猫-735张图像 

    椅子-648张图像 

    杯-519张图像 

    狗-801张图像 

    摩托车-503张图像 

    人-609张图像 

    表-505张图像 

    总计：7,363张图像

  - 训练 - 3,000张图片（每类250张图片）
    验证 - 1,800张图片（每类150张图片）
    测试 - 2,563张图片



### ACDC

不良条件下的**语义分割**

- **下载链接**: [ACDC Dataset Download](https://acdc.vision.ee.ethz.ch/download)
- 项目链接: [ACDC Homepage](https://acdc.vision.ee.ethz.ch/)
- 论文链接: [Arxiv - ACDC: The Adverse Conditions Dataset with Correspondences for Robust Semantic Driving Scene Perception](https://arxiv.org/abs/2104.13395)
- 介绍: 
  - 包含4006张图像，(1,000雾天、1,000雪天、1,000雨天和1,006夜间)
  - 分辨率为1,080×1,920
  - 每张不良条件图像都附带高质量的精细**像素级全景语义标注**、同一场景在正常条件下的对应图像以及一个二值掩码，用于区分图像内清晰和不确定语义内容的不同区域。共19类
  - 夜间包含400张训练图像、106张验证图像和500张测试图像



### SGZ

- **下载链接**: [Google Drive - SGZ](https://drive.google.com/drive/folders/1RIQsP5ap5QU7LstHPknOffQZeqht_FCh)
- 代码仓库: [Github - ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement)
- 论文地址 : [Arxiv - Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement](https://arxiv.org/abs/2110.00970)
- 介绍:
  - 包含1**50张合成的低光图像**
  - 分辨率为1,024 × 2,048
  - 通过对原始CityScape 数据集进行伽玛校正而生成，其中包含了带有**精细分割标签的城市场景（30类）**



> LIME, NPE, MEF, DICM, VV 可用作主观评测



### MEF

- 代码仓库:  [Github - h4nwei/MEF-SSIMd](https://github.com/h4nwei/MEF-SSIMd)
- 论文地址: [IEEEXplore - Perceptual Quality Assessment for Multi-Exposure Image Fusion](https://ieeexplore.ieee.org/document/7120119)
- 介绍: 
  - **136 张融合图像**
  - 包括室内和室外视图、自然风景和人造建筑
  - 所有图像序列都包含至少 3 张图像，分别代表**欠曝、过曝和中间情况**



### NPE

- 介绍:
  - 由 46 张使用佳能数码相机捕获的图像和 110 张从一些组织/公司网站（如 NASA 和 Google）下载的图像组成。总计**156张真实低亮度图像**
  - 数据集中的所有图像在**局部区域对比度低，在全局空间中照度变化严重**。



### LIME

- 论文链接: [LIME: Low-light Image Enhancement via Illumination Map Estimation](https://www3.cs.stonybrook.edu/~hling/publication/LIME-tip.pdf)



### VV

- 项目主页: [Vasileios Vnikakis](https://sites.google.com/site/vonikakis/datasets)
- 介绍: 
  - 包含24张真实的多曝光图片，分辨率各异。其中包含旅行照片，包括室内外人物和自然风光
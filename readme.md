

# :zap:Fast_Seg:zap:

This repo try to implement **state-of-art fast semantic segmentation model** s on **road scene dataset**(CityScape, 
Mapillary, Camvid).


# What is purpose of this repo?
This repo aims to do experiments and verify the idea of fast semantic segmentation and this repo
also provide some fast models.  
 
Our ICnet implementation achieves **74.5% mIoU** which is **5%** point higher than orginal paper. !!!!! Here: [model](https://drive.google.com/open?id=1A6z87_GCHEuKeZfbGpEvnkZ0POdW2Q_U)

# Another Link For Accurate Seg: 
[GALD-net](https://github.com/lxtGH/GALD-Net) provides some state-of-art accurate methods implementation.   

# Model Zoo (Updating)
1. ICNet:ICnet for real-time semantic segmentation on high-resolution images. ECCV-2018, [paper](https://arxiv.org/abs/1704.08545) 
2. DF-Net: Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search.CVPR-2019, [paper](https://arxiv.org/abs/1903.03777)   
3. Bi-Seg: Bilateral segmentation network for real-time semantic segmentation.ECCV-2018, [paper](https://arxiv.org/pdf/1808.00897.pdf)  
4. DFA-Net: Deep feature aggregation for real-time semantic segmentation.CVPR-2019,[paper](https://arxiv.org/abs/1904.02216)  
5. ESP-Net: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation. ECCV-2018,[paper](https://arxiv.org/abs/1803.06815)  
6. SwiftNet: In defense of pre-trained imagenet architectures for real-time semantic segmentation of road-driving images. CVPR2019, [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Orsic_In_Defense_of_Pre-Trained_ImageNet_Architectures_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.pdf)  
7. Real-Time Semantic Segmentation via Multiply Spatial Fusion Network.(face++) arxiv,[paper](https://arxiv.org/abs/1911.07217)  
8. Fast-SCNN: Fast Semantic Segmentation Network.BMVC-2019 [paper](https://arxiv.org/abs/1902.04502)  




# Usage
1. use train_distribute.py for training For example, use scripts in exp floder for training and evaluation.
2. use prediction_test_different_size.py for prediction with different size input.


## Datasets Perparation
- You can download [cityscapes] dataset (https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).
- You can download camvid dataset from [here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
- You can download pretrained XceptionA with RGB input and ResNet18 with bgr input  and ResNet50 with bgr input
[link]:(https://pan.baidu.com/s/1mM_Lc44iX9CT1nPq6tjOAA)  password:bnfv.


# Some Advice on Training
1. use syn-bn(apex).
2. use batch-size >=8.
3. use deep supervised loss for easier optimation.
4. use large crop size during training. 
5. longer training time for small models(60,000 interaction or more).
6. use Mapillary data for pretraining for boosting performance.
7. Deeply based resnet runs slowly than torch pretrained resnet but with higher accuracy.
8. The small network doesn't need ImageNet pretraining if training longer time on Cityscape.(Fast-SCNN paper)


<img src="./data/fig/frankfurt_000000_002196_leftImg8bit.png" width="290" /><img src="./data/fig/frankfurt_000000_002196_gtFine_color.png" width="290" /><img src="./data/fig/frankfurt_000000_002196_leftImg8bit_pred.png" width="290" />
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(a) test image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) ground truth &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(c) predicted result

# License
This project is released under the Apache 2.0 license.




# Acknowledgement

Thanks to previous open-sourced repo:  
[Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)    
[CCNet](https://github.com/speedinghzl/CCNet)   
[TorchSeg](https://github.com/ycszen/TorchSeg)  
[pytorchseg](https://github.com/meetshah1995/pytorch-semseg) 

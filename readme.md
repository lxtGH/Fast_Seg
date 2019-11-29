# Note: still building(please wait a couple of days)

# What is Fast_Seg?
This repo try to implement state-of-art fast semantic segmentation models on road scene dataset(CityScape, Camvid).


# What is purpose of this repo?
This repo aims to do experiments and verify the idea of fast semantic segmentation and this repo
also provide some fast models.
 

# Model Zoo (Updating)
1. ICNet:ICnet for real-time semantic segmentation on high-resolution images.
2. DF-Net: Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search.
3. Bi-Seg: Bilateral segmentation network for real-time semantic segmentation.
4. DFA-Net: Deep feature aggregation for real-time semantic segmentation.
5. ESP-Net: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
6. SwiftNet: In defense of pre-trained imagenet architectures for real-time semantic segmentation of road-driving images.
7. Real-Time Semantic Segmentation via Multiply Spatial Fusion Network
8. Fast-SCNN: Fast Semantic Segmentation Network 


# Usage
1. use train_distribute.py for training 
2. use prediction_test_different_size.py for prediction with different size input.
3. use eval.py for evaluation on validation dataset.


## Datasets Perparation
- You can download [cityscapes] dataset (https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).
- You can download camvid dataset from [here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).


# Some Advice on Training
1. use syn-bn(apex).
2. use batch-size >=8.
3. use deep supervised loss for easier optimation.
4. use large crop size during training. 
5. longer training time for small models(60,000 interaction or more).
6. use Mapillary data for pretraining for boosting performance.
7. Deeply based resnet runs slowly than torch pretrained resnet but with higher accuracy.


<img src="./data/fig/frankfurt_000000_002196_leftImg8bit.png" width="290" /><img src="./data/fig/frankfurt_000000_002196_gtFine_color.png" width="290" /><img src="./data/fig/frankfurt_000000_002196_leftImg8bit_pred.png" width="290" />
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(a) test image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) ground truth &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(c) predicted result

# License
This project is released under the Apache 2.0 license.


# Acknowledgement

Thanks to previous open-sourced repo:  
[Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)  
[CCNet](https://github.com/speedinghzl/CCNet)

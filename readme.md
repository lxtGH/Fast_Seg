# What is Fast_Seg?
This repo try to implement state-of-art fast semantic segmentation models on road scene dataset(CityScape, Camvid).


# What is purpose of this repo?
This repo aims to do experiments and verify the idea of fast semantic segmentation and this repo
also provide some fast models.



# Model Zoo 
1. ICNet:ICnet for real-time semantic segmentation on high-resolution images.
2. DF-Net: 
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





# Some Advice on Training
1. use syn-bn(apex).
2. use batch-size >=8.
3. use deep supervised loss for easier optimation.
4. use large crop size during training.
5. longer training time for small models(60,0000 interaction or more).


<p align="center"><img width="100%" src="./data/fig/frankfurt_000000_002196_leftImg8bit_pred.png" /></p>
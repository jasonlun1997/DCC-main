# DCC-main
Downsampling Consistency Correction-Based Quality Enhancement for CNN-Based Light Field Image Super-Resolution (2023)

## News
* Apr 6, 2023: upload the pre-trained models of our method to [GoogleDrive](https://drive.google.com/drive/folders/12eQfFK2Lm102WqTK5BCa1H6mp5VK7PBk?usp=sharing).

## Contributions
A cascaded Swin Transformer-based recognizer is proposed to identify the downsampled position and downsampling scheme
used in the LR testing LF image, and then the proposed DCC-based method is used to significantly improve the quality of
the upsampled LF image.

## Dataset
We use the processed data by [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855), including EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and testing. Please download the dataset in the official repository of [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855).

## Results
![f5](/Figs/Fig5.png)  
Fig. 5. For 2x LF SR, the perceptual merits of our method for DHR,testing = 4:2:0(Direct). (a) The ground truth LF image, HCI-bedroom. (b) The two
magnified subimages of (a). (c) LF-InterNet. (d) LF-DFnet. (e) LF-IINet. (f) DistgSSR. (g) Ours for LF-InterNet. (h) Ours for LF-DFnet.
(i) Ours for LF-IINet. (j) Ours for DistgSSR.

![f6](/Figs/Fig6.png)  

Fig. 6. For 2x LF SR, the perceptual merits of our method for DHR,testing = 4:2:0(A). (a) The ground truth LF image, Stanford Gantry-Tarot Cards S. (b)
The two magnified subimages of (a). (c) LF-InterNet. (d) LF-DFnet. (e) LF-IINet. (f) DistgSSR. (g) Ours for LF-InterNet. (h) Ours for
LF-DFnet. (i) Ours for LF-IINet. (j) Ours for DistgSSR.  
More Visual Results can visit [1](https://github.com/jasonlun1997/DCC-main/tree/main/Figs/420A) [2](https://github.com/jasonlun1997/DCC-main/tree/main/Figs/420D)
## Code
### Dependecies
* Python 3.9.16
* Pytorch 1.13.1 + torchvision 0.14.1 + cuda 11.7.1
* Matlab  
### Recognizer
2x/4x position and scheme recognizer can be download from [GoogleDrive](https://drive.google.com/drive/folders/12eQfFK2Lm102WqTK5BCa1H6mp5VK7PBk?usp=sharing)

## Acknowledgement
Our work and implementations are inspired and based on the following projects:  
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)  
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)  
[LF-IINet](https://github.com/GaoshengLiu/LF-IINet)  
[DistgSSR](https://github.com/YingqianWang/DistgSSR)  
[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)  
We sincerely thank the authors for sharing their code and amazing research work!

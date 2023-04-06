# DCC-main
Downsampling Consistency Correction-Based Quality Enhancement for CNN-Based Light Field Image Super-Resolution (2023) by Kuo-Liang Chung and Tsung-Lun Hsieh

## News
* Apr 6, 2023: upload the pre-trained models of our method to [GoogleDrive](https://drive.google.com/drive/folders/12eQfFK2Lm102WqTK5BCa1H6mp5VK7PBk?usp=sharing).

## Contributions
A cascaded Swin Transformer-based recognizer is proposed to identify the downsampled position and downsampling scheme
used in the LR testing LF image, and then the proposed Downsampling Consistency Correction (DCC)-based method is used to significantly improve the quality of
the upsampled LF image.

## Dataset
We use the datasets which can be accessed from [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855), including EPFL, HCI, HCI_old, INRIA_Lytro and Stanford_Gantry datasets for training and testing. Please download the dataset in the official repository of [LF-DFnet]([https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855](https://github.com/YingqianWang/LF-DFnet)).

## Results
![f5](/Figs/2x420A_Stanford_Gantry_Tarot_Card_S.png)  
Fig. 5. For 2x LF SR, the perceptual merits of our method for D_{HR,testing} = 4:2:0(Direct). (a) The ground truth LF image, HCI-bedroom. (b) The two
magnified subimages of (a). (c) LF-InterNet. (d) LF-DFnet. (e) LF-IINet. (f) DistgSSR. (g) Ours for LF-InterNet. (h) Ours for LF-DFnet.
(i) Ours for LF-IINet. (j) Ours for DistgSSR.

![f6](/Figs/2x420D_HCI_bedroom.png)  

Fig. 6. For 2x LF SR, the perceptual merits of our method for D_{HR,testing} = 4:2:0(A). (a) The ground truth LF image, Stanford Gantry-Tarot Cards S. (b)
The two magnified subimages of (a). (c) LF-InterNet. (d) LF-DFnet. (e) LF-IINet. (f) DistgSSR. (g) Ours for LF-InterNet. (h) Ours for
LF-DFnet. (i) Ours for LF-IINet. (j) Ours for DistgSSR.  

More visual results can visit the link [./Figs/420A](https://github.com/jasonlun1997/DCC-main/tree/main/Figs/420A) or [./Figs/420D](https://github.com/jasonlun1997/DCC-main/tree/main/Figs/420D).  
## Code
### Dependecies
* Python 3.9.16
* Pytorch 1.13.1 + torchvision 0.14.1 + cuda 11.7.1
* Matlab  
### Recognizer
For 2x and 4x LF SR, the downsampled position and the downsampling scheme recognizers can be download from [GoogleDrive](https://drive.google.com/drive/folders/12eQfFK2Lm102WqTK5BCa1H6mp5VK7PBk?usp=sharing)  
After downloading the five datasets, the recognizer is used to create the training or testing data:
* First, set the parameters in 'DownsampleMat2png.m'
  ```
  src_dataset_for = 'training'; 
  ```
  ```
  src_dataset_for = 'test'; 
  ```  
  Secordly, generate the kind of data set which you want, and then run
  ```
  ./recognizer/DownsampleMat2png.m 
  ```  
### Prepare Training Data
* First, set the parameters  in 'Generate_Data_for_Training.m'
  ```
  patch_Sr_y = imresize(patch_Hr_y, downRatio); % bicubic down
  ```
  ```
  patch_Sr_y = convert420_A(patch_Hr_y,factor); % 420A down
  ```
  Secordly, generate the kind of data set which you want, and run
  ```
  Generate_Data_for_Training.m
  ```  
### Prepare Testing Data
* First, set the parameters in 'Generate_Data_for_Test.m'
  ```
  temp_Lr_y = convert420_A(temp_Hr_y,factor);   % 420A down
  ```
  ```
  temp_Lr_y = convert420_D(temp_Hr_y,factor);   % 420D down     
  ```
  ```
  temp_Lr_y = imresize(temp_Hr_y, downRatio);   % bicubic down
  ```
  Secordly, generate the kind of data set which you want, and run
  ```
  Generate_Data_for_Test.m
  ```
### Train  
* Run DistgSSR
  ```
  python train.py --angRes 5 --upscale_factor 2 --model_name DistgSSR --use_pre_ckpt False --path_pre_pth [pre-trained dir] --path_for_train [path of training set] --path_for_test [path of test set] --batchsize 8 --lr 2e-4 --n_steps 15 --epoch 50
  ```
* Run LF-IINet
  ```
  python train.py --angRes 5 --upscale_factor 2 --model_name DistgSSR --use_pre_ckpt False --path_pre_pth [pre-trained dir] --path_for_train [path of training set] --path_for_test [path of test set] --batchsize 10 --lr 2e-4 --n_steps 10 --epoch 50
  ```
* Run LF-DFnet
  ```
  python train.py --angRes 5 --upscale_factor 2 --model_name DistgSSR --use_pre_ckpt False --path_pre_pth [pre-trained dir] --path_for_train [path of training set] --path_for_test [path of test set] --batchsize 8 --lr 2e-4 --n_steps 15 --epoch 50
  ```
* Run LF_InterNet
  ```
  python train.py --angRes 5 --upscale_factor 2 --model_name DistgSSR --use_pre_ckpt False --path_pre_pth [pre-trained dir] --path_for_train [path of training set] --path_for_test [path of test set] --batchsize 12 --lr 5e-4 --n_steps 10 --epoch 40
  ```
### Test  
* Run 
  ```
  python test.py --angRes 5 --upscale_factor 2 --model_name [model name] --use_pre_ckpt True --path_pre_pth [pre-trained dir] --path_for_train [path of training set] --path_for_test [path of test set] --path_re_pth [re-train dir] --position_recongnizer [path of position recongnizer] --scheme_recongnizer [path of scheme recongnizer] --result_dir [name of result dir]
  ```
  |[model name]|[pre-trained dir]|[re-trained dir]|
  |---|---|---|
  | DistgSSR|```./pth/SR/bicubic/DistgSSR_2xSR_5x5.pth.tar```|```./pth/SR/420A/DistgSSR_5x5_2x_420A_model.pth```|
  |LF_IINet|```./pth/SR/bicubic/IINet_2xSR.pth.tar```|```./pth/SR/420A/LF_IINet_5x5_2x_420A_model.pth```|
  |DFnet|```./pth/SR/bicubic/DFnet_2xSR_5x5.pth```|```./pth/SR/420A/DFnet_5x5_2x_420A_model.pth```|
  |LF_InterNet|```./pth/SR/bicubic/InterNet_5x5_2xSR_C64.pth.tar```|```./pth/SR/420A/LF_InterNet_5x5_2x_420A_model.pth```|
  
 
  
## Acknowledgement
Our work and implementations are inspired and based on the following projects:  
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)  
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)  
[LF-IINet](https://github.com/GaoshengLiu/LF-IINet)  
[DistgSSR](https://github.com/YingqianWang/DistgSSR)  
[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)  
We appreciate the authors for sharing their codes. Their pre-trained models with the related training parameters can be access from the link.

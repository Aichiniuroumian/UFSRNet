# UFSRNet: U-shaped Face Super-resolution Reconstruction Network Based on Wavelet Transform


## Installation and Requirements 


I have tested the codes on
- Ubuntu 18.04
- CUDA 10.1  
- Python 3.7, install required packages by `pip3 install -r requirements.txt`  


### Test with Pretrained Models

We provide example test commands in script `test.sh` for both SPARNet and SPARNetHD. Two models with difference configurations are provided for each of them, refer to [section below](#differences-with-the-paper) to see the differences. Here are some test tips:

- SPARNet upsample a 16x16 bicubic downsampled face image to 128x128, and there is **no need to align the LR face**.   
- Please specify test input directory with `--dataroot` option.  
- Please specify save path with `--save_as_dir`, otherwise the results will be saved to predefined directory `results/exp_name/test_latest`.  



### Train the Model

The commands used to train the released models are provided in script `train.sh`. Here are some train tips:

- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and to train UFSRNet . Please change the `--dataroot` to the path where your training images are stored.  
- To train UFSRNet, we simply crop out faces from CelebA without pre-alignment, because for ultra low resolution face SR, it is difficult to pre-align the LR images. 
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `check_points/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.  
- `--gpus` specify number of GPUs used to train. The script will use GPUs with more available memory first. To specify the GPU index, uncomment the `export CUDA_VISIBLE_DEVICES=` 


*All models are trained with CelebA and tested on Helen test set provided by [DICNet](https://github.com/Maclory/Deep-Iterative-Collaboration)*



## Acknowledgement

The codes are based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The project also benefits from [DICNet](https://github.com/Maclory/Deep-Iterative-Collaboration).  The project also benefits from [SPARNet](https://github.com/chaofengc/Face-SPARNet).

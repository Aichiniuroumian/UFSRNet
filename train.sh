# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train UFSRNet
# =================================================================================

python train.py --gpus 1 --name UFSRNet_sy92 --model ufsrnet \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot ../celeba_crop_train --dataset_name celeba --batch_size 32 --total_epochs 20 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train 

#python train.py --gpus 1 --name SPARNetLight_Attn3D --model sparnet \
#    --res_depth 1 --att_name spar3d \
#    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 --total_epochs 20 \
#    --dataroot ../celeba_crop_train --dataset_name celeba --batch_size 32 \
#    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train

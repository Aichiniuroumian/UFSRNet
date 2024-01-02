# export CUDA_VISIBLE_DEVICES=$1
# ================================================================================
# Test SPARNet on Helen test dataset provided by DICNet
# ================================================================================

python test.py --gpus 1 --model ufsrnet --name UFSRNet_sy92 \
    --load_size 128 --dataset_name single --dataroot test_dirs/Helen/LR \
    --pretrain_model_path ./pretrain_models/UFSRNet_sy92/latest_net_G.pth \
    --save_as_dir results_helen/UFSRNet_sy92/

  python test.py --gpus 1 --model ufsrnet --name UFSRNet_sy92 \
      --load_size 128 --dataset_name single --dataroot test_dirs/CelebA/LR \
      --pretrain_model_path ./UFSRNet_sy92/check_points/latest_net_G.pth\
      --save_as_dir ./UFSRNet_sy92/result


# ----------------- calculate PSNR/SSIM scores ----------------------------------
python psnr_ssim.py
# ------------------------------------------------------------------------------- 

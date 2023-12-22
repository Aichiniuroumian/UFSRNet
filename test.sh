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

# ================================================================================
# Test SPARNetHD for aligned images
# ================================================================================

python test.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn2D \
    --res_depth 10 --att_name spar --Gnorm 'in' \
    --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/ \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn2D_net_H-epoch10.pth \
    --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn2D/

python test.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn3D \
    --res_depth 10 --att_name spar3d --Gnorm 'in' \
    --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/ \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn3D_net_H-epoch10.pth \
    --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn3D/

# ----------------- calculate FID scores ----------------------------------
python -m pytorch_fid results_CelebA-TestN/SPARNetHD_V4_Attn2D/ test_dirs/CelebAHQ-Test-HR 
python -m pytorch_fid results_CelebA-TestN/SPARNetHD_V4_Attn3D/ test_dirs/CelebAHQ-Test-HR 
# ------------------------------------------------------------------------------- 


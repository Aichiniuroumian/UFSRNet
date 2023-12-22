import argparse
import os
import lpips




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/home/wang107552002794/compare results/HR/Helen')  # 真实图像路径
parser.add_argument('-d1','--dir1', type=str, default='/home/wang107552002794/compare results/MRRNet/MRRNet_fx01_helen')  # 生成图像路径
parser.add_argument('-o','--out', type=str, default='/home/wang107552002794/compare results/Helen_MRRNet.txt')  # 输出的结果保存位置
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_false', help='turn on flag to use GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
dist_sum = 0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		print('%s: %.3f'%(file,dist01))

		f.writelines('%s: %.6f\n'%(file,dist01))
		dist01 = float(dist01)
		dist_sum += dist01
		# dist_sum += dist01
		# print(dist_sum/len(files))
f.close()
print(dist_sum / len(files))

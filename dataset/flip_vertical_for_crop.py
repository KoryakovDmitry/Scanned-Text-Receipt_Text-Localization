# flip crop image

import cv2
import os
import shutil
from tqdm.auto import tqdm

# root = 'val'
# img_path = f'preprocessed/path_for_{root}/flip_vertical_cr_img'
# img_path_root = f'preprocessed/path_for_{root}/cr_img'
img_path = '../test_params_on_val/output/eval_rslt_test_fl2Origin_9171'
img_path_root = '../test_params_on_val/output/eval_rslt_test_keep_fl_9171/img'

if os.path.exists(img_path):
	shutil.rmtree(img_path)

os.mkdir(img_path)

for f in tqdm(os.listdir(img_path_root)):
	f_path_root = os.path.join(img_path_root, f)
	img = cv2.imread(f_path_root)
	img_flip = cv2.flip(img, 0)
	cv2.imwrite(os.path.join(img_path, f), img_flip)

print(f'{img_path} has {len(os.listdir(img_path))} flipped imgs')


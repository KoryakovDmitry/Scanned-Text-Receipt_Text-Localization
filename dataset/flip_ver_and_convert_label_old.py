"""
augment with vertical flip
"""

import os
import shutil
import cv2
import numpy as np
from tqdm.auto import tqdm



def get_points(file_path, h):
	with open(file_path) as f:
		dt = f.readlines()
		# print(dt[-1])
	pointsList = []
	
	for i in range(len(dt)):
		bb = np.asarray(dt[i][:-2].split(',')[:8] +[0, 0])
		bb = bb.reshape(-1, 2)
		# print(bb.shape)
		# print(bb)
		pointsList.append(bb)
		
	pointsList = np.asarray(pointsList)
	pointsList = pointsList.astype(int)
	# print(pointsList.shape)
	# print(type(pointsList))
	pointsList[:, :, 1] = h - pointsList[:, :, 1]
	return pointsList


root = 'test'
img_path_root = f'path_for_{root}/img'
gt_path_root = f'path_for_{root}/gt'

save_folder = f'path_for_{root}_OnlyVerFlip'
img_path_save = f'path_for_{root}_OnlyVerFlip/img'
gt_path_save = f'path_for_{root}_OnlyVerFlip/gt'

save_file_txt_for_training = '/home/hana/pvxFunix_folder/dataset/train_dataset_task1_all_files_w_ver_flip1128.txt'

if root == 'train' and os.path.exists(save_file_txt_for_training):
	os.remove(save_file_txt_for_training)

for i in [save_folder, gt_path_save, img_path_save]:
	if os.path.exists(i):
		shutil.rmtree(i)
	os.mkdir(i)

for f in tqdm(os.listdir(img_path_root)):
	name_of_f = f.split('.')[0]
	path_of_f_img_root = os.path.join(img_path_root, f'{name_of_f}.jpg')
	path_of_f_txt_root = os.path.join(gt_path_root, f'{name_of_f}.txt')
	
	# shutil.copy(path_of_f_img_root, img_path_save)
	# shutil.copy(path_of_f_txt_root, gt_path_save)

	# path_of_f_img_save = os.path.join(img_path_save, f'{name_of_f}0.jpg')
	# path_of_f_txt_save = os.path.join(gt_path_save, f'{name_of_f}0.txt')
	path_of_f_img_save = os.path.join(img_path_save, f'{name_of_f}.jpg')
	path_of_f_txt_save = os.path.join(gt_path_save, f'{name_of_f}.txt')
	
	if root == 'train':
		with open(save_file_txt_for_training, 'a') as f_save_train:
			f_save_train.write(f'{path_of_f_img_save}\t{path_of_f_txt_save}\n')

	
	img_root = cv2.imread(path_of_f_img_root)
	img_root_flipped = cv2.flip(img_root, 0)
	h = img_root.shape[0]

	cv2.imwrite(path_of_f_img_save, img_root_flipped)

	boxes_list = get_points(path_of_f_txt_root, h)
	np.savetxt(path_of_f_txt_save, boxes_list.reshape(-1, 10), delimiter=',', fmt='%d')


print(f'{img_path_save} has {len(os.listdir(img_path_save))} flipped imgs')
print(f'{gt_path_save} has {len(os.listdir(gt_path_save))} flipped gts')


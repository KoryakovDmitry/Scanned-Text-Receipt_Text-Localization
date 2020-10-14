"""
augment with vertical flip
"""

import os
import shutil
import cv2
import numpy as np
from tqdm.auto import tqdm
import random


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

def adjust_label(txt_path, h_dif):
    new_info = ''
    file = open(txt_path, 'r', errors='ignore')
    for line in file.readlines():
        info = line.strip('\r\n').split(',')
        # print(info[0])
        # print('----')
        new_info += str(int(info[0]) - 0) + ','
        new_info += str(h_dif - int(info[1])) + ','

        new_info += str(int(info[2]) - 0) + ','
        new_info += str(h_dif - int(info[3])) + ','

        new_info += str(int(info[4]) - 0) + ','
        new_info += str(h_dif - int(info[5])) + ','

        new_info += str(int(info[6]) - 0) + ','
        new_info += str(h_dif - int(info[7]))
        for word in info[8:]:
            new_info += ','
            new_info += word
        new_info += '\r\n'
    file.close()
    return new_info


root = 'test'
img_path_root = f'path_for_{root}/img'
gt_path_root = f'path_for_{root}/gt'


save_folder = f'path_for_{root}_OnlyVerFlip'
img_path_save = f'path_for_{root}_OnlyVerFlip/img'
gt_path_save = f'path_for_{root}_OnlyVerFlip/gt'

# save_folder = f'path_for_{root}_aug_denoise_randomVerflip'
# img_path_save = f'path_for_{root}_aug_denoise_randomVerflip/img'
# gt_path_save = f'path_for_{root}_aug_denoise_randomVerflip/gt'

save_file_txt_for_training = 'train_dataset_task1_all_files_denoise_randomVerflip.txt'

if os.path.exists(save_file_txt_for_training) and root=='train':
	os.remove(save_file_txt_for_training)

for i in [save_folder, gt_path_save, img_path_save]:
	if os.path.exists(i):
		shutil.rmtree(i)
	os.mkdir(i)

for f in tqdm(os.listdir(img_path_root)):
  # print(f)
  name_of_f = f.split('.')[0]
  path_of_f_img_root = os.path.join(img_path_root, f'{name_of_f}.jpg')
  path_of_f_txt_root = os.path.join(gt_path_root, f'{name_of_f}.txt')
  
  path_of_f_img_root_to_save = os.path.join(img_path_save, f'{name_of_f}.jpg')
  path_of_f_txt_root_to_save = os.path.join(gt_path_save, f'{name_of_f}.txt')
  # shutil.copy(path_of_f_img_root, img_path_save)
  shutil.copy(path_of_f_txt_root, gt_path_save)

  path_of_f_img_save = os.path.join(img_path_save, f'{name_of_f}0.jpg')
  path_of_f_txt_save = os.path.join(gt_path_save, f'{name_of_f}0.txt')
  
  if root == 'train' or root == 'val':
    with open(save_file_txt_for_training, 'a') as f_save_train:
      f_save_train.write(f'../dataset/{path_of_f_img_root_to_save}\t../dataset/{path_of_f_txt_root_to_save}\n')
  
    img_root = cv2.imread(path_of_f_img_root)
    img_root = cv2.cvtColor(img_root, cv2.COLOR_BGR2GRAY)
    img_root = cv2.fastNlMeansDenoising(img_root, None, 5, 7, 21)
  
    cv2.imwrite(path_of_f_img_root_to_save, img_root)


  if root == 'train':
    if random.random() < 0.5: # or root == 'val'
      img_root = cv2.GaussianBlur(img_root, (3, 3), 0)
    if random.random() < 0.5:
      img_root_flipped = cv2.flip(img_root, 0)
  
      cv2.imwrite(path_of_f_img_save, img_root_flipped)
      h = img_root.shape[0]
      boxes_list = adjust_label(path_of_f_txt_root, h)
      
      with open(save_file_txt_for_training, 'a') as f_save_train:
        f_save_train.write(f'../dataset/{path_of_f_img_save}\t../dataset/{path_of_f_txt_save}\n')
    
      with open(path_of_f_txt_save, 'w') as file_label:
        file_label.write(boxes_list)

  if root == 'test':
    # print(f)
    path_of_f_img_save = os.path.join(img_path_save, f'{name_of_f}.jpg')
    path_of_f_txt_save = os.path.join(gt_path_save, f'{name_of_f}.txt')    
    img_root = cv2.imread(path_of_f_img_root)
    img_root = cv2.cvtColor(img_root, cv2.COLOR_BGR2GRAY)
    img_root_flipped = cv2.flip(img_root, 0)
    cv2.imwrite(path_of_f_img_save, img_root_flipped)
    h = img_root.shape[0]
    boxes_list = adjust_label(path_of_f_txt_root, h)
    with open(path_of_f_txt_save, 'w') as file_label:
      file_label.write(boxes_list)


print(f'{img_path_save} has {len(os.listdir(img_path_save))} flipped imgs')
print(f'{gt_path_save} has {len(os.listdir(gt_path_save))} flipped gts')


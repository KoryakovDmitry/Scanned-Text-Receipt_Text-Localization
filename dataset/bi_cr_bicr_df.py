import os
import shutil



target 			= 'train'
# target 			= 'test'
# target 			= 'val'

img_save_path 	= f'/home/hana/pvxFunix_folder/dataset/preprocessed/path_for_{target}/bi_cr_bicr_df_img/'
gt_save_path 	= f'/home/hana/pvxFunix_folder/dataset/preprocessed/path_for_{target}/bi_cr_bicr_df_gt/'

img_root_df		= f'path_for_{target}/img/'
gt_root_df		= f'path_for_{target}/gt/'
print(f'{img_root_df} has {len(os.listdir(img_root_df))}')
print(f'{gt_root_df} has {len(os.listdir(gt_root_df))}')

img_root_cr		= f'preprocessed/path_for_{target}/cr_img/'
gt_root_cr		= f'preprocessed/path_for_{target}/cr_gt/'
print(f'{img_root_cr} has {len(os.listdir(img_root_cr))}')
print(f'{gt_root_cr} has {len(os.listdir(gt_root_cr))}')

img_root_bi		= f'preprocessed/path_for_{target}/bi_img/'
gt_root_bi		= f'path_for_{target}/gt/'
print(f'{img_root_bi} has {len(os.listdir(img_root_bi))}')
print(f'{gt_root_bi} has {len(os.listdir(gt_root_bi))}')

img_root_bicr	= f'preprocessed/path_for_{target}/bi_cr_img/'
gt_root_bicr	= f'preprocessed/path_for_{target}/cr_gt/'
print(f'{img_root_bicr} has {len(os.listdir(img_root_bicr))}')
print(f'{gt_root_bicr} has {len(os.listdir(gt_root_bicr))}')



print('processing ... ')

for f in os.listdir(img_root_df):
	current_file = os.path.join(img_root_df, f)
	shutil.copy(current_file, img_save_path)

for f in os.listdir(gt_root_df):
	current_file = os.path.join(gt_root_df, f)
	shutil.copy(current_file, gt_save_path)	

print('done for df')


for f in os.listdir(img_root_cr):
	current_file = os.path.join(img_root_cr, f)
	dst = os.path.join(img_save_path, f'{f.split(".")[0]}0.jpg')
	shutil.copy(current_file, dst)	

for f in os.listdir(gt_root_cr):
	current_file = os.path.join(gt_root_cr, f)
	dst = os.path.join(gt_save_path, f'{f.split(".")[0]}0.txt')
	shutil.copy(current_file, dst)	

print('done for cr')


for f in os.listdir(img_root_bi):
	current_file = os.path.join(img_root_bi, f)
	dst = os.path.join(img_save_path, f'{f.split(".")[0]}1.jpg')
	shutil.copy(current_file, dst)	

for f in os.listdir(gt_root_bi):
	current_file = os.path.join(gt_root_bi, f)
	dst = os.path.join(gt_save_path, f'{f.split(".")[0]}1.txt')
	shutil.copy(current_file, dst)	

print('done for bi')


for f in os.listdir(img_root_bicr):
	current_file = os.path.join(img_root_bicr, f)
	dst = os.path.join(img_save_path, f'{f.split(".")[0]}2.jpg')
	shutil.copy(current_file, dst)	

for f in os.listdir(gt_root_bicr):
	current_file = os.path.join(gt_root_bicr, f)
	dst = os.path.join(gt_save_path, f'{f.split(".")[0]}2.txt')
	shutil.copy(current_file, dst)	

print('done for bicr')

img_save_quantity 	= len(os.listdir(img_root_df)) + len(os.listdir(img_root_cr)) + len(os.listdir(img_root_bi)) + len(os.listdir(img_root_bicr))
gt_save_quantity 	= len(os.listdir(gt_root_df)) + len(os.listdir(gt_root_cr)) + len(os.listdir(gt_root_bi)) + len(os.listdir(gt_root_bicr))

print(f'{img_save_path} should have {img_save_quantity} files')
print(f'{gt_save_path} should have {gt_save_quantity} files')

print('result are:')
print(f'{img_save_path} has {len(os.listdir(img_save_path))}')
print(f'{gt_save_path} has {len(os.listdir(gt_save_path))}')

condition1 = img_save_quantity == len(os.listdir(img_save_path))
condition2 = gt_save_quantity == len(os.listdir(gt_save_path))

if  condition1 and condition2 and target == 'train':
	file_txt = 'train_dataset_task1_all_files_bi_cr_bicr_df.txt'
	if os.path.exists(file_txt):
		print(file_txt, 'existed')
		os.remove(file_txt)
		print('removed', file_txt)
	for f in os.listdir(img_save_path):
		name = f.split('.')[0]
		with open(file_txt, 'a') as file_txt_train:
			file_txt_train.write(f'{img_save_path}{name}.jpg\t{gt_save_path}{name}.txt\n')

	print('created', file_txt)


"""
sudo chmod +777 path_for_train/bi_cr_bicr_df_img/*
sudo chmod +777 path_for_train/bi_cr_bicr_df_gt/*
sudo chmod +777 path_for_val/bi_cr_bicr_df_gt/*
sudo chmod +777 path_for_val/bi_cr_bicr_df_img/*
sudo chmod +777 path_for_test/bi_cr_bicr_df_img/*
sudo chmod +777 path_for_test/bi_cr_bicr_df_gt/*
"""
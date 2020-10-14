import os
from tqdm.auto import tqdm


root = 'train'

img_path = f'path_for_{root}_aug_denoise_randomVerflip/img/'
gt_path = f'path_for_{root}_aug_denoise_randomVerflip/gt/'
# img_path = f'preprocessed/path_for_{root}/cr_img_verflip/'
# gt_path = f'preprocessed/path_for_{root}/cr_gt_verflip/'

save_file_txt_for_training = 'train_dataset_task1_all_files_denoise_randomVerflip.txt'
# save_file_txt_for_training = 'train_dataset_task1_all_files_CropAndVerflip1128.txt'

if os.path.exists(save_file_txt_for_training) and root=='train':
	os.remove(save_file_txt_for_training)

for file in tqdm(os.listdir(img_path)):
	name = file.split('.')[0]
	name_ext_jpg = file
	name_ext_txt = name + '.txt'

	name_to_write = f'../dataset/{img_path}{name_ext_jpg}\t../dataset/{gt_path}{name_ext_txt}\n'
	with open(save_file_txt_for_training, 'a') as f:
		f.write(name_to_write)


"""
for txt file contains label
-> convert label after predicting on cr_img, bi_cr_img
"""

import os
import shutil
import shutil
import cv2
import numpy as np


def adjust_label(txt_path, w_dif, h_dif):
    new_info = ''
    file = open(txt_path, 'r')
    for line in file.readlines():
        info = line.strip('\r\n').split(',')
        new_info += str(int(info[0]) + w_dif) + ','
        new_info += str(int(info[1]) + h_dif) + ','

        new_info += str(int(info[2]) + w_dif) + ','
        new_info += str(int(info[3]) + h_dif) + ','

        new_info += str(int(info[4]) + w_dif) + ','
        new_info += str(int(info[5]) + h_dif) + ','

        new_info += str(int(info[6]) + w_dif) + ','
        new_info += str(int(info[7]) + h_dif)
        new_info += '\r\n'
    file.close()
    return new_info


def convert_label():

	path_predicted_labels = '../test_params_on_val/output/eval_rslt_test_cr90/result'
	target = 'test'
	path_diff_files = f'preprocessed/path_for_{target}/diff_noblur_gt/'
	path_save_rslt_final = f'preprocessed/path_for_{target}/converted_gt_cr90/'

	if os.path.exists(path_save_rslt_final):
		print(path_save_rslt_final, 'existed')
		shutil.rmtree(path_save_rslt_final)
		print(path_save_rslt_final, 'removed')
	os.mkdir(path_save_rslt_final)
	print(path_save_rslt_final, 'created')
	print('==============')

	for file in os.listdir(path_diff_files):
		d_file_path = os.path.join(path_diff_files, file)
		with open(d_file_path) as d_content:
			d_data = d_content.read()
			w_dif = int(d_data.split(',')[0])
			h_dif = int(d_data.split(',')[1])

		txt_predicted_path = os.path.join(path_predicted_labels, file)
		final_label = adjust_label(txt_predicted_path, w_dif, h_dif)

		txt_converted_path = os.path.join(path_save_rslt_final, file)
		with open(txt_converted_path, "w") as converted_f:
			converted_f.write(final_label)

	print(f'saved at {path_save_rslt_final}')
	print(f'{path_save_rslt_final} has {len(os.listdir(path_save_rslt_final))} files')

convert_label()	
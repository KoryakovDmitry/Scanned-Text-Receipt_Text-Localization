# merge vertical flip result with the original
# must have same file name and number of files
import Polygon as plg
import numpy as np
import os
import shutil
from tqdm.auto import tqdm
from utils import cal_recall_precison_f1, draw_bbox
import cv2



def polygon_from_points(points):
	resBoxes = np.empty([1, 8], dtype='int32')
	resBoxes[0, 0] = int(points[0])
	resBoxes[0, 4] = int(points[1])
	resBoxes[0, 1] = int(points[2])
	resBoxes[0, 5] = int(points[3])
	resBoxes[0, 2] = int(points[4])
	resBoxes[0, 6] = int(points[5])
	resBoxes[0, 3] = int(points[6])
	resBoxes[0, 7] = int(points[7])
	pointMat = resBoxes[0].reshape([2, 4]).T
	return plg.Polygon(pointMat)

def get_union(pD, pG):
	areaA = pD.area();
	areaB = pG.area();
	return areaA + areaB - get_intersection(pD, pG);

def get_intersection_over_union(pD, pG):
	try:
		return get_intersection(pD, pG) / get_union(pD, pG)
	except:
		return 0

def get_intersection(pD, pG):
	pInt = pD & pG
	if len(pInt) == 0:
		return 0
	return pInt.area()

def get_points_get_pols(file_path):
	pols = []
	with open(file_path) as f:
		dt = f.readlines()
		# print(dt[-1])
	pointsList = np.empty([len(dt), 8])
	for i in range(len(dt)):
		bb = np.asarray(dt[i][:-1].split(','))
		pointsList[i] = bb.astype(np.int32)
		pol = polygon_from_points(pointsList[i])
		pols.append(pol)
	return pols, pointsList


root = 'test'
baseline_txt_folder_path = f'../dataset/preprocessed/path_for_{root}/converted_gt/'  # best
img_folder_path = f'../dataset/path_for_{root}/img/'

# cr90_txt_folder_path = f'../dataset/preprocessed/path_for_{root}/converted_gt_cr90/' 
# combined_root = f'../dataset/preprocessed/path_for_{root}/ensemble_baseline_cr90/'
# combined_txt_folder_path = f'../dataset/preprocessed/path_for_{root}/ensemble_baseline_cr90/gt/'
# combined_img_folder_path = f'../dataset/preprocessed/path_for_{root}/ensemble_baseline_cr90/img/'

cr90_txt_folder_path = f'output/eval_rslt_test_vflip/result'
combined_root = f'../dataset/preprocessed/path_for_{root}/ensemble_baseline_baselineVflip/'
combined_txt_folder_path = f'../dataset/preprocessed/path_for_{root}/ensemble_baseline_baselineVflip/gt/'
combined_img_folder_path = f'../dataset/preprocessed/path_for_{root}/ensemble_baseline_baselineVflip/img/'


if os.path.exists(combined_root):
	shutil.rmtree(combined_root)
os.mkdir(combined_root)
print(f'{combined_root} created')

if os.path.exists(combined_txt_folder_path):
	shutil.rmtree(combined_txt_folder_path)
os.mkdir(combined_txt_folder_path)
print(f'{combined_txt_folder_path} created')

if os.path.exists(combined_img_folder_path):
	shutil.rmtree(combined_img_folder_path)
os.mkdir(combined_img_folder_path)
print(f'{combined_img_folder_path} created')


# if any box in second doesnt appear in best -> move to best
for f in tqdm(os.listdir(baseline_txt_folder_path)):
	# print(f)
	f_name = f'{f.split(".")[0]}.jpg' # get img name
	f_baseline_path = os.path.join(baseline_txt_folder_path, f) # path for baseline img
	f_cr90_path = os.path.join(cr90_txt_folder_path, f) # path for cr90 img
	f_save_rslt_path = os.path.join(combined_txt_folder_path, f) # path to save ensemble gt result
	f_img_root_path = os.path.join(img_folder_path, f_name) # path of root img to -> will be used to draw ensemble bb and save to combined_img_folder_path

	pols_baseline, pointsList_baseline = get_points_get_pols(f_baseline_path)
	pols_cr90, pointsList_cr90 = get_points_get_pols(f_cr90_path)
	
	# print(pointsList_cr90[[0,1,2,3]])
	# print(pointsList_cr90.shape)
	# break

	adding_bb = []
	for n_cr90 in range(len(pols_cr90)):
		max_iou = 0
		for n_base in range(len(pols_baseline)):
			pbase = pols_baseline[n_base]
			pcr90 = pols_cr90[n_cr90]
			# IoU_mat[n_base, n_cr90] = get_intersection_over_union(pcr90, pbase)
			IoU_val = get_intersection_over_union(pcr90, pbase)
			if IoU_val > max_iou:
				max_iou = IoU_val
		if max_iou <= 0.2:
			# print('hello')
			adding_bb.append(n_cr90)
			# print(n_cr90)
	
	rslt0 = pointsList_baseline
	# print(rslt0.shape)
	rslt1 = np.asarray(pointsList_cr90[adding_bb])
	# print(rslt1.shape)
	rslt = np.concatenate((rslt0, rslt1), axis=0).reshape(-1,4,2)
	# print(rslt.reshape(-1,4,2).shape)
	
	
	np.savetxt(f_save_rslt_path, rslt.reshape(-1,8), delimiter=',', fmt='%d')

	rslt0_reshape = rslt0.reshape(-1,4,2)
	rslt1_reshape = rslt1.reshape(-1,4,2)
	img = draw_bbox(f_img_root_path, rslt0_reshape, color=(0, 0, 255))
	cv2.imwrite(os.path.join(combined_img_folder_path, '{}'.format(f_name)), img)
	img = draw_bbox(os.path.join(combined_img_folder_path, '{}'.format(f_name)), rslt1_reshape, color=(255, 0, 0))
	cv2.imwrite(os.path.join(combined_img_folder_path, '{}'.format(f_name)), img)








# x = get_points_get_pols('../test_params_on_val/output/eval_rslt/result/X00016469670.txt')
# print(x)
# print(len(x))
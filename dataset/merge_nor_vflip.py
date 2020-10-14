# merge vertical flip result with the original
# must have same file name and number of files
import Polygon as plg
import numpy as np
import os
import shutil
from tqdm.auto import tqdm
from utils import cal_recall_precison_f1, draw_bbox

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
		bb = np.asarray(dt[i][:-2].split(','))
		pointsList[i] = bb.astype(np.int32)
		pol = polygon_from_points(pointsList[i])
		pols.append(pol)
	return pols, pointsList


baseline_txt_folder_path = '../test_params_on_val/output/eval_rslt_val_cr/result'
vflip_txt_folder_path = '../test_params_on_val/output/eval_rslt_val_fl_cr/result'
combined_txt_folder_path = '../test_params_on_val/output/combined_val_cr_fl'


if os.path.exists(combined_txt_folder_path):
	shutil.rmtree(combined_txt_folder_path)
os.mkdir(combined_txt_folder_path)


for f in tqdm(os.listdir(baseline_txt_folder_path)):
	# print(f)
	f_baseline_path = os.path.join(baseline_txt_folder_path, f)
	f_vflip_path = os.path.join(vflip_txt_folder_path, f)
	f_save_rslt_path = os.path.join(combined_txt_folder_path, f)

	pols_baseline, pointsList_baseline = get_points_get_pols(f_baseline_path)
	pols_vflip, pointsList_vflip = get_points_get_pols(f_vflip_path)
	
	output_shape = [len(pols_baseline), len(pols_vflip)]
	IoU_mat = np.empty(output_shape)

	for n_base in range(len(pols_baseline)):
		for n_vflip in range(len(pols_vflip)):
			pBase = pols_baseline[n_base]
			pVflip = pols_vflip[n_vflip]
			IoU_mat[n_base, n_vflip] = get_intersection_over_union(pVflip, pBase)

	# print(IoU_mat.shape)
	# x = np.argmax(IoU_mat, axis=-1)
	# print(x)
	# print(x.shape)
	# not implement for 0 intersection, diff boxes yet.
	base_order_by_vflip = np.argmax(IoU_mat, axis=-1)
	for i in range(len(pointsList_baseline)):
		bb_baseline = pointsList_baseline[i]
		bb_vflip_IoU_mat = pointsList_vflip[base_order_by_vflip[i]]

		# x_min = int(min(bb_baseline[0], bb_vflip_IoU_mat[0], bb_baseline[6], bb_vflip_IoU_mat[6]))
		# x_max = int(max(bb_baseline[2], bb_vflip_IoU_mat[2], bb_baseline[4], bb_vflip_IoU_mat[4]))
		# y_min = int(min(bb_baseline[1], bb_vflip_IoU_mat[1], bb_baseline[3], bb_vflip_IoU_mat[3]))
		# y_max = int(max(bb_baseline[5], bb_vflip_IoU_mat[5], bb_baseline[7], bb_vflip_IoU_mat[7]))

		x_min = int(min(bb_baseline[0], bb_vflip_IoU_mat[0]))
		x_max = int(max(bb_baseline[2], bb_vflip_IoU_mat[2]))
		y_min = int(min(bb_baseline[1], bb_vflip_IoU_mat[1]))
		y_max = int(max(bb_baseline[5], bb_vflip_IoU_mat[5]))
		
		rslt = f'{x_min},{y_min},{x_max},{y_min},{x_max},{y_max},{x_min},{y_max}\n'

		with open(f_save_rslt_path, 'a') as f_save:
			f_save.write(rslt)








# x = get_points_get_pols('../test_params_on_val/output/eval_rslt/result/X00016469670.txt')
# print(x)
# print(len(x))
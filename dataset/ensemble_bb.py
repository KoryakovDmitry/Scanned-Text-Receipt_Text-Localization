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
		bb = np.asarray(dt[i][:-2].split(','))
		pointsList[i] = bb.astype(np.int32)
		pol = polygon_from_points(pointsList[i])
		pols.append(pol)
	return pols, pointsList


baseline_txt_folder_path = 'output/eval_rslt_val_cr/result'
vflip_txt_folder_path = 'output/eval_rslt_val_fl_cr/result'
combined_txt_folder_path = 'output/combined_val_cr_fl/txt'
img_folder_path = '../dataset/preprocessed/path_for_val/cr_img'
combined_img_folder_path = 'output/combined_val_cr_fl/img'


if os.path.exists(combined_txt_folder_path):
	shutil.rmtree(combined_txt_folder_path)
os.mkdir(combined_txt_folder_path)

if os.path.exists(combined_img_folder_path):
	shutil.rmtree(combined_img_folder_path)
os.mkdir(combined_img_folder_path)


for f in tqdm(os.listdir(baseline_txt_folder_path)[:2]):
	print(f)
	f_name = f'{f.split(".")[0]}.jpg'
	f_baseline_path = os.path.join(baseline_txt_folder_path, f)
	f_vflip_path = os.path.join(vflip_txt_folder_path, f)
	f_save_rslt_path = os.path.join(combined_txt_folder_path, f)
	f_img_root_path = os.path.join(img_folder_path, f_name)


# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import cv2
import torch
import shutil
import numpy as np
from tqdm.auto import tqdm
from predict import Pytorch_model
from utils import cal_recall_precison_f1, draw_bbox

torch.backends.cudnn.benchmark = True


def main(model_path, img_folder, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
    model = Pytorch_model(model_path, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        #save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        save_name = os.path.join(save_txt_folder, img_name + '.txt')
        _, boxes_list, t = model.predict(img_path)
        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        # img = draw_bbox('../dataset/preprocessed/path_for_val/cr_img/X51006350763.jpg', boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('1')
    model_path = 'output/tuning_lr_large_PAN_resnet18_FPEM_FFM/checkpoint/model_best.pth'
    # model_path = 'output/PAN_resnet18_FPEM_FFM/checkpoint/PANNet_latest.pth'
    # model_path = 'output/PAN_resnet18_FPEM_FFM/checkpoint/model_best.pth'    
    # model_path = '../PAN.pytorch/output/crop_90/checkpoint/model_best.pth'
    # model_path = '../PAN.pytorch/output/PAN_resnet18_FPEM_FFM/checkpoint/model_best.pth'
    # model_path = '../PAN.pytorch/output/PAN_resnet18_FPEM_FFM/checkpoint/PANNet_latest.pth'
    # model_path = '../CEIR/ceir_pretrained/best_loss0.000151.pth'
    # model_path = 'sroie_pretrained/task1_epoch_0203.pth'
    # model_path = 'pretrained_offical_PAN/PAN_shufflenetv2_FPEM_FFM.pth'
    # fps:2.587515300778842
    # {'precision': 0.34507407407407414, 'recall': 0.26827776078311405, 'hmean': 0.30186817625956625}
    

    # model_path = 'pretrained_offical_PAN/PAN_resnet18_FPEM_FFM.pth'
    # fps:2.5758246383876755
    # {'precision': 0.39872830383227365, 'recall': 0.35484857754665045, 'hmean': 0.3755109131892964}

    # img_path = '../dataset/path_for_train/img'
    # gt_path = '../dataset/path_for_train/gt'
    # img_path = '../dataset/path_for_val_aug_verflip_and_ori/img'
    # gt_path = '../dataset/path_for_val_aug_verflip_and_ori/gt'
    # img_path = 'path_for_val/img'
    # gt_path = 'path_for_val/gt'
    # gt_path = '../dataset/preprocessed/path_for_val/cr_gt'
    # img_path = '../dataset/preprocessed/path_for_val/cr_img'
    # gt_path = '../dataset/preprocessed/path_for_val/cr_gt_aug_denoise_randomVerflip'
    # img_path = '../dataset/preprocessed/path_for_val/cr_img_aug_denoise_randomVerflip'
    # gt_path = '../dataset/preprocessed/path_for_val/cr_gt_verflip'
    # img_path = '../dataset/preprocessed/path_for_val/cr_img_verflip'
    # gt_path = '../dataset/preprocessed/path_for_val/cr_noblur_gt'
    # img_path = '../dataset/preprocessed/path_for_val/cr_noblur_img'
    # img_path = '../dataset/preprocessed/path_for_val/bi_cr_img'
    # img_path = '../dataset/preprocessed/path_for_val/bi_img'
    # img_path = '../dataset/preprocessed/path_for_val/flip_vertical_cr_img'
    # img_path1 = 'output/eval_rslt_val_cr/result'
    # img_path = '../dataset/test_val_flip_img'
    # gt_path = '../dataset/path_for_val/gt'
    # img_path = '../dataset/path_for_val/img'
    
    # gt_path = '../dataset/preprocessed/path_for_train/cr_gt'
    # img_path = '../dataset/preprocessed/path_for_train/bi_cr_img'
    # img_path = '../dataset/preprocessed/path_for_train/bi_img'
    # img_path = '../dataset/preprocessed/path_for_train/cr_img'
    
    # img_path = '../dataset/preprocessed/path_for_test/flip_vertical_cr_img'
    # img_path = '../dataset/preprocessed/path_for_test/cr_img'
    # gt_path = '../dataset/preprocessed/path_for_test/cr_gt'
    # img_path = '../dataset/preprocessed/path_for_test/cr_img_aug_denoise_randomVerflip'
    # gt_path = '../dataset/preprocessed/path_for_test/cr_gt_aug_denoise_randomVerflip'
    # gt_path = '../dataset/preprocessed/path_for_test/cr_noblur_gt'
    # img_path = '../dataset/preprocessed/path_for_test/cr_noblur_img'
    # img_path = '../dataset/preprocessed/path_for_test/bi_img'
    # img_path = '../dataset/preprocessed/path_for_test/bi_cr_img'
    # gt_path = '../dataset/preprocessed/path_for_test/cr_gt_verflip'
    # img_path = '../dataset/preprocessed/path_for_test/cr_img_verflip'
    # img_path = '../dataset/path_for_test/img'
    gt_path = '../dataset/path_for_test/gt'
    img_path = '../dataset/path_for_test_OnlyVerFlip/img'
    # gt_path = '../dataset/path_for_test_OnlyVerFlip/gt'  
      

    print('==========================')
    print(model_path)
    print(img_path)
    print(gt_path)
    print('==========================')


    # save_path = 'output/eval_rslt'
    # save_path = 'output/eval_rslt_test_baseline/result'
    # save_path = '../test_params_on_val/output/eval_rslt_val_cr/result'
    # save_path = '../corner/outputs_eval/ic15/15/res'
    # save_path = '../corner/outputs/imgs/ic13/60/res'
    # save_path = '../dataset/preprocessed/path_for_test/converted_gt_cr90'
    # save_path = '../dataset/preprocessed/path_for_test/converted_gt'
    # save_path = '../dataset/preprocessed/path_for_test/combined_converted_gt'
    # save_path = '../dataset/preprocessed/path_for_test/91.7converted_gt'
    # save_path = '../dataset/preprocessed/path_for_test/ensemble_baseline_cr90/gt'
    save_path = '../dataset/preprocessed/path_for_test/ensemble_baseline_baselineVflip/gt'
    # save_path = '../dataset/preprocessed/path_for_test/converted_gt/'
    gpu_id = 0

    # save_path = main(model_path, img_path, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)

# # -*- coding: utf-8 -*-
# """
# Created on Wed May  2 15:33:52 2018

# @author: Yi (Robin) Fan
# """
# #%% Parameters for user
# c = 0.4                   # in the range[0, 1] recommend 0.2-0.3
# bl = 260                # range[230 - 300]  recommend 260

# #%% get File
# # print("Please Input File Name (Example: image_name)")
# # FILE_NAME = input("FILE_NAME: ")
# # print("Please Input File Format (Example: .png)")
# # FORMAT = input("FORMAT: ")

# #%% Program begin here
# import cv2
# import os
# import shutil
# import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
# from binarization_helpers import *

# # print(FILE_NAME)
# def get_TLI(save_path, f_name, FILE_NAME, FORMAT):
#     # print(FILE_NAME + FORMAT)
#     im = cv2.imread(FILE_NAME + FORMAT)
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     gray = gray.astype(np.float32)

#     width = gray.shape[1]
#     height = gray.shape[0]

#     #%% STEP: contrast enhancement
#     # print("Enhancing Contrast")
#     hp = get_hist(im)
#     sqrt_hw = np.sqrt(height * width)
#     hr = get_hr(hp, sqrt_hw)
#     cei = get_CEI(gray, hr, c)
#     # cv2.imwrite(FILE_NAME + "_Cei" + FORMAT, cei)

#     #%% STEP: Edge detection
#     # print("Edge Detection")
#     # build four filters
#     m1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
#     m2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
#     m3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
#     m4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

#     eg1 = np.abs(cv2.filter2D(gray, -1, m1))
#     eg2 = np.abs(cv2.filter2D(gray, -1, m2))
#     eg3 = np.abs(cv2.filter2D(gray, -1, m3))
#     eg4 = np.abs(cv2.filter2D(gray, -1, m4))
#     eg_avg = scale((eg1 + eg2 + eg3 + eg4) / 4)

#     bins_1 = np.arange(0, 265, 5)
#     #threshold = get_th2(eg_avg, bins_1)
#     eg_bin = img_threshold(30, eg_avg,"H2H") #threshold is hard coded to 30 (based
#                                              #on the paper). Uncomment above to replace
#     # cv2.imwrite(FILE_NAME + "_EdgeBin" + FORMAT, eg_bin)


#     #%% STEP: Text location
#     # print("Locating the Text")
#     bins_2 = np.arange(0, 301, 40)
#     #threshold_c = 255 - get_th2(cei, bins_2)
#     cei_bin = img_threshold(60, cei, "H2L")#threshold is hard coded to 60 (based
#                                            #on the paper). Uncomment above to replace
#     # cv2.imwrite(FILE_NAME + "_CeiBin" + FORMAT, cei_bin)
#     tli = merge(eg_bin, cei_bin)
#     # cv2.imwrite(save_path + f_name + FORMAT, tli)
#     kernel = np.ones((3,3),np.uint8)
#     erosion = cv2.erode(tli,kernel,iterations = 1)
#     # cv2.imwrite(FILE_NAME + "_TLI_erosion" + FORMAT, erosion)


#     # #%% STEP: Light distribution
#     # print("Estimate Light Distribution")
#     int_img = np.array(cei)
#     ratio = int(width / 20)
#     for y in range(width):
#         # if y % ratio == 0 :
#             # print(int(y / width * 100), "%")
#         for x in range(height):
#             if erosion[x][y] == 0:
#                 x = set_intp_img(int_img, x, y, erosion, cei)
#     mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
#     ldi = cv2.filter2D(scale(int_img), -1, mean_filter)
#     # print(ldi)
#     # cv2.imwrite(FILE_NAME + "_LDI" + FORMAT, ldi)


#     #%% STEP: Light Balancing
#     # print("Balancing Light and Generating Result")
#     result = np.divide(cei, ldi) * bl
#     result[np.where(erosion != 0)] *= 1.5
    
#     cv2.imwrite(save_path + f_name + FORMAT, result)


# root = '../dataset/preprocessed/path_for_val/cr_img/'
# save_path = '../dataset/preprocessed/path_for_val/bi_cr_img/'

# target = 'test'
# # root = f'path_for_{target}/img/'
# # save_path = f'preprocessed/path_for_{target}/bi_img/'
# root = f'preprocessed/path_for_{target}/cr_img/'
# save_path = f'preprocessed/path_for_{target}/bi_cr_img_rslt/'
# # save_path = f'preprocessed/path_for_{target}/bi_cr_img/'

# if os.path.exists(save_path):
#     print(save_path, 'existed')
#     shutil.rmtree(save_path)
#     print(save_path, 'removed')
#     os.mkdir(save_path)
#     print(save_path, 'created')
#     print('==============')

# count = 0
# files = os.listdir(root)
# percent = len(files)
# print('Runing binarization.p')
# for f in files:
#     f_name = f.split('.')[0]
#     f_ext = f.split('.')[1]
#     if f_ext == 'jpg':
#         FILE_NAME = root + f_name
#         FORMAT = '.' + f_ext
#         get_TLI(save_path, f_name, FILE_NAME, FORMAT)

#     count += 1
#     print('Finished ', '%.2f%%' % (count/percent * 100))
# print(f'Finised all binarization for {percent} files')
# print(f'save at {save_path}')

################################################################
################################################################
################################################################


import numpy as np
import cv2
import os
import shutil


def black_white(f_name, img_path, output_path):
    # Read image using opencv
    # print(img_path)
    img = cv2.imread(img_path)
    # Extract the file name without the file extension
    # file_name = os.path.basename(img_path).split('.')[0]
    # print(file_name)
    # file_name = file_name.split()[0]
    # Create a directory for outputs
    # output_path = os.path.join('output_path', "ocr")
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # Rescale the image, if needed.
    # img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)  # increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1)  # erodes away the boundaries of foreground object

    # Apply blur to smooth out the edges
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Save the filtered image in the output directory
    save_path = os.path.join(output_path, f_name)
    cv2.imwrite(save_path, img)

# root = '../dataset/preprocessed/path_for_val/cr_img/'
# save_path = '../dataset/preprocessed/path_for_val/bi_cr_img/'

target = 'test'

# root = f'path_for_{target}/img/'
# save_path = f'preprocessed/path_for_{target}/bi_img/'


root = f'preprocessed/path_for_{target}/cr_img/'
save_path = f'preprocessed/path_for_{target}/bi_cr_img/'

# root = 'test_crop_in/img/'
# save_path = 'test_crop_out/img'


if os.path.exists(save_path):
    print(save_path, 'existed')
    shutil.rmtree(save_path)
    print(save_path, 'removed')
    os.mkdir(save_path)
    print(save_path, 'created')
    print('==============')
else:
    os.mkdir(save_path)
    print(save_path, 'created')
    print('==============')

count = 0
files = os.listdir(root)
percent = len(files)
print('Runing binarization.p')
for f in files:
    # print(f)
    f_name = f.split('.')[0]
    f_ext = f.split('.')[1]
    if f_ext == 'jpg':
        FILE_NAME = root + f
        # FORMAT = '.' + f_ext
        black_white(f_name = f, img_path = FILE_NAME, output_path = save_path)

    count += 1
    print('Finished ', '%.2f%%' % (count/percent * 100))
print(f'Finised all binarization for {percent} files')
print(f'save at {save_path}')
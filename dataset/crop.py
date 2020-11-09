import os
import shutil
import cv2
import numpy as np
from tqdm.auto import tqdm


def get_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def step1(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # Sobel_gradient
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # thresh_and_blur
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    return thresh


def step2(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thresh.shape[1]/40), int(thresh.shape[0]/18)))
    morpho_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morpho_image = cv2.erode(morpho_image, None, iterations=1)
    morpho_image = cv2.dilate(morpho_image, None, iterations=1)
    return morpho_image


# Get contour's points, draw Contours and crop image.
def step3(morpho_image, original_img):
    cnts, hierarchy = cv2.findContours(morpho_image.copy(),
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    height = original_img.shape[0]
    weight = original_img.shape[1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = max(min(Xs), 0)
    x2 = min(max(Xs), weight)
    y1 = max(min(Ys), 0)
    y2 = min(max(Ys), height)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]
    return draw_img, crop_img, x1, y1


# Adjust labels for crop image.
def adjust_label(txt_path, w_dif, h_dif):
    new_info = ''
    file = open(txt_path, 'r')
    for line in file.readlines():
        info = line.strip('\r\n').split(',')
        new_info += str(int(info[0]) - w_dif) + ','
        new_info += str(int(info[1]) - h_dif) + ','

        new_info += str(int(info[2]) - w_dif) + ','
        new_info += str(int(info[3]) - h_dif) + ','

        new_info += str(int(info[4]) - w_dif) + ','
        new_info += str(int(info[5]) - h_dif) + ','

        new_info += str(int(info[6]) - w_dif) + ','
        new_info += str(int(info[7]) - h_dif)
        for word in info[8:]:
            new_info += ','
            new_info += word
        new_info += '\r\n'
    file.close()
    return new_info


def crop(target='test'):
    image_path = f'path_for_{target}/img/'
    label_path = f'path_for_{target}/gt/'
    image_save_path = f'preprocessed/path_for_{target}/img/'
    label_save_path = f'preprocessed/path_for_{target}/gt/'
    label_diff_save_path = f'preprocessed/path_for_{target}/diff_gt/'
    
    for save_path in [image_save_path, label_save_path, label_diff_save_path]:
        if os.path.exists(save_path):
            print(save_path, 'existed')
            shutil.rmtree(save_path)
            print(save_path, 'removed')
        os.mkdir(save_path)
        print(save_path, 'created')
        print('==============')

    count_croped = 0

    for _, _, names in os.walk(image_path):
        filenames = []
        for name in names:
            if name[-3:] == 'jpg':
                filenames.append(name)
        for file in tqdm(filenames):
            file_path = image_path + file
            txt_path = label_path + file[:-3] + 'txt'
            try:
                original_img, gray = get_image(file_path)
                if original_img.shape[1] > 990:
                    gradient = step1(gray)
                    morpho_image = step2(gradient)
                    draw_img, crop_img, weight, height = step3(morpho_image, original_img)
                    new_image_save_path = image_save_path + file
                    if crop_img.size == 0:
                        crop_img = original_img
                    cv2.imwrite(new_image_save_path, crop_img)

                    new_label = adjust_label(txt_path, weight, height)
                    file_txt_path = label_save_path + file[:-3] + 'txt'
                    with open(file_txt_path, "w") as f:
                        f.write(new_label)
                    
                    diff = f'{weight},{height}'
                    diff_files = label_diff_save_path + file[:-3] + 'txt'
                    with open(diff_files, "w") as f_d:
                        f_d.write(diff)

                    count_croped += 1

                else:
                    shutil.copy(file_path, image_save_path)
                    shutil.copy(txt_path, label_save_path)

                    diff = f'0,0'
                    diff_files = label_diff_save_path + file[:-3] + 'txt'
                    with open(diff_files, "w") as f_d:
                        f_d.write(diff)

            except BaseException:
                print(BaseException)
                shutil.copy(file_path, image_save_path)
                shutil.copy(txt_path, label_save_path)

                diff = f'0,0'
                diff_files = label_diff_save_path + file[:-3] + 'txt'
                with open(diff_files, "w") as f_d:
                    f_d.write(diff)

    print(f'saved at {image_save_path}')
    print(f'saved at {label_save_path}')
    print(f'saved at {label_diff_save_path}')
    print(f'crop {count_croped} files over {len(filenames)} files')


if __name__ == '__main__':
    list_target = ['test', 'val']
    
    for target in list_target:
        crop(target=target)

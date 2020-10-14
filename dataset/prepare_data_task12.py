import os
import shutil
import random


# train_path = '0325updated_task1train626p'
# val_ratio = 0.1
# val_img_path = 'path_for_val/img'
# val_gt_path = 'path_for_val/gt'
train_file_txt = 'train_dataset_task1_all_files.txt'


def prepare_train_val(train_path = '0325updated_task1train626p',
                      val_ratio = 0.1,
                      train_img_path = 'path_for_train/img',
                      train_gt_path = 'path_for_train/gt',
                      val_img_path = 'path_for_val/img',
                      val_gt_path = 'path_for_val/gt',
                      train_file_txt = 'train_dataset_task1_all_files.txt',
                      train_file_txt_crop = 'train_dataset_task1_all_files_crop.txt',
                      train_file_txt_bi = 'train_dataset_task1_all_files_bi.txt',
                      train_file_txt_crop_bi = 'train_dataset_task1_all_files_crop_bi.txt'):
    """
    split train and val based on val_ratio
    removing files which aren't matched the format:  'name.ext' without spaces between
    train_path : folder includes all training files: imgs, txts
    val_img_path : storages imgs for val
    val_gt_path : storages txt for val
    train_file_txt : records the path to training samples with format: path/to/train/imgs\tpath/to/train/txt
    val_ratio : ratio for val
    """
    print('==================')
    print('preparing train - val')
    print('==================')
    # for each time we want to split train - val
    for path in [train_file_txt, train_file_txt_bi, train_file_txt_crop, train_file_txt_crop_bi]:
        if os.path.exists(path):
            print(path, 'existed')
            os.remove(path)
            print(path, 'deleted')

    for path in [val_img_path, val_gt_path, train_img_path, train_gt_path]:
        if os.path.exists(path):
            print(path, 'existed')
            shutil.rmtree(path)
            print(path, 'removed')
            os.mkdir(path)
            print(path, 'created')
            print('==============')


    all_files_for_train_txt     =   []
    root_for_default_files      =   '/home/hana/pvxFunix_folder/dataset/path_for_train/img'
    root_for_default_files_gt   =   '/home/hana/pvxFunix_folder/dataset/path_for_train/gt'
    root_for_crop_files         =   '/home/hana/pvxFunix_folder/dataset/preprocessed/path_for_train/cr_img'
    root_for_crop_files_gt      =   '/home/hana/pvxFunix_folder/dataset/preprocessed/path_for_train/cr_gt'
    root_for_bi_files           =   '/home/hana/pvxFunix_folder/dataset/preprocessed/path_for_train/bi_img'
    root_for_crop_bi_files      =   '/home/hana/pvxFunix_folder/dataset/preprocessed/path_for_train/bi_cr_img'


    for root, _, files in os.walk(train_path):
        # get file name for both jpg and txt
        # print(root)
        # print(files)
        # break

        for f in files:
            if len(f.split()) > 1:
                os.remove(f'{root}/{f}') # remove duplicates
            elif f.split('.')[-1] == 'jpg':
                file_name = f.split('.')[0]
                all_files_for_train_txt.append(file_name)
                # start to check error of readin g file
                try:
                    with open(f'{root}/{file_name}.txt') as read2check_errors:
                        data = read2check_errors.read()
                except ValueError as vle:
                    print(f'file {root}/{file_name}')
                    print(vle)
                    break # or remove
                    # finished checking error of reading file

        # get validation files ~ .1
        val_files = random.sample(all_files_for_train_txt, k = int(len(all_files_for_train_txt) * val_ratio))
        print(f'len of all_files: {len(set(all_files_for_train_txt))}')
        print(f'len of val_files: {len(set(val_files))}')
        # copying val_files to val_path
        for val_f in val_files:
            jpg_file = f'{root}/{val_f}.jpg'
            txt_file = f'{root}/{val_f}.txt'
            shutil.copy2(jpg_file, val_img_path)
            shutil.copy2(txt_file, val_gt_path)
        print(f'moved {len(os.listdir(val_img_path))} imgs to {val_img_path}')
        print(f'moved {len(os.listdir(val_gt_path))} txts to {val_gt_path}')
        # get train files
        train_files = set(all_files_for_train_txt) - set(val_files)
        print(f'len of train_files: {len(train_files)}')
        for train_f in train_files:
            jpg_file = f'{root}/{train_f}.jpg'
            txt_file = f'{root}/{train_f}.txt'
            # print(jpg_file)
            shutil.copy2(jpg_file, train_img_path)
            shutil.copy2(txt_file, train_gt_path)
            with open(train_file_txt, 'a') as dataset_file:
                dataset_file.write(f'{root_for_default_files}/{train_f}.jpg\t{root_for_default_files_gt}/{train_f}.txt\n')
            with open(train_file_txt_crop, 'a') as dataset_file_crop:
                dataset_file_crop.write(f'{root_for_crop_files}/{train_f}.jpg\t{root_for_crop_files_gt}/{train_f}.txt\n')
            with open(train_file_txt_bi, 'a') as dataset_file_bi:
                dataset_file_bi.write(f'{root_for_bi_files}/{train_f}.jpg\t{root_for_default_files_gt}/{train_f}.txt\n')
            with open(train_file_txt_crop_bi, 'a') as dataset_file_crop_bi:
                dataset_file_crop_bi.write(f'{root_for_crop_bi_files}/{train_f}.jpg\t{root_for_crop_files_gt}/{train_f}.txt\n')
        print(f'moved {len(os.listdir(train_img_path))} imgs to {train_img_path}')
        print(f'moved {len(os.listdir(train_gt_path))} txts to {train_gt_path}')

def prepare_test(path_to_txts_test = 'path_for_test/gt',
               path_to_imgs_test = 'path_for_test/img'):
    """
    remove mismatched files
    """
    print('==================')
    print('prepare test data')
    print('==================')
    for f in os.listdir(path_to_txts_test):
        txt_name = f.split('.')[0]
        if not os.path.exists(f'{path_to_imgs_test}/{txt_name}.jpg'):
            print(txt_name)
            os.remove(f'{path_to_txts_test}/{txt_name}.txt')
            print(f'removed {path_to_txts_test}/{txt_name}.txt')

    print(f'{path_to_txts_test} has {len(os.listdir(path_to_txts_test))} files')
    print(f'{path_to_imgs_test} has {len(os.listdir(path_to_imgs_test))} files')

    # print("remove sample when it's txt file can not read")
    # for f in os.listdir(path_to_txts_test):
    #     try:
    #         with open(f'{path_to_txts_test}/{f}') as txt_file:
    #             dt = txt_file.read()
    #     except ValueError as vle:
    #         print(f'file {f} has error: {vle}')
    #         os.remove(f'{path_to_txts_test}/{f}')
    #         print(f'{path_to_txts_test}/{f}')
    #         os.remove(f'{path_to_imgs_test}/{f.split(".")[0]}.jpg')
    #         print(f'{path_to_imgs_test}/{f.split(".")[0]}.jpg') 

    print(f'{path_to_txts_test} has {len(os.listdir(path_to_txts_test))} files')
    print(f'{path_to_imgs_test} has {len(os.listdir(path_to_imgs_test))} files')
    print('==================')
    print('finished')
    print('==================')


if __name__ == '__main__':
    prepare_train_val()
    #recheck len of train_file_txt
    print('==============')
    print(f'recheck {train_file_txt}')
    with open (train_file_txt) as f:
        dt = f.readlines()
        print(len(dt))
        print(f'last line of {train_file_txt}')
        print(dt[-1])

    print('==================')
    print('finished')
    print('==================')

    prepare_test()

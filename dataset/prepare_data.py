import os
import shutil
import random


train_file_txt = 'train_dataset_all_files.txt'

def prepare_train_val(train_path = 'all_imgs_gts_training_file',
                      val_ratio = 0.1,
                      train_img_path = 'path_for_train/img',
                      train_gt_path = 'path_for_train/gt',
                      val_img_path = 'path_for_val/img',
                      val_gt_path = 'path_for_val/gt',
                      train_file_txt = train_file_txt):
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

    if os.path.exists(train_file_txt):
        print(train_file_txt, 'existed')
        os.remove(train_file_txt)
        print(train_file_txt, 'deleted')

    for path in [val_img_path, val_gt_path, train_img_path, train_gt_path]:
        if os.path.exists(path):
            print(path, 'existed')
            shutil.rmtree(path)
            print(path, 'removed')
        os.mkdir(path)
        print(path, 'created')
        print('==============')

    all_files_for_train_txt     =   []

    for root, _, files in os.walk(train_path):
        # get file name for both jpg and txt
        for f in files:
            if len(f.split()) > 1:
                os.remove(f'{root}/{f}') # remove duplicates
            elif f.split('.')[-1] == 'jpg':
                file_name = f.split('.')[0]
                all_files_for_train_txt.append(file_name)
                # start to check error when reading file
                try:
                    with open(f'{root}/{file_name}.txt') as read2check_errors:
                        data = read2check_errors.read()
                except ValueError as vle:
                    print(f'file {root}/{file_name}')
                    print(vle)
                    break # or remove
                    # finished checking error when reading file

        # get validation files ~ val_ratio
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
                dataset_file.write(f'{train_img_path}/{train_f}.jpg\t{train_gt_path}/{train_f}.txt\n')
           
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

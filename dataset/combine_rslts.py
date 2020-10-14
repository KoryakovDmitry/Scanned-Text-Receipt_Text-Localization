"""
idea is just save all in one file

NOT RIGHT!!
"""
import os
import shutil




path_rslt1 = 'preprocessed/path_for_test/converted_gt'
path_rslt2 = 'preprocessed/path_for_test/91.7converted_gt'
path_combined = 'preprocessed/path_for_test/combined_converted_gt'


print('------------------------------------------------')

for file in os.listdir(path_rslt1):
	path_file_rslt1 = os.path.join(path_rslt1, file)
	# print(path_file_rslt1)
	path_file_rslt2 = os.path.join(path_rslt2, file)
	# print(path_file_rslt2)
	path_file_combined = os.path.join(path_combined, file)
	# print(path_file_combined)
	
	with open(path_file_rslt1) as f_rslt1:
		dt_f_rslt1 = f_rslt1.readlines()[:-1]

	with open(path_file_rslt2) as f_rslt2:
		dt_f_rslt2 = f_rslt2.readlines()

	dt_combined = ''.join(cor for cor in dt_f_rslt1 + dt_f_rslt2)
	with open(path_file_combined, 'w') as f_combined:
		f_combined.write(dt_combined)

print(f'{len(os.listdir(path_combined))} files in {path_combined}')

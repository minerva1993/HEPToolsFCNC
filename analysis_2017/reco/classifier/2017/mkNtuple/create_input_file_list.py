import os

path_to_prod = '/data/users/minerva1993/ntuple_Run2017/V9_2/180614/production/'
print("Looking for files in %s"%path_to_prod)

input_list_signal_file_name = 'file_signal.txt'
string_for_signal_processing = ''

input_list_bkg_file_name = 'file_other.txt'
string_for_bkg_processing = ''

for dataset_folder in os.listdir(path_to_prod):
  dataset_path = os.path.join(path_to_prod, dataset_folder)
  for file_name in os.listdir(dataset_path):
    tmp_string = ''
    file_id = file_name.split('_')[-1].split('.')[0]
    tmp_string += os.path.join(dataset_path, file_name)
    output_file_name = dataset_folder.replace("_",'')
    tmp_string += ' ' + output_file_name + "_" + file_id
    if 'ST' in dataset_folder:
      string_for_signal_processing += tmp_string + '\n'
    else:
      string_for_bkg_processing += tmp_string + '\n'

with open(input_list_signal_file_name, 'w') as f:
  f.write(string_for_signal_processing) 

with open(input_list_bkg_file_name, 'w') as f: 
  f.write(string_for_bkg_processing) 

print("{0} and {1} written.\nPlease check the signal file list as the criteria is to have 'ST' in the name to be signal (some background with this pattern may appear in the future).".format(input_list_signal_file_name, input_list_bkg_file_name))

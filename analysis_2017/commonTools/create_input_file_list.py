import os

version = 'V9_2/180703/'
path_to_prod = '/data/users/minerva1993/ntuple_Run2017/' + version + 'production/'
path_to_prod_noreco = '/data/users/minerva1993/ntuple_Run2017/' + version
print("Looking for files in %s"%path_to_prod)

merge_file_name = 'merge_ntuples.sh'
string_for_merge = ''

input_list_signal_file_name = 'file_signal.txt'
string_for_signal_processing = ''

input_list_bkg_file_name = 'file_other.txt'
string_for_bkg_processing = ''


#This part is for making script file for ntuple merging
string_for_merge += '#!/bin/sh\n'
string_for_merge += 'for i in ' + path_to_prod + '*; do hadd $i.root $i/*.root; done\n'

for dataset_folder in os.listdir(path_to_prod):
  if "part2" in dataset_folder:
    dataset_folder = dataset_folder[:-6]
    string_for_merge += 'hadd ' + path_to_prod + dataset_folder + '_v2.root ' + path_to_prod + dataset_folder+ '.root ' + path_to_prod + dataset_folder + '_part2.root\n'
    string_for_merge += 'rm ' + path_to_prod + dataset_folder+ '.root ' + path_to_prod + dataset_folder + '_part2.root\n'
string_for_merge += 'hadd ' + path_to_prod + 'SingleElectron_Run2017.root ' + path_to_prod + 'SingleElectron_Run2017*.root\n'
string_for_merge += 'hadd ' + path_to_prod + 'SingleMuon_Run2017.root ' + path_to_prod + 'SingleMuon_Run2017*.root\n'
string_for_merge += 'rm ' + path_to_prod + 'SingleMuon_Run2017[B-F].root ' + path_to_prod + 'SingleElectron_Run2017[B-F].root\n'

string_for_merge += 'mv ' + path_to_prod + '*.root ' + path_to_prod_noreco + '\n'

with open(merge_file_name, 'w') as f:
  f.write(string_for_merge)


#This part is for reconstruction
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



print("{0} and {1}  written.\nPlease check the signal file list as the criteria is to have 'ST' in the name to be signal (some background with this pattern may appear in the future).".format(input_list_signal_file_name, input_list_bkg_file_name))

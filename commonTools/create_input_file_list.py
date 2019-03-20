import os, sys

if len(sys.argv) < 2:
  print("Specify year: 2017/2018")
  sys.exit()
era = sys.argv[1]

ttbb = False
file_postfix = ''
if len(sys.argv) > 2:
  if sys.argv[2] == 'ttbb':
    ttbb = True
    file_postfix = '_ttbb'

if era == '2017':
  version = 'V9_5/190117/'
  path_to_prod = '/data/users/minerva1993/ntuple_Run2017/' + version + 'production/'
  path_to_prod_noreco = '/data/users/minerva1993/ntuple_Run2017/' + version
elif era == '2018':
  version = 'V10_1/190318/'
  path_to_prod = '/data/users/minerva1993/ntuple_Run2018/' + version + 'production/'
  path_to_prod_noreco = '/data/users/minerva1993/ntuple_Run2018/' + version

print("Looking for files in %s"%path_to_prod)

merge_file_name = 'merge_' + era + '_ntuples.sh'
string_for_merge = ''

input_list_signal_file_name = 'file_' + era + '_top' + file_postfix + '.txt'
string_for_signal_processing = ''

input_list_bkg_file_name = 'file_' + era + '_other' + file_postfix + '.txt'
string_for_bkg_processing = ''

input_list_syst_file_name = 'file_' + era + '_syst' + file_postfix + '.txt'
string_for_syst_processing = ''

input_list_all_file_name = 'file_' + era + '_all' + file_postfix + '.txt'
string_for_all_processing = ''

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
    if ('hdampup' in dataset_folder) or ('hdampdown' in dataset_folder) or ('TuneCP5up' in dataset_folder) or ('TuneCP5down' in dataset_folder):
      string_for_syst_processing += tmp_string + '\n'
    elif ('ST_' in dataset_folder) or ('TT_powheg' in dataset_folder) or ('TTHad_powheg' in dataset_folder) or ('TTLL_powheg' in dataset_folder) or ('TT_TH' in dataset_folder):
      string_for_signal_processing += tmp_string + '\n'
    else:
      string_for_bkg_processing += tmp_string + '\n'

with open(input_list_signal_file_name, 'w') as f:
  f.write(string_for_signal_processing) 

with open(input_list_bkg_file_name, 'w') as f: 
  f.write(string_for_bkg_processing) 

with open(input_list_syst_file_name, 'w') as f:
  f.write(string_for_syst_processing)

with open(input_list_all_file_name, 'w') as f:
  string_for_all_processing += string_for_signal_processing
  string_for_all_processing += string_for_bkg_processing
  string_for_all_processing += string_for_syst_processing
  f.write(string_for_all_processing)


print("{0}, {1} and {2}  written.".format(input_list_signal_file_name, input_list_bkg_file_name, input_list_syst_file_name))

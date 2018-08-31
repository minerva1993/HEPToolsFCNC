import os
import shutil

version = 'V9_2/180815/'
path_to_prod_noreco = '/data/users/minerva1993/ntuple_Run2017/' + version

run_file_name = 'runNoReco.py'
string_for_run = ''

merge_file_name = 'job_merge.sh'
string_for_merge = ''

print("Write options for bypassing reconstruction")
#This part is for bypass reconstruction
noreco_list = []
for file_name in os.listdir(path_to_prod_noreco):
  if file_name.endswith(".root"):
    dataset_path = os.path.join(path_to_prod_noreco, file_name)
    tmp_string = ''
    file_id = file_name.split('_')[-1].split('.')[0]
    tmp_string += dataset_path
    output_file_name = file_name.replace("_",'')
    output_file_name = output_file_name.replace(".root",'')
    tmp_string += ' ' + output_file_name
    noreco_list.append(tmp_string)

if os.path.exists(run_file_name): os.remove(run_file_name)
shutil.copy2('template_runNoReco', run_file_name)

for lines in noreco_list:
  string_for_run += '  runAna("' + lines.split(' ')[0] + '", "' + (lines.split(' ')[1]) + '")\n'

with open(run_file_name, 'a') as target:
  target.write(string_for_run)


print("Write script for mergeing histograms")
ext_dataset = []
string_for_merge += '#!/bin/sh\n'
string_for_merge += 'rm hist_*.root\n'

for lines in noreco_list:
  if "v2" in lines.split(' ')[1]: ext_dataset.append(lines.split(' ')[1])
  elif "Run2017" in lines.split(' ')[1]:
    string_for_merge += "hadd hist_" + (lines.split(' ')[1]) + ".root temp/hist_" + (lines.split(' ')[1]) + "*.root\n"
  else: string_for_merge += "hadd hist_" + (lines.split(' ')[1]) + ".root temp/hist_" + (lines.split(' ')[1]) + "_*.root\n"

for exts in ext_dataset:
  string_for_merge += "hadd hist_" + (exts) + ".root temp/hist_" + (exts)[:-2] + "_*.root temp/hist_" + (exts)[:-2] + "part2_*.root\n"

if os.path.exists('doReco/' + merge_file_name): os.remove('doReco/' + merge_file_name)
with open('doReco/' + merge_file_name, 'w') as g:
  g.write(string_for_merge)
  


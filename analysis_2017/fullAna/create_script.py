import os
import shutil

path_to_fileList = '../commonTools/file_noreco.txt'

run_file_name = 'runNoReco.py'
string_for_run = ''

merge_file_name = 'job_merge.sh'
string_for_merge = ''

print("Write options for bypassing reconstruction script from %s"%path_to_fileList)
with open(path_to_fileList, 'r') as f:
  if os.path.exists(run_file_name): os.remove(run_file_name)
  shutil.copy2('template_runNoReco', run_file_name)

  with open(run_file_name, 'a') as target:
    content = f.readlines()

    for lines in content:
      string_for_run += 'runAna("' + lines.split(' ')[0] + '", "' + (lines.split(' ')[1])[:-1] + '")\n'

    target.write(string_for_run)


print("Write script for mergeing histograms")
string_for_merge += '#!/bin/sh\n'
string_for_merge += 'rm hist_*.root\n'
with open(path_to_fileList, 'r') as f:
  content = f.readlines()

  for lines in content:
    string_for_merge += "hadd hist_" + (lines.split(' ')[1])[:-1] + ".root temp/hist_" + (lines.split(' ')[1])[:-1] + "*.root\n"

if os.path.exists(merge_file_name): os.remove(merge_file_name)
with open(merge_file_name, 'w') as g:
  g.write(string_for_merge)
  


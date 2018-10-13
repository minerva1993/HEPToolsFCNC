import os, shutil, re

version = 'V9_3/181009/'
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

syst = ["__jecup", "__jecdown", "__jerup", "__jerdown",]
syst1 = ["__TuneCP5up", "__TuneCP5down", "__hdampup", "__hdampdown",]
syst2 = [""]

print("Write script for mergeing histograms")
ext_dataset = []
fcnc_dataset = []
string_for_merge += '#!/bin/sh\n'
string_for_merge += 'mkdir pre_process\n'
string_for_merge += 'rm pre_process/hist_*.root\n'

for lines in noreco_list:
  if "v2" in lines.split(' ')[1]: ext_dataset.append(lines.split(' ')[1])
  elif "TTTH" in lines.split(' ')[1]: fcnc_dataset.append(lines.split(' ')[1])
  elif "Run2017" in lines.split(' ')[1]:
    string_for_merge += "hadd pre_process/hist_" + (lines.split(' ')[1]) + ".root temp/hist_" + (lines.split(' ')[1]) + "[A-F]_*[0-9].root\n"
  else:
    for syst_ext in syst + syst1 + syst2:
      if (syst_ext in syst1) and (not syst_ext.replace('__','') in lines.split(' ')[1]): continue
      if (syst_ext in syst) and any(tmp.replace('__','') in lines.split(' ')[1] for tmp in syst1): continue
      if any(tmp.replace('__','') in lines.split(' ')[1] for tmp in syst1) and (syst_ext == ""): continue
      if (syst_ext in syst1):
        syst_ext = syst_ext.replace('__','')
        string_for_merge += "hadd pre_process/hist_" + (lines.split(' ')[1].replace(syst_ext,'')) + "__" + syst_ext + ".root temp/hist_" + (lines.split(' ')[1].replace(syst_ext,'')) + "_*[0-9]" + "__" + syst_ext + ".root\n"
      else: string_for_merge += "hadd pre_process/hist_" + (lines.split(' ')[1]) + syst_ext + ".root temp/hist_" + (lines.split(' ')[1]) + "_" + "*[0-9]" + syst_ext + ".root\n"

for exts in ext_dataset:
  for syst_ext in syst + syst1 + syst2:
    if (syst_ext in syst1) and (not syst_ext.replace('__','') in exts): continue
    if (syst_ext in syst) and any(tmp.replace('__','') in exts for tmp in syst1): continue
    if any(tmp.replace('__','') in exts for tmp in syst1) and (syst_ext == ""): continue
    if (syst_ext in syst1):
      syst_ext = syst_ext.replace('__','')
      string_for_merge += "hadd pre_process/hist_" + (exts.replace(syst_ext,'')) + "__" + syst_ext + ".root temp/hist_" + (exts) + "_" + "*[0-9].root\n"
    else: string_for_merge += "hadd pre_process/hist_" + (exts) + syst_ext + ".root temp/hist_" + (exts)[:-2] + "_*[0-9]" + syst_ext + ".root temp/hist_" + (exts)[:-2] + "part2_*[0-9]" + syst_ext + ".root\n"

sig_reg = re.compile('H[u,c]t')

for ttsig in fcnc_dataset:
  if "aTLep" in ttsig: continue
  chs = ['Hct', 'Hut']
  for ch in chs:
    if ch not in ttsig: continue
    for syst_ext in syst + syst2:
      string_for_merge += "hadd pre_process/hist_TTTH1L3B" + ch + syst_ext + ".root temp/hist_TTTH*" + ch + "_*[0-9]" + syst_ext + ".root\n"

if os.path.exists('doReco/' + merge_file_name): os.remove('doReco/' + merge_file_name)
with open('doReco/' + merge_file_name, 'w') as g:
  g.write(string_for_merge)

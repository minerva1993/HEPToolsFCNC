import sys, os
import re
from subprocess import call

if len(sys.argv) < 3:
  print("Ver of 2017 and 2018")
  sys.exit()
ver17 = sys.argv[1]
ver18 = sys.argv[2]

config_path = '../../plotIt/configs/'
dest_path = './full1718'
common_syst = 'systematics:\n'
#not include prefire and elzvtx which exist only in 2017
common_syst_list = ['pu', 'muid', 'muiso', 'mutrg', 'elid', 'elreco', 'eltrg',
                    'jec', 'jer',
                    'lf', 'hf', 'lfstat1', 'lfstat2', 'hfstat1', 'hfstat2', 'cferr1', 'cferr2']
#reco_scenario = ['STFCNC', 'TTFCNC', 'TTBKG']
reco_scenario = ['STFCNC']

for item in common_syst_list:
  common_syst += '  - ' + item + '\n'

for scenario in reco_scenario:
  string_for_files = ''
  #Firstly, merge file list + scale
  with open(config_path + 'files_2017.yml') as f:
    lines = f.readlines()
    skip_merged_TT = False
    for line in lines:
      if '#' in line[0]: line = line[1:]
      if skip_merged_TT and 'hist' in line: skip_merged_TT = False
      if 'hist_TTTH1L3BH' in line: skip_merged_TT = True
      if 'hist' in line:
        line = line[0] + '2017/' + scenario + ver17 + '/post_process/' + line[1:]
        if not any(i in line for i in ['TH1L3B', 'Run201']):
          line += '  scale: ' + str(41529/101270.0) + '\n'
      if not skip_merged_TT: string_for_files += line

  with open(config_path + 'files_2018.yml') as f:
    lines = f.readlines()
    skip_merged_TT = False
    for line in lines:
      if '#' in line[0]: line = line[1:]
      if skip_merged_TT and 'hist' in line: skip_merged_TT = False
      if 'hist_TTTH1L3BH' in line: skip_merged_TT = True
      if 'hist' in line:
        line = line[0] + '2018/' + scenario + ver18 + '/post_process/' + line[1:]
        if not any(i in line for i in ['TH1L3B', 'Run201']):
          line += '  scale: ' + str(59741/101270.0) + '\n'
      if not skip_merged_TT and 'order' not in line: string_for_files += line

  with open(config_path + 'files_1718.yml', 'w+') as fnew:
    fnew.write(string_for_files)

  with open(config_path + 'template_1718.yml') as f:
    lines = f.readlines()
    with open(config_path + 'config_1718.yml', 'w+') as f1:
      for line in lines: f1.write(line)
      f1.write(common_syst)
      f1.write("\nplots:\n  include: ['histos_" + scenario.lower() + ".yml']\n")

  call(['../../plotIt/plotIt', '-o ' + dest_path + '/' + scenario, config_path + 'config_1718.yml'], shell=False)

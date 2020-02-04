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
reco_scenario = ['STFCNC', 'TTFCNC', 'TTBKG']
#reco_scenario = ['STFCNC']

for item in common_syst_list:
  common_syst += '  - ' + item + '\n'

for scenario in reco_scenario:
  string_for_files = ''
  #Firstly, merge file list + scale
  with open(config_path + 'files_2017.yml') as f:
    lines = f.readlines()
    skip_signal = False
    for line in lines:
      if '#' in line[0]: line = line[1:]
      if skip_signal and 'hist' in line: skip_signal = False
      if 'TH1L3B' in line: skip_signal = True
      if 'hist_QCD' in line: skip_signal = True
      if 'hist' in line:
        line = line[0] + '2017/' + scenario + ver17 + '/post_process/' + line[1:]
        if not any(i in line for i in ['TH1L3B', 'Run201']):
          line += '  scale: ' + str(41529/101270.0) + '\n'
      if not skip_signal and not any(i in line for i in ['yields-group']): string_for_files += line

  with open(config_path + 'files_2018.yml') as f:
    lines = f.readlines()
    skip_signal = False
    for line in lines:
      if '#' in line[0]: line = line[1:]
      if skip_signal and 'hist' in line: skip_signal = False
      if 'TH1L3B' in line: skip_signal = True
      if 'hist_QCD' in line: skip_signal = True
      if 'hist' in line:
        line = line[0] + '2018/' + scenario + ver18 + '/post_process/' + line[1:]
        if not any(i in line for i in ['TH1L3B', 'Run201']):
          line += '  scale: ' + str(59741/101270.0) + '\n'
      if not skip_signal and not any(i in line for i in ['yields-group']): string_for_files += line

  with open(config_path + 'files_1718.yml', 'w+') as fnew:
    print>>fnew, """
'full1718/{0}/hist_STTH1L3BHct.root':
  type: signal
  pretty-name: 'STTH1L3BHct'
  cross-section: 0.076
  generated-events: 1000000
  group: GSTHct
  order: 1

'full1718/{0}/hist_TTTH1L3BHct.root':
  type: signal
  pretty-name: 'TTTH1L3BHct'
  cross-section: 1.86
  generated-events: 1000000
  group: GTTHct
  order: 2

'full1718/{0}/hist_STTH1L3BHut.root':
  type: signal
  pretty-name: 'STTH1L3BHut'
  cross-section: 0.55
  generated-events: 1000000
  group: GSTHut
  order: 3

'full1718/{0}/hist_TTTH1L3BHut.root':
  type: signal
  pretty-name: 'TTTH1L3BHut'
  cross-section: 1.86
  generated-events: 1000000
  group: GTTHut
  order: 4
    """.format(scenario)
    fnew.write(string_for_files)

  file_syst = ''
  with open(config_path + 'config_2017.yml') as f:
    lines = f.readlines()
    for line in lines:
      if 'type' in line:
        if 'const' in line:
          file_syst += line[:line.find(':')] + '_17' + line[line.find(':'):line.find('hist')] + '2017/' + scenario + ver17 + '/post_process/' + line[line.find('hist'):]
        elif 'shape' in line:
          file_syst += line[:line.find('hist')] + '2017/' + scenario + ver17 + '/post_process/' + line[line.find('hist'):]

  with open(config_path + 'config_2018.yml') as f:
    lines = f.readlines()
    for line in lines:
      if 'type' in line:
        if 'const' in line:
          file_syst += line[:line.find(':')] + '_18' + line[line.find(':'):line.find('hist')] + '2018/' + scenario + ver18 + '/post_process/' + line[line.find('hist'):]
        elif 'shape' in line:
          file_syst += line[:line.find('hist')] + '2018/' + scenario + ver18 + '/post_process/' + line[line.find('hist'):]

  with open(config_path + 'template_1718.yml') as f:
    lines = f.readlines()
    with open(config_path + 'config_1718.yml', 'w+') as f1:
      for line in lines: f1.write(line)
      f1.write(common_syst)
      f1.write(file_syst)
      f1.write("\nplots:\n  include: ['histos_" + scenario.lower() + ".yml']\n")

  call(['../../plotIt/plotIt', '-o ' + dest_path + '/' + scenario, config_path + 'config_1718.yml'], shell=False)

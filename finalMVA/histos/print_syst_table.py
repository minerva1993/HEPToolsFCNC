import os, shutil, re, sys
from subprocess import call
from collections import OrderedDict

if len(sys.argv) < 3:
  print("Specify year: 2017/2018, Ver, systematic type: all / source by source (a/s)")
  sys.exit()
era = sys.argv[1]
ver = sys.argv[2]

config_path = '../../plotIt/configs/'
dest_path = './' + era + '/Hct_merged_' + ver + '/'
tmp_file_name = 'temp_' + era + '_' + ver + '.yml'
string_to_add = 'systematics:\n'
plot_to_add = "plots:\n  include: ['histos_unc.yml']\n\n"

if os.path.exists(config_path + tmp_file_name):
  os.remove(config_path + tmp_file_name)
#shutil.copy2(config_path + 'config_' + era + '.yml', config_path + tmp_file_name)

unc_cat = OrderedDict([
('all', ['xsec', 'pu', 'muid', 'muiso', 'mutrg', 'elid', 'elreco', 'elzvtx', 'eltrg', 'jec', 'jer', 'scale', 'ps', 'hdamp', 'pdf', 'TuneCP5', 'lf', 'hf', 'lfstat1', 'lfstat2', 'hfstat1', 'hfstat2', 'cferr1', 'cferr2']),
('pu', ['pu']),
('xsec', ['xsec']),
('lep', ['muid', 'muiso', 'mutrg', 'elid', 'elreco', 'elzvtx', 'eltrg']),
('jec', ['jec']),
('jer', ['jer']),
('scale', ['scale']),
('ps', ['ps']),
('hdamp', ['hdamp']),
('pdf', ['pdf']),
('tune', ['TuneCP5']),
('lf', ['lf']),
('hf', ['hf']),
('lfstat1', ['lfstat1']),
('lfstat2', ['lfstat2']),
('hfstat1', ['hfstat1']),
('hfstat2', ['hfstat2']),
('cferr1', ['cferr1']),
('cferr2', ['cferr2']),
('bAll', ['lf', 'hf', 'lfstat1', 'lfstat2', 'hfstat1', 'hfstat2', 'cferr1', 'cferr2']),
])

if era == "2017":
  unc_cat['prefire'] = ['prefire']
  unc_cat['all'].append('prefire')

for key, value in unc_cat.items():
  print "Print yield tables with source: " + key
  #prep for systematics
  with open(config_path + 'config_' + era + '.yml', 'r') as f:
    with open(config_path + tmp_file_name, 'w') as f1:
      syst_delete = False
      plot_delete = False
      for line in f:
        if 'luminosity-error:' in line:
          line = line[:line.find(':')+2]
          line += "0.0\n"
        if 'root:' in line:
          line = line[:line.find(':')+2]
          line += "'" + dest_path + "merged/'\n"
        if 'systematics' in line: syst_delete = True
        if 'plots:' in line:
          syst_delete = False
          plot_delete = True
        if 'legend' in line: plot_delete = False
        if syst_delete == True: continue
        if plot_delete == True: continue
        f1.write(line)

  with open(config_path + "syst_all.yml") as f:
    lines = f.readlines()
    with open(config_path + tmp_file_name, "a") as f1:
      f1.writelines(plot_to_add)
      for line in lines:
        if any(x in line for x in value): string_to_add += str(line)
      f1.writelines(string_to_add)
  syst_postfix = key
  call(['../../plotIt/plotIt', '-o ' + dest_path, config_path + tmp_file_name, '-y'], shell=False)

  #Remove signals in TT specific sources
  with open(dest_path + 'yields.tex', 'r') as f:
    with open(dest_path + 'yields_' + syst_postfix + '.tex', 'w+') as f1:
      isTT = False
      if any(x in key for x in ['scale', 'ps', 'hdamp', 'tune', 'pdf', 'xsec']): isTT = True
      for line in f:
        if isTT and 'ttbar' in line: isTT = False
        if isTT: continue
        if any(x in line[0] for x in ['0','1','2','3']): line = line[1:]
        f1.write(line)

  string_to_add = 'systematics:\n'

unc_summary = OrderedDict([
('xsec', 'Cross section'), ('pu', 'Pileup'), ('lep', 'Lepton SF'), ('jec', 'JES'), ('jer', 'JER'),
('scale', 'ME scale'), ('ps', 'PS scale'), ('hdamp', 'ME-PS matching'), ('pdf', 'PDF'), ('tune', 'Underlying event'),
('bAll', 'DeepCSV shape'), ('all', 'Total sys. unc.'),
])

if era == "2017": unc_summary['prefire'] = 'Prefire Reweight'

print "Generating summary table..."
#Gather all results into one summary table
with open("total_syst_template.tex") as f:
  lines = f.readlines()
  with open(dest_path + 'total_syst.tex', "w") as f1:
    for line in lines:
      if era != '2017' and 'Prefire' in line: continue
      for key, value in unc_summary.items():
        if value in line:
          with open(dest_path + 'yields_' + key + '.tex','r') as f2:
            lines = f2.read().splitlines()
            last_line = lines[-1]
            last_line = last_line[last_line.find("&")+1:]
            line = line.rstrip('\n') + last_line + '\n'
      f1.write(line)

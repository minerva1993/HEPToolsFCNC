import os, sys

if len(sys.argv) < 2:
  print("Specify year: 2017/2018")
  sys.exit()
era = sys.argv[1]

if era == '2017':
  version = 'V9_6/190702/'
  path_to_prod = '/data/users/minerva1993/ntuple/' + version + 'production/'
  #path_to_prod = '/pnfs/knu.ac.kr/data/cms/store/user/jipark/ntuple_jw/' + version + 'production/'
elif era == '2018':
  version = 'V10_2/190702/'
  path_to_prod = '/data/users/minerva1993/ntuple/' + version + 'production/'
  #path_to_prod = '/pnfs/knu.ac.kr/data/cms/store/user/jipark/ntuple_jw/' + version + 'production/'

merge_file_name = 'arrange_' + era + '_ntuples.sh'
string_for_merge = ''

string_for_merge += '#!/bin/sh\n'
for i in ['DYJets_10to50', 'QCD*', 'TTHad*', 'WW*', 'WZ*', 'ZZ*']:
  string_for_merge += 'for i in ' + path_to_prod + i + '; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm -f $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; chmod 777 $i/Tree_fcncLepJets_000.root; done\n'

with open(merge_file_name, 'w') as f:
  f.write(string_for_merge)

print("{0} written.".format(merge_file_name))

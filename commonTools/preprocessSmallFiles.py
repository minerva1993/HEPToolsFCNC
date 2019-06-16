import os, sys

if len(sys.argv) < 2:
  print("Specify year: 2017/2018")
  sys.exit()
era = sys.argv[1]

if era == '2017':
  #version = 'V9_5/190117/'
  version = 'V9_5/test/'
  path_to_prod = '/data/users/minerva1993/ntuple_Run2017/' + version + 'production/'
  path_to_prod_noreco = '/data/users/minerva1993/ntuple_Run2017/' + version
elif era == '2018':
  version = 'V10_1/190318/'
  path_to_prod = '/data/users/minerva1993/ntuple_Run2018/' + version + 'production/'
  path_to_prod_noreco = '/data/users/minerva1993/ntuple_Run2018/' + version

merge_file_name = 'arrange_' + era + '_ntuples.sh'
string_for_merge = ''

string_for_merge += '#!/bin/sh\n'
string_for_merge += 'for i in ' + path_to_prod + 'DY*; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'
string_for_merge += 'for i in ' + path_to_prod + 'QCD*; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'
string_for_merge += 'for i in ' + path_to_prod + 'TTHad*; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'
string_for_merge += 'for i in ' + path_to_prod + 'W*Jets*; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'
string_for_merge += 'for i in ' + path_to_prod + 'WW*; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'
string_for_merge += 'for i in ' + path_to_prod + 'WZ*; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'
string_for_merge += 'for i in ' + path_to_prod + 'ZZ; do hadd $i/Tree_fcncLepJets.root $i/*.root; rm $i/Tree_fcncLepJets_*.root; mv $i/Tree_fcncLepJets.root  $i/Tree_fcncLepJets_000.root; done\n'

with open(merge_file_name, 'w') as f:
  f.write(string_for_merge)

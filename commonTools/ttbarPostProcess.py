#!/usr/bin/env python
import os, sys
from ROOT import *
from subprocess import call
gROOT.SetBatch(True)

basedir = '/data/users/minerva1993/ntuple/V9_5/190117'
split_path = os.path.join(basedir, 'production')

if not os.path.exists( basedir+'_genLepSel' ):
  os.makedirs( basedir+'_genLepSel' )
if not os.path.exists( os.path.join(basedir+'_genLepSel', 'production') ):
  os.makedirs( os.path.join(basedir+'_genLepSel', 'production') )
if not os.path.exists( os.path.join(basedir+'_genLepSel', 'production', 'TT_powheg_ttbkg') ):
  os.makedirs( os.path.join(basedir+'_genLepSel', 'production', 'TT_powheg_ttbkg') )
os.chmod( basedir+'_genLepSel', 0o777 )

split_list = os.listdir(split_path)

#iii = 0

for path in split_list:
  if path.startswith('TT_powheg'):
    if any(i in path for i in ['hdamp','Tune']): continue
    if not 'ttlf' in path: continue
    syst = ''
    if len(path.split('_')) > 3:
      syst = '_' + path.split('_')[3]
      if not os.path.exists( os.path.join(basedir+'_genLepSel', 'production', 'TT_powheg_ttbkg'+syst) ):
        os.makedirs( os.path.join(basedir+'_genLepSel', 'production', 'TT_powheg_ttbkg'+syst) )

    if not os.path.exists( os.path.join(basedir+'_genLepSel', 'production', path) ):
      os.makedirs( os.path.join(basedir+'_genLepSel', 'production', path) )

    for tmp in os.listdir( os.path.join(split_path, path) ):
#      if iii > 0: continue
      tmp_file = TFile.Open( os.path.join(split_path, path, tmp) )
      tmp_tree = tmp_file.Get('fcncLepJets/tree')

      histoEvt = tmp_file.Get('fcncLepJets/EventInfo')
      histoScale = tmp_file.Get('fcncLepJets/ScaleWeights')
      histoPS = tmp_file.Get('fcncLepJets/PSWeights')
      histoPDF = tmp_file.Get('fcncLepJets/PDFWeights')

      # nGenLep = 1
      target_file = TFile.Open( os.path.join(basedir+'_genLepSel', 'production', path, tmp), 'RECREATE' )
      target_file.cd()
      target_file.mkdir('fcncLepJets')
      target_file.cd('fcncLepJets')
      new_tree = tmp_tree.CopyTree('genconecatid[0] == 1')
      histoEvt.Write()
      histoScale.Write()
      histoPS.Write()
      histoPDF.Write()
      target_file.Write()
      target_file.Close()

      # nGenLep = 0: ttbkg
      target_file2 = TFile.Open( os.path.join(basedir+'_genLepSel', 'production', 'TT_powheg_ttbkg'+syst, path.split('_')[2] + '_' + tmp), 'RECREATE')
      target_file2.cd()
      target_file2.mkdir('fcncLepJets')
      target_file2.cd('fcncLepJets')
      new_tree2 = tmp_tree.CopyTree('genconecatid[0] == 0')
      histoEvt.Write()
      histoScale.Write()
      histoPS.Write()
      histoPDF.Write()
      target_file2.Write()
      target_file2.Close()

#      iii +=1

for root,dirs,_ in os.walk(basedir+'_genLepSel'):
  for d in dirs:
    os.chmod(os.path.join(root,d) , 0o777)
    for dd in os.listdir( os.path.join(root, d) ):
      os.chmod(os.path.join(root, d, dd) , 0o777)

for path in os.listdir( os.path.join(basedir+'_genLepSel', 'production') ):
  if path.startswith('TT_powheg'):
    call(['hadd ' +  basedir+'_genLepSel/'+path+'.root '+basedir+'_genLepSel/production/'+path+'/*.root'], shell=True)
#    print 'hadd', basedir+'_genLepSel/'+path+'.root '+ basedir+'_genLepSel/production/'+path+'/*.root'

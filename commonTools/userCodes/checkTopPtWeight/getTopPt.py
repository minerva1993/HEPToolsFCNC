from __future__ import print_function
from ROOT import *
import ROOT
import sys, os
from collections import OrderedDict
gROOT.SetBatch(True)
hostname = os.environ["HOSTNAME"]

basedir = "/pnfs/knu.ac.kr/data/cms"
ver="V9_7"
#ver="V10_3"
datasets = OrderedDict()
filelists = OrderedDict()

if 'V9' in ver:
  datasets['ttbkg'] = ['/store/group/CAT/V9_7/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_7_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/200723_184504',]
  datasets['tthut'] = ['/store/group/CAT/V9_7/TT_FCNC-TtoHJ_aTleptonic_HTobb_eta_hut_TuneCP5-MadGraph5-pythia8/V9_7_RunIIFall17MiniAODv2-PU2017_12Apr2018_tauDecays_94X_mc2017_realistic_v14-v1/200724_072443','/store/group/CAT/V9_7/TT_FCNC-aTtoHJ_Tleptonic_HTobb_eta_hut_TuneCP5-MadGraph5-pythia8/V9_7_RunIIFall17MiniAODv2-PU2017_12Apr2018_tauDecays_94X_mc2017_realistic_v14-v1/200724_031632',]
  datasets['tthct'] = ['/store/group/CAT/V9_7/TT_FCNC-T2HJ_aTleptonic_HTobb_eta_hct_TuneCP5-MadGraph5-pythia8/V9_7_RunIIFall17MiniAODv2-PU2017_12Apr2018_tauDecays_94X_mc2017_realistic_v14-v1/200724_031848','/store/group/CAT/V9_7/TT_FCNC-aTtoHJ_Tleptonic_HTobb_eta_hct_TuneCP5-MadGraph5-pythia8/V9_7_RunIIFall17MiniAODv2-PU2017_12Apr2018_tauDecays_94X_mc2017_realistic_v14-v1/200724_032059']

else:
  datasets['ttbkg'] = ['/store/group/CAT/V10_3/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/V10_3_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/200725_030313',]
  datasets['tthut'] = ['/store/group/CAT/V10_3/TT_FCNC-TtoHJ_aTleptonic_HTobb_eta_hut_TuneCP5-MadGraph5-pythia8/V10_3_RunIIAutumn18MiniAOD-tauDecays_102X_upgrade2018_realistic_v15-v1/191230_005512','/store/group/CAT/V10_3/TT_FCNC-aTtoHJ_Tleptonic_HTobb_eta_hut_TuneCP5-MadGraph5-pythia8/V10_3_RunIIAutumn18MiniAOD-tauDecays_102X_upgrade2018_realistic_v15-v1/191230_005602']
  datasets['tthct'] = ['/store/group/CAT/V10_3/TT_FCNC-T2HJ_aTleptonic_HTobb_eta_hct_TuneCP5-MadGraph5-pythia8/V10_3_RunIIAutumn18MiniAOD-tauDecays_102X_upgrade2018_realistic_v15-v1/191230_005652','/store/group/CAT/V10_3/TT_FCNC-aTtoHJ_Tleptonic_HTobb_eta_hct_TuneCP5-MadGraph5-pythia8/V10_3_RunIIAutumn18MiniAOD-tauDecays_102X_upgrade2018_realistic_v15-v1/191230_005745']

for name, dataset in datasets.items():
  for path in dataset:
    filelists[name] = []
    if os.path.isdir(basedir+path+'/0000/'):
      for tmps in os.listdir(basedir+path+'/0000/'):
        if 'fail' in tmps: continue
        if int(tmps.rstrip('.root').split('_')[-1]) < 100:
          filelists[name].append(basedir+path+'/0000/'+tmps)

  #if os.path.isdir(basedir+path+'/0001/'):
  #  for tmps in os.listdir(basedir+path+'/0001/'):
  #    if 'fail' in tmps: continue
  #    filelist_ttbkg.append(basedir+path+'/0001/'+tmps)

for name, files in filelists.items():
  entries = 0
  h = TH1F('h', 'h', 40, 0, 2000)
  for file in files:
    tmp_file = './' + ver + '/' + file.split("/")[-5]
    f = TFile.Open(file,'READ')
    print("processing "+file.split("/")[-5] + " " + file.split("/")[-1], end="\r")
    tree = f.Get("Events")
    htmp1 = TH1F('htmp1', 'htmp1', 40, 0, 2000)
    htmp2 = TH1F('htmp2', 'htmp2', 40, 0, 2000)
    tree.Draw("catGenTops_catGenTops__CAT.obj.topquark1().Pt()>>htmp1")
    tree.Draw("catGenTops_catGenTops__CAT.obj.topquark2().Pt()>>htmp2")
    h.Add(htmp1, 1.0)
    h.Add(htmp2, 1.0)
    entries += tree.GetEntries()
    
  tmp_string = '['
  for i in xrange(1,h.GetNbinsX()+1):
    tmp_string += str(h.GetBinContent(i)) + ', '
  tmp_string += ']\n'
  tmp_string += str(entries)
  with open(ver+'_'+name+'.txt', 'w') as f: f.write(tmp_string)

from __future__ import print_function
from ROOT import *
import sys, os
gROOT.SetBatch(True)
hostname = os.environ["HOSTNAME"]

basedir = "/pnfs/knu.ac.kr/data/cms/store/group/CAT/V9_4"
ver="V9_4"
baselen = len(basedir)
dataset = [
#'/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181207_055659',
#'/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14_ext1-v1/181206_111046',
#'/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181207_055748',
#'/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14_ext1-v2/181210_062206',
#'/W1JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/181206_111256',
#'/W2JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v4/181207_055520',
#'/W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_111443',
#'/W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/181213_053509',
#'/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/181207_055248',
#'/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181206_104220',
#'/TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181207_055935',
#'/TTTo2L2Nu_hdampUP_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181207_055849',
#'/TTTo2L2Nu_hdampDOWN_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_062709',
#'/TTToSemiLeptonic_hdampUP_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/181206_062759',
#'/TTToSemiLeptonic_hdampDOWN_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_062849',
#'/TTToHadronic_hdampUP_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181206_062938',
#'/TTToHadronic_hdampDOWN_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181212_052120',
#'/TTTo2L2Nu_TuneCP5up_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_063114',
#'/TTTo2L2Nu_TuneCP5down_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181207_055343',
#'/TTToSemiLeptonic_TuneCP5up_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/181206_063301',
#'/TTToSemiLeptonic_TuneCP5down_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_063350',
#'/TTToHadronic_TuneCP5up_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_063437',
#'/TTToHadronic_TuneCP5down_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v2/181206_063525',
#'/ST_s-channel_4f_leptonDecays_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181212_052205',
#'/ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_055351',
#'/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181206_055442',
#'/ST_tW_top_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_055529',
#'/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181212_052249',
#'/WW_TuneCP5_13TeV-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_055704',
#'/WZ_TuneCP5_13TeV-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_055756',
#'/ZZ_TuneCP5_13TeV-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_055847',
#'/TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_055936',
#'/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181206_060027',
#'/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/181206_060120',
'/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181213_053646',
#'/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_060302',
#'/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_060349',
#'/ST_FCNC-TH_Tleptonic_HTobb_eta_hut-MadGraph5-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_051918',
#'/ST_FCNC-TH_Tleptonic_HTobb_eta_hct-MadGraph5-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/181206_052011',
#'/TT_FCNC-TtoHJ_aTleptonic_HTobb_eta_hut-MadGraph5-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/181206_052059',
#'/TT_FCNC-aTtoHJ_Tleptonic_HTobb_eta_hut-MadGraph5-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181206_052146',
#'/TT_FCNC-T2HJ_aTleptonic_HTobb_eta_hct-MadGraph5-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/181206_052233',
#'/TT_FCNC-aTtoHJ_Tleptonic_HTobb_eta_hct-MadGraph5-pythia8/V9_4_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v3/181206_052323',
          ]

c1 = TCanvas( 'c1', 'c1', 500, 500 )

for path in dataset:
  filelist = []
  if os.path.isdir(basedir+path+'/0000/'):
    for tmps in os.listdir(basedir+path+'/0000/'):
      if 'fail' in tmps: continue
      filelist.append(basedir+path+'/0000/'+tmps)
  if os.path.isdir(basedir+path+'/0001/'):
    for tmps in os.listdir(basedir+path+'/0001/'):
      if 'fail' in tmps: continue
      filelist.append(basedir+path+'/0001/'+tmps)


  nTruePV = TH1F('nTruePV','nTruePV',100,0,100)
  nTruePV.SetDirectory(0)

  tmp_file = ''
  for files in filelist:
    if 'sdfarm' in hostname: tmp_file = './'+ver+'/'+ files.split("/")[6]
    elif 'knu' in hostname: tmp_file = './'+ver+'/'+ files.split("/")[9]
    cat = TFile.Open(files,'READ')
    print("processing "+files.split("/")[9] + " " + files.split("/")[-1], end="\r")
    tmp = TH1F('tmp','tmp',100,0,100)

    tree = cat.Get('Events')
    tree.Draw("nTrueInteraction>>tmp")
    nTruePV.Add(tmp,1.0)

  nTruePV.Draw("hist")
  tmp_string = 'cms.vdouble(\n'

  for i in range(1,101):
    tmp_string += str(nTruePV.GetBinContent(i)) + ', '
    if i%5 == 0: tmp_string += '\n'

  tmp_string += ')'

  if os.path.isfile(tmp_file +".pdf"):
    c1.Print(tmp_file+"_v2.pdf")
    with open(tmp_file+'_v2.txt', 'w') as f: f.write(tmp_string)
  else:
    c1.Print(tmp_file+".pdf")
    with open(tmp_file+'.txt', 'w') as f: f.write(tmp_string)

#!/usr/bin/python
from ROOT import *
import os, sys

inputdir = '/data/users/minerva1993/ntuple_Run2017/V9_2/180614'

test = os.listdir("/home/minerva1993/fcnc/analysis_2017/fullAna/temp")
for item in test:
  if item.endswith(".root"):
    os.remove(os.path.join("/home/minerva1993/fcnc/analysis_2017/fullAna/temp", item))

def runAna(dir, file, name):
  print 'processing '+dir+'/'+file
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(dir+'/'+file)
  chain.SetProof();
  chain.Process("MyAnalysis.C+",name)
  #print chain.GetCurrentFile().GetName()

  f = TFile.Open(dir+'/'+file, "READ")

  ## save Event Summary histogram ##
  out = TFile("temp/hist_"+name+".root","update")
  hevt = f.Get("fcncLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()

p = TProof.Open("", "workers=8")

runAna(inputdir,"SingleMuon_Run2017.root","DataSingleMu")
runAna(inputdir,"SingleElectron_Run2017.root","DataSingleEG")
runAna(inputdir,"TT_powheg_ttbb.root","ttbb")
runAna(inputdir,"TT_powheg_ttbj.root","ttbj")
runAna(inputdir,"TT_powheg_ttcc.root","ttcc")
runAna(inputdir,"TT_powheg_ttlf.root","ttLF")
runAna(inputdir,"TT_powheg_ttother.root","ttother")
runAna(inputdir,"TTLL_powheg_ttbb.root","ttLLbb")
runAna(inputdir,"TTLL_powheg_ttbj.root","ttLLbj")
runAna(inputdir,"TTLL_powheg_ttcc.root","ttLLcc")
runAna(inputdir,"TTLL_powheg_ttlf.root","ttLLLF")
runAna(inputdir,"TTLL_powheg_ttother.root","ttLLother")
runAna(inputdir,"TTHad_powheg_ttbb.root","ttHadbb")
runAna(inputdir,"TTHad_powheg_ttbj.root","ttHadbj")
runAna(inputdir,"TTHad_powheg_ttcc.root","ttHadcc")
runAna(inputdir,"TTHad_powheg_ttlf.root","ttHadLF")
runAna(inputdir,"TTHad_powheg_ttother.root","ttHadother")
runAna(inputdir,"SingleTop_s.root","schannel")
runAna(inputdir,"SingleTop_t.root","tchannel")
runAna(inputdir,"SingleTbar_t.root","tbarchannel")
runAna(inputdir,"SingleTop_tW.root","tWchannel")
runAna(inputdir,"SingleTbar_tW.root","tbarWchannel")
runAna(inputdir,"ttHTobb.root","ttHTobb")
runAna(inputdir,"ttHToNonbb.root","ttHToNonbb")
runAna(inputdir,"TTWJetsToLNu_PSweight.root","ttwjetsToLNu")
runAna(inputdir,"TTWJetsToQQ.root","ttwjetsToQQ")
runAna(inputdir,"TTZToLLNuNu.root","ttzToLLNuNu")
runAna(inputdir,"TTZToQQ.root","ttzToQQ")
runAna(inputdir,"DYJets_v2.root","zjets")
runAna(inputdir,"DYJets_4to50_HT70to100.root","zjets4to50HT70to100")
runAna(inputdir,"DYJets_4to50_HT100to200_v2.root","zjets4to50HT100to200")
runAna(inputdir,"DYJets_4to50_HT200to400.root","zjets4to50HT200to400")
runAna(inputdir,"DYJets_4to50_HT400to600.root","zjets4to50HT400to600")
runAna(inputdir,"DYJets_4to50_HT600toinf.root","zjets4to50HT600toinf")
runAna(inputdir,"W1JetsToLNu_50-150.root","w1jets50to150")
runAna(inputdir,"W1JetsToLNu_150-250.root","w1jets150to250")
runAna(inputdir,"W1JetsToLNu_250-400.root","w1jets250to400")
runAna(inputdir,"W1JetsToLNu_400-inf.root","w1jets400toinf")
runAna(inputdir,"W2JetsToLNu_50-150.root","w2jets50to150")
#runAna(inputdir,"W2JetsToLNu_150-250.root","w2jets150to250")
runAna(inputdir,"W2JetsToLNu_250-400.root","w2jets250to400")
runAna(inputdir,"W2JetsToLNu_400-inf.root","w2jets400toinf")
runAna(inputdir,"W3JetsToLNu.root","w3jets")
runAna(inputdir,"W4JetsToLNu.root","w4jets")
runAna(inputdir,"WW.root","ww")
runAna(inputdir,"WZ.root","wz")
runAna(inputdir,"ZZ.root","zz")
#runAna(inputdir,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
#runAna(inputdir,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
#runAna(inputdir,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
#runAna(inputdir,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")
runAna(inputdir,"ST_TH_1L3B_Hct.root","singleTopHct")
runAna(inputdir,"ST_TH_1L3B_Hut.root","singleTopHut")

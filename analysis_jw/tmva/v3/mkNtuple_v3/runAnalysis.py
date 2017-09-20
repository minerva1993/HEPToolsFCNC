#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas
from ROOT import *
import os
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  chain.SetProof();
  chain.Process("tmvaAnalysis.C+",name)

  #f = TFile(dir+"/"+file,"read")

  ## save Event Summary histogram ##
  #out = TFile("hist_"+name+".root","update")
  #hevt = f.Get("ttbbLepJets/EventInfo")
  #hevt.Write()
  #out.Write()
  #out.Close()


p = TProof.Open("", "workers=8")

version = "v3"
inputdir = "/home/minerva1993/fcnc/ntuple_jw/"

#BigFiles
"""
#runAna(inputdir+version,"SingleMuon_Run2016.root","DataSingleMu")
for mi in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016B"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016B", mi,"SingleMuon_Run2016B_splitted_"+ str(mi).rstrip(".root"))

for mj in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016C"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016C", mj,"SingleMuon_Run2016C_splitted_"+ str(mj).rstrip(".root"))

for mk in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016D"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016D", mk,"SingleMuon_Run2016D_splitted_"+ str(mk).rstrip(".root"))

for ml in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016E"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016E", ml,"SingleMuon_Run2016E_splitted_"+ str(ml).rstrip(".root"))

for mm in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016F"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016F", mm,"SingleMuon_Run2016F_splitted_"+ str(mm).rstrip(".root"))

for mn in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016G"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016G", mn,"SingleMuon_Run2016G_splitted_"+ str(mn).rstrip(".root"))

for mo in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016H_v2"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016H_v2", mo,"SingleMuon_Run2016H_v2_splitted_"+ str(mo).rstrip(".root"))

for mp in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016H_v3"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleMuon_Run2016H_v3", mp,"SingleMuon_Run2016H_v3_splitted_"+ str(mp).rstrip(".root"))
"""
#runAna(inputdir+version,"SingleElectron_Run2016.root","DataSingleEG")
for ei in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016B"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016B", ei,"SingleElectron_Run2016B_splitted_"+ str(ei).rstrip(".root"))

for ej in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016C"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016C", ej,"SingleElectron_Run2016C_splitted_"+ str(ej).rstrip(".root"))

for ek in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016D"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016D", ek,"SingleElectron_Run2016D_splitted_"+ str(ek).rstrip(".root"))

for el in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016E"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016E", el,"SingleElectron_Run2016E_splitted_"+ str(el).rstrip(".root"))

for em in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016F"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016F", em,"SingleElectron_Run2016F_splitted_"+ str(em).rstrip(".root"))

for en in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016G"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016G", en,"SingleElectron_Run2016G_splitted_"+ str(en).rstrip(".root"))

for eo in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016H_v2"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016H_v2", eo,"SingleElectron_Run2016H_v2_splitted_"+ str(eo).rstrip(".root"))

for ep in os.listdir("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016H_v3"):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleElectron_Run2016H_v3", ep,"SingleElectron_Run2016H_v3_splitted_"+ str(ep).rstrip(".root"))
"""
#runAna(inputdir+version,"TT_powheg_ttbb.root","ttbb")
for o in range(0, 238):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/TT_powheg_ttbb","Tree_ttbbLepJets_"+ str(o) + ".root","ttbb_split_"+ str(o))

#runAna(inputdir+version,"TT_powheg_ttlf.root","ttLF")
for j in range(0, 238):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/TT_powheg_ttlf","Tree_ttbbLepJets_"+ str(j) + ".root","ttLF_split_"+ str(j))

#runAna(inputdir+version,"TT_powheg_ttbj.root","ttbj")
for p in range(0, 238):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/TT_powheg_ttbj","Tree_ttbbLepJets_"+ str(p) + ".root","ttbj_split_"+ str(p))

#runAna(inputdir+version,"TT_powheg_ttcc.root","ttcc")
for q in range(0, 238):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/TT_powheg_ttcc","Tree_ttbbLepJets_"+ str(q) + ".root","ttcc_split_"+ str(q))

#runAna(inputdir+version,"TT_powheg_ttbkg.root","ttBkg")
for r in range(0, 20):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/TT_powheg_ttBkg","Tree_ttbbLepJets_"+ str(r) + ".root","ttBkg_split_"+ str(r))

#runAna(inputdir+version,"TT_powheg_ttother.root","tt")
for k in range(0, 238):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/TT_powheg_ttother","Tree_ttbbLepJets_"+ str(k) + ".root","tt_split_"+ str(k))

#runAna(inputdir+version,"DYJets.root","zjets")
for i in range(0, 145):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/DYJets","Tree_ttbbLepJets_"+ str(i) + ".root","zjets_split_"+ str(i))

#runAna(inputdir+version,"WJets.root","wjets")
for l in range(0, 30):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/WJets","Tree_ttbbLepJets_"+ str(l) + ".root","wjets_split_"+ str(l))

#runAna(inputdir+version,"SingleTop_t.root","tchannel")
for m in range(0, 72):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleTop_t","Tree_ttbbLepJets_"+ str(m) + ".root","tchannel_split_"+ str(m))

#runAna(inputdir+version,"SingleTbar_t.root","tbarchannel")
for n in range(0, 41):
  runAna("/data/users/minerva1993/ntuple_Run2016/v3/v3_production/SingleTbar_t","Tree_ttbbLepJets_"+ str(n) + ".root","tbarchannel_split_"+ str(n))

runAna(inputdir+version,"DYJets_10to50.root","zjets10to50")
runAna(inputdir+version,"WW.root","ww")
runAna(inputdir+version,"WZ.root","wz")
runAna(inputdir+version,"ZZ.root","zz")
runAna(inputdir+version,"SingleTop_tW.root","tWchannel")
runAna(inputdir+version,"SingleTbar_tW.root","tbarWchannel")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")
runAna(inputdir+version,"QCD_EGEnr_20to30.root","QCD_EGEnr_20to30")
runAna(inputdir+version,"QCD_EGEnr_30to50.root","QCD_EGEnr_30to50")
runAna(inputdir+version,"QCD_EGEnr_50to80.root","QCD_EGEnr_50to80")
runAna(inputdir+version,"QCD_EGEnr_80to120.root","QCD_EGEnr_80to120")
runAna(inputdir+version,"QCD_EGEnr_120to170.root","QCD_EGEnr_120to170")
runAna(inputdir+version,"QCD_EGEnr_170to300.root","QCD_EGEnr_170to300")
runAna(inputdir+version,"QCD_EGEnr_300toInf.root","QCD_EGEnr_300toInf")
runAna(inputdir+version,"QCD_MuEnr_20to30.root","QCD_MuEnr_20to30")
runAna(inputdir+version,"QCD_MuEnr_30to50.root","QCD_MuEnr_30to50")
runAna(inputdir+version,"QCD_MuEnr_50to80.root","QCD_MuEnr_50to80")
runAna(inputdir+version,"QCD_MuEnr_80to120.root","QCD_MuEnr_80to120")
runAna(inputdir+version,"QCD_MuEnr_120to170.root","QCD_MuEnr_120to170")
runAna(inputdir+version,"QCD_MuEnr_170to300.root","QCD_MuEnr_170to300")
runAna(inputdir+version,"QCD_MuEnr_300to470.root","QCD_MuEnr_300to470")
runAna(inputdir+version,"QCD_MuEnr_470to600.root","QCD_MuEnr_470to600")
runAna(inputdir+version,"QCD_MuEnr_600to800.root","QCD_MuEnr_600to800")
runAna(inputdir+version,"QCD_MuEnr_800to1000.root","QCD_MuEnr_800to1000")
runAna(inputdir+version,"QCD_MuEnr_1000toInf.root","QCD_MuEnr_1000toInf")
"""

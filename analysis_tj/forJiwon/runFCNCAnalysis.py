#!/usr/bin/python

from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas

Proof = True

def runAna(dir, file, name):
  chain = TChain("TopTree/events","events")
  chain.Add(dir+"/"+file)
  if Proof :
    chain.SetProof();
  chain.Process("FCNCAnalysis.C+",name)

  f = TFile(dir+"/"+file,"read")

  ## save Event Summary histogram ##
  out = TFile("hist_"+name+".root","update")
  hevt = f.Get("TopTree/EventSummary")
  hevt.Write()
  out.Write()
  out.Close()


if Proof:
  p = TProof.Open("", "workers=8")

version = "V1"
inputdir = "/xrootd/store/user/tjkim/ntuples/hep2017/"


runAna(inputdir+version+"/","vallot_SingleMuon_Run2016B.root","SingleMuon_Run2016B")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016C.root","SingleMuon_Run2016C")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016D.root","SingleMuon_Run2016D")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016E.root","SingleMuon_Run2016E")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016F.root","SingleMuon_Run2016F")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016G.root","SingleMuon_Run2016G")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016H_v2.root","SingleMuon_Run2016H_v2")
runAna(inputdir+version+"/","vallot_SingleMuon_Run2016H_v3.root","SingleMuon_Run2016H_v3")
runAna(inputdir+version+"/","vallot_WJets.root","WJets")
runAna(inputdir+version+"/","vallot_DYJets.root","DYjets")
runAna(inputdir+version+"/","vallot_DYJets_10to50.root","DYJets_10to50")
runAna(inputdir+version+"/","vallot_WW.root","WW")
runAna(inputdir+version+"/","vallot_WZ.root","WZ")
runAna(inputdir+version+"/","vallot_ZZ.root","ZZ")
runAna(inputdir+version+"/","vallot_SingleTbar_t.root","SingleTbar_t")
runAna(inputdir+version+"/","vallot_SingleTop_t.root","SingleTop_t")
runAna(inputdir+version+"/","vallot_SingleTop_tW.root","SingleTop_tW")
runAna(inputdir+version+"/","vallot_SingleTbar_tW.root","SingleTbar_tW")
runAna(inputdir+version+"/","vallot_TT_powheg.root","ttbb")
runAna(inputdir+version+"/","vallot_TT_powheg.root","ttbj")
runAna(inputdir+version+"/","vallot_TT_powheg.root","ttcc")
runAna(inputdir+version+"/","vallot_TT_powheg.root","ttLF")
runAna(inputdir+version+"/","vallot_TT_powheg.root","ttothers")
runAna(inputdir+version+"/","vallot_TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct")
runAna(inputdir+version+"/","vallot_TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut")
runAna(inputdir+version+"/","vallot_TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","TT_TopLeptonicDecay_TH_1L3B_Eta_Hct")
runAna(inputdir+version+"/","vallot_TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","TT_TopLeptonicDecay_TH_1L3B_Eta_Hut")



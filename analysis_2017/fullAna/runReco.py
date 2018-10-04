#!/usr/bin/python
from ROOT import *
import os, sys
gROOT.SetBatch(True)

reco_scheme = sys.argv[1]
file_path = sys.argv[2]
name = sys.argv[3]
syst = ["","jecup","jecdown","jerup","jerdown"]

test = os.listdir("./doReco/temp")
dupl = False
for item in test:
  if item.endswith(name + ".root"):
    dupl = True
if dupl == True: print 'Previous verion of histogram root file exists!! Please remove them first.'

def runAna(file_path, name):
  #Needs some options to bypass jec jer hdamp tune for real data
  print 'processing ' + file_path

  for syst_ext in syst:
    if ("Run2017" in name) and syst_ext != "": continue
    else:
      chain = TChain("fcncLepJets/tree","events")
      chain.Add(file_path)
      chain.Process("MyAnalysis.C+", name + syst_ext)
      #print chain.GetCurrentFile().GetName()

      ## save Event Summary histogram ##
      if syst_ext == "": postfix = ""
      else:              postfix = "__"

      f = TFile.Open(file_path, "READ")
      out = TFile("doReco/temp/hist_" + name.replace("-" + reco_scheme,"") + postfix + syst_ext + ".root","update")
      hevt = f.Get("fcncLepJets/EventInfo")
      hevt.Write()
      out.Write()
      out.Close()

if not dupl: runAna(file_path, name + '-' + reco_scheme)

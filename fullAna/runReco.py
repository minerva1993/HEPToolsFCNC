#!/usr/bin/python
from ROOT import *
import os, sys
gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")

#Channel and version
if len(sys.argv) < 4:
  print("Not enough arguements: Ver, Input, Output")
  sys.exit()
reco_scheme = sys.argv[1]
era = sys.argv[2]
file_path = sys.argv[3]
name = sys.argv[4]
syst = ["","jecup","jecdown","jerup","jerdown",]
syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist

if not os.path.exists("./doReco/temp"):
  os.makedirs("./doReco/temp")
test = os.listdir("./doReco/temp")
dupl = False
for item in test:
  if item.endswith(name + ".root"):
    dupl = True
if dupl == True: print 'Previous verion of histogram root file exists!! Please remove them first.'

def runAna(file_path, name):
  print 'processing ' + file_path.split('/')[-2] + '/' + file_path.split('/')[-1]

  for syst_ext in syst + syst2:
    if ("Run201" in name) and syst_ext != "": continue
    elif (syst_ext in syst2) and not (syst_ext in name): continue
    elif (syst_ext in syst) and any(tmp in name for tmp in syst2): continue
    else:
      if (syst_ext in syst2): name = name.replace(syst_ext,"")

    chain = TChain("fcncLepJets/tree","events")
    chain.Add(file_path)
    chain.Process("MyAnalysis.C+", name + syst_ext)
    #print chain.GetCurrentFile().GetName()

    ## save Event Summary histogram ##
    if syst_ext == "": postfix = ""
    else:              postfix = "__"

    f = TFile.Open(file_path, "READ")
    out = TFile("doReco/temp/hist_" + name.replace("-" + era + reco_scheme,"") + postfix + syst_ext + ".root","update")
    hevt = f.Get("fcncLepJets/EventInfo")
    hevt.Write()
    hscale = f.Get("fcncLepJets/ScaleWeights")
    hscale.Write()
    hps = f.Get("fcncLepJets/PSWeights")
    hps.Write()
    #hpdf = f.Get("fcncLepJets/PDFWeights")
    #hpdf.Write()
    out.Write()
    out.Close()

if not dupl: runAna(file_path, name + '-' + era + reco_scheme)

#!/usr/bin/python
from ROOT import *
import os, sys
gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")

#Channel and version
if len(sys.argv) < 4:
  print("Not enough arguements: Ver, Input, Output")
  sys.exit()
train_scheme = sys.argv[1] #supposed to be Hct_j3b2_01 or similar, change to Hct_0101010101 (ver1~5)
era = sys.argv[2]
file_path = sys.argv[3]
name = sys.argv[4]
syst = ["","jecup","jecdown","jerup","jerdown",]
syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist

#if not os.path.exists("./histos/" + train_scheme):
#  os.makedirs("./histos/" + train_scheme)
#test = os.listdir("./histos/" + train_scheme)

if not os.path.exists("./histos/temp"):
  os.makedirs("./histos/temp")
test = os.listdir("./histos/temp")
dupl = False
#for item in test:
#  if item.endswith(name + ".root"):
#    dupl = True
#if dupl == True: print 'Previous verion of histogram root file exists!! Please remove them first.'

def runAna(file_path, name):
  #import time
  #start_time = time.time()

  print 'processing ' + file_path.split('/')[-2] + '/' + file_path.split('/')[-1]

  for syst_ext in syst + syst2:
    if ("Run201" in name) and syst_ext != "": continue #process only nominal for data
    elif (syst_ext in syst2) and not (syst_ext in name): continue #skip non-dedicated samples
    elif (syst_ext in syst) and any(tmp in name for tmp in syst2): continue #skip jer/c for syst2
    else:
      if (syst_ext in syst2): name = name.replace(syst_ext,"") #syst2 already in 'name', remove in (handle in postfix)

    chain = TChain("fcncLepJets/tree","events")
    chain.Add(file_path)
    chain.Process("MyAnalysis.C+", name + "_" + syst_ext) #one '_'!
    #print chain.GetCurrentFile().GetName()

    ## save Event Summary histogram ##
    if syst_ext == "": postfix = ""
    else:              postfix = "__"

    f = TFile.Open(file_path, "READ")
    #out = TFile("histos/" + train_scheme + "/hist_" + name.replace("-" + train_scheme,"") + postfix + syst_ext + ".root","update")
    out = TFile("histos/temp/hist_" + name.replace("-" + era+train_scheme,"") + postfix + syst_ext + ".root","update")
    hevt = f.Get("fcncLepJets/EventInfo")
    hevt.Write()
    hscale = f.Get("fcncLepJets/ScaleWeights")
    hscale.Write()
    hps = f.Get("fcncLepJets/PSWeights")
    hps.Write()
    hpdf = f.Get("fcncLepJets/PDFWeights")
    hpdf.Write()
    out.Write()
    out.Close()

    #print("--- %s seconds ---" % (time.time() - start_time))

if not dupl: runAna(file_path, name + '-' + era + train_scheme)

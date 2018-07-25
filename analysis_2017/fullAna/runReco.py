#!/usr/bin/python
from ROOT import *
import os, sys
gROOT.SetBatch(True)

file_path = sys.argv[1]
name = sys.argv[2]

test = os.listdir("./doReco")
dupl = False
for item in test:
  if item.endswith(name + ".root"):
    dupl = True
if dupl == True: print 'Previous verion of histogram root file exists!! Please remove them first.'

def runAna(file_path, name):
  print 'processing ' + file_path
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(file_path)
  chain.Process("MyAnalysis.C+",name)
  #print chain.GetCurrentFile().GetName()

  f = TFile.Open(file_path, "READ")

  ## save Event Summary histogram ##
  out = TFile("doReco/hist_"+name+".root","update")
  hevt = f.Get("fcncLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()

if not dupl: runAna(file_path, name)

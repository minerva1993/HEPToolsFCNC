#!/usr/bin/python
from ROOT import TFile, TChain, gSystem
import os, sys

test = os.listdir("/home/minerva1993/fcnc/analysis_2017/fullAna/temp")
for item in test:
  if item.endswith(".root"):
    os.remove(os.path.join("/home/minerva1993/fcnc/analysis_2017/fullAna/temp", item))

inputdir = sys.argv[1]
dataset = sys.argv[2]
tuples = sys.argv[3]
name = sys.argv[4]

def runAna(dir, file, name):
  print 'processing '+dir+'/'+file
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(dir+'/'+file)
  chain.Process("MyAnalysis.C+",name)
  #print chain.GetCurrentFile().GetName()

  f = TFile.Open(dir+'/'+file, "READ")

  ## save Event Summary histogram ##
  out = TFile("temp/hist_"+name+".root","update")
  hevt = f.Get("fcncLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()

runAna(inputdir+'/'+dataset, tuples, name)

#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
from subprocess import call
from os.path import isfile
import numpy as np

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

ch = 'Hct32'
#ch = 'Hut15'
#ch = 'Hct30'

tuples = sys.argv[1]

# Load data
data = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v5/mkNtuple_v5/'+tuples)
data_tree = data.Get('tmva_tree')

target = TFile('output_'+ch+'_'+tuples,'RECREATE')
tree = TTree("tree","tree")

branches = {}
for branch in data_tree.GetListOfBranches():
  branchName = branch.GetName()
  if branchName not in ["lepDPhi", "bjetmDR", "bjetmDEta", "bjetmDPhi", "dibjetsMass", "bjetPt_dibjetsm", "cjetPt", "transverseMass", "jet1phi", "jet2phi", "jet3phi", "jet4phi", "KinLepWMass", "KinLepTopMass", "KinHadWMass", "KinHadTopMass", "FCNHKinLepWMass", "FCNHKinLepTopMass", "FCNHKinHMass", "FCNHKinHucTopMass", "M3LepWMass", "M3HadWMass", "M3HMass", "M3DR", "M3LepTopMass", "M3HucTopMass", "DRlepWphi", "DRjet0phi", "DRjet1phi", "DRjet2phi", "DRjet3phi", "DRjet12phi", "DRjet23phi", "DRjet31phi", "DRlepTphi", "DRhadTphi", "EventWeight", "totnevt", "nevt", "GoodPV",
  #"nbjets_m", 
  #"jet1cvsl", "jet2cvsl", "jet3cvsl", "jet4cvsl", 
  #"jet1cvsb","jet2cvsb", "jet3cvsb", "jet4cvsb",
  #"DRjet0cvsl", "DRjet1cvsl","DRjet2cvsl","DRjet3cvsl",
  #"DRjet0cvsb", "DRjet1cvsb", "DRjet2cvsb", "DRjet3cvsb", 
  #"ncjets_m"
  ]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

  elif branchName in ["GoodPV"]:
      branches["GoodPV"] = array('f', [-999])
      reader.AddSpectator("GoodPV", branches["GoodPV"])
      #data_tree.SetBranchAddress("GoodPV", branches["GoodPV"])

reader.BookMVA('PyKeras', TString('/home/minerva1993/fcnc/analysis_jw/tmva/v5/keras5_'+ch+'/weights/TMVAClassification_Keras_TF.weights.xml'))
reader.BookMVA('BDT', TString('/home/minerva1993/fcnc/analysis_jw/tmva/v5/keras5_'+ch+'/weights/TMVAClassification_BDT.weights.xml'))

#reader.BookMVA('BDT', TString('/home/minerva1993/fcnc/analysis_jw/tmva/v5/bdt_test_Hct11/weights/TMVAClassification_BDT.weights.xml'))

print "processing "+tuples
totalevt = data_tree.GetEntries()
print "this sample contains "+str(totalevt)+" events"

score1 = np.zeros(1, dtype=float)
score2 = np.zeros(1, dtype=float)
event_weight = np.zeros(1, dtype=float)

tree.Branch('KerasScore', score1, 'KerasScore/D')
tree.Branch('BDTScore', score2, 'BDTScore/D')
tree.Branch('Event_Weight', event_weight, 'Event_Weight/D')

for i in xrange(totalevt):
  data_tree.GetEntry(i)
  nevt = data_tree.GetLeaf("nevt").GetValue(0)

  if tuples in ["tmva_SingleLepton_Run2016.root", "tmva_wjets.root", "tmva_zjets10to50.root", "tmva_zjets.root"]:
    score1[0] = reader.EvaluateMVA('PyKeras')
    score2[0] = reader.EvaluateMVA('BDT')
    event_weight[0] = data_tree.EventWeight

  else:
    if float(nevt)/totalevt > 0.8:
      score1[0] = reader.EvaluateMVA('PyKeras')
      score2[0] = reader.EvaluateMVA('BDT')
      event_weight[0] = data_tree.EventWeight
      #print data_tree.EventWeight
    else:
      score1[0] = 2
      score2[0] = 2
      event_weight[0] = 10

  s = str(i)

  if totalevt > 100000:
    if i%20000 == 0: print "processing "+tuples+" "+s
  elif totalevt > 10000 and nevt < 100000:
    if i%10000 == 0: print "processing "+tuples+" "+s
  elif totalevt < 10000:
    if i%1000 == 0: print "processing "+tuples+" "+s

  tree.Fill()

  if i == totalevt:
    del score1, score2, branches

target.Write()
target.Close()

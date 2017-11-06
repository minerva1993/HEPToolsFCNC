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

#ch = 'Hct15'
#ch = 'Hut15'
ch = 'Hct21'

tuples = sys.argv[1]

# Load data
data = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v5/mkNtuple_v5/'+tuples)
data_tree = data.Get('tmva_tree')

target = TFile('output_'+ch+'_'+tuples,'RECREATE')
tree = TTree("tree","tree")

branches = {}
for branch in data_tree.GetListOfBranches():
  branchName = branch.GetName()
  if branchName not in ["lepDPhi", "bjetmDR", "bjetmDEta", "bjetmDPhi", "dibjetsMass", "bjetPt_dibjetsm", "cjetPt", "transverseMass", "jet1phi", "jet2phi", "jet3phi", "jet4phi", "KinLepWMass", "KinLepTopMass", "KinHadWMass", "KinHadTopMass", "FCNHKinLepWMass", "FCNHKinLepTopMass", "FCNHKinHMass", "FCNHKinHucTopMass", "M3LepWMass", "M3HadWMass", "M3HMass", "M3DR", "M3LepTopMass", "M3HucTopMass", "DRlepWphi", "DRjet0phi", "DRjet1phi", "DRjet2phi", "DRjet3phi", "DRjet12phi", "DRjet23phi", "DRjet31phi", "DRlepTphi", "DRhadTphi", "EventWeight", "totnevt", "nevt", "GoodPV"]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

reader.BookMVA('PyKeras', TString('/home/minerva1993/fcnc/analysis_jw/tmva/v5/keras5_'+ch+'/weights/TMVAClassification_Keras_TF.weights.xml'))
reader.BookMVA('BDT', TString('/home/minerva1993/fcnc/analysis_jw/tmva/v5/keras5_'+ch+'/weights/TMVAClassification_BDT_optimized.weights.xml'))

print "processing "+tuples
nevt = data_tree.GetEntries()
print "this sample contains "+str(nevt)+" events"

score1 = np.zeros(1, dtype=float)
score2 = np.zeros(1, dtype=float)
event_weight = np.zeros(1, dtype=float)

tree.Branch('KerasScore', score1, 'KerasScore/D')
tree.Branch('BDTScore', score2, 'BDTScore/D')
tree.Branch('Event_Weight', event_weight, 'Event_Weight/D')

for i in xrange(nevt):
  data_tree.GetEntry(i)
  score1[0] = reader.EvaluateMVA('PyKeras')
  score2[0] = reader.EvaluateMVA('BDT')

  event_weight[0] = data_tree.EventWeight
  #print data_tree.EventWeight

  s = str(i)

  if nevt > 100000:
    if i%20000 == 0: print "processing "+tuples+" "+s
  elif nevt > 10000 and nevt < 100000:
    if i%10000 == 0: print "processing "+tuples+" "+s
  elif nevt < 10000:
    if i%1000 == 0: print "processing "+tuples+" "+s

  tree.Fill()

  if i == nevt:
    del score1, score2, branches

target.Write()
target.Close()

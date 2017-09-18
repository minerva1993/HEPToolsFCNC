#!/usr/bin/env python

from ROOT import *
from array import array
from subprocess import call
from os.path import isfile
import numpy as np

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

# Load data
data = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/mkNtuple_v3/tmva_SingleLepton_Run2016.root')
data_tree = data.Get('tmva_tree')

target = TFile("rd_output.root","RECREATE" )
tree = TTree("tree","tree")

branches = {}
for branch in data_tree.GetListOfBranches():
  branchName = branch.GetName()
  if branchName not in ["transverseMass", "jet1phi", "jet2phi", "jet3phi", "jet4phi", "KinLepWMass", "KinLepTopMass", "KinHadWMass", "KinHadTopMass", "FCNHKinLepWMass", "FCNHKinLepTopMass", "FCNHKinHMass", "FCNHKinHucTopMass", "M3LepWMass", "M3HadWMass", "M3HMass", "M3DR", "M3LepTopMass", "M3HucTopMass", "DRlepWphi", "DRjet0phi", "DRjet1phi", "DRjet2phi", "DRjet3phi", "DRjet12phi", "DRjet23phi", "DRjet31phi", "DRlepTphi", "DRhadTphi"]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

# Book methods
reader.BookMVA('PyKeras', TString('keras_Hct2v2/weights/TMVAClassification_PyKeras.weights.xml'))
reader.BookMVA('BDT', TString('keras_Hct2v2/weights/TMVAClassification_BDT.weights.xml'))
nevt = data_tree.GetEntries()

score1 = np.zeros(1, dtype=float)
score2 = np.zeros(1, dtype=float)

tree.Branch('KerasScore', score1, 'KerasScore/D')
tree.Branch('BDTScore', score2, 'BDTScore/D')

for i in xrange(nevt):
  data_tree.GetEntry(i)
  score1[0] = reader.EvaluateMVA('PyKeras')
  score2[0] = reader.EvaluateMVA('BDT')
  s = str(i)
  if i%5000 == 0: print "process "+s
  #if i%500 == 0: print score[0]
  tree.Fill()

target.Write()
target.Close()

"""
# Print some example classifications
print('Some signal example classifications:')
for i in range(20):
    data_tree.GetEntry(i)
    print(reader.EvaluateMVA('PyKeras'))
print('')
"""

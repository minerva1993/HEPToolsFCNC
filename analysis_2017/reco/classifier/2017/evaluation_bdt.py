#from __future__ import print_function
#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
import numpy as np

# Setup TMVA
TMVA.Tools.Instance()
#TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

#ver = sys.argv[1]
ver = '05'
filename = sys.argv[2]

# Load data
data = TFile.Open('mkNtuple/dummy/deepReco_'+filename+'.root')
data_tree = data.Get('test_tree')

target = TFile.Open('scoreSTFCNC'+ver+'/score_deepReco_'+filename+'.root','RECREATE')
tree = TTree("tree","tree")

branches = {}
for branch in data_tree.GetListOfBranches():
  branchName = branch.GetName()
  if branchName in ['jet0pt', 'jet0eta', 'jet0m', 'jet1pt', 'jet1eta', 'jet1m', 'jet2pt', 'jet2eta', 'jet2m',
                  'jet12pt', 'jet12eta', 'jet12deta', 'jet12dphi', #'jet12dR', 'jet12m',
                  'lepWpt', 'lepWdphi', 'lepWm', 'lepTdphi', 'lepTm',]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

  elif branchName in ["nevt", "EventCategory", "genMatch", "njets", "nbjets_m"]:
    branches[branchName] = array('f', [-999])
    reader.AddSpectator(branchName, branches[branchName])

reader.BookMVA('BDT', TString('training/recoSTFCNC'+ver+'/weights/TMVAClassification_BDT.weights.xml'))

totalevt = data_tree.GetEntries()
#print("this sample contains "+str(totalevt)+" combinations")

score2        = np.zeros(1, dtype=np.float32)
nevt          = np.zeros(1, dtype=int)
njet          = np.zeros(1, dtype=int)
nbjets_m      = np.zeros(1, dtype=int)
EventCategory = np.zeros(1, dtype=int)
genMatch      = np.zeros(1, dtype=int)
jet0Idx       = np.zeros(1, dtype=int)
jet1Idx       = np.zeros(1, dtype=int)
jet2Idx       = np.zeros(1, dtype=int)
jet3Idx       = np.zeros(1, dtype=int)
lepPt         = np.zeros(1, dtype=np.float32)
missinget     = np.zeros(1, dtype=np.float32)
whMass        = np.zeros(1, dtype=np.float32)
leptMass      = np.zeros(1, dtype=np.float32)
hadtMass      = np.zeros(1, dtype=np.float32)

tree.Branch('MLScore'      , score2       , 'MLScore/F')
tree.Branch('nevt'         , nevt         , 'nevt/I')
tree.Branch('njet'         , njet         , 'njet/I')
tree.Branch('nbjets_m'     , nbjets_m     , 'nbjets_m/I')
tree.Branch('EventCategory', EventCategory, 'EventCategory/I')
tree.Branch('genMatch'     , genMatch     , 'genMatch/I')
tree.Branch('jet0Idx'      , jet0Idx      , 'jet0Idx/I')
tree.Branch('jet1Idx'      , jet1Idx      , 'jet1Idx/I')
tree.Branch('jet2Idx'      , jet2Idx      , 'jet2Idx/I')
tree.Branch('jet3Idx'      , jet3Idx      , 'jet3Idx/I')
tree.Branch('lepPt'        , lepPt        , 'lepPt/F')
tree.Branch('missinget'    , missinget    , 'missinget/F')
tree.Branch('whMass'       , whMass       , 'whMass/F')
tree.Branch('leptMass'     , leptMass     , 'leptMass/F')
tree.Branch('hadtMass'     , hadtMass     , 'hadtMass/F')


for i in xrange(totalevt):
  if totalevt == 0: tree.Write()
  else:
    data_tree.GetEntry(i)
    Nevt = data_tree.nevt

    score2[0]        = reader.EvaluateMVA('BDT')
    nevt[0]          = data_tree.nevt
    njet[0]          = data_tree.njets
    nbjets_m[0]      = data_tree.nbjets_m
    EventCategory[0] = data_tree.GetLeaf("EventCategory").GetValue(0)
    genMatch[0]      = data_tree.genMatch
    jet0Idx[0]       = data_tree.jet0Idx
    jet1Idx[0]       = data_tree.jet1Idx
    jet2Idx[0]       = data_tree.jet2Idx
    jet3Idx[0]       = data_tree.jet3Idx
    lepPt[0]         = data_tree.lepton_pt
    missinget[0]     = data_tree.MET
    whMass[0]        = data_tree.jet12m
    leptMass[0]      = data_tree.lepTm
    hadtMass[0]      = data_tree.hadTm
    tree.Fill()
    #print('processing '+str(Nevt)+'th event', end='\r')

target.Write()
target.Close()

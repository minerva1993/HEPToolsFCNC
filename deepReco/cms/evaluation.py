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

ver = sys.argv[1]
filetype = sys.argv[2]
filename = sys.argv[3]

# Load data

data = TFile.Open('mkNtuple/j4b2/ntuple_'+filetype+'/'+filename)
data_tree = data.Get('test_tree')

target = TFile('score_'+filename,'RECREATE')
tree = TTree("tree","tree")

branches = {}
for branch in data_tree.GetListOfBranches():
  branchName = branch.GetName()
  if branchName not in ["nevt", "file", "combiCount", "genMatch", "GoodPV", "EventCategory", "EventWeight",
    "njets", "nbjets_m", "lepton_pt", "lepton_eta", "lepton_phi", "MET", "MET_phi", "lepDPhi",
    "jet0phi", "jet0Idx", "jet1phi", "jet1Idx", "jet2phi", "jet2Idx", "jet3phi", "jet3Idx",
    "jet12phi", "jet23phi", "jet31phi", "lepWphi", "lepTphi", "hadTphi", 
    "genLep_pt", "genLep_eta", "genLep_phi", "gennu_pt", "gennu_eta", "gennu_phi",
    "genLepB_pt", "genLepB_eta", "genLepB_phi", "genLepB_m",
    "genLepW_pt", "genLepW_eta", "genLepW_phi", "genLepW_m",
    "genLepT_pt", "genLepT_eta", "genLepT_phi", "genLepT_m",
    "genHadJ1_pt", "genHadJ1_eta", "genHadJ1_phi", "genHadJ1_m",
    "genHadJ2_pt", "genHadJ2_eta", "genHadJ2_phi", "genHadJ2_m",
    "genHadJ3_pt", "genHadJ3_eta", "genHadJ3_phi", "genHadJ3_m",
    "genHadW_pt", "genHadW_eta", "genHadW_phi", "genHadW_m", "genHadW_dR",
    "genHadT_pt", "genHadT_eta", "genHadT_phi", "genHadT_m",
    "jet0csv", "jet1csv", "jet2csv", "jet3csv",
    "jet0cvsl", "jet1cvsl", "jet2cvsl", "jet3cvsl",
    "jet0cvsb", "jet1cvsb", "jet2cvsb", "jet3cvsb",
    "genHadW",
    "lepWeta", "lepTeta", "lepTdeta", "lepTdR",
    "hadTpt", "lepTpt",
  ]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

  elif branchName in ["nevt", "file", "EventCategory", "genMatch", "njets", "nbjets_m"]:
    branches[branchName] = array('f', [-999])
    reader.AddSpectator(branchName, branches[branchName])

reader.BookMVA('BDT', TString('deepReco'+ver+'/weights/TMVAClassification_BDT.weights.xml'))

totalevt = data_tree.GetEntries()
#print("this sample contains "+str(totalevt)+" combinations")

score2        = np.zeros(1, dtype=float)
nevt          = np.zeros(1, dtype=int)
nfile         = np.zeros(1, dtype=int)
EventCategory = np.zeros(1, dtype=int)
genMatch      = np.zeros(1, dtype=int)
jet0Idx       = np.zeros(1, dtype=int)
jet1Idx       = np.zeros(1, dtype=int)
jet2Idx       = np.zeros(1, dtype=int)
jet3Idx       = np.zeros(1, dtype=int)
lepPt         = np.zeros(1, dtype=float)
missinget     = np.zeros(1, dtype=float)

tree.Branch('BDTScore'     , score2       , 'BDTScore/D')
tree.Branch('nevt'         , nevt         , 'nevt/I')
tree.Branch('file'         , nfile        , 'file/I')
tree.Branch('EventCategory', EventCategory, 'EventCategory/I')
tree.Branch('genMatch'     , genMatch     , 'genMatch/I')
tree.Branch('jet0Idx'      , jet0Idx      , 'jet0Idx/I')
tree.Branch('jet1Idx'      , jet1Idx      , 'jet1Idx/I')
tree.Branch('jet2Idx'      , jet2Idx      , 'jet2Idx/I')
tree.Branch('jet3Idx'      , jet3Idx      , 'jet3Idx/I')
tree.Branch('lepPt'        , lepPt        , 'lepPt/D')
tree.Branch('missinget'    , missinget    , 'missinget/D')


for i in xrange(totalevt):
  data_tree.GetEntry(i)
  Nevt = data_tree.nevt

  score2[0]        = reader.EvaluateMVA('BDT')
  nevt[0]          = data_tree.nevt
  nfile[0]         = data_tree.GetLeaf("file").GetValue(0)
  EventCategory[0] = data_tree.GetLeaf("EventCategory").GetValue(0)
  genMatch[0]      = data_tree.genMatch
  jet0Idx[0]       = data_tree.jet0Idx
  jet1Idx[0]       = data_tree.jet1Idx
  jet2Idx[0]       = data_tree.jet2Idx
  jet3Idx[0]       = data_tree.jet3Idx
  lepPt[0]         = data_tree.lepton_pt
  missinget[0]     = data_tree.MET

  #print('processing '+str(Nevt)+'th event', end='\r')
  
  tree.Fill()

target.Write()
target.Close()

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
ver = '01'
verBDT= '02'
filename = sys.argv[2]

# Load data

#data = TFile.Open('root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/ntuple/'+filename)
data = TFile.Open('/home/minerva1993/fcnc/analysis_2017/reco/classifier/2017/mkNtuple/j3b2_st/'+filename)
data_tree = data.Get('test_tree')

#target = TFile.Open('/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/training/scoreFCNC'+ver+'/score_'+filename,'RECREATE')
#target = TFile.Open('/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/training/scoreTT'+ver+'/score_'+filename,'RECREATE')
target = TFile.Open('/home/minerva1993/fcnc/analysis_2017/reco/classifier/2017/scoreST'+ver+'/score_'+filename,'RECREATE')
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
    "jet3pt", "jet3eta", "jet3m", #STFCNC
    "jet23pt", "jet23eta", "jet23deta", "jet23dphi", "jet23dR", "jet23m", #STFCNC
    "jet31pt", "jet31eta", "jet31deta", "jet31dphi", "jet31dR", "jet31m", #STFCNC
    "hadTeta", "hadT12_3deta", "hadT23_1deta", "hadT31_2deta",#STFCNC
    "hadT12_3dphi", "hadT23_1dphi", "hadT31_2dphi", "hadT12_3dR", "hadT23_1dR", "hadT31_2dR", "hadTm",#STFCNC
  ]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

  elif branchName in ["nevt", "file", "EventCategory", "genMatch", "njets", "nbjets_m"]:
    branches[branchName] = array('f', [-999])
    reader.AddSpectator(branchName, branches[branchName])

reader.BookMVA('BDT', TString('/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/training/deepRecoSTFCNC'+verBDT+'/weights/TMVAClassification_BDT.weights.xml'))

totalevt = data_tree.GetEntries()
#print("this sample contains "+str(totalevt)+" combinations")

score2        = np.zeros(1, dtype=float)
nevt          = np.zeros(1, dtype=int)
njet          = np.zeros(1, dtype=int)
nbjet         = np.zeros(1, dtype=int)
EventCategory = np.zeros(1, dtype=int)
genMatch      = np.zeros(1, dtype=int)
jet0Idx       = np.zeros(1, dtype=int)
jet1Idx       = np.zeros(1, dtype=int)
jet2Idx       = np.zeros(1, dtype=int)
jet3Idx       = np.zeros(1, dtype=int)
lepPt         = np.zeros(1, dtype=float)
missinget     = np.zeros(1, dtype=float)
whMass        = np.zeros(1, dtype=float)
leptMass      = np.zeros(1, dtype=float)
hadtMass      = np.zeros(1, dtype=float)

tree.Branch('BDTScore'     , score2       , 'BDTScore/D')
tree.Branch('nevt'         , nevt         , 'nevt/I')
tree.Branch('njet'         , njet         , 'njet/I')
tree.Branch('nbjet'        , nbjet        , 'nbjet/I')
tree.Branch('EventCategory', EventCategory, 'EventCategory/I')
tree.Branch('genMatch'     , genMatch     , 'genMatch/I')
tree.Branch('jet0Idx'      , jet0Idx      , 'jet0Idx/I')
tree.Branch('jet1Idx'      , jet1Idx      , 'jet1Idx/I')
tree.Branch('jet2Idx'      , jet2Idx      , 'jet2Idx/I')
tree.Branch('jet3Idx'      , jet3Idx      , 'jet3Idx/I')
tree.Branch('lepPt'        , lepPt        , 'lepPt/D')
tree.Branch('missinget'    , missinget    , 'missinget/D')
tree.Branch('whMass'       , whMass       , 'whMass/D')
tree.Branch('leptMass'     , leptMass     , 'leptMass/D')
tree.Branch('hadtMass'     , hadtMass     , 'hadtMass/D')


for i in xrange(totalevt):
  if totalevt == 0: tree.Write()
  else:
    data_tree.GetEntry(i)
    Nevt = data_tree.nevt

    score2[0]        = reader.EvaluateMVA('BDT')
    nevt[0]          = data_tree.nevt
    njet[0]          = data_tree.njets
    nbjet[0]         = data_tree.nbjets_m
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

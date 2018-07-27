#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
from subprocess import call
from os.path import isfile
import numpy as np
#import google.protobuf

tuples = sys.argv[1]
ch = sys.argv[2]
"""
if tuples == 'tmva_SingleLepton_Run2016.root':
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif tuples == 'tmva_ttLF.root':
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif tuples == 'tmva_ttother.root':
  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
else:
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

path='/home/minerva1993/fcnc/analysis_2017/tmva/v1/score_mva/'
if not os.path.exists(path+ch):
  os.makedirs(path+ch)

if   'j3b2' in ch: jetcut = [3,2]
elif 'j3b3' in ch: jetcut = [3,3]
elif 'j4b2' in ch: jetcut = [4,2]
elif 'j4b3' in ch: jetcut = [4,3]
elif 'j4b4' in ch: jetcut = [4,4]

keras = False

# Load data

data = TFile.Open('/home/minerva1993/fcnc/analysis_2017/tmva/v1/mkNtuple/'+tuples)
data_tree = data.Get('tmva_tree')

target = TFile(path+ch+'/output_'+ch+'_'+tuples,'RECREATE')
tree = TTree("tree","tree")

branches = {}
for branch in data_tree.GetListOfBranches():
  branchName = branch.GetName()
  """
  if branchName not in ["lepDPhi", "bjetmDR", "bjetmDEta", "bjetmDPhi", "dibjetsMass", "bjetPt_dibjetsm", "cjetPt", "transverseMass", "jet1phi", "jet2phi", "jet3phi", "jet4phi", "DRlepWphi", "DRjet0phi", "DRjet1phi", "DRjet2phi", "DRjet3phi", "DRjet12phi", "DRjet23phi", "DRjet31phi", "DRlepTphi", "DRhadTphi", "EventWeight", "totnevt", "nevt", "GoodPV", "EventCategory","GenMatch",
  "njets", "nbjets_m", "ncjets_m", "missingET",
  "jet1pt", "jet2pt", "jet3pt", "jet4pt",
  "jet1eta", "jet2eta", "jet3eta", "jet4eta",
  "jet1m", "jet2m", "jet3m", "jet4m",
  "jet1csv", "jet2csv", "jet3csv", "jet4csv",
  "jet1cvsl", "jet2cvsl", "jet3cvsl", "jet4cvsl", 
  "jet1cvsb","jet2cvsb", "jet3cvsb", "jet4cvsb",
  #"DRlepWpt", "DRlepWeta", "DRlepWdeta", "DRlepWdphi", "DRlepWm",
  "DRjet0pt", "DRjet0eta", "DRjet0m", "DRjet0csv","DRjet0cvsl","DRjet0cvsb",
  "DRlepTpt", "DRlepTeta", "DRlepTdeta","DRlepTdphi","DRlepTm",
  "DRjet0cvsl", "DRjet1cvsl","DRjet2cvsl","DRjet3cvsl",
  "DRjet0cvsb", "DRjet1cvsb", "DRjet2cvsb", "DRjet3cvsb", 
  "DRlepWdeta"
  ]:
  """
  if branchName in ["lepWpt", "lepWdphi","lepWm", "jet0pt", "jet0eta", "jet0m", "jet1pt", "jet1eta", "jet1m","jet2pt", "jet2eta", "jet2m","jet12pt", "jet12eta","jet12deta","jet12dphi","jet12m","jet12DR","lepTdphi","lepTm",
  ]:
      branches[branchName] = array('f', [-999])
      reader.AddVariable(branchName, branches[branchName])
      data_tree.SetBranchAddress(branchName, branches[branchName])

  elif branchName in ["GoodPV", "EventCategory", "GenMatch"]:
    branches[branchName] = array('f', [-999])
    reader.AddSpectator(branchName, branches[branchName])

if keras:
  reader.BookMVA('PyKeras', TString('/home/minerva1993/fcnc/analysis_2017/tmva/v1/'+ch+'/weights/TMVAClassification_Keras_TF.weights.xml'))
reader.BookMVA('BDT', TString('/home/minerva1993/fcnc/analysis_2017/tmva/v1/'+ch+'/weights/TMVAClassification_BDT.weights.xml'))

print "processing "+tuples
totalevt = data_tree.GetEntries()
print "this sample contains "+str(totalevt)+" events"

score1 = np.zeros(1, dtype=float)
score2 = np.zeros(1, dtype=float)
event_weight = np.zeros(1, dtype=float)
pileup =  np.zeros(1, dtype=int)
category = np.zeros(1, dtype=int)
genMatch = np.zeros(1, dtype=int)

tree.Branch('KerasScore', score1, 'KerasScore/D')
tree.Branch('BDTScore', score2, 'BDTScore/D')
tree.Branch('Event_Weight', event_weight, 'Event_Weight/D')
tree.Branch("PU", pileup, 'PU/I')
tree.Branch("category", category, 'category/I')
tree.Branch("genMatch", genMatch, 'genMatch/I')

for i in xrange(totalevt-1):
  data_tree.GetEntry(i)
  nevt = data_tree.GetLeaf("nevt").GetValue(0)
#  nevt = data_tree.nevt
  if data_tree.njets != jetcut[0] or data_tree.nbjets_m != jetcut[1]: continue

  if tuples in ["tmva_SingleLepton_Run2017.root"]:
    if keras:
      score1[0] = reader.EvaluateMVA('PyKeras')
    score2[0] = reader.EvaluateMVA('BDT')
    event_weight[0] = data_tree.EventWeight
    pileup[0] = data_tree.GoodPV
    category[0] = data_tree.EventCategory
    genMatch[0] = data_tree.GenMatch
  else:
    #if nevt == i:
    if tuples in ["tmva_w1jets50to150.root", "tmva_w1jets150to250.root", "tmva_w1jets250to400.root", "tmva_w1jets400toinf.root", "tmva_w2jets250to400.root", "tmva_w2jets400toinf.root", "tmva_w3jets.root", "tmva_w4jets.root", "tmva_zjets10to50.root", "tmva_zjets.root", "tmva_ww.root", "tmva_wz.root", "tmva_zz.root"]:
      if keras:
        score1[0] = reader.EvaluateMVA('PyKeras')
      score2[0] = reader.EvaluateMVA('BDT')
      event_weight[0] = data_tree.EventWeight
      pileup[0] = data_tree.GoodPV
      category[0] = data_tree.EventCategory
      genMatch[0] = data_tree.GenMatch
    else:
      if nevt % 3 == 0:
        if keras:
          score1[0] = reader.EvaluateMVA('PyKeras')
        score2[0] = reader.EvaluateMVA('BDT')
        event_weight[0] = data_tree.EventWeight
        pileup[0] = data_tree.GoodPV
        category[0] = data_tree.EventCategory
        genMatch[0] = data_tree.GenMatch
      else:
        score1[0] = 2
        score2[0] = 2
        event_weight[0] = 10
        pileup[0] = -1
        category[0] = -1
        genMatch[0] = 5
     
  s = str(nevt)

  if totalevt > 100000:
    if nevt%20000 == 0: print "processing "+tuples+" "+s
  elif totalevt > 10000 and nevt < 100000:
    if nevt%10000 == 0: print "processing "+tuples+" "+s
  elif totalevt < 10000:
    if nevt%1000 == 0: print "processing "+tuples+" "+s

  tree.Fill()

  if i == totalevt:
    del score1, score2, branches

target.Write()
target.Close()

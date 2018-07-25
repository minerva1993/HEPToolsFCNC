#!/usr/bin/env python
import sys, os

from ROOT import *

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader("deepRecoSTFCNC03")
#loader.AddVariable("njets", "I")
#loader.AddVariable("nbjets_m",'I')
loader.AddVariable("jet0pt",'F')
loader.AddVariable("jet0eta",'F')
loader.AddVariable("jet0m",'F')
#loader.AddVariable("jet0csv",'I')
loader.AddVariable("jet1pt",'F')
loader.AddVariable("jet1eta",'F')
loader.AddVariable("jet1m",'F')
#loader.AddVariable("jet1csv",'I')
loader.AddVariable("jet2pt",'F')
loader.AddVariable("jet2eta",'F')
loader.AddVariable("jet2m",'F')
#loader.AddVariable("jet2csv",'I')

loader.AddVariable("jet12pt",'F')
loader.AddVariable("jet12eta",'F')
loader.AddVariable("jet12deta",'F')
loader.AddVariable("jet12dphi",'F')
loader.AddVariable("jet12dR",'F')
loader.AddVariable("jet12m",'F')

loader.AddVariable("lepWpt",'F')
loader.AddVariable("lepWdphi",'F')
loader.AddVariable("lepWm",'F')
#loader.AddVariable("lepTpt",'F')
loader.AddVariable("lepTdphi",'F')
#loader.AddVariable("lepTdR",'F')
loader.AddVariable("lepTm",'F')


loader.SetWeightExpression("EventWeight")
loader.AddSpectator("nevt")
loader.AddSpectator("file")
loader.AddSpectator("EventCategory")
loader.AddSpectator("genMatch")
loader.AddSpectator("njets")
loader.AddSpectator("nbjets_m")

## Load input files
inputs = ["deepReco_STTH1L3BHct_000.root"
          ]

trees = []
for fName in inputs:
    if "Hut" in fName: fileWeight = 1
    elif "Hct" in fName: fileWeight = 0.15
    f = TFile("../dummy/"+fName)
    t = f.Get("sig_tree")
    loader.AddSignalTree(t, fileWeight)
    trees.append([f, t])
for fName in inputs:
    if "Hut" in fName: fileWeight = 1
    elif "Hct" in fName: fileWeight = 0.15
    f = TFile("../dummy/"+fName)
    t = f.Get("bkg_tree")
    loader.AddBackgroundTree(t, fileWeight)
    trees.append([f, t])

sigCut = TCut("")
bkgCut = TCut("")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=15000:nTrain_Background=15000:nTest_Signal=40000:nTest_Background=50000:SplitMode=Random:NormMode=NumEvents:!V"
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=1000:MinNodeSize=5%:MaxDepth=8:BoostType=Grad:Shrinkage=0.5:SeparationType=GiniIndex:nCuts=60")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()


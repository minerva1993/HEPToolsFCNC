#!/usr/bin/env python
import sys, os

from ROOT import *

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader("deepReco04")
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
loader.AddVariable("jet3pt",'F')
loader.AddVariable("jet3eta",'F')
loader.AddVariable("jet3m",'F')
#loader.AddVariable("jet3csv",'I')

loader.AddVariable("jet12pt",'F')
loader.AddVariable("jet12eta",'F')
loader.AddVariable("jet12deta",'F')
loader.AddVariable("jet12dphi",'F')
loader.AddVariable("jet12dR",'F')
loader.AddVariable("jet12m",'F')
loader.AddVariable("jet23pt",'F')
loader.AddVariable("jet23eta",'F')
loader.AddVariable("jet23deta",'F')
loader.AddVariable("jet23dphi",'F')
loader.AddVariable("jet23dR",'F')
loader.AddVariable("jet23m",'F')
loader.AddVariable("jet31pt",'F')
loader.AddVariable("jet31eta",'F')
loader.AddVariable("jet31deta",'F')
loader.AddVariable("jet31dphi",'F')
loader.AddVariable("jet31dR",'F')
loader.AddVariable("jet31m",'F')

loader.AddVariable("lepWpt",'F')
#loader.AddVariable("lepWeta",'F')
loader.AddVariable("lepWdphi",'F')
loader.AddVariable("lepWm",'F')
#loader.AddVariable("lepTpt",'F')
#loader.AddVariable("lepTeta",'F')
#loader.AddVariable("lepTdeta",'F')
loader.AddVariable("lepTdphi",'F')
#loader.AddVariable("lepTdR",'F')
loader.AddVariable("lepTm",'F')

#loader.AddVariable("hadTpt",'F')
loader.AddVariable("hadTeta",'F')
loader.AddVariable("hadT12_3deta",'F')
loader.AddVariable("hadT23_1deta",'F')
loader.AddVariable("hadT31_2deta",'F')
loader.AddVariable("hadT12_3dphi",'F')
loader.AddVariable("hadT23_1dphi",'F')
loader.AddVariable("hadT31_2dphi",'F')
loader.AddVariable("hadT12_3dR",'F')
loader.AddVariable("hadT23_1dR",'F')
loader.AddVariable("hadT31_2dR",'F')
loader.AddVariable("hadTm",'F')

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("nevt")
loader.AddSpectator("file")
loader.AddSpectator("EventCategory")
loader.AddSpectator("genMatch")
loader.AddSpectator("njets")
loader.AddSpectator("nbjets_m")

## Load input files
#inputs = ["deepReco_ttbb_0.root", "deepReco_ttbb_1.root","deepReco_ttbb_2.root", "deepReco_ttbb_3.root", "deepReco_ttbb_4.root", "deepReco_ttbb_5.root", "deepReco_ttbb_6.root", "deepReco_ttbj_0.root", "deepReco_ttbj_1.root", "deepReco_ttbj_2.root", "deepReco_ttbj_3.root", "deepReco_ttbj_4.root", "deepReco_ttbj_5.root", "deepReco_ttbj_6.root", "deepReco_ttcc_0.root", "deepReco_ttcc_1.root", "deepReco_ttcc_2.root", "deepReco_ttcc_3.root", "deepReco_ttcc_4.root", "deepReco_ttcc_5.root", "deepReco_ttcc_6.root", "deepReco_ttLF_0.root", "deepReco_ttLF_1.root", "deepReco_ttLF_2.root", "deepReco_ttLF_3.root", "deepReco_ttLF_4.root", "deepReco_ttLF_5.root", "deepReco_ttLF_6.root", "deepReco_ttother_0.root", "deepReco_ttother_1.root", "deepReco_ttother_2.root", "deepReco_ttother_3.root", "deepReco_ttother_4.root", "deepReco_ttother_5.root", "deepReco_ttother_5.root",  ]#ttbar
inputs = ["deepReco_TopHct_0.root", "deepReco_TopHct_1.root", "deepReco_TopHct_2.root", "deepReco_TopHct_3.root", "deepReco_TopHct_4.root", "deepReco_TopHct_5.root", "deepReco_TopHct_6.root", "deepReco_TopHct_7.root", "deepReco_AntiTopHct_0.root", "deepReco_AntiTopHct_1.root", "deepReco_AntiTopHct_2.root", "deepReco_AntiTopHct_3.root", "deepReco_AntiTopHct_4.root", "deepReco_AntiTopHct_5.root", "deepReco_AntiTopHct_6.root", "deepReco_AntiTopHct_7.root"]
#for mcfile in os.listdir("/home/minerva1993/fcnc/deepReco/mkNtuple/cms/ntuple_tt/"):
#  inputs.append(mcfile)
#inputs = ["test.root"]

trees = []
for fName in inputs:
    f = TFile("/home/minerva1993/fcnc/deepReco/cms/mkNtuple/j4b2/ntuple_mc/"+fName)
    t = f.Get("sig_tree")
    loader.AddSignalTree(t, 1)
    trees.append([f, t])
for fName in inputs:
    f = TFile("/home/minerva1993/fcnc/deepReco/cms/mkNtuple/j4b2/ntuple_mc/"+fName)
    t = f.Get("bkg_tree")
    loader.AddBackgroundTree(t, 1)
    trees.append([f, t])

sigCut = TCut("")
bkgCut = TCut("")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    #"nTrain_Signal=50000:nTrain_Background=50000:nTest_Signal=20000:nTest_Background=20000:SplitMode=Random:NormMode=NumEvents:!V"#ttbar
    "nTrain_Signal=13000:nTrain_Background=50000:nTest_Signal=5000:nTest_Background=10000:SplitMode=Random:NormMode=NumEvents:!V"#fcnc
)

#factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=800:MinNodeSize=5%:MaxDepth=4:BoostType=Grad:Shrinkage=0.4:SeparationType=GiniIndex:nCuts=40")#tt
factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=200:MinNodeSize=5%:MaxDepth=4:BoostType=Grad:Shrinkage=0.4:SeparationType=GiniIndex:nCuts=20")#fcnc

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()


#!/usr/bin/env python

from ROOT import *
from collections import OrderedDict
import sys, os
import re

if len(sys.argv) < 2:
    print "Usage: %s INPUTFILE.root" % sys.argv[0]
    sys.exit(1)

def buildROC(fName, objPath):
    f = TFile(fName)
    h = f.Get(objPath)

    grp = TGraph()
    for i in range(h.GetNbinsX()):
      x = h.GetXaxis().GetBinCenter(i+1)
      y = h.GetBinContent(i+1)
      grp.SetPoint(i, x, y)
    grp.SetLineWidth(2)
    return grp

def buildLegend(x1=0.6, y1=0.75, x2=0.85, y2=0.87):
    leg = TLegend(x1, y1, x2, y2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(1111)
    return leg

gROOT.ProcessLine(".L tdrstyle.C")
gROOT.ProcessLine("setTDRStyle()")
gStyle.SetOptTitle(0)
gStyle.SetOptStat(0)
gStyle.SetPaintTextFormat(".3f")
#gStyle.SetPalette(kDarkBodyRadiator)
gStyle.SetPalette(kRainBow)

grps = OrderedDict()

## Load ROC histograms and compute AUC
for fName in sys.argv[1:]:
    f = TFile(fName)
    if f == None: continue

    for dBaseName0 in [key.GetName() for key in f.GetListOfKeys()]:
        dBase = f.Get(dBaseName0)
        for dName in [key.GetName() for key in dBase.GetListOfKeys()]:
            if dName.split('_', 1)[0] != 'Method': continue
            d = dBase.GetDirectory(dName)
            if d == None: continue

            methodName = dName.split('_', 1)[-1]
            d = d.GetDirectory(methodName)

            hROC = d.Get("MVA_%s_rejBvsS" % methodName)
            objPath = d.GetPath().split(':',1)[-1]+'/'+hROC.GetName()
            auc = hROC.Integral()/hROC.GetNbinsX()

            hMVA_TestS = d.Get("MVA_%s_S" % methodName)
            hMVA_TestB = d.Get("MVA_%s_B" % methodName)
            hMVA_TrainS = d.Get("MVA_%s_Train_S" % methodName)
            hMVA_TrainB = d.Get("MVA_%s_Train_B" % methodName)

            chi2S = hMVA_TestS.Chi2Test(hMVA_TrainS, "WW CHI2/NDF")
            chi2B = hMVA_TestB.Chi2Test(hMVA_TrainB, "WW CHI2/NDF")

            grps[methodName] = [(fName, objPath), auc, chi2S, chi2B]

grps = OrderedDict(reversed(sorted(grps.iteritems(), key=lambda x: x[1][1])))

## Fill histograms
colors = [kRed, kBlue+1, kGreen+3, kOrange-3, kMagenta+2]
colors.extend([kRed-1, kBlue-1, kGreen-5, kOrange+3, kMagenta-1])

hAUC = TH1D("hAUC", "AUC;;Area under curve", len(grps), 0, len(grps)) ## Just for frame
grpAUC = TGraph()
grpAUCRefs = OrderedDict()
#grpAUCRefs["BDT_.*minNode2p5_nTree850"] = ["BDT >=2.5node 850tree", kAzure+1, grpAUC.Clone()]
#grpAUCRefs["DNN_ReLU_.*"] = ["DNN (ReLU)", kRed+2, TGraph()]
#grpAUCRefs["BDTG_.*"] = ["Gradient BDT", kMagenta+1, TGraph()]
grpAUCRefs["BDT_nCuts40_maxDepth3_minNode2p5_nTree850"] = ["BDT default", kBlue+1, TGraph()]
"""
grpAUCRefs["Keras+TF"] = ["Keras", kGreen+2, TGraph()]
"""
#grpAUCRefs["BDTD_.*"] = ["Decorrelated BDT", kCyan+1, grpAUC.Clone()]
for name, (title, color, grp) in grpAUCRefs.iteritems():
    grp.SetMarkerStyle(kFullCircle)
    grp.SetMarkerSize(0.5)
    grp.SetMarkerColor(color)
    grp.SetLineColor(color)

grpAUCRefs["BDT_nCuts40_maxDepth3_minNode2p5_nTree850"][-1].SetMarkerSize(2)
grpAUCRefs["BDT_nCuts40_maxDepth3_minNode2p5_nTree850"][-1].SetMarkerStyle(kFullTriangleUp)
"""
grpAUCRefs["Keras+TF"][-1].SetMarkerSize(2.5)
grpAUCRefs["Keras+TF"][-1].SetMarkerStyle(kFullStar)
"""
grpChi2S, grpChi2B = TGraph(), TGraph()
grpChi2S.SetMarkerStyle(kFullTriangleUp)
grpChi2B.SetMarkerStyle(kFullTriangleUp)
grpChi2S.SetTitle("BDT sig.")
grpChi2B.SetTitle("BDT bkg.")
grpChi2S.SetMarkerColor(kAzure+1)
grpChi2B.SetMarkerColor(kGray+1)
grpChi2S.SetFillColor(kAzure+1)
grpChi2B.SetFillColor(kGray+1)
grpChi2S.SetMarkerSize(0.7)
grpChi2B.SetMarkerSize(0.7)
"""
grpChi2RefS, grpChi2RefB = TGraph(), TGraph()
grpChi2RefS.SetMarkerStyle(kFullStar)
grpChi2RefB.SetMarkerStyle(kFullStar)
grpChi2RefS.SetTitle("Keras+TF sig.")
grpChi2RefB.SetTitle("Keras+TF bkg.")
grpChi2RefS.SetMarkerSize(2.5)
grpChi2RefB.SetMarkerSize(2.5)
grpChi2RefS.SetMarkerColor(kGreen+2)
grpChi2RefB.SetMarkerColor(kMagenta)
"""
grpChi2Ref2S, grpChi2Ref2B = TGraph(), TGraph()
grpChi2Ref2S.SetTitle("BDT sig. (opt.)")
grpChi2Ref2B.SetTitle("BDT bkg. (opt.)")
grpChi2Ref2S.SetMarkerSize(2.0)
grpChi2Ref2B.SetMarkerSize(2.0)
grpChi2Ref2S.SetMarkerStyle(kFullTriangleUp)
grpChi2Ref2B.SetMarkerStyle(kFullTriangleUp)
grpChi2Ref2S.SetMarkerColor(kBlue+1)
grpChi2Ref2B.SetMarkerColor(kRed)

for i, (name, [(fName, objPath), auc, chi2S, chi2B]) in enumerate(grps.iteritems()):
    grpAUC.SetPoint(i, i, auc)
    grpChi2S.SetPoint(i, auc, chi2S)
    grpChi2B.SetPoint(i, auc, chi2B)
    for pattern0, (title, color, grp) in grpAUCRefs.iteritems():
        pattern = pattern0.replace("+", "\\+")
        if not re.match('^'+pattern+'$', name): continue
        grp.SetPoint(i, i, auc)
    if name == "Keras+TF":
        grpChi2RefS.SetPoint(grpChi2RefS.GetN(), auc, chi2S)
        grpChi2RefB.SetPoint(grpChi2RefB.GetN(), auc, chi2B)
    if name == "BDT_nCuts40_maxDepth3_minNode2p5_nTree850":
        grpChi2Ref2S.SetPoint(grpChi2Ref2S.GetN(), auc, chi2S)
        grpChi2Ref2B.SetPoint(grpChi2Ref2B.GetN(), auc, chi2B)

grpsToDraw = []
legROC = buildLegend(0.2, 0.15, 0.8, 0.6)
if len(grps) > 5:
    legROC.AddEntry(0, "Top 5", "p").SetMarkerColor(kWhite)
for i, (name, [(fName, objPath), auc, chi2S, chi2B]) in enumerate(grps.iteritems()):
    if len(grps) > 10 and i >= 5: break
    grp = buildROC(fName, objPath)
    grp.SetLineColor(colors[i])
    grpsToDraw.append(grp)
    legROC.AddEntry(grp, name, "l")
if len(grps) > 10:
    legROC.AddEntry(0, "", "p").SetMarkerColor(kWhite)
    legROC.AddEntry(0, "Worst 5", "p").SetMarkerColor(kWhite)
    for i, (name, [(fName, objPath), auc, chi2S, chi2B]) in enumerate(reversed([x for x in grps.iteritems()])):
        if i >= 5: break
        grp = buildROC(fName, objPath)
        grp.SetLineColor(colors[i+5])
        grpsToDraw.append(grp)
        legROC.AddEntry(grp, name, "l")

## Start drawing
cROC = TCanvas("cROC", "cROC", 500, 500)
hFrameROC = TH2F("hFrameROC", "hFrame;Signal efficiency;Background rejection", 100, 0, 1, 100, 0, 1)
hFrameROC.Draw()
legROC.Draw()
for grp in grpsToDraw: grp.Draw("l")
cROC.SetGridx()
cROC.SetGridy()

cAUC = TCanvas("cAUC", "cAUC", 500, 500)
hAUC.SetMinimum(0.70)
hAUC.SetMaximum(0.76)
for i, (name, [(fName, objPath), auc, chi2S, chi2B]) in enumerate(grps.iteritems()):
    hAUC.GetXaxis().SetBinLabel(i+1, name)
    hAUC.SetBinContent(i+1, auc)
hAUC.Draw()
legAUC = buildLegend(0.2, 0.2, 0.6, 0.4)
grpAUC.SetMarkerStyle(kFullCircle)
grpAUC.SetMarkerSize(0.5)
grpAUC.Draw("p")
grpAUC.SetEditable(False)
legAUC.AddEntry(grpAUC, "All MVA methods", "p")
for name, (title, color, grp) in grpAUCRefs.iteritems():
    legAUC.AddEntry(grp, "%s" % title, "p")
    grp.Draw("p")
    grp.SetEditable(False)
legAUC.Draw()

cChi2 = TCanvas("cChi2", "cChi2", 500, 500)
cChi2.SetLogy()
hFrameChi2 = TH1F("hChi2Frame", "hChi2Frame;Area under curve;#chi^{2}/dof", 100, 0.735, 0.755)
hFrameChi2.SetMinimum(0.5)
hFrameChi2.SetMaximum(1e2)
hFrameChi2.GetXaxis().SetNdivisions(505)
hFrameChi2.Draw()
legChi2 = buildLegend(0.2, 0.6, 0.5, 0.8)
for grp in (grpChi2S, grpChi2B):
    grp.Draw("p")
    grp.SetEditable(False)
    legChi2.AddEntry(grp, grp.GetTitle(), "f")
for grp in (grpChi2Ref2S, grpChi2Ref2B):#grpChi2RefS, grpChi2RefB, 
    grp.Draw("p")
    grp.SetEditable(False)
    legChi2.AddEntry(grp, grp.GetTitle(), "p")
legChi2.Draw()

cROC.Print("cROC.C")
cAUC.Print("cAUC.C")
cChi2.Print("cChi2.C")

for x in grps.keys()[:5]:
    print grps[x][0][0], grps[x][0][1].split('/')[-2], grps[x][1]

#! /usr/bin/env python

from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1F, TH2F, THStack, TLegend, gROOT, TPad
gROOT.SetBatch(True)
import ROOT
import os

from style import *

log = False
each_plot = True
printscale = False

from collections import OrderedDict
data17=OrderedDict()
data18=OrderedDict()
dataA=OrderedDict()
dataB=OrderedDict()
dataC=OrderedDict()
dataD=OrderedDict()

def Add17(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  data17[fname] = tmp
 
def Add18(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  data18[fname] = tmp

def AddA(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  dataA[fname] = tmp

def AddB(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  dataB[fname] = tmp

def AddC(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  dataC[fname] = tmp

def AddD(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  dataD[fname] = tmp

####Users should provide these information 
Add17("2017/hist_SingleMuonRun2017.root","2017", 41529)
Add17("2017/hist_SingleElectronRun2017.root","2017", 41529)
Add18("2018/hist_SingleMuonRun2018.root","2018", 59970)
Add18("2018/hist_SingleElectronRun2018.root","2018", 59970)
AddA("2018/hist_SingleMuonRun2018A.root","2018A", 14000)
AddA("2018/hist_SingleElectronRun2018A.root","2018A", 14000)
AddB("2018/hist_SingleMuonRun2018B.root","2018B", 7100)
AddB("2018/hist_SingleElectronRun2018B.root","2018B", 7100)
AddC("2018/hist_SingleMuonRun2018C.root","2018C", 6940)
AddC("2018/hist_SingleElectronRun2018C.root","2018C", 6940)
AddD("2018/hist_SingleMuonRun2018Dv2.root","2018D", 31930)
AddD("2018/hist_SingleElectronRun2018Dv2.root","2018D", 31930)

noRecoList = [ x for x in data17[data17.keys()[0]]["hname"] if "FCNH" not in x ]
noRecoList = [ x for x in noRecoList if "gen" not in x ]
noRecoList = [ x for x in noRecoList if "match" not in x ]
N_17_noReco = len(noRecoList)
N_17 = len(data17[data17.keys()[0]]["hname"])
N_A = len(dataA)
N_B = len(dataB)
N_C = len(dataC)
N_D = len(dataD)

count = 0
for i in range(0, N_17):
  if "Ch0" in data17[data17.keys()[0]]["hname"][i]:   mode = 0
  elif "Ch1" in data17[data17.keys()[0]]["hname"][i]: mode = 1
  elif "Ch2" in data17[data17.keys()[0]]["hname"][i]: mode = 2
  if any(reco in data17[data17.keys()[0]]["hname"][i] for reco in ["FCNH","gen","match"] ): continue

  string_fname = ''
  string_nevt =  ''
  hnames = data17[data17.keys()[0]]["hname"][i].split("_")

  #print hnames[1], " ", hnames[2], " ", hnames[3]  

  l = TLegend(0.15,0.71,0.89,0.87)
  l.SetNColumns(4);
  l.SetTextSize(0.05);
  l.SetLineColor(0);
  l.SetFillColor(0);

  if   mode == 0 or mode == 1: h_17 = data17[data17.keys()[mode]]["file"].Get(data17[data17.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_17 = data17[data17.keys()[0]]["file"].Get(data17[data17.keys()[0]]["hname"][i])
    h_17.Add(data17[data17.keys()[1]]["file"].Get(data17[data17.keys()[1]]["hname"][i]))
  nbins = h_17.GetNbinsX()
  h_17.AddBinContent( nbins, h_17.GetBinContent( nbins+1 ) )  #overflow
  scale = 0.0
  if h_17.Integral() > 0: scale = 1.0/h_17.Integral()
  h_17.Scale(scale)
  l.AddEntry(h_17,"2017","P")

  if   mode == 0 or mode == 1: h_18 = data18[data18.keys()[mode]]["file"].Get(data18[data18.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_18 = data18[data18.keys()[0]]["file"].Get(data18[data18.keys()[0]]["hname"][i])
    h_18.Add(data18[data18.keys()[1]]["file"].Get(data18[data18.keys()[1]]["hname"][i]))
  nbins = h_18.GetNbinsX()
  h_18.AddBinContent( nbins, h_18.GetBinContent( nbins+1 ) )  #overflow
  scale = 0.0
  if h_18.Integral() > 0: scale = 1.0/h_18.Integral()
  h_18.Scale(scale)
  l.AddEntry(h_18,"2018","P")

  if   mode == 0 or mode == 1: h_A = dataA[dataA.keys()[mode]]["file"].Get(dataA[dataA.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_A = dataA[dataA.keys()[0]]["file"].Get(dataA[dataA.keys()[0]]["hname"][i])
    h_A.Add(dataA[dataA.keys()[1]]["file"].Get(dataA[dataA.keys()[1]]["hname"][i]))
  nbins = h_A.GetNbinsX()
  h_A.AddBinContent( nbins, h_A.GetBinContent( nbins+1 ) )  #overflow
  scale = 0.0
  if h_A.Integral() > 0: scale = 1.0/h_A.Integral()
  h_A.Scale(scale)
  h_A.SetLineColor(ROOT.kRed-7)
  l.AddEntry(h_A,"2018A","F")

  if   mode == 0 or mode == 1: h_B = dataB[dataB.keys()[mode]]["file"].Get(dataB[dataB.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_B = dataB[dataB.keys()[0]]["file"].Get(dataB[dataB.keys()[0]]["hname"][i])
    h_B.Add(dataB[dataB.keys()[1]]["file"].Get(dataB[dataB.keys()[1]]["hname"][i]))
  nbins = h_B.GetNbinsX()
  h_B.AddBinContent( nbins, h_B.GetBinContent( nbins+1 ) )  #overflow
  scale = 0.0
  if h_B.Integral() > 0: scale = 1.0/h_B.Integral()
  h_B.Scale(scale)
  h_B.SetLineColor(ROOT.kYellow+1)
  l.AddEntry(h_B,"2018B","F")

  if   mode == 0 or mode == 1: h_C = dataC[dataC.keys()[mode]]["file"].Get(dataC[dataC.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_C = dataC[dataC.keys()[0]]["file"].Get(dataC[dataC.keys()[0]]["hname"][i])
    h_C.Add(dataC[dataC.keys()[1]]["file"].Get(dataC[dataC.keys()[1]]["hname"][i]))
  nbins = h_C.GetNbinsX()
  h_C.AddBinContent( nbins, h_C.GetBinContent( nbins+1 ) )  #overflow
  scale = 0.0
  if h_C.Integral() > 0: scale = 1.0/h_C.Integral()
  h_C.Scale(scale)
  h_C.SetLineColor(ROOT.kBlue+1)
  l.AddEntry(h_C,"2018C","F")

  if   mode == 0 or mode ==1: h_D = dataD[dataD.keys()[mode]]["file"].Get(dataD[dataD.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_D = dataD[dataD.keys()[0]]["file"].Get(dataD[dataD.keys()[0]]["hname"][i])
    h_D.Add(dataD[dataD.keys()[1]]["file"].Get(dataD[dataD.keys()[1]]["hname"][i]))
  nbins = h_D.GetNbinsX()
  h_D.AddBinContent( nbins, h_D.GetBinContent( nbins+1 ) )  #overflow
  scale = 0.0
  if h_D.Integral() > 0: scale = 1.0/h_D.Integral()
  h_D.Scale(scale)
  h_D.SetLineColor(ROOT.kCyan+1)
  l.AddEntry(h_D,"2018D","F")


  def createCanvasPads():
    #creat canvas
    c = TCanvas("c_"+"{}".format(i),"c", 450, 450)

    # Upper histogram plot is pad1
    pad1 = TPad("pad1", "pad1", 0.0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.Draw()
    c.cd()  # returns to main canvas before defining pad2
    pad2 = TPad("pad2", "pad2", 0.0, 0.0, 1, 0.28)
    pad2.SetBottomMargin(0.3)
    pad2.SetTopMargin(0.02)
    pad2.SetGridx()
    pad2.SetGridy()
    pad2.Draw()

    return c, pad1, pad2

  def createRatio(mc, data):
    h4 = mc.Clone("h4")
    h3 = data.Clone("h3")
    h3.SetDirectory(0)
    h3.SetLineColor(1)
    h3.SetMarkerStyle(20)
    h3.SetMarkerSize(0.4)
    h3.SetTitle("")
    #h3.Sumw2()
    h3.SetStats(0)

    y = h3.GetYaxis()
    y.SetTitle("2018/2017")
    y.SetNdivisions(505)
    y.SetTitleSize(0.11)
    y.SetTitleOffset(0.35)
    y.SetLabelSize(0.1)

    # Adjust x-axis settings
    x = h3.GetXaxis()
    x.SetTitleSize(0.11)
    x.SetTitleOffset(1.0)
    x.SetLabelSize(0.1)
    #x.SetLabelSize(0)
    h3.Divide(h4)
    h3.SetMinimum(0.5)
    h3.SetMaximum(1.5)

    return h3

  def ratioplot():
    h_1718 = createRatio(h_17, h_18)
    h_17A = createRatio(h_17, h_A)
    h_17B = createRatio(h_17, h_B)
    h_17C = createRatio(h_17, h_C)
    h_17D = createRatio(h_17, h_D)
    c, pad1, pad2 = createCanvasPads()

    #Draw each plot
    pad1.cd()
    if log:
      pad1.SetLogy()
    h_17.SetMarkerStyle(41)#20
    h_17.SetMarkerSize(0.5)
    h_18.SetMarkerStyle(43)
    h_18.SetMarkerSize(0.7)
    maxs = [h_17.GetMaximum(), h_A.GetMaximum(), h_B.GetMaximum(), h_C.GetMaximum(), h_D.GetMaximum(),]
    maxHisto = max(maxs)
    maxfrac = 0.5
    if log:
      maxfrac = 100
      h_17.SetMaximum(maxHisto + maxHisto * maxfrac)
    else:
      h_17.SetMaximum(maxHisto * 1.5)
    if log: h_17.SetMinimum(0.5)

    h_17.Draw("p")
    h_17.SetTitle("")
    h_17.GetYaxis().SetTitle("Events")
    h_17.GetYaxis().SetTitleOffset(1.2)
    h_17.GetYaxis().SetTitleSize(0.045)
    h_17.GetXaxis().SetLabelSize(0)#
    #h_17.GetXaxis().SetTitle("")
    h_17.GetXaxis().SetTitleOffset(5.0)
    h_A.SetLineWidth(2)
    h_B.SetLineWidth(2)
    h_C.SetLineWidth(2)
    h_D.SetLineWidth(2)
    h_A.Draw("hist same")
    h_B.Draw("hist same")
    h_C.Draw("hist same")
    h_D.Draw("hist same")
    h_17.Draw("P SAME")
    h_18.Draw("P SAME")
    h_18.Draw("AXIS P SAME")

    l.Draw()
    label = TPaveText()
    label.SetX1NDC(gStyle.GetPadLeftMargin())
    label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
    label.SetX2NDC(1.0-gStyle.GetPadRightMargin()+0.03)
    label.SetY2NDC(1.0)
    label.SetTextFont(62)
    label.AddText("Work in Progress            CMS, Run2 at #sqrt{s} = 13 TeV")
    label.SetFillStyle(0)
    label.SetBorderSize(0)
    label.SetTextSize(0.05)
    label.SetTextAlign(32)
    label.Draw("same")
    
    pad2.cd()
    h_17A.SetLineColor(ROOT.kRed-7)
    h_17B.SetLineColor(ROOT.kYellow+1)
    h_17C.SetLineColor(ROOT.kBlue+1)
    h_17D.SetLineColor(ROOT.kCyan+1)
    h_17A.Draw("hist")
    h_17B.Draw("hist same")
    h_17C.Draw("hist same")
    h_17D.Draw("hist same")
    h_1718.Draw("p same")
    

    logname = ""
    if log:
      logname = "_log"
    if hnames[2] == 'Ch0':
      h_17.SetTitle("#mu+jets")
      h_17.SetTitleSize(0.7)
    elif hnames[2] == 'Ch1':
      h_17.SetTitle("e+jets")
      h_17.SetTitleSize(0.7)
    else:
      h_17.SetTitle("#mu+e")
      h_17.SetTitleSize(0.7)

    if each_plot:
      if ( hnames[3] == "S0" or hnames[3] == "S9" ) and not( hnames[1] == "csv" or hnames[1] == "cvsl" or hnames[1] == "cvsb" ) and not( "noSF" in hnames[1] or "EventWeight" in hnames[1] ):
        if mode != 2: c.Print(data17[data17.keys()[mode]]["hname"][i]+logname+".pdf")
        else: c.Print(data17[data17.keys()[0]]["hname"][i]+logname+".pdf")
        ##h_17.SetTitle(hnames[2]+"_"+hnames[3])

    filename = "result_ratio"+logname+".pdf"
    if i == 0 and N_17 > 1:
      c.Print( (filename+"(") )
#    elif i > 0 and i == N_17-1:
    elif i > 0 and count == N_17_noReco-1:
      c.Print( (filename+")") ) 
    else:
      c.Print(filename)
    
    c.Clear()

  ratioplot()
  count += 1


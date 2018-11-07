#! /usr/bin/env python

from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1F, TH2F, THStack, TLegend, gROOT, TPad
gROOT.SetBatch(True)
import ROOT
import os

from style import *

log = False
each_plot = False
onlyelmu = True

from collections import OrderedDict
datasamples=OrderedDict()
bkgsamples=OrderedDict()
hctsamples=OrderedDict()
hutsamples=OrderedDict()
sthctsamples=OrderedDict()
sthutsamples=OrderedDict()

def SetData(fname, name, lumi):
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
  datasamples[fname] = tmp
 
def AddBkg(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  nevt = 1
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["total"] = nevt 
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  bkgsamples[fname] = tmp

def AddHct(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  nevt = 1
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["total"] = nevt
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  hctsamples[fname] = tmp

def AddHut(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  nevt = 1
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["total"] = nevt
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  hutsamples[fname] = tmp

def AddSTHct(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  nevt = 1
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["total"] = nevt
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  sthctsamples[fname] = tmp

def AddSTHut(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  nevt = 1
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)
  tmp["hname"].remove("ScaleWeights")
  tmp["hname"].remove("PSWeights")
  tmp["total"] = nevt
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  sthutsamples[fname] = tmp

####Users should provide these information 
SetData("hist_SingleMuonRun2017.root","data", 41529)
SetData("hist_SingleElectronRun2017.root","data", 41529)
AddBkg("hist_TTpowhegttbb.root","ttbb",ROOT.kRed+4, 365.34)#(831.76)*2[(0.1086*3)*(0.6741)]
AddBkg("hist_TTpowhegttbj.root","ttbj",ROOT.kRed+3, 365.34)
AddBkg("hist_TTpowhegttcc.root","ttcc",ROOT.kRed+2, 365.34)
AddBkg("hist_TTpowhegttlf.root","ttLF",ROOT.kRed, 365.34)
AddBkg("hist_TTpowhegttother.root","ttBkg",ROOT.kRed-7, 365.34)
AddBkg("hist_TTHadpowheg.root","ttBkg",ROOT.kRed-7, 377.96)
AddBkg("hist_TTLLpowheg.root","ttBkg",ROOT.kRed-7, 88.29)
AddBkg("hist_ttHTobb.root","ttV",ROOT.kMagenta-7, 0.2934)
AddBkg("hist_ttHToNonbb.root","ttV",ROOT.kMagenta-7, 0.2151)
AddBkg("hist_TTWJetsToLNuPSweight.root","ttV",ROOT.kMagenta-7, 0.2043)
AddBkg("hist_TTWJetsToQQ.root","ttV",ROOT.kMagenta-7, 0.4062)
AddBkg("hist_TTZToLLNuNu.root","ttV",ROOT.kMagenta-7, 0.2529)
AddBkg("hist_TTZToQQ.root","ttV",ROOT.kMagenta-7, 0.5297)
AddBkg("hist_W1JetsToLNu.root","WJets",ROOT.kYellow, 9625) 
AddBkg("hist_W2JetsToLNu.root","WJets",ROOT.kYellow, 2793)
AddBkg("hist_W3JetsToLNu.root","WJets",ROOT.kYellow, 992.5)
AddBkg("hist_W4JetsToLNu.root","WJets",ROOT.kYellow, 544.3)
AddBkg("hist_DYJetsv2.root","ZJets",ROOT.kBlue, 5765.4)
AddBkg("hist_DYJets10to50.root","ZJets",ROOT.kBlue, 18610.0)
AddBkg("hist_SingleTops.root","Single t",6, 11.36)
AddBkg("hist_SingleTopt.root","Single t",6, 136.02)
AddBkg("hist_SingleTbart.root","Single t",6, 80.95)
AddBkg("hist_SingleToptW.root","Single t",6, 35.85)
AddBkg("hist_SingleTbartW.root","Single t",6, 35.85)
AddBkg("hist_WW.root","VV",ROOT.kCyan, 118.7)
AddBkg("hist_WZ.root","VV",ROOT.kCyan, 47.13)
AddBkg("hist_ZZ.root","VV",ROOT.kCyan, 16.523)
AddHct("hist_TTTH1L3BHct.root", "Hct", 433, 1.86)
AddHut("hist_TTTH1L3BHut.root", "Hut", 401, 1.86)
#AddHct("hist_TTTH1L3BaTLepHct.root", "Hct", 433, 1.86)
#AddHct("hist_TTTH1L3BTLepHct.root", "Hct", 433, 1.86) 
#AddHut("hist_TTTH1L3BaTLepHut.root", "Hut", 401, 1.86)
#AddHut("hist_TTTH1L3BTLepHut.root", "Hut", 401, 1.86)
AddSTHct("hist_STTH1L3BHct.root", "STHct", 435, 0.076)#1.9*0.04
AddSTHut("hist_STTH1L3BHut.root", "STHut", 403, 0.55)#13.84*0.04

N_hist = len(datasamples[datasamples.keys()[0]]["hname"])
N_bkgsamples = len(bkgsamples)
N_Hctsamples = len(hctsamples)
N_Hutsamples = len(hutsamples)
N_stHctsamples = len(sthctsamples)
N_stHutsamples = len(sthutsamples)

fNevt = open("Nevt_ratio.txt",'w')

for i in range(0, N_hist):
  if "Ch0" in datasamples[datasamples.keys()[0]]["hname"][i]:   mode = 0
  elif "Ch1" in datasamples[datasamples.keys()[0]]["hname"][i]: mode = 1
  elif "Ch2" in datasamples[datasamples.keys()[0]]["hname"][i]: mode = 2
  if onlyelmu:
    if mode == 0 or mode == 1: continue

  string_fname = ''
  string_nevt =  ''
  hnames = datasamples[datasamples.keys()[0]]["hname"][i].split("_")
  printHistName = "NJet"
  if hnames[1] == printHistName:
    print hnames[2] + "_" + hnames[3]
    string_fname += "%s \n" %hnames
    string_nevt += "%s \n" %hnames

  ##if hnames[1] == printHistName :
  #print hnames[1], " ", hnames[2], " ", hnames[3]  

  hs = THStack()
  #l = TLegend(0.30, 0.99 - 0.8 * N_bkgsamples / 20., 0.89, 0.85)
  l = TLegend(0.15,0.71,0.89,0.87)
  l.SetNColumns(4);
  l.SetTextSize(0.05);
  l.SetLineColor(0);
  l.SetFillColor(0);

  if   mode == 0 or mode == 1: h_data = datasamples[datasamples.keys()[mode]]["file"].Get(datasamples[datasamples.keys()[mode]]["hname"][i])
  elif mode == 2 :
    h_data = datasamples[datasamples.keys()[0]]["file"].Get(datasamples[datasamples.keys()[0]]["hname"][i])
    h_data.Add(datasamples[datasamples.keys()[1]]["file"].Get(datasamples[datasamples.keys()[1]]["hname"][i]))
  nbins = h_data.GetNbinsX()
  h_data.AddBinContent( nbins, h_data.GetBinContent( nbins+1 ) )  #overflow

  h_sub = h_data.Clone("h_sub")

  ntotalbkg = 0
  k = 0
  for fname in bkgsamples.keys():
    h_tmp = bkgsamples[fname]["file"].Get(datasamples[datasamples.keys()[0]]["hname"][i])
    nbins = h_tmp.GetNbinsX()
    h_tmp.AddBinContent( nbins, h_tmp.GetBinContent( nbins+1 ) ) #overflow
    h_tmp.SetFillColor(bkgsamples[fname]["col"])
    ## normalization
    if mode != 2: scale = datasamples[datasamples.keys()[mode]]["lumi"]/(bkgsamples[fname]["total"]/bkgsamples[fname]["xsection"])
    else: scale = datasamples[datasamples.keys()[0]]["lumi"]/(bkgsamples[fname]["total"]/bkgsamples[fname]["xsection"])

    h_tmp.Scale(scale)

    ## check if the sample is the same as previous process. 
    if k < N_bkgsamples-1 :
      post_name = bkgsamples.keys()[k+1]
      if bkgsamples[fname]["name"] is bkgsamples[post_name]["name"]:
        h_tmp.SetLineColor(bkgsamples[fname]["col"]) 
      else:
        l.AddEntry(h_tmp, bkgsamples[fname]["name"]  ,"F") 
    else: 
      l.AddEntry(h_tmp, bkgsamples[fname]["name"]  ,"F")

    ## print out number of events
    numevt = h_tmp.Integral()
    rawevt = h_tmp.GetEntries()
    ntotalbkg = ntotalbkg + numevt
    if hnames[1] == printHistName:
      string_nevt += "%f \n"%(numevt)
      string_fname += "%s :  %s = %f \n"%(fname,bkgsamples[fname]["name"],numevt)
      print fname, " : ", bkgsamples[fname]["name"], " = ", "{0:.5g}".format(numevt), " scale : " ,"{0:.3g}".format(scale)
    #print scale

    ## Add to Stack
    hs.Add( h_tmp ) #hh_tmp -> add h tmp sig, hs->other
    k = k+1

  h_bkg = hs.GetStack().Last()

  #Sig Stack 
  hsHct = THStack()
  ntotalHct = 0
  m = 0

  for fname in hctsamples.keys():
    h_Hct = hctsamples[fname]["file"].Get(datasamples[datasamples.keys()[0]]["hname"][i])
    nbins = h_Hct.GetNbinsX()
    h_Hct.AddBinContent( nbins, h_Hct.GetBinContent( nbins+1 ) ) 
    h_Hct.SetLineColor(hctsamples[fname]["col"])

    ## normalization
    if mode != 2: scale = datasamples[datasamples.keys()[mode]]["lumi"]/(hctsamples[fname]["total"]/hctsamples[fname]["xsection"])
    else: scale = datasamples[datasamples.keys()[0]]["lumi"]/(hctsamples[fname]["total"]/hctsamples[fname]["xsection"])
    h_Hct.Scale(scale)

    ## check if the sample is the same as previous process. 
    if m < N_Hctsamples-1 :
      post_name = hctsamples.keys()[m+1]
      if hctsamples[fname]["name"] is hctsamples[post_name]["name"]:
        h_Hct.SetLineColor(hctsamples[fname]["col"]) 
      else:
        l.AddEntry(h_Hct, hctsamples[fname]["name"]  ,"F")
    else: 
      l.AddEntry(h_Hct, hctsamples[fname]["name"]  ,"F")

    ## print out number of events
    numevt = h_Hct.Integral()
    rawevt = h_Hct.GetEntries()
    if hnames[1] == printHistName:
      string_nevt += "%f \n"%(numevt)
      string_fname += "%s :  %s = %f \n"%(fname,hctsamples[fname]["name"],numevt)
      print fname, " : ", hctsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.3g}".format(scale)
    #print scale

    ## Add to Stack
    hsHct.Add( h_Hct ) #hh_tmp -> add h tmp sig, hs->other
    m = m+1 

  hs_Hct = hsHct.GetStack().Last()
  if h_data.Integral > 0 and hs_Hct.Integral() > 0 and h_bkg.Integral() != 0: hs_Hct.Scale(h_data.Integral()/hs_Hct.Integral())


  #Add Hut
  hsHut = THStack()
  ntotalHut = 0
  n = 0

  for fname in hutsamples.keys():
    h_Hut = hutsamples[fname]["file"].Get(datasamples[datasamples.keys()[0]]["hname"][i])
    nbins = h_Hut.GetNbinsX()
    h_Hut.AddBinContent( nbins, h_Hut.GetBinContent( nbins+1 ) ) 
    h_Hut.SetLineColor(hutsamples[fname]["col"])

    ## normalization
    if mode != 2: scale = datasamples[datasamples.keys()[mode]]["lumi"]/(hutsamples[fname]["total"]/hutsamples[fname]["xsection"])
    else: scale = datasamples[datasamples.keys()[0]]["lumi"]/(hutsamples[fname]["total"]/hutsamples[fname]["xsection"])
    h_Hut.Scale(scale)

    ## check if the sample is the same as previous process. 
    if n < N_Hutsamples-1 :
      post_name = hutsamples.keys()[n+1]
      if hutsamples[fname]["name"] is hutsamples[post_name]["name"]:
        h_Hut.SetLineColor(hutsamples[fname]["col"])
      else:
        l.AddEntry(h_Hut, hutsamples[fname]["name"]  ,"F")
    else:
      l.AddEntry(h_Hut, hutsamples[fname]["name"]  ,"F")

    ## print out number of events
    numevt = h_Hut.Integral()
    rawevt = h_Hut.GetEntries()
    if hnames[1] == printHistName:
      string_nevt += "%f \n"%(numevt)
      string_fname += "%s :  %s = %f \n"%(fname,hutsamples[fname]["name"],numevt)
      print fname, " : ", hutsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.3g}".format(scale)
    #print scale

    ## Add to Stack
    hsHut.Add( h_Hut ) #hh_tmp -> add h tmp sig, hs->other
    n = n+1

  hs_Hut = hsHut.GetStack().Last()
  if h_data.Integral > 0 and hs_Hut.Integral() > 0 and h_bkg.Integral() != 0: hs_Hut.Scale(h_data.Integral()/hs_Hut.Integral())

  #Add singletop Hct
  #hsSTHct = THStack()
  ntotalSTHct = 0
  stm = 0

  for fname in sthctsamples.keys():
    h_stHct = sthctsamples[fname]["file"].Get(datasamples[datasamples.keys()[0]]["hname"][i])
    nbins = h_stHct.GetNbinsX()
    h_stHct.AddBinContent( nbins, h_stHct.GetBinContent( nbins+1 ) )
    h_stHct.SetLineColor(sthctsamples[fname]["col"])

    ## normalization
    if mode != 2: scale = datasamples[datasamples.keys()[mode]]["lumi"]/(sthctsamples[fname]["total"]/sthctsamples[fname]["xsection"])
    else: scale = datasamples[datasamples.keys()[0]]["lumi"]/(sthctsamples[fname]["total"]/sthctsamples[fname]["xsection"])
    h_stHct.Scale(scale)

    ## check if the sample is the same as previous process. 
    if stm < N_stHctsamples-1 :
      post_name = sthctsamples.keys()[m+1]
      if sthctsamples[fname]["name"] is sthctsamples[post_name]["name"]:
        h_stHct.SetLineColor(sthctsamples[fname]["col"])
      else:
        l.AddEntry(h_stHct, sthctsamples[fname]["name"]  ,"F")
    else:
      l.AddEntry(h_stHct, sthctsamples[fname]["name"]  ,"F")

    ## print out number of events
    numevt = h_stHct.Integral()
    rawevt = h_stHct.GetEntries()
    if hnames[1] == printHistName:
      string_nevt += "%f \n"%(numevt)
      string_fname += "%s :  %s = %f \n"%(fname,sthctsamples[fname]["name"],numevt)
      print fname, " : ", sthctsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.3g}".format(scale)
    #print scale

    ## Add to Stack
    #hsSTHct.Add( h_stHct ) #hh_tmp -> add h tmp sig, hs->other
    stm = stm+1

  #hs_stHct = hsSTHct.GetStack().Last()
  hs_stHct = h_stHct.Clone("hs_shHct")
  if h_data.Integral > 0 and hs_stHct.Integral() > 0 and h_bkg.Integral() != 0: hs_stHct.Scale(h_data.Integral()/hs_stHct.Integral())

  #Add singletop Hut
  #hsSTHut = THStack()
  ntotalSTHut = 0
  stn = 0

  for fname in sthutsamples.keys():
    h_stHut = sthutsamples[fname]["file"].Get(datasamples[datasamples.keys()[0]]["hname"][i])
    nbins = h_stHut.GetNbinsX()
    h_stHut.AddBinContent( nbins, h_stHut.GetBinContent( nbins+1 ) )
    h_stHut.SetLineColor(sthutsamples[fname]["col"])

    ## normalization
    if mode != 2: scale = datasamples[datasamples.keys()[mode]]["lumi"]/(sthutsamples[fname]["total"]/sthutsamples[fname]["xsection"])
    else: scale = datasamples[datasamples.keys()[0]]["lumi"]/(sthutsamples[fname]["total"]/sthutsamples[fname]["xsection"])
    h_stHut.Scale(scale)

    ## check if the sample is the same as previous process. 
    if stn < N_stHutsamples-1 :
      post_name = sthutsamples.keys()[n+1]
      if sthutsamples[fname]["name"] is sthutsamples[post_name]["name"]:
        h_stHut.SetLineColor(sthutsamples[fname]["col"])
      else:
        l.AddEntry(h_stHut, sthutsamples[fname]["name"]  ,"F")
    else:
      l.AddEntry(h_stHut, sthutsamples[fname]["name"]  ,"F")

    ## print out number of events
    numevt = h_stHut.Integral()
    rawevt = h_stHut.GetEntries()
    if hnames[1] == printHistName:
      string_nevt += "%f \n"%(numevt)
      string_fname += "%s :  %s = %f \n"%(fname,sthutsamples[fname]["name"],numevt)
      print fname, " : ", sthutsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.3g}".format(scale)
    #print scale

    ## Add to Stack
    #hsSTHut.Add( h_stHut ) #hh_tmp -> add h tmp sig, hs->other
    stn = stn+1

  #hs_stHut = hsSTHut.GetStack().Last()
  hs_stHut = h_stHut.Clone("hs_shHut")
  if h_data.Integral > 0 and hs_stHut.Integral() > 0 and h_bkg.Integral() != 0: hs_stHut.Scale(h_data.Integral()/hs_stHut.Integral())

  ndata= h_data.Integral()
  nsub = ndata-ntotalbkg
  if hnames[1] == printHistName:
    string_nevt += "%f \n" % ntotalbkg
    string_nevt += "%d \n" % ndata
    string_nevt += "%f \n" % nsub
    string_fname += "ntotal = %f \n" % ntotalbkg
    string_fname += "ndata = %d \n" % ndata
    string_fname += "nsub = %f \n" % nsub
    fNevt.write(string_nevt)
    fNevt.write(string_fname)
    print "ntotal = " , "{0:.6g}".format(ntotalbkg)
    print "ndata = " , "{0:.0f}".format(ndata)
    print "nsub = ", "{0:.6g}".format(nsub)

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
    h3.SetMarkerSize(0.5)
    h3.SetTitle("")
    h3.SetMinimum(0.6)
    h3.SetMaximum(1.4)
    #h3.Sumw2()
    h3.SetStats(0)

    y = h3.GetYaxis()
    y.SetTitle("Data/MC")
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

    return h3

  def ratioplot():
    h3 = createRatio(h_bkg, h_data)
    c, pad1, pad2 = createCanvasPads()

    #Draw each plot
    pad1.cd()
    if log:
      pad1.SetLogy()
    h_data.SetMarkerStyle(20)
    h_data.SetMarkerSize(0.5)
    max_data = h_data.GetMaximum()
    max_hs = hs.GetMaximum()
    sigmaximums = [hs_Hct.GetMaximum(), hs_Hut.GetMaximum(), hs_stHct.GetMaximum(), hs_stHut.GetMaximum()]
    #sigmaximums = [hs_stHct.GetMaximum(), hs_stHut.GetMaximum()]
    max_sig = max(sigmaximums)
    maxfrac = 0.5
    if(max_hs > max_sig):
      if log :
        if max_data > 100000:
          maxfrac = 1000
        else:
          maxfrac = 100
      if max_hs > max_data :
        h_data.SetMaximum(max_hs+max_hs*maxfrac)
      else:
        h_data.SetMaximum(max_data+max_data*maxfrac)
    else:
      if log:
        maxfrac = 100
        h_data.SetMaximum(max_sig+max_sig*maxfrac)
      else:
        h_data.SetMaximum(max_sig*1.5)
    if log: h_data.SetMinimum(0.5)
    h_data.Draw("p")
    h_data.SetTitle("")
    h_data.GetYaxis().SetTitle("Events")
    h_data.GetYaxis().SetTitleOffset(1.2)
    h_data.GetYaxis().SetTitleSize(0.045)
    h_data.GetXaxis().SetLabelSize(0)#
    #h_data.GetXaxis().SetTitle("")
    h_data.GetXaxis().SetTitleOffset(5.0)
    hs.Draw("histsame")
    hs_Hct.SetLineWidth(2)
    hs_Hut.SetLineWidth(2)
    hs_stHct.SetLineWidth(2)
    hs_stHut.SetLineWidth(2)
    hs_Hct.Draw("hist same")
    hs_Hut.Draw("hist same")
    hs_stHct.Draw("hist same")
    hs_stHut.Draw("hist same")
    h_data.Draw("psame")
    h_data.Draw("AXIS P SAME")

    l.AddEntry(h_data,"Data","P")
    l.Draw()
    label = TPaveText()
    label.SetX1NDC(gStyle.GetPadLeftMargin())
    label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
    label.SetX2NDC(1.0-gStyle.GetPadRightMargin()+0.03)
    label.SetY2NDC(1.0)
    label.SetTextFont(62)
    label.AddText("Work in Progress        CMS, 41.5 fb^{-1} at #sqrt{s} = 13 TeV")
    label.SetFillStyle(0)
    label.SetBorderSize(0)
    label.SetTextSize(0.05)
    label.SetTextAlign(32)
    label.Draw("same")
    
    pad2.cd()
    h3.Draw("ep")

    logname = ""
    if log:
      logname = "_log"
    """
    if hnames[2] == 'Ch0':
      h_data.SetTitle("#mu ch")
      h_data.SetTitleSize(0.7)
    else:
      h_data.SetTitle("e ch")
      h_data.SetTitleSize(0.7)
    """

    if each_plot:
      if mode != 2: c.Print(datasamples[datasamples.keys()[mode]]["hname"][i]+logname+".pdf")
      else: c.Print(datasamples[datasamples.keys()[0]]["hname"][i]+logname+".pdf")
      ##h_data.SetTitle(hnames[2]+"_"+hnames[3])

    filename = "result_ratio"+logname+".pdf"
    if onlyelmu: nhist = 2*N_hist/3
    else: nhist = 0
    if i == nhist and N_hist > 1:
      c.Print( (filename+"(") )
    elif i > 0 and i == N_hist-1:
      c.Print( (filename+")") ) 
    else:
      c.Print(filename)
    
    c.Clear()

  ratioplot()

fNevt.close()

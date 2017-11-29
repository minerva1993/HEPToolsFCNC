#! /usr/bin/env python

from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1F, TH2F, THStack, TLegend, gROOT 
import ROOT
import os

from style import *

QCDestimate = False
log = False 

from collections import OrderedDict
datasamples=OrderedDict()
bkgsamples=OrderedDict()
hctsamples=OrderedDict()
hutsamples=OrderedDict()

def SetData(fname, name, lumi):
  tmp = {}
  f = TFile(fname)
  #fnames = f.GetName().split('.')
  #fname = fnames[0]
  fname = os.path.basename(fname)[:-5]

  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  tmp["hname"].remove("EventInfo")
  tmp["lumi"] = lumi 
  tmp["name"] = name
  datasamples[fname] = tmp
 
def AddBkg(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  #fnames = f.GetName().split('.')
  #fname = fnames[0]

  nevt = 1
 
  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)

  tmp["total"] = nevt 
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  bkgsamples[fname] = tmp

def AddHct(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  #fnames = f.GetName().split('.')
  #fname = fnames[0]

  nevt = 1

  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)

  tmp["total"] = nevt
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  hctsamples[fname] = tmp

def AddHut(fname, name, color, xsection):
  tmp = {}
  f = TFile(fname)
  fname = os.path.basename(fname)[:-5]
  #fnames = f.GetName().split('.')
  #fname = fnames[0]

  nevt = 1

  tmp["file"] = f
  tmp["hname"] = [x.GetName() for x in f.GetListOfKeys()]
  if xsection is not 1:
    tmp["hname"].remove("EventInfo")
    h = f.Get("EventInfo")
    nevt = h.GetBinContent(2)

  tmp["total"] = nevt
  tmp["col"] = color
  tmp["xsection"] = xsection
  tmp["name"] = name
  hutsamples[fname] = tmp

####Users should provide these information 
SetData("hist_DataSingleMu.root","data", 35922) #35867
SetData("hist_DataSingleEG.root","data", 35922) # for now, combination of muon and electron
AddBkg("hist_ttbb.root","ttbb",ROOT.kRed+4, 365.3*1.25)#(831.76)*2[(0.1086*3)*(0.6741)]
AddBkg("hist_ttbj.root","ttbj",ROOT.kRed+3, 365.3)
AddBkg("hist_ttcc.root","ttcc",ROOT.kRed+2, 365.3)
AddBkg("hist_ttLF.root","ttLF",ROOT.kRed, 365.3)
AddBkg("hist_tt.root","ttLF",ROOT.kRed, 365.3)
#AddBkg("hist_wjets.root","WJets",ROOT.kYellow,61524)
AddBkg("hist_wjets2.root","WJets",ROOT.kYellow,61526.7)
AddBkg("hist_zjets.root","ZJets",ROOT.kBlue, 6025.2)
AddBkg("hist_zjets10to50.root","ZJets",ROOT.kBlue, 18610.0)
AddBkg("hist_tchannel.root","Single t",6, 136.02)#44.33 = 136.02*(0.1086*3)
AddBkg("hist_tbarchannel.root","Single t",6, 80.95)#26.38 = 80.95*(0.1086*3)
AddBkg("hist_tWchannel.root","Single t",6, 35.85)#35.85?, 
AddBkg("hist_tbarWchannel.root","Single t",6, 35.85)
AddBkg("hist_ww.root","DiBoson",ROOT.kCyan, 118.7)
AddBkg("hist_wz.root","DiBoson",ROOT.kCyan, 47.13)
AddBkg("hist_zz.root","DiBoson",ROOT.kCyan, 16.523)
#AddBkg("hist_QCD.root","QCD",ROOT.kGray, 1000)
"""
AddBkg("hist_QCD_EGEnr_20to30.root","QCD",ROOT.kGray, 5352960)
AddBkg("hist_QCD_EGEnr_30to50.root","QCD",ROOT.kGray, 9928000)
AddBkg("hist_QCD_EGEnr_50to80.root","QCD",ROOT.kGray, 2890800)
AddBkg("hist_QCD_EGEnr_80to120.root","QCD",ROOT.kGray, 350000)
AddBkg("hist_QCD_EGEnr_120to170.root","QCD",ROOT.kGray, 62964)
AddBkg("hist_QCD_EGEnr_170to300.root","QCD",ROOT.kGray, 18810)
AddBkg("hist_QCD_EGEnr_300toInf.root","QCD",ROOT.kGray, 1350)
AddBkg("hist_QCD_MuEnr_20to30.root","QCD",ROOT.kGray, 2960198.4)
AddBkg("hist_QCD_MuEnr_30to50.root","QCD",ROOT.kGray, 1652471.46)
AddBkg("hist_QCD_MuEnr_50to80.root","QCD",ROOT.kGray, 437504.1)
AddBkg("hist_QCD_MuEnr_80to120.root","QCD",ROOT.kGray, 106033.6648)
AddBkg("hist_QCD_MuEnr_120to170.root","QCD",ROOT.kGray, 25190.51514)
AddBkg("hist_QCD_MuEnr_170to300.root","QCD",ROOT.kGray, 8654.49315)
AddBkg("hist_QCD_MuEnr_300to470.root","QCD",ROOT.kGray, 797.35269)
AddBkg("hist_QCD_MuEnr_470to600.root","QCD",ROOT.kGray, 79.02553776)
AddBkg("hist_QCD_MuEnr_600to800.root","QCD",ROOT.kGray, 25.09505908)
AddBkg("hist_QCD_MuEnr_800to1000.root","QCD",ROOT.kGray, 4.707368272)
AddBkg("hist_QCD_MuEnr_1000toInf.root","QCD",ROOT.kGray, 1.62131692)
"""

AddHct("hist_Top_Hct.root", "Hct", 433, 3.70) #here, 36.98 * 0.1
#AddHct("hist_AntiTop_Hct.root", "Hct", 3, 3.33) # Top Hct ->xsection twice!
AddHut("hist_Top_Hut.root", "Hut", 401, 3.70)
#AddHut("hist_AntiTop_Hut.root", "Hut", 5, 4.57) #used 1610.04857 values
#### each signal cross section for t + tbar!

qcd = []

N_hist = len(datasamples[datasamples.keys()[0]]["hname"])
N_bkgsamples = len(bkgsamples)
N_Hctsamples = len(hctsamples)
N_Hutsamples = len(hutsamples)

fNevt = open("Nevt.txt",'w')

for i in range(0, N_hist):
  if "Ch0" in datasamples[datasamples.keys()[0]]["hname"][i]:
    mode = 0
  else:
    mode = 1 

  hnames = datasamples[datasamples.keys()[mode]]["hname"][i].split("_")
  string0 = "%s \n" %hnames
  fNevt.write(string0)

  #printHistName = "LepIsoQCD"
  printHistName = "NJet"

  ##if hnames[1] == printHistName :
  ##  print hnames[1], " ", hnames[2], " ", hnames[3]  

  hs = THStack()
  #l = TLegend(0.30, 0.99 - 0.8 * N_bkgsamples / 20., 0.89, 0.85)
  l = TLegend(0.15,0.71,0.89,0.87)
  l.SetNColumns(4);
  l.SetTextSize(0.04);
  l.SetLineColor(0);
  l.SetFillColor(0);

  h_data = datasamples[datasamples.keys()[mode]]["file"].Get(datasamples[datasamples.keys()[mode]]["hname"][i])
  nbins = h_data.GetNbinsX()
  h_data.AddBinContent( nbins, h_data.GetBinContent( nbins+1 ) )  #overflow

  h_sub = h_data.Clone("h_sub") 
  if QCDestimate : 
    h_sub.SetName(hnames[0]+"_"+hnames[1]+"_"+hnames[2]+"_"+hnames[3]+"_qcd")

  ntotalbkg = 0
  k = 0
  for fname in bkgsamples.keys():
    h_tmp = bkgsamples[fname]["file"].Get(bkgsamples[fname]["hname"][i])
    nbins = h_tmp.GetNbinsX()
    h_tmp.AddBinContent( nbins, h_tmp.GetBinContent( nbins+1 ) ) #overflow
    h_tmp.SetFillColor(bkgsamples[fname]["col"])
    ## normalization
    scale = 1.0
    #if bkgsamples[fname]["name"] == "QCD": 
      #scale = 1.0
    #else: 
    scale = datasamples[datasamples.keys()[mode]]["lumi"]/(bkgsamples[fname]["total"]/bkgsamples[fname]["xsection"])

    print fname
    print scale
    h_tmp.Scale(scale)

    if bkgsamples[fname]["name"] is not "QCD" and QCDestimate: 
      h_sub.Add(h_tmp, -1)
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
    if bkgsamples[fname]["name"] == "QCD": numqcd = numevt
    if hnames[1] == printHistName:
      string = "%s :  %s = %d \n"%(fname,bkgsamples[fname]["name"],numevt)
      fNevt.write(string)
      print fname, " : ", bkgsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.1g}".format(scale)  
    ## Add to Stack
    #if bkgsamples[fname]["name"] == "WJets":
    #  hs.Add( h_tmp, "E" ) #hh_tmp -> add h tmp sig, hs->other
    #else:
    #  hs.Add( h_tmp )
    hs.Add( h_tmp )
    k = k+1

#Sig Stack
  hsHct = THStack()

  ntotalHct = 0
  m = 0

  for fname in hctsamples.keys():
    h_Hct = hctsamples[fname]["file"].Get(hctsamples[fname]["hname"][i])
    nbins = h_Hct.GetNbinsX()
    #h_tmp.AddBinContent( nbins, h_tmp.GetBinContent( nbins+1 ) ) 
    h_Hct.SetLineColor(hctsamples[fname]["col"])
    h_Hct.SetFillColorAlpha(hctsamples[fname]["col"],0.0)
    ## normalization
    """
    scale = 1.0
    if hctsamples[fname]["name"] == "QCD": 
      scale = 1.0
    else: 
      scale = datasamples[datasamples.keys()[mode]]["lumi"]/(hctsamples[fname]["total"]/hctsamples[fname]["xsection"])
    """
    scale = 0.1

    #print fname
    #print scale
    h_Hct.Scale(scale)

    if hctsamples[fname]["name"] is not "QCD" and QCDestimate: 
      h_HctSub.Add(h_Hct, -1)
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
    if hctsamples[fname]["name"] == "QCD": numqcd = numevt
    if hnames[1] == printHistName:
      string = "%s :  %s = %d \n"%(fname,hctsamples[fname]["name"],numevt)
      fNevt.write(string)
      print fname, " : ", hctsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.1g}".format(scale)
 
    ## Add to Stack
    hsHct.Add( h_Hct ) #hh_tmp -> add h tmp sig, hs->other
    m = m+1 

  hs_Hct = hsHct.GetStack().Last()

#Add Hut
  hsHut = THStack()

  ntotalHut = 0
  n = 0

  for fname in hutsamples.keys():
    h_Hut = hutsamples[fname]["file"].Get(hutsamples[fname]["hname"][i])
    nbins = h_Hut.GetNbinsX()
    #h_tmp.AddBinContent( nbins, h_tmp.GetBinContent( nbins+1 ) ) 
    h_Hut.SetLineColor(hutsamples[fname]["col"])
    h_Hut.SetFillColorAlpha(hutsamples[fname]["col"],0.0)
    ## normalization
    """
    scale = 1.0
    if hutsamples[fname]["name"] == "QCD":
      scale = 1.0
    else:
      scale = datasamples[datasamples.keys()[mode]]["lumi"]/(hutsamples[fname]["total"]/hutsamples[fname]["xsection"])
    """
    scale = 0.1

    #print fname
    #print scale
    h_Hut.Scale(scale)

    if hutsamples[fname]["name"] is not "QCD" and QCDestimate:
      h_HutSub.Add(h_Hut, -1)
    ## check if the sample is the same as previous process. 
    if m < N_Hutsamples-1 :
      post_name = hutsamples.keys()[m+1]
      if hutsamples[fname]["name"] is hutsamples[post_name]["name"]:
        h_Hut.SetLineColor(hutsamples[fname]["col"])
      else:
        l.AddEntry(h_Hut, hutsamples[fname]["name"]  ,"F")
    else:
      l.AddEntry(h_Hut, hutsamples[fname]["name"]  ,"F")

    ## print out number of events
    numevt = h_Hut.Integral()
    rawevt = h_Hut.GetEntries()
    if hutsamples[fname]["name"] == "QCD": numqcd = numevt
    if hnames[1] == printHistName:
      string = "%s :  %s = %d \n"%(fname,hutsamples[fname]["name"],numevt)
      fNevt.write(string)
      print fname, " : ", hutsamples[fname]["name"], " = ", "{0:.5g}".format(numevt),  " scale : " ,"{0:.1g}".format(scale)

    ## Add to Stack
    hsHut.Add( h_Hut ) #hh_tmp -> add h tmp sig, hs->other
    n = n+1

  hs_Hut = hsHut.GetStack().Last()

  if QCDestimate:
    qcd.append(h_sub)

  c = TCanvas("c_"+"{}".format(i),"c",1)
  if log:
    c.SetLogy()

  h_data.SetMarkerStyle(20)
  h_data.SetMarkerSize(0.3)
  max_data = h_data.GetMaximum()
  max_hs = hs.GetMaximum()
  max_hct = h_Hct.GetMaximum()
  maxfrac = 0.5
  if(max_hs > max_hct):
    if log :
      if max_data > 100000:
        maxfrac = 1000 
      else:
        maxfrac = 100
    if max_hs > max_data :
      h_data.SetMaximum(max_hs+max_hs*maxfrac)
    else:
      h_data.SetMaximum(max_data+max_data*maxfrac)
  else: h_data.SetMaximum(max_hct*1.5)
  h_data.Draw("p")
  h_data.SetTitle("")
  h_data.GetYaxis().SetTitle("Entries")
  h_data.GetYaxis().SetTitleOffset(1.2)
  hs.Draw("histsame")
  h_data.Draw("psame")
  hs_Hct.Draw("hist same")
  hs_Hut.Draw("hist same")


  l.AddEntry(h_data,"Data","P")
  l.Draw()
  label = TPaveText()
  label.SetX1NDC(gStyle.GetPadLeftMargin())
  label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
  label.SetX2NDC(1.0-gStyle.GetPadRightMargin())
  label.SetY2NDC(1.0)
  label.SetTextFont(42)
  label.AddText("Work in progress, CMS, 35.9 fb^{-1} at #sqrt{s} = 13 TeV")
  label.SetFillStyle(0)
  label.SetBorderSize(0)
  label.SetTextSize(0.04)
  label.SetTextAlign(32)
  label.Draw("same")

  ndata = h_data.Integral()
  nsub = ndata-ntotalbkg
  if hnames[1] == printHistName:
    string1 = "ntotal = %d \n" % ntotalbkg
    string2 = "ndata = %d \n" % ndata
    string3 = "nsub = %d \n" % nsub
    #string4 = "ratio(qcd/total) = %f \n" % (numqcd/ntotalbkg)
    fNevt.write(string1)
    fNevt.write(string2)
    fNevt.write(string3)
    #fNevt.write(string4)
    print "ntotal = " , "{0:.6g}".format(ntotalbkg)
    print "ndata = " , "{0:.0f}".format(ndata)
    print "nsub = ", "{0:.6g}".format(nsub)

  logname = ""
  if log:
    logname = "_log"

  #c.Print(datasamples[datasamples.keys()[mode]]["hname"][i]+logname+".pdf")
  h_data.SetTitle(hnames[2]+"_"+hnames[3])
  filename = "result"+logname+".pdf"
  if i == 0 and N_hist > 1:
    c.Print( (filename+"(") )
  elif i > 0 and i == N_hist-1:
    c.Print( (filename+")") ) 
  else:
    c.Print(filename)

if QCDestimate :
 f = ROOT.TFile("hist_qcd.root", "recreate")
 f.cd()

 N_qcd = len(qcd)

 for i in range(0,N_qcd):
   qcd[i].Write()

 f.cd()
 f.Write()
 f.Close()
 fNevt.close()

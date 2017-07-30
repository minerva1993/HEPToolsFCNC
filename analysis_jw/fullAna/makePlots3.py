#! /usr/bin/env python

from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1F, TH2F, THStack, TLegend, gROOT , TPad
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
SetData("hist_DataSingleMu.root","data", 35867) # for now, combination of muon and electron
SetData("hist_DataSingleEG.root","data", 35867) # for now, combination of muon and electron
AddBkg("hist_ttbb.root","ttbb",ROOT.kRed+4, 365.4)
AddBkg("hist_ttbj.root","ttbj",ROOT.kRed+3, 365.4)
AddBkg("hist_ttcc.root","ttcc",ROOT.kRed+2, 365.4)
AddBkg("hist_ttLF.root","ttLF",ROOT.kRed, 365.4)
AddBkg("hist_tt.root","ttLF",ROOT.kRed, 356.4)
#AddBkg("hist_ttBkg.root","ttLF",ROOT.kRed, 831.8)
AddBkg("hist_wjets.root","WJets",ROOT.kYellow,61524)
AddBkg("hist_zjets.root","ZJets",ROOT.kBlue, 6025.2)
AddBkg("hist_zjets10to50.root","ZJets",ROOT.kBlue, 18610.0)
AddBkg("hist_tchannel.root","Single t",6, 44.33)
AddBkg("hist_tbarchannel.root","Single t",6, 26.38)
AddBkg("hist_tWchannel.root","Single t",6, 35.6)
AddBkg("hist_tbarWchannel.root","Single t",6, 35.6)
AddBkg("hist_ww.root","DiBoson",ROOT.kCyan, 118.7)
AddBkg("hist_wz.root","DiBoson",ROOT.kCyan, 47.13)
AddBkg("hist_zz.root","DiBoson",ROOT.kCyan, 16.523)
#AddBkg("../noniso/hist_qcd.root","QCD",ROOT.kGray, 1)

AddHct("hist_Top_Hct.root", "Hct", 602, 6.66)
#AddHct("hist_AntiTop_Hct.root", "Hct", 3, 3.33) # Top Hct ->xsection twice!
AddHut("hist_Top_Hut.root", "Hut", 419, 9.14)
#AddHut("hist_AntiTop_Hut.root", "Hut", 5, 4.57) #used 1610.04857 values
#### 

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
  #fNevt.write(string0)
  print string0

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
  #h_data.AddBinContent( nbins, h_data.GetBinContent( nbins+1 ) )  #overflow

  h_sub = h_data.Clone("h_sub") 
  if QCDestimate : 
    h_sub.SetName(hnames[0]+"_"+hnames[1]+"_"+hnames[2]+"_"+hnames[3]+"_qcd")

  ntotalbkg = 0
  k = 0
  for fname in bkgsamples.keys():
    h_tmp = bkgsamples[fname]["file"].Get(bkgsamples[fname]["hname"][i])
    nbins = h_tmp.GetNbinsX()
    #h_tmp.AddBinContent( nbins, h_tmp.GetBinContent( nbins+1 ) ) #overflow
    h_tmp.SetFillColor(bkgsamples[fname]["col"])
    ## normalization
    scale = 1.0
    if bkgsamples[fname]["name"] == "QCD": 
      scale = 1.0
    else: 
      scale = datasamples[datasamples.keys()[mode]]["lumi"]/(bkgsamples[fname]["total"]/bkgsamples[fname]["xsection"])

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
      print fname, " : ", bkgsamples[fname]["name"], " = ", "{0:.5g}".format(numevt) # " scale : " ,"{0:.1g}".format(scale)  

    ## Add to Stack
    hs.Add( h_tmp ) #hh_tmp -> add h tmp sig, hs->other
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
    scale = 1.0
    if hctsamples[fname]["name"] == "QCD": 
      scale = 1.0
    else: 
      scale = datasamples[datasamples.keys()[mode]]["lumi"]/(hctsamples[fname]["total"]/hctsamples[fname]["xsection"])

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
 
    ## Add to Stack
    hsHct.Add( h_Hct ) #hh_tmp -> add h tmp sig, hs->other
    m = m+1 

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
    scale = 1.0
    if hutsamples[fname]["name"] == "QCD":
      scale = 1.0
    else:
      scale = datasamples[datasamples.keys()[mode]]["lumi"]/(hutsamples[fname]["total"]/hutsamples[fname]["xsection"])

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

    ## Add to Stack
    hsHut.Add( h_Hut ) #hh_tmp -> add h tmp sig, hs->other
    n = n+1

  if QCDestimate:
    qcd.append(h_sub)

#creat canvas
  c = TCanvas("c_"+"{}".format(i),"c", 450, 450)
  if log:
    c.SetLogy()

  # Upper histogram plot is pad1
  pad1 = TPad("pad1", "pad1", 0.02, 0.20, 1, 1.0)
  #pad1.SetGridx()
  pad1.Draw()
  c.cd()  # returns to main canvas before defining pad2
  pad2 = TPad("pad2", "pad2", 0.02, 0.05, 1, 0.29)
  #pad2.SetBottomMargin(-0.5)
  #pad2.SetGridx()
  pad2.Draw()
"""
#creat ratio plot
  h3 = h_data.Clone("h3")
  #h3 = hs.GetStack().Last()
  ROOT.SetOwnership( h3,  True )
  h3.SetDirectory(0)
  h3.SetLineColor(1)
  h3.SetMarkerStyle(6)
  h3.SetTitle("")
  #h3.SetMinimum(0.8)
  #h3.SetMaximum(1.2)
  # Set up plot for markers and errors
  #h3.Sumw2()
  h3.SetStats(0)
  #h3.Divide(hs)

  # Adjust y-axis settings
  y = h3.GetYaxis()
  y.SetTitle("Data/MC")
  y.SetNdivisions(505)
  y.SetTitleSize(0.14)
  y.SetTitleOffset(0.3)
  y.SetLabelSize(0.1)

  # Adjust x-axis settings
  x = h3.GetXaxis()
  x.SetTitleSize(0.14)
  x.SetTitleOffset(0.8)
  x.SetLabelSize(0.1)
"""

  #Draw each plot...
  pad1.cd()
  h_data.SetMarkerStyle(20)
  h_data.SetMarkerSize(0.4)
  max_data = h_data.GetMaximum()
  max_hs = hs.GetMaximum()
  maxfrac = 0.5
  if log :
    if max_data > 100000:
      maxfrac = 1000 
    else:
      maxfrac = 100
  if max_hs > max_data :
    h_data.SetMaximum(max_hs+max_hs*maxfrac)
  else:
    h_data.SetMaximum(max_data+max_data*maxfrac)

  h_data.Draw("p")
  h_data.SetTitle("")
  h_data.GetYaxis().SetTitle("Entries")
  h_data.GetYaxis().SetTitleOffset(1.2)
  h_data.GetXaxis().SetLabelSize(0)
  #hs.GetXaxis().SetTitle("")
  hs.Draw("histsame")
  h_data.Draw("psame")
  hsHct.Draw("hist same")
  hsHut.Draw("hist same")

  l.AddEntry(h_data,"Data","P")
  l.Draw()
  label = TPaveText()
  label.SetX1NDC(gStyle.GetPadLeftMargin())
  label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
  label.SetX2NDC(1.0-gStyle.GetPadRightMargin())
  label.SetY2NDC(1.0)
  label.SetTextFont(42)
  label.AddText("CMS, 35.9 fb^{-1} at #sqrt{s} = 13 TeV")
  label.SetFillStyle(0)
  label.SetBorderSize(0)
  label.SetTextSize(0.04)
  label.SetTextAlign(32)
  label.Draw("same")
  h4 = hs.GetHistogram().Clone("h4")
  ROOT.SetOwnership( h4,  True )
  h4.SetDirectory(0)

  pad2.cd()
  #creat ratio plot
  h3 = h_data.Clone("h3")
  #h3 = hs.GetStack().Last()
  ROOT.SetOwnership( h3,  True )
  h3.SetDirectory(0)
  h3.SetLineColor(1)
  h3.SetMarkerStyle(6)
  h3.SetTitle("")
  #h3.SetMinimum(0.8)
  #h3.SetMaximum(1.2)
  # Set up plot for markers and errors
  #h3.Sumw2()
  h3.SetStats(0)
  #h3.Divide(hs)

  # Adjust y-axis settings
  y = h3.GetYaxis()
  y.SetTitle("Data/MC")
  y.SetNdivisions(505)
  y.SetTitleSize(0.14)
  y.SetTitleOffset(0.3)
  y.SetLabelSize(0.1)

  # Adjust x-axis settings
  x = h3.GetXaxis()
  x.SetTitleSize(0.14)
  x.SetTitleOffset(0.8)
  x.SetLabelSize(0.1)
  h3.Divide(h4)
  h3.Draw("ep")

  ndata= h_data.Integral()
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
  filename = "result_ratio"+logname+".pdf"

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

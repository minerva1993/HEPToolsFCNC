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
sigsamples=OrderedDict()

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
#### 

qcd = []

N_hist = len(datasamples[datasamples.keys()[0]]["hname"])
N_bkgsamples = len(bkgsamples)

fNevt = open("Nevt.txt",'w')

for i in range(0, N_hist):
  if "Ch0" in datasamples[datasamples.keys()[0]]["hname"][i]:
    mode = 0
  else:
    mode = 1 

  hnames = datasamples[datasamples.keys()[mode]]["hname"][i].split("_")

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
  h_data.AddBinContent( nbins, h_data.GetBinContent( nbins+1 ) )

  h_sub = h_data.Clone("h_sub") 
  if QCDestimate : 
    h_sub.SetName(hnames[0]+"_"+hnames[1]+"_"+hnames[2]+"_"+hnames[3]+"_qcd")

  ntotalbkg = 0
  k = 0
  for fname in bkgsamples.keys():
    h_tmp = bkgsamples[fname]["file"].Get(bkgsamples[fname]["hname"][i])
    nbins = h_tmp.GetNbinsX()
    h_tmp.AddBinContent( nbins, h_tmp.GetBinContent( nbins+1 ) ) 
    h_tmp.SetFillColor(bkgsamples[fname]["col"])
    ## normalization
    scale = 1.0
    scales_Ch0 = {'tWchannel' : '0.948619', 'tbarWchannel' : '0.948696', 'tbarchannel' : '0.986703',
	'tchannel' : '0.986703', 'tt' : '0.971293', 'ttBkg' : '0.843443', 'ttLF' : '0.961737',
	'ttbb' : '0.969366', 'ttbj' : '0.961783', 'ttcc' : '0.961522', 'wjets' : '0.988420',
	'ww' : '0.951653', 'wz' : '0.980968', 'zjets' : '1.014968', 'zjets10to50' : '0.986675',
	'zz' : '1.006082'}
    scales_Ch1 = {'tWchannel' : '1.012115', 'tbarWchannel' : '1.013038', 'tbarchannel' : '1.012247',
	'tchannel' : '1.013092', 'tt' : '1.034054', 'ttBkg' : '1.007072', 'ttLF' : '1.039719',
	'ttbb' : '1.090047', 'ttbj' : '1.059840', 'ttcc' : '1.049882', 'wjets' : '0.968534',
	'ww' : '0.966891', 'wz' : '0.991173', 'zjets' : '1.060363', 'zjets10to50' : '0.972615',
	'zz' : '1.039911'}
    hName = h_tmp.GetName()
    hProc = hName.split('_')[4]
    if bkgsamples[fname]["name"] != "QCD": 
      if "Ch_0" in hName : scale *= scales_Ch0[hProc]
      if "Ch_1" in hName : scale *= scales_Ch1[hProc]
      scale *= datasamples[datasamples.keys()[mode]]["lumi"]/(bkgsamples[fname]["total"]/bkgsamples[fname]["xsection"])
    else: 
      scaleQCD_Ch0 = [0.832826, 0.738475, 0.738475, 0.738475, 0.738475, 0.738475, 0.776416, 0.682633, 0.682633, 0.584223, 0.590185, 0.748903, 0.748903, -1.27058, 0.748903, 0.748903, 0.792743, 0.792743, 0.792743, 0.792743, 0.792743]
      scaleQCD_Ch1 = [2.19208, 2.15573, 2.14465, 0.0357605, 2.03599, 2.03599, 0.943209, 0.943209, 1.16091, 0.943209, 0.943209, 1.9208, 0.366675, 0.366675, 0.316396, 0.289221, 0.962235, 0.962235, 0.962235, 0.962235, 0.962235]
      if bkgsamples[fname]["name"] == "QCD":
          step = int(hName.split('_')[3][1:])
	  if "Ch0_" in hName: scale *= scaleQCD_Ch0[step]
	  #if scaleQCD_Ch0[step] > 0: scale *= scaleQCD_Ch0[step]
          #  else: scale *= 0.832826
	  if "Ch1_" in hName: scale *= scaleQCD_Ch1[step]
	  #if scaleQCD_Ch1[step] > 0: scale *= scaleQCD_Ch1[step]
          #  else: scale *= 0.832826
 
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
    hs.Add( h_tmp )
    k = k+1 

  if QCDestimate:
    qcd.append(h_sub)

  c = TCanvas("c_"+"{}".format(i),"c",1)
  if log:
    c.SetLogy()

  h_data.SetMarkerStyle(20)
  h_data.SetMarkerSize(0.3)
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
  hs.Draw("histsame")
  h_data.Draw("psame")

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

  c.Print(datasamples[datasamples.keys()[mode]]["hname"][i]+logname+".pdf")
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

#!/usr/bin/python

from ROOT import *
gROOT.SetBatch(True)

f1 = TFile.Open("hist_TTpowhegttlf.root","READ")
f2 = TFile.Open("hist_TTTH1L3BTLepHct.root","READ")

print "one lept/sum of weight"
print "ttLF 16 mu, el, all"
print (f1.Get("h_NJet_Ch0_S0").Integral())/f1.Get("EventInfo").GetBinContent(2), (f1.Get("h_NJet_Ch0_S1").Integral())/f1.Get("EventInfo").GetBinContent(2), (f1.Get("h_NJet_Ch0_S0").Integral()+f1.Get("h_NJet_Ch0_S1").Integral())/f1.Get("EventInfo").GetBinContent(2)
print "TLepHct 16 mu, el, all"
print (f2.Get("h_NJet_Ch0_S0").Integral())/f2.Get("EventInfo").GetBinContent(2), (f2.Get("h_NJet_Ch0_S1").Integral())/f2.Get("EventInfo").GetBinContent(2), (f2.Get("h_NJet_Ch0_S0").Integral()+f2.Get("h_NJet_Ch0_S1").Integral())/f2.Get("EventInfo").GetBinContent(2)

print "al jetcat /one lept"
print "ttLF 17 mu, el, all"
print (f1.Get("h_NJet_Ch0_S2").Integral()+f1.Get("h_NJet_Ch0_S3").Integral()+f1.Get("h_NJet_Ch0_S6").Integral()+f1.Get("h_NJet_Ch0_S7").Integral()+f1.Get("h_NJet_Ch0_S8").Integral())/f1.Get("h_NJet_Ch0_S0").Integral(), (f1.Get("h_NJet_Ch1_S2").Integral()+f1.Get("h_NJet_Ch1_S3").Integral()+f1.Get("h_NJet_Ch1_S6").Integral()+f1.Get("h_NJet_Ch1_S7").Integral()+f1.Get("h_NJet_Ch1_S8").Integral())/f1.Get("h_NJet_Ch1_S0").Integral(), (f1.Get("h_NJet_Ch0_S2").Integral()+f1.Get("h_NJet_Ch0_S3").Integral()+f1.Get("h_NJet_Ch0_S6").Integral()+f1.Get("h_NJet_Ch0_S7").Integral()+f1.Get("h_NJet_Ch0_S8").Integral()+f1.Get("h_NJet_Ch1_S2").Integral()+f1.Get("h_NJet_Ch1_S3").Integral()+f1.Get("h_NJet_Ch1_S6").Integral()+f1.Get("h_NJet_Ch1_S7").Integral()+f1.Get("h_NJet_Ch1_S8").Integral())/(f1.Get("h_NJet_Ch0_S0").Integral()+f1.Get("h_NJet_Ch1_S0").Integral())

print "TLepHct 17 mu, el, all"
print (f2.Get("h_NJet_Ch0_S2").Integral()+f2.Get("h_NJet_Ch0_S3").Integral()+f2.Get("h_NJet_Ch0_S6").Integral()+f2.Get("h_NJet_Ch0_S7").Integral()+f2.Get("h_NJet_Ch0_S8").Integral())/f2.Get("h_NJet_Ch0_S0").Integral(), (f2.Get("h_NJet_Ch1_S2").Integral()+f2.Get("h_NJet_Ch1_S3").Integral()+f2.Get("h_NJet_Ch1_S6").Integral()+f2.Get("h_NJet_Ch1_S7").Integral()+f2.Get("h_NJet_Ch1_S8").Integral())/f2.Get("h_NJet_Ch1_S0").Integral(), (f2.Get("h_NJet_Ch0_S2").Integral()+f2.Get("h_NJet_Ch0_S3").Integral()+f2.Get("h_NJet_Ch0_S6").Integral()+f2.Get("h_NJet_Ch0_S7").Integral()+f2.Get("h_NJet_Ch0_S8").Integral()+f2.Get("h_NJet_Ch1_S2").Integral()+f2.Get("h_NJet_Ch1_S3").Integral()+f2.Get("h_NJet_Ch1_S6").Integral()+f2.Get("h_NJet_Ch1_S7").Integral()+f2.Get("h_NJet_Ch1_S8").Integral())/(f2.Get("h_NJet_Ch0_S0").Integral()+f2.Get("h_NJet_Ch1_S0").Integral())

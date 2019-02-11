import os
from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1D, TH1F, THStack, TLegend, gROOT, TColor
import ROOT
from style import *

c1 = TCanvas( 'c1', 'c1', 450, 450 ) 
#c1.SetLogy()

label = TPaveText()
label.SetX1NDC(gStyle.GetPadLeftMargin())
label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
label.SetX2NDC(1.0-gStyle.GetPadRightMargin()+0.03)
label.SetY2NDC(1.0)
label.SetTextFont(62)
label.AddText("Work in Progress")
label.SetFillStyle(0)
label.SetBorderSize(0)
label.SetTextSize(0.035)
label.SetTextAlign(32)

label2 = TPaveText()
label2.SetX1NDC(gStyle.GetPadLeftMargin()+0.6)
label2.SetY1NDC(1.0-gStyle.GetPadTopMargin()-0.35)
label2.SetX2NDC(1.0-gStyle.GetPadRightMargin()-0.03)
label2.SetY2NDC(1.0)
label2.SetTextFont(62)
label2.SetFillStyle(0)
label2.SetBorderSize(0)
label2.SetTextSize(0.03)
label2.SetTextAlign(32)

l = TLegend(0.15,0.8,0.89,0.87)
l.SetNColumns(2);
l.SetTextSize(0.03);
l.SetLineColor(0);
l.SetFillColor(0);

##all histos with event selection!
###########################################################
for ver in ["deepRecoSTFCNC02",]:
  out = TFile.Open('/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/training/'+ver+'/output.root')
  testTree = out.Get(ver+'/TestTree')
  trainS = out.Get(ver+'/Method_BDT/BDT/MVA_BDT_Train_S')
  trainB = out.Get(ver+'/Method_BDT/BDT/MVA_BDT_Train_B')
  testS = out.Get(ver+'/Method_BDT/BDT/MVA_BDT_S')
  testB = out.Get(ver+'/Method_BDT/BDT/MVA_BDT_B')
  corrS = out.Get(ver+'/CorrelationMatrixS')
  corrB = out.Get(ver+'/CorrelationMatrixB')
  rocBDT = out.Get(ver+'/Method_BDT/BDT/MVA_BDT_rejBvsS')

  testS.SetStats(0)
  testS.SetTitle('')

  testS.GetXaxis().SetTitle('MVA score')
  testS.GetYaxis().SetTitle('A.U. (Normalized)')
  testB.SetFillStyle(3554)
  testS.SetFillColor(38)
  testB.SetFillColor(2)
  testS.SetLineColor(4)
  testB.SetLineColor(2)
  trainS.SetMarkerStyle(20)
  trainB.SetMarkerStyle(20)
  trainS.SetMarkerSize(0.7)
  trainB.SetMarkerSize(0.7)
  trainS.SetMarkerColor(4)
  trainB.SetMarkerColor(2)

  histoMax = [trainS.GetMaximum(), trainB.GetMaximum(), testS.GetMaximum(), testB.GetMaximum()]
  testS.SetMaximum(max(histoMax)*1.25)

  l.AddEntry(trainS, 'Signal (training)', 'p')
  l.AddEntry(trainB, 'Background (training)', 'p')
  l.AddEntry(testS, 'Signal (testing)', 'f')
  l.AddEntry(testB, 'Background (testing)', 'f')
  #label2.AddText(ch+', '+BDT)

  testS.Draw('hist')
  testB.Draw('hist same')
  trainS.Draw('ep same')
  trainB.Draw('ep same')
  label.Draw('same')
  label2.Draw('same')
  l.Draw('same')
  testS.Draw('axis same')
  c1.Print(ver+'_BDT_score.pdf')
  l.Clear()
  label2.Clear()
  c1.Clear()

  #Correlation part
  #axislabel = ["# of jets", "# of b jets", "LepW pT", "LepW #eta", "LepW #Delta#dphi", "LepW mass", "jet0 pT", "jet0 #eta", "jet0 mass", "jet0 CSVv2", "jet0 CvsL", "jet0 CvsB", "jet1 pT", "jet1 #eta", "jet1 mass", "jet1 CSVv2", "jet1 CvsL", "jet1 CvsB", "jet2 pT", "jet2 #eta", "jet2 mass", "jet2 CSV", "jet2 CvsL", "jet2 CvsB", "jet3 pT", "jet3 #eta", "jet3 mass", "jet3 CSVv2", "jet3 CvsL", "jet3 CvsB", "jet12 pT", "jet12 #eta", "jet12 #Delta#eta", "jet12 #Delta#dphi", "jet12 mass", "jet12 #Delta R", "jet23 pT", "jet23 #eta", "jet23 #Delta#eta", "jet23 #Delta#phi", "jet23 mass", "jet31 pT", "jet31 #eta", "jet31 #Delta#eta", "jet31 #Delta#phi", "jet31 mass", "LepT pT", "LepT #eta", "LepT #Delta#eta", "LepT #Delta#phi", "LepT mass", "HadT pT", "HadT #eta", "HadT_Hq #Delta#eta", "HadT_Wb #Delta#eta", "HadT_Hq #Delta#phi", "HadT_Wb #Delta#phi", "HadT mass"]#TTFCNC,old
  axislabel = ["jet0 pT", "jet0 #eta", "jet0 mass", "jet1 pT", "jet1 #eta", "jet1 mass", "jet2 pT", "jet2 #eta", "jet2 mass", "jet12 pT", "jet12 #eta", "jet12 #Delta#eta", "jet12 #Delta#dphi", "jet12 #Delta R", "jet12 mass", "LepW pT", "LepW #Delta#phi", "LepW mass", "LepT #Delta#phi", "LepT mass", ]

  corrS.SetTitle('')
  #for i in range(0,58):
  for i in range(0,20):
    corrS.GetXaxis().SetBinLabel(i+1, axislabel[i])
    corrS.GetXaxis().LabelsOption("v");
    corrS.GetYaxis().SetBinLabel(i+1, axislabel[i])
  corrS.GetXaxis().SetLabelSize(0.025)#0.02
  corrS.GetYaxis().SetLabelSize(0.025)
  corrS.GetZaxis().SetLabelSize(0.025)
  corrS.GetXaxis().SetLabelOffset(0.002)
  corrS.GetYaxis().SetLabelOffset(0.002)
  c1.SetGrid(1,1)
  corrS.Draw('colz')
  label.Draw('same')
  c1.Print(ver+'_corr_sig.pdf')
  c1.Clear()

  corrB.SetTitle('')
  #for i in range(0,58):
  for i in range(0,20):
    corrB.GetXaxis().SetBinLabel(i+1, axislabel[i])
    corrB.GetXaxis().LabelsOption("v");
    corrB.GetYaxis().SetBinLabel(i+1, axislabel[i])
  corrB.GetXaxis().SetLabelSize(0.025)#0.02
  corrB.GetYaxis().SetLabelSize(0.025)
  corrB.GetZaxis().SetLabelSize(0.025)
  corrB.GetXaxis().SetLabelOffset(0.002)
  corrB.GetYaxis().SetLabelOffset(0.002)
  c1.SetGrid(1,1)
  corrS.Draw('colz')
  label.Draw('same')
  c1.Print(ver+'_corr_bkg.pdf')
  c1.Clear()


  #ROC part
  aucBDT = str(round(rocBDT.Integral()/100, 3))

  l2 = TLegend(0.16,0.17,0.5,0.25)
  l2.SetTextSize(0.03);
  l2.SetLineColor(0);
  l2.SetFillColor(0);
  l2.AddEntry(rocBDT, 'BDT, AUC = '+aucBDT, 'l')

  rocBDT.SetTitle('')
  rocBDT.GetXaxis().SetTitle('Signal Efficiency')
  rocBDT.GetYaxis().SetTitle('Background Rejection')
  rocBDT.SetLineColor(1)
  rocBDT.SetLineWidth(3)
  c1.SetGrid(0,0)
  rocBDT.Draw('hist l')
  label.Draw('same')
  l2.Draw('same')
  c1.Print(ver+'_roc.pdf')

  c2 = TCanvas( 'c2', 'c2', 450, 450 )
  #Jet category
  h_j3b2_sig = TH1D("h_j3b2_sig","",40,-1,1)
  h_j3b3_sig = TH1D("h_j3b3_sig","",40,-1,1)
  h_j4b2_sig = TH1D("h_j4b2_sig","",40,-1,1)
  h_j4b3_sig = TH1D("h_j4b3_sig","",40,-1,1)
  j3b2_sig = testTree.Draw("BDT>>h_j3b2_sig","classID==0 && njets==3 && nbjets_m==2","norm Hist")
  j3b3_sig = testTree.Draw("BDT>>h_j3b3_sig","classID==0 && njets==3 && nbjets_m==3","norm same Hist")
  j4b2_sig = testTree.Draw("BDT>>h_j4b2_sig","classID==0 && njets==4 && nbjets_m==2","norm same Hist")
  j4b3_sig = testTree.Draw("BDT>>h_j4b3_sig","classID==0 && njets==4 && nbjets_m==3","norm same Hist")
  h_j3b2_sig.SetTitle('')
  h_j3b2_sig.GetXaxis().SetTitle('BDT Score (sig)')
  h_j3b2_sig.GetYaxis().SetTitle('A.U.')
  h_j3b2_sig.SetMaximum(h_j3b2_sig.GetMaximum()*1.5)
  h_j3b2_sig.SetLineColor(ROOT.kRed+1)
  h_j3b3_sig.SetLineColor(ROOT.kBlue+1)
  h_j4b2_sig.SetLineColor(ROOT.kCyan+1)
  h_j4b3_sig.SetLineColor(ROOT.kMagenta+1)
  h_j3b2_sig.SetLineWidth(2)
  h_j3b3_sig.SetLineWidth(2)
  h_j4b2_sig.SetLineWidth(2)
  h_j4b3_sig.SetLineWidth(2)
  l1 = TLegend(0.17, 0.7, 0.4, 0.85)
  l1.AddEntry(h_j3b2_sig, 'j3b2 signal', 'l')
  l1.AddEntry(h_j3b3_sig, 'j3b3 signal', 'l')
  l1.AddEntry(h_j4b2_sig, 'j4b2 signal', 'l')
  l1.AddEntry(h_j4b3_sig, 'j4b4 signal', 'l')
  l1.Draw('SAME')
  label.Draw('same')
  c2.Print('h_score_sig.pdf')

  #Jet category
  h_j3b2_bkg = TH1D("h_j3b2_bkg","",40,-1,1)
  h_j3b3_bkg = TH1D("h_j3b3_bkg","",40,-1,1)
  h_j4b2_bkg = TH1D("h_j4b2_bkg","",40,-1,1)
  h_j4b3_bkg = TH1D("h_j4b3_bkg","",40,-1,1)
  j3b2_bkg = testTree.Draw("BDT>>h_j3b2_bkg","classID==1 && njets==3 && nbjets_m==2","norm Hist")
  j3b3_bkg = testTree.Draw("BDT>>h_j3b3_bkg","classID==1 && njets==3 && nbjets_m==3","norm same Hist")
  j4b2_bkg = testTree.Draw("BDT>>h_j4b2_bkg","classID==1 && njets==4 && nbjets_m==2","norm same Hist")
  j4b3_bkg = testTree.Draw("BDT>>h_j4b3_bkg","classID==1 && njets==4 && nbjets_m==3","norm same Hist")
  h_j3b2_bkg.SetTitle('')
  h_j3b2_bkg.GetXaxis().SetTitle('BDT Score (bkg)')
  h_j3b2_bkg.GetYaxis().SetTitle('A.U.')
  h_j3b2_bkg.SetMaximum(h_j3b2_bkg.GetMaximum()*1.5)
  h_j3b2_bkg.SetLineColor(ROOT.kRed+1)
  h_j3b3_bkg.SetLineColor(ROOT.kBlue+1)
  h_j4b2_bkg.SetLineColor(ROOT.kCyan+1)
  h_j4b3_bkg.SetLineColor(ROOT.kMagenta+1)
  h_j3b2_bkg.SetLineWidth(2)
  h_j3b3_bkg.SetLineWidth(2)
  h_j4b2_bkg.SetLineWidth(2)
  h_j4b3_bkg.SetLineWidth(2)
  l3 = TLegend(0.6, 0.7, 0.67, 0.85)
  l3.AddEntry(h_j3b2_bkg, 'j3b2 background', 'l')
  l3.AddEntry(h_j3b3_bkg, 'j3b3 background', 'l')
  l3.AddEntry(h_j4b2_bkg, 'j4b2 background', 'l')
  l3.AddEntry(h_j4b3_bkg, 'j4b4 background', 'l')
  l3.Draw('SAME')
  label.Draw('same')
  c2.Print('h_score_bkg.pdf')

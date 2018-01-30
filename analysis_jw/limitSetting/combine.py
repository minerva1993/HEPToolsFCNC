from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1D, TH1F, THStack, TLegend, gROOT, TColor, TPaveText, TLine, TPolyLine
import ROOT
from style import *
from array import array

c1 = TCanvas("c1","c1",600,450)

label = TPaveText()
label.SetX1NDC(gStyle.GetPadLeftMargin())
label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
label.SetX2NDC(1.0-gStyle.GetPadRightMargin()+0.03)
label.SetY2NDC(1.0)
label.SetTextFont(62)
label.AddText("Work in Progress                      CMS, 35.9 fb^{-1} at #sqrt{s} = 13 TeV")
label.SetFillStyle(0)
label.SetBorderSize(0)
label.SetTextSize(0.04)
label.SetTextAlign(32)

xsec = 3.698

###########################################################
comb = TH1F("comb","",8,0,8)
comb.SetTitle('')
comb.GetXaxis().SetBinLabel(1, '#splitline{Hct}{Keras+TF}')
comb.GetXaxis().SetBinLabel(2, '#splitline{Hct}{BDT}')
comb.GetXaxis().SetBinLabel(3, '#splitline{#splitline{Hct}{Keras+TF}}{matched}')
comb.GetXaxis().SetBinLabel(4, '#splitline{#splitline{Hct}{BDT}}{matched}')
comb.GetXaxis().SetBinLabel(5, '#splitline{Hut}{Keras+TF}')
comb.GetXaxis().SetBinLabel(6, '#splitline{Hut}{BDT}')
comb.GetXaxis().SetBinLabel(7, '#splitline{#splitline{Hut}{Keras+TF}}{matched}')
comb.GetXaxis().SetBinLabel(8, '#splitline{#splitline{Hut}{BDT}}{matched}')
comb.GetYaxis().SetTitle('#sigma (95% CL) [pb]')
comb.GetYaxis().SetTitleSize(0.045)
comb.GetYaxis().SetTitleOffset(1.1)
comb.GetYaxis().SetRangeUser(0,1.3)
comb.Draw('')
label.Draw('same')

#sigma band
sigboxX1 = array('d',[0,1,1,0])
sigboxX2 = array('d',[1,2,2,1])
sigboxX3 = array('d',[2,3,3,2])
sigboxX4 = array('d',[3,4,4,3])
sigboxX5 = array('d',[4,5,5,4])
sigboxX6 = array('d',[5,6,6,5])
sigboxX7 = array('d',[6,7,7,6])
sigboxX8 = array('d',[7,8,8,7])
sig2boxY1 = array('d',[xsec*0.0734, xsec*0.0734, xsec*0.2518, xsec*0.2518])
sig2boxY2 = array('d',[xsec*0.0526, xsec*0.0526, xsec*0.1820, xsec*0.1820])
sig2boxY3 = array('d',[xsec*0.0640, xsec*0.0640, xsec*0.2196, xsec*0.2196])
sig2boxY4 = array('d',[xsec*0.0536, xsec*0.0536, xsec*0.1869, xsec*0.1869])
sig2boxY5 = array('d',[xsec*0.0787, xsec*0.0787, xsec*0.2694, xsec*0.2694])
sig2boxY6 = array('d',[xsec*0.0603, xsec*0.0603, xsec*0.2084, xsec*0.2084])
sig2boxY7 = array('d',[xsec*0.0758, xsec*0.0758, xsec*0.2585, xsec*0.2585])
sig2boxY8 = array('d',[xsec*0.0630, xsec*0.0630, xsec*0.2194, xsec*0.2194])
sig2box1 = TPolyLine(4, sigboxX1, sig2boxY1)
sig2box2 = TPolyLine(4, sigboxX2, sig2boxY2)
sig2box3 = TPolyLine(4, sigboxX3, sig2boxY3)
sig2box4 = TPolyLine(4, sigboxX4, sig2boxY4)
sig2box5 = TPolyLine(4, sigboxX5, sig2boxY5)
sig2box6 = TPolyLine(4, sigboxX6, sig2boxY6)
sig2box7 = TPolyLine(4, sigboxX7, sig2boxY7)
sig2box8 = TPolyLine(4, sigboxX8, sig2boxY8)
sig2box = [sig2box1, sig2box2, sig2box3, sig2box4, sig2box5, sig2box6, sig2box7, sig2box8]
for box2 in sig2box:
  box2.SetLineColor(5)
  box2.SetFillColor(5)
  box2.Draw('f same')

sig1boxY1 = array('d',[xsec*0.0980, xsec*0.0980, xsec*0.1894, xsec*0.1894])
sig1boxY2 = array('d',[xsec*0.0702, xsec*0.0702, xsec*0.1362, xsec*0.1362])
sig1boxY3 = array('d',[xsec*0.0853, xsec*0.0853, xsec*0.1655, xsec*0.1655])
sig1boxY4 = array('d',[xsec*0.0717, xsec*0.0717, xsec*0.1400, xsec*0.1400])
sig1boxY5 = array('d',[xsec*0.1050, xsec*0.1050, xsec*0.2024, xsec*0.2024])
sig1boxY6 = array('d',[xsec*0.0804, xsec*0.0804, xsec*0.1559, xsec*0.1559])
sig1boxY7 = array('d',[xsec*0.1008, xsec*0.1008, xsec*0.1948, xsec*0.1948])
sig1boxY8 = array('d',[xsec*0.0843, xsec*0.0843, xsec*0.1641, xsec*0.1641])
sig1box1 = TPolyLine(4, sigboxX1, sig1boxY1)
sig1box2 = TPolyLine(4, sigboxX2, sig1boxY2)
sig1box3 = TPolyLine(4, sigboxX3, sig1boxY3)
sig1box4 = TPolyLine(4, sigboxX4, sig1boxY4)
sig1box5 = TPolyLine(4, sigboxX5, sig1boxY5)
sig1box6 = TPolyLine(4, sigboxX6, sig1boxY6)
sig1box7 = TPolyLine(4, sigboxX7, sig1boxY7)
sig1box8 = TPolyLine(4, sigboxX8, sig1boxY8)
sig1box = [sig1box1, sig1box2, sig1box3, sig1box4, sig1box5, sig1box6, sig1box7, sig1box8]
for box1 in sig1box:
  box1.SetLineColor(3)
  box1.SetFillColor(3)
  box1.Draw('f same')

#################################
observed = [0.0777, 0.1223, 0.0791, 0.0597, 0.1496, 0.1103, 0.0890, 0.0726]
expected = [0.1362, 0.0977, 0.1187, 0.1001, 0.1460, 0.1118, 0.1396, 0.1177]

obs_hct_keras     = TLine(0, xsec*observed[0], 1, xsec*observed[0])
obs_hct_bdt       = TLine(1, xsec*observed[1], 2, xsec*observed[1])
obs_hct_keras_mat = TLine(2, xsec*observed[2], 3, xsec*observed[2])
obs_hct_bdt_mat   = TLine(3, xsec*observed[3], 4, xsec*observed[3])
obs_hut_keras     = TLine(4, xsec*observed[4], 5, xsec*observed[4])
obs_hut_bdt       = TLine(5, xsec*observed[5], 6, xsec*observed[5])
obs_hut_keras_mat = TLine(6, xsec*observed[6], 7, xsec*observed[6])
obs_hut_bdt_mat   = TLine(7, xsec*observed[7], 8, xsec*observed[7])
obs = [obs_hct_keras, obs_hct_bdt, obs_hct_keras_mat, obs_hct_bdt_mat, obs_hut_keras, obs_hut_bdt, obs_hut_keras_mat, obs_hut_bdt_mat]
for lines in obs:
  lines.SetLineColor(2)
  lines.SetLineWidth(2)
  lines.Draw('same')

exp_hct_keras     = TLine(0, xsec*expected[0], 1, xsec*expected[0])
exp_hct_bdt       = TLine(1, xsec*expected[1], 2, xsec*expected[1])
exp_hct_keras_mat = TLine(2, xsec*expected[2], 3, xsec*expected[2])
exp_hct_bdt_mat   = TLine(3, xsec*expected[3], 4, xsec*expected[3])
exp_hut_keras     = TLine(4, xsec*expected[4], 5, xsec*expected[4])
exp_hut_bdt       = TLine(5, xsec*expected[5], 6, xsec*expected[5])
exp_hut_keras_mat = TLine(6, xsec*expected[6], 7, xsec*expected[6])
exp_hut_bdt_mat   = TLine(7, xsec*expected[7], 8, xsec*expected[7])
exp = [exp_hct_keras, exp_hct_bdt, exp_hct_keras_mat, exp_hct_bdt_mat, exp_hut_keras, exp_hut_bdt, exp_hut_keras_mat, exp_hut_bdt_mat]
for lines2 in exp:
  lines2.SetLineColor(1)
  lines2.SetLineWidth(2)
  lines2.SetLineStyle(7)
  lines2.Draw('same')

################################
l = TLegend(0.17,0.75,0.6,0.87)
l.SetNColumns(2)
l.SetTextSize(0.03)
l.SetLineColor(0)
l.SetFillColor(0)
l.AddEntry(exp_hct_keras, 'Expected', 'l')
l.AddEntry(obs_hct_keras, 'Observed', 'l')
l.AddEntry(sig1box1, '68% expected', 'f')
l.AddEntry(sig2box1, '95% expected', 'f')
l.Draw('same')

c1.Print('out.pdf')


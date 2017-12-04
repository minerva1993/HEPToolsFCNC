import os, sys
from ROOT import *

ch = sys.argv[1]

if 'Hct' in ch: filename = 'output_keras_Hct.root'
elif 'Hut' in ch: filename = 'output_keras_Hut.root'

out = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v7/keras7_'+ch+'/'+filename)

tree = out.Get('keras7_'+ch+'/TestTree')

c1 = TCanvas( 'c1', 'c1', 500, 400 )
ch_ttbb_bdt = tree.Draw("BDT>>h_ttbb_bdt(40,-1,1)","classID == 1 && EventCategory == 1","norm HIST")
ch_ttbj_bdt = tree.Draw("BDT>>h_ttbj_bdt(40,-1,1)","classID == 1 && EventCategory == 2","norm same HIST")
ch_ttcc_bdt = tree.Draw("BDT>>h_ttcc_bdt(40,-1,1)","classID == 1 && EventCategory == 3","norm same HIST")
ch_ttLF_bdt = tree.Draw("BDT>>h_ttLF_bdt(40,-1,1)","classID == 1 && EventCategory == 4","norm same HIST")
ch_singletop_bdt = tree.Draw("BDT>>h_singletop_bdt(40,-1,1)","classID == 1 && EventCategory == 5","norm same HIST")
ch_others_bdt = tree.Draw("BDT>>h_others_bdt(40,-1,1)","classID == 1 && (EventCategory == 6 || EventCategory == 7)","norm same HIST")

h_ttbb_bdt.SetStats(0)
h_ttbb_bdt.SetMaximum(h_ttbb_bdt.GetMaximum() *1.3)
h_ttbb_bdt.SetTitle('Score w.r.t. decay channel')
h_ttbb_bdt.SetLineColor(1)
h_ttbj_bdt.SetLineColor(2)
h_ttcc_bdt.SetLineColor(3)
h_ttLF_bdt.SetLineColor(4)
h_singletop_bdt.SetLineColor(402)
h_others_bdt.SetLineColor(6)

l1 = TLegend(0.7, 0.7, 0.9, 0.9)
l1.AddEntry(h_ttbb_bdt, 'ttbb', 'l')
l1.AddEntry(h_ttbj_bdt, 'ttbj', 'l')
l1.AddEntry(h_ttcc_bdt, 'ttcc', 'l')
l1.AddEntry(h_ttLF_bdt, 'ttLF', 'l')
l1.AddEntry(h_singletop_bdt, 'ST', 'l')
l1.AddEntry(h_others_bdt, 'Wjets+DY', 'l')
l1.Draw('SAME')
c1.Print('h_bdt_ch_'+ch+'.pdf')

#######################

c2 = TCanvas( 'c2', 'c2', 500, 400 )
pu0to10_bdt = tree.Draw("BDT>>h_pu0to10_bdt(40,-1,1)","GoodPV < 10","norm HIST")
pu10to30_bdt = tree.Draw("BDT>>h_pu10to30_bdt(40,-1,1)","GoodPV > 10 && GoodPV < 30","norm same HIST")
pu30toinf_bdt = tree.Draw("BDT>>h_pu30toinf_bdt(40,-1,1)","GoodPV > 30","norm same HIST")

h_pu0to10_bdt.SetStats(0)
h_pu0to10_bdt.SetTitle('Score w.r.t. Pileup')
h_pu0to10_bdt.SetMaximum(h_pu0to10_bdt.GetMaximum() *1.2)
h_pu0to10_bdt.SetLineColor(1)
h_pu10to30_bdt.SetLineColor(2)
h_pu30toinf_bdt.SetLineColor(3)

l2 = TLegend(0.7, 0.75, 0.9, 0.9)
l2.AddEntry(h_pu0to10_bdt, 'pu 0~10', 'l')
l2.AddEntry(h_pu10to30_bdt, 'pu 10~30', 'l')
l2.AddEntry(h_pu30toinf_bdt, 'pu 30~', 'l')
l2.Draw('SAME')
c2.Print('h_bdt_pu_'+ch+'.pdf')

######################

c3 = TCanvas( 'c3', 'c3', 500, 400 )
ch_ttbb_keras = tree.Draw("Keras_TF>>h_ttbb_keras(20,0,0.5)","classID == 1 && EventCategory == 1","norm HIST")
ch_ttbj_keras = tree.Draw("Keras_TF>>h_ttbj_keras(20,0,0.5)","classID == 1 && EventCategory == 2","norm same HIST")
ch_ttcc_keras = tree.Draw("Keras_TF>>h_ttcc_keras(20,0,0.5)","classID == 1 && EventCategory == 3","norm same HIST")
ch_ttLF_keras = tree.Draw("Keras_TF>>h_ttLF_keras(20,0,0.5)","classID == 1 && EventCategory == 4","norm same HIST")
ch_singletop_keras = tree.Draw("Keras_TF>>h_singletop_keras(20,0,0.5)","classID == 1 && EventCategory == 5","norm same HIST")
ch_others_keras = tree.Draw("Keras_TF>>h_others_keras(20,0,0.5)","classID == 1 && (EventCategory == 6 || EventCategory == 7)","norm same HIST")

h_ttbb_keras.SetStats(0)
h_ttbb_keras.SetMaximum(h_ttcc_keras.GetMaximum() *1.2)
h_ttbb_keras.SetTitle('Score w.r.t. decay channel')
h_ttbb_keras.SetLineColor(1)
h_ttbj_keras.SetLineColor(2)
h_ttcc_keras.SetLineColor(3)
h_ttLF_keras.SetLineColor(4)
h_singletop_keras.SetLineColor(402)
h_others_keras.SetLineColor(6)

l3 = TLegend(0.7, 0.7, 0.9, 0.9)
l3.AddEntry(h_ttbb_keras, 'ttbb', 'l')
l3.AddEntry(h_ttbj_keras, 'ttbj', 'l')
l3.AddEntry(h_ttcc_keras, 'ttcc', 'l')
l3.AddEntry(h_ttLF_keras, 'ttLF', 'l')
l3.AddEntry(h_singletop_keras, 'ST', 'l')
l3.AddEntry(h_others_keras, 'Wjets+DY', 'l')
l3.Draw('SAME')
c3.Print('h_keras_ch_'+ch+'.pdf')

######################

c4 = TCanvas( 'c4', 'c4', 500, 400 )
pu0to10_keras = tree.Draw("Keras_TF>>h_pu0to10_keras(20,0,0.5)","GoodPV < 10","norm HIST")
pu10to30_keras = tree.Draw("Keras_TF>>h_pu10to30_keras(20,0,0.5)","GoodPV > 10 && GoodPV < 30","norm same HIST")
pu30toinf_keras = tree.Draw("Keras_TF>>h_pu30toinf_keras(20,0,0.5)","GoodPV > 30","norm same HIST")

h_pu0to10_keras.SetStats(0)
h_pu0to10_keras.SetTitle('Score w.r.t. Pileup')
h_pu0to10_keras.SetMaximum(h_pu0to10_keras.GetMaximum() *1.2)
h_pu0to10_keras.SetLineColor(1)
h_pu10to30_keras.SetLineColor(2)
h_pu30toinf_keras.SetLineColor(3)

l4 = TLegend(0.7, 0.75, 0.9, 0.9)
l4.AddEntry(h_pu0to10_keras, 'pu 0~10', 'l')
l4.AddEntry(h_pu10to30_keras, 'pu 10~30', 'l')
l4.AddEntry(h_pu30toinf_keras, 'pu 30~', 'l')
l4.Draw('SAME')
c4.Print('h_keras_pu_'+ch+'.pdf')

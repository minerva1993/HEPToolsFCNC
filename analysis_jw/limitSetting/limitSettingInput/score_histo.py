import os
from ROOT import *

tmva_version = 'v5'

ch = 'Hct11'
#ch = 'Hut11'

c1 = TCanvas( 'c1', 'c1', 500, 400 ) 
c2 = TCanvas( 'c2', 'c2', 500, 400 )

bdt_data = TH1F('bdt_data',ch+' BDT Score',20,-0.3,0.3)
bdt_sig = TH1F('bdt_sig','',20,-0.3,0.3)
bdt_bkg = TH1F('bdt_bkg','',20,-0.3,0.3)

keras_data = TH1F('keras_data',ch+' Keras Score',20,0,1)
keras_sig = TH1F('keras_sig','',20,0,1)
keras_bkg = TH1F('keras_bkg','',20,0,1)

shape_file = TFile.Open('/home/minerva1993/fcnc/analysis_jw/limitSetting/limitSettingInput/shape_'+tmva_version+'_'+ch+'.root')

bdt_ttbb = shape_file.Get('bdt_ttbb')
bdt_ttbj = shape_file.Get('bdt_ttbj')
bdt_ttcc = shape_file.Get('bdt_ttcc')
bdt_ttLF = shape_file.Get('bdt_ttLF')
bdt_singletop = shape_file.Get('bdt_singletop')
bdt_others = shape_file.Get('bdt_others')
bdt_data = shape_file.Get('bdt_data_obs')
bdt_sig = shape_file.Get('bdt_sig')

bdt_bkg = bdt_ttbb.Clone('bdt_bkg')
bdt_bkg.Add(bdt_ttbj, 1.0)
bdt_bkg.Add(bdt_ttcc, 1.0)
bdt_bkg.Add(bdt_ttLF, 1.0)
bdt_bkg.Add(bdt_singletop, 1.0)
#bdt_bkg.Add(bdt_others, 1.0)

bdt_sig.Scale(bdt_data.Integral()/bdt_sig.Integral())

keras_ttbb = shape_file.Get('keras_ttbb')
keras_ttbj = shape_file.Get('keras_ttbj')
keras_ttcc = shape_file.Get('keras_ttcc')
keras_ttLF = shape_file.Get('keras_ttLF')
keras_singletop = shape_file.Get('keras_singletop')
keras_others = shape_file.Get('keras_others')
keras_data = shape_file.Get('keras_data_obs')
keras_sig = shape_file.Get('keras_sig')

keras_bkg = keras_ttbb.Clone('keras_bkg')
keras_bkg.Add(keras_ttbj, 1.0)
keras_bkg.Add(keras_ttcc, 1.0)
keras_bkg.Add(keras_ttLF, 1.0)
keras_bkg.Add(keras_singletop, 1.0)
keras_bkg.Add(keras_others, 1.0)

keras_sig.Scale(keras_data.Integral()/keras_sig.Integral())

c1.cd()
bdt_data.Draw('P')
bdt_bkg.Draw('HIST SAME E')
bdt_data.SetMarkerStyle(8)
bdt_sig.Draw('HIST SAME E')
bdt_sig.SetLineStyle(2)
bdt_data.SetStats(0)
l1 = TLegend(0.7, 0.8, 0.9, 0.9)
l1.AddEntry(bdt_data, 'data', 'p')
l1.AddEntry(bdt_bkg, 'background', 'l')
l1.AddEntry(bdt_sig, 'signal', 'l')
l1.Draw('SAME')
c1.Print('histo_'+tmva_version+'_'+ch+'_bdt.pdf')

c2.cd()
keras_data.Draw('P')
keras_bkg.Draw('HIST SAME')
keras_data.SetMarkerStyle(8)
keras_sig.Draw('HIST SAME')
keras_sig.SetLineStyle(2)
keras_data.SetStats(0)
l2 = TLegend(0.7, 0.8, 0.9, 0.9)
l2.AddEntry(bdt_data, 'data', 'p')
l2.AddEntry(bdt_bkg, 'background', 'l')
l2.AddEntry(bdt_sig, 'signal', 'l')
l2.Draw('SAME')
c2.Print('histo_'+tmva_version+'_'+ch+'_keras.pdf')

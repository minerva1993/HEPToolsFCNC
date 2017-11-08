import os
from ROOT import *

tmva_version = 'v5'

ch = 'Hct28'
#ch = 'Hut15'

target = TFile('shape_'+tmva_version+'_'+ch+'_test.root','RECREATE')

mva_out = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v5/bdt_test_Hct9/output_bdt.root')
#mva_out = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v5/keras5_Hut15/output_keras_Hut.root')

#bdt_bkg = mva_out.Get('/keras5_Hut15/Method_BDT_optimized/BDT_optimized/MVA_BDT_optimized_B')
bdt_bkg = mva_out.Get('/bdt_test_Hct9/Method_BDT/BDT/MVA_BDT_B')
bdt_bkg.Scale(55873/bdt_bkg.Integral())
bdt_bkg.SetName('bdt_bkg')
target.cd()
bdt_bkg.Write()

bdt_sig = mva_out.Get('/bdt_test_Hct9/Method_BDT/BDT/MVA_BDT_S')
bdt_sig.Scale(2786/bdt_sig.Integral())
bdt_sig.SetName('bdt_sig')
target.cd()
bdt_sig.Write()

rd = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_SingleLepton_Run2016.root')
bdt_data_obs = rd.Get('h_scoreBDT_'+ch+'_SingleLepton_Run2016')
bdt_data_obs.SetName('bdt_data_obs')
target.cd()
bdt_data_obs.Write()
 
target.Close()

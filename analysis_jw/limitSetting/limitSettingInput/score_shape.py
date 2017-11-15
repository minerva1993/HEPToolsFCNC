import os
from ROOT import *

tmva_version = 'v5'

ch = 'Hct32'
#ch = 'Hut15'

ttbar = 0.0910581123792

target = TFile('shape_'+tmva_version+'_'+ch+'.root','RECREATE')

for scores in os.listdir('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'):

  if scores == 'shape_'+ch+'_ttbb.root':
    ttbb = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+scores)

    bdt_ttbb = ttbb.Get('h_scoreBDT_'+ch+'_ttbb')
    bdt_ttbb.Scale(ttbar)#0.0888153017294
    bdt_ttbb.SetName('bdt_ttbb')

    keras_ttbb = ttbb.Get('h_scoreKeras_'+ch+'_ttbb')
    keras_ttbb.Scale(ttbar)
    keras_ttbb.SetName('keras_ttbb')

    bdt_ttbb.Scale(5)
    keras_ttbb.Scale(5)

    target.cd()
    bdt_ttbb.Write()
    keras_ttbb.Write()
 
  elif scores == 'shape_'+ch+'_ttLF.root':
    ttLF = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+scores)
    tt = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_tt.root')
    
    bdt_ttLF = ttLF.Get('h_scoreBDT_'+ch+'_ttLF')
    bdt_ttLF.Scale(ttbar)
    bdt_tt = tt.Get('h_scoreBDT_'+ch+'_tt')
    bdt_tt.Scale(ttbar)    
    bdt_ttLF.Add(bdt_ttLF, bdt_tt, 1.0, 1.0)
    bdt_ttLF.SetName('bdt_ttLF')

    keras_ttLF = ttLF.Get('h_scoreKeras_'+ch+'_ttLF')
    keras_ttLF.Scale(ttbar)
    keras_tt = tt.Get('h_scoreKeras_'+ch+'_tt')
    keras_tt.Scale(ttbar)
    keras_ttLF.Add(keras_ttLF, keras_tt, 1.0, 1.0)
    keras_ttLF.SetName('keras_ttLF')

    bdt_ttLF.Scale(5)
    keras_ttLF.Scale(5)

    target.cd()
    bdt_ttLF.Write()
    keras_ttLF.Write()
 
  elif scores == 'shape_'+ch+'_ttcc.root':
    ttcc = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+scores)

    bdt_ttcc = ttcc.Get('h_scoreBDT_'+ch+'_ttcc')
    bdt_ttcc.Scale(ttbar)
    bdt_ttcc.SetName('bdt_ttcc')

    keras_ttcc = ttcc.Get('h_scoreKeras_'+ch+'_ttcc')
    keras_ttcc.Scale(ttbar)
    keras_ttcc.SetName('keras_ttcc')

    bdt_ttcc.Scale(5)
    keras_ttcc.Scale(5)

    target.cd()
    bdt_ttcc.Write()
    keras_ttcc.Write()
 
  elif scores == 'shape_'+ch+'_ttbj.root':
    ttbj = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+scores)

    bdt_ttbj = ttbj.Get('h_scoreBDT_'+ch+'_ttbj')
    bdt_ttbj.Scale(ttbar)
    bdt_ttbj.SetName('bdt_ttbj')

    keras_ttbj = ttbj.Get('h_scoreKeras_'+ch+'_ttbj')
    keras_ttbj.Scale(ttbar)
    keras_ttbj.SetName('keras_ttbj')

    bdt_ttbj.Scale(5)
    keras_ttbj.Scale(5)

    target.cd()
    bdt_ttbj.Write()
    keras_ttbj.Write()

  elif scores == 'shape_'+ch+'_tchannel.root':
    st = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+scores)
    stbar = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_tbarchannel.root')
    tw = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_tWchannel.root')
    tbarw = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_tbarWchannel.root')

    bdt_singletop = st.Get('h_scoreBDT_'+ch+'_tchannel')
    bdt_singletop.Scale(0.023844899675)
    bdt_stbar = stbar.Get('h_scoreBDT_'+ch+'_tbarchannel')
    bdt_stbar.Scale(0.024575262909)
    bdt_singletop.Add(bdt_singletop, bdt_stbar, 1.0, 1.0)
    bdt_tw = tw.Get('h_scoreBDT_'+ch+'_tWchannel')
    bdt_tw.Scale(0.190335714074)
    bdt_singletop.Add(bdt_singletop, bdt_tw, 1.0, 1.0)
    bdt_tbarw =tbarw.Get('h_scoreBDT_'+ch+'_tbarWchannel')
    bdt_tbarw.Scale(0.193026936331)
    bdt_singletop.Add(bdt_singletop, bdt_tbarw, 1.0, 1.0)
    bdt_singletop.SetName('bdt_singletop')

    keras_singletop = st.Get('h_scoreKeras_'+ch+'_tchannel')
    keras_singletop.Scale(0.023844899675)
    keras_stbar = stbar.Get('h_scoreKeras_'+ch+'_tbarchannel')
    keras_stbar.Scale(0.024575262909)
    keras_singletop.Add(keras_singletop, keras_stbar, 1.0, 1.0)
    keras_tw = tw.Get('h_scoreKeras_'+ch+'_tWchannel')
    keras_tw.Scale(0.190335714074)
    keras_singletop.Add(keras_singletop, keras_tw, 1.0, 1.0)
    keras_tbarw = tbarw.Get('h_scoreKeras_'+ch+'_tbarWchannel')
    keras_tbarw.Scale(0.193026936331)
    keras_singletop.Add(keras_singletop, keras_tbarw, 1.0, 1.0)
    keras_singletop.SetName('keras_singletop')

    bdt_singletop.Scale(5)
    keras_singletop.Scale(5)

    target.cd()
    bdt_singletop.Write()
    keras_singletop.Write()

  elif scores == 'shape_'+ch+'_zjets.root':
    zjets = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+scores)
    zjets10to50 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_zjets10to50.root')
    wjets = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_wjets.root')
    #ww = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_ww.root')
    #wz = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_wz.root')
    #zz = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_zz.root')

    bdt_others = zjets.Get('h_scoreBDT_'+ch+'_zjets')
    bdt_others.Scale(2.65549294802)
    bdt_zjets10to50 = zjets10to50.Get('h_scoreBDT_'+ch+'_zjets10to50')
    bdt_zjets10to50.Scale(22.7143748772)
    bdt_others.Add(bdt_others, bdt_zjets10to50, 1.0, 1.0)
    bdt_wjets = wjets.Get('h_scoreBDT_'+ch+'_wjets')
    bdt_wjets.Scale(134.276449237)
    bdt_others.Add(bdt_others, bdt_wjets, 1.0, 1.0)
    """
    bdt_ww = ww.Get('h_scoreBDT_'+ch+'_ww')
    bdt_ww.Scale(4.70407414855)
    bdt_others.Add(bdt_others, bdt_ww, 1.0, 1.0)
    bdt_wz = wz.Get('h_scoreBDT_'+ch+'_wz')
    bdt_wz.Scale(1.69041171)
    bdt_others.Add(bdt_others, bdt_wz, 1.0, 1.0)
    bdt_zz = zz.Get('h_scoreBDT_'+ch+'_zz')
    bdt_zz.Scale(0.598577911125)
    bdt_others.Add(bdt_others, bdt_zz, 1.0, 1.0)
    """
    bdt_others.SetName('bdt_others')

    keras_others = zjets.Get('h_scoreKeras_'+ch+'_zjets')
    keras_others.Scale(2.65549294802)
    keras_zjets10to50 = zjets10to50.Get('h_scoreKeras_'+ch+'_zjets10to50')
    keras_zjets10to50.Scale(22.7143748772)
    keras_others.Add(keras_others, keras_zjets10to50, 1.0, 1.0)
    keras_wjets = wjets.Get('h_scoreKeras_'+ch+'_wjets')
    keras_wjets.Scale(134.276449237)
    keras_others.Add(keras_others, keras_wjets, 1.0, 1.0)
    """
    keras_ww = ww.Get('h_scoreKeras_'+ch+'_ww')
    keras_ww.Scale(4.70407414855)
    keras_others.Add(keras_others, keras_ww, 1.0, 1.0)
    keras_wz = wz.Get('h_scoreKeras_'+ch+'_wz')
    keras_wz.Scale(1.69041171)
    keras_others.Add(keras_others, keras_wz, 1.0, 1.0)
    keras_zz = zz.Get('h_scoreKeras_'+ch+'_zz')
    keras_zz.Scale(0.598577911125)
    keras_others.Add(keras_others, keras_zz, 1.0, 1.0)
    """
    keras_others.SetName('keras_others')
    """
    #QCD
    eg120to170 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_EGEnr_120to170.root')
    eg120to170_tree = eg120to170.Get('tree')
    eg170to300 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_EGEnr_170to300.root')
    eg170to300_tree = eg170to300.Get('tree')
    eg300toinf = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_EGEnr_300toInf.root')
    eg300toinf_tree = eg300toinf.Get('tree')
    mu120to170 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_MuEnr_120to170.root')
    mu120to170_tree = mu120to170.Get('tree')
    mu170to300 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_MuEnr_170to300.root')
    mu170to300_tree = mu170to300.Get('tree')
    mu300to470 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_MuEnr_300to470.root')
    mu300to470_tree = mu300to470.Get('tree')
    mu80to120 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/output_'+ch+'_tmva_QCD_MuEnr_80to120.root')
    mu80to120_tree = mu80to120.Get('tree')

    bdt_eg120to170 = TH1F('bdt_eg120to170', '', 20, -0.3, 0.3)
    eg120to170_tree.Draw('BDTScore>>bdt_eg120to170')
    bdt_eg120to170.Scale(66.8174520871)
    bdt_others.Add(bdt_others, bdt_eg120to170, 1.0, 1.0)
    bdt_eg170to300 = TH1F('bdt_eg170to300', '', 20, -0.3, 0.3)
    eg170to300_tree.Draw('BDTScore>>bdt_eg170to300')
    bdt_eg170to300.Scale(58.6084594604)
    bdt_others.Add(bdt_others, bdt_eg170to300, 1.0, 1.0)
    bdt_eg300toinf = TH1F('bdt_eg300toinf', '', 20, -0.3, 0.3)
    eg300toinf_tree.Draw('BDTScore>>bdt_eg300toinf')
    bdt_eg300toinf.Scale(6.60760047123)
    bdt_others.Add(bdt_others, bdt_eg300toinf, 1.0, 1.0)
    bdt_mu120to170 = TH1F('bdt_mu120to170', '', 20, -0.3, 0.3)
    mu120to170_tree.Draw('BDTScore>>bdt_mu120to170')
    bdt_mu120to170.Scale(75.6824937994)
    bdt_others.Add(bdt_others, bdt_mu120to170, 1.0, 1.0)
    bdt_mu170to300 = TH1F('bdt_mu170to300', '', 20, -0.3, 0.3)
    mu170to300_tree.Draw('BDTScore>>bdt_mu170to300')
    bdt_mu170to300.Scale(39.0593299833)
    bdt_others.Add(bdt_others, bdt_mu170to300, 1.0, 1.0)
    bdt_mu300to470 = TH1F('bdt_mu300to470', '', 20, -0.3, 0.3)
    mu300to470_tree.Draw('BDTScore>>bdt_mu300to470')
    bdt_mu300to470.Scale(1.82404861053)
    bdt_others.Add(bdt_others, bdt_mu300to470, 1.0, 1.0)
    bdt_mu80to120 = TH1F('bdt_mu80to120', '', 20, -0.3, 0.3)
    mu80to120_tree.Draw('BDTScore>>bdt_mu80to120')
    bdt_mu80to120.Scale(278.226438007)
    bdt_others.Add(bdt_others, bdt_mu80to120, 1.0, 1.0)

    keras_eg120to170 = TH1F('keras_eg120to170', '', 20, -0, 1)
    eg120to170_tree.Draw('KerasScore>>keras_eg120to170')
    keras_eg120to170.Scale(66.8174520871)
    keras_others.Add(keras_others, keras_eg120to170, 1.0, 1.0)
    keras_eg170to300 = TH1F('keras_eg170to300', '', 20, -0, 1)
    eg170to300_tree.Draw('KerasScore>>keras_eg170to300')
    keras_eg170to300.Scale(58.6084594604)
    keras_others.Add(keras_others, keras_eg170to300, 1.0, 1.0)
    keras_eg300toinf = TH1F('keras_eg300toinf', '', 20, -0, 1)
    eg300toinf_tree.Draw('KerasScore>>keras_eg300toinf')
    keras_eg300toinf.Scale(6.60760047123)
    keras_others.Add(keras_others, keras_eg300toinf, 1.0, 1.0)
    keras_mu120to170 = TH1F('keras_mu120to170', '', 20, -0, 1)
    mu120to170_tree.Draw('KerasScore>>keras_mu120to170')
    keras_mu120to170.Scale(75.6824937994)
    keras_others.Add(keras_others, keras_mu120to170, 1.0, 1.0)
    keras_mu170to300 = TH1F('keras_mu170to300', '', 20, -0, 1)
    mu170to300_tree.Draw('KerasScore>>keras_mu170to300')
    keras_mu170to300.Scale(39.0593299833)
    keras_others.Add(keras_others, keras_mu170to300, 1.0, 1.0)
    keras_mu300to470 = TH1F('keras_mu300to470', '', 20, -0, 1)
    mu300to470_tree.Draw('KerasScore>>keras_mu300to470')
    keras_mu300to470.Scale(1.824048611)
    keras_others.Add(keras_others, keras_mu300to470, 1.0, 1.0)
    keras_mu80to120 = TH1F('keras_mu80to120', '', 20, -0, 1)
    mu80to120_tree.Draw('KerasScore>>keras_mu80to120')
    keras_mu80to120.Scale(278.226438007)
    keras_others.Add(keras_others, keras_mu80to120, 1.0, 1.0)
    """

    target.cd()
    bdt_others.Write()
    keras_others.Write()

  elif scores == 'shape_'+ch+'_SingleLepton_Run2016.root':
    rd = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_SingleLepton_Run2016.root')
    
    bdt_data_obs = rd.Get('h_scoreBDT_'+ch+'_SingleLepton_Run2016')
    bdt_data_obs.SetName('bdt_data_obs')

    keras_data_obs = rd.Get('h_scoreKeras_'+ch+'_SingleLepton_Run2016')
    keras_data_obs.SetName('keras_data_obs')

    target.cd()
    bdt_data_obs.Write()
    keras_data_obs.Write()
   
  elif scores == 'shape_'+ch+'_Top_Hct.root' and ch == 'Hct32':
    tch = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_Top_Hct.root')
    tbarch = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_AntiTop_Hct.root')

    bdt_sig = tch.Get('h_scoreBDT_'+ch+'_Top_Hct')
    #bdt_sig.Scale(0.113541253338/2)
    bdt_sig.Scale(0.1/2)
    bdt_tbarch = tbarch.Get('h_scoreBDT_'+ch+'_AntiTop_Hct')
    #bdt_tbarch.Scale(0.113541253338/2)
    bdt_tbarch.Scale(0.1/2)
    bdt_sig.Add(bdt_sig, bdt_tbarch, 1.0, 1.0)
    bdt_sig.SetName('bdt_sig')

    keras_sig = tch.Get('h_scoreKeras_'+ch+'_Top_Hct')
    #keras_sig.Scale(0.113541253338/2)
    keras_sig.Scale(0.1/2)
    keras_tbarch = tbarch.Get('h_scoreKeras_'+ch+'_AntiTop_Hct')
    #keras_tbarch.Scale(0.113541253338/2)
    keras_tbarch.Scale(0.1/2)
    keras_sig.Add(keras_sig, keras_tbarch, 1.0, 1.0)
    keras_sig.SetName('keras_sig')

    bdt_sig.Scale(5)
    keras_sig.Scale(5)

    target.cd()
    bdt_sig.Write()
    keras_sig.Write()

  elif scores == 'shape_'+ch+'_Top_Hut.root' and ch == 'Hut15':
    tuh = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_Top_Hut.root')
    tbaruh = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/'+tmva_version+'/score_mva/'+ch+'/'+'shape_'+ch+'_AntiTop_Hut.root')

    bdt_sig = tuh.Get('h_scoreBDT_'+ch+'_Top_Hut')
    #bdt_sig.Scale(0.15613733157/2)
    bdt_sig.Scale(0.1/2)
    bdt_tbaruh = tbaruh.Get('h_scoreBDT_'+ch+'_AntiTop_Hut')
    #bdt_tbaruh.Scale(0.15613733157/2)
    bdt_tbaruh.Scale(0.1/2)
    bdt_sig.Add(bdt_sig, bdt_tbaruh, 1.0, 1.0)
    bdt_sig.SetName('bdt_sig')

    keras_sig = tuh.Get('h_scoreKeras_'+ch+'_Top_Hut')
    #keras_sig.Scale(0.15613733157/2)
    keras_sig.Scale(0.1/2)
    keras_tbaruh = tbaruh.Get('h_scoreKeras_'+ch+'_AntiTop_Hut')
    #keras_tbaruh.Scale(0.15613733157/2)
    keras_tbaruh.Scale(0.1/2)
    keras_sig.Add(keras_sig, keras_tbaruh, 1.0, 1.0)
    keras_sig.SetName('keras_sig')

    bdt_sig.Scale(5)
    keras_sig.Scale(5)

    target.cd()
    bdt_sig.Write()
    keras_sig.Write()

  else: continue
  
target.Close()

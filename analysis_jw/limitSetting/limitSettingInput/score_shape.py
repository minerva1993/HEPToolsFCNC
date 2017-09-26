import os
from ROOT import *

ch = 'Hct3'
#ch = 'Hut3'

target = TFile('shape_'+ch+'_v2.root','RECREATE')

for scores in os.listdir('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'):

  if scores == 'output_'+ch+'_tmva_ttbb.root':
    ttbb = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+scores)
    ttbb_tree = ttbb.Get('tree')

    bdt_ttbb = TH1F('bdt_ttbb', '', 100, -1, 1)
    ttbb_tree.Draw('BDTScore>>bdt_ttbb')
    bdt_ttbb.Scale(0.0888153017294)

    keras_ttbb = TH1F('keras_ttbb', '', 100, 0, 1)
    ttbb_tree.Draw('KerasScore>>keras_ttbb')
    keras_ttbb.Scale(0.0888153017294)

    target.cd()
    bdt_ttbb.Write()
    keras_ttbb.Write()
 
  elif scores == 'output_'+ch+'_tmva_ttLF.root':
    ttLF = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+scores)
    ttLF_tree = ttLF.Get('tree')
    tt = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_tt.root')
    tt_tree = tt.Get('tree')
    
    bdt_ttLF = TH1F('bdt_ttLF', '', 100, -1, 1)
    ttLF_tree.Draw('BDTScore>>bdt_ttLF')
    bdt_ttLF.Scale(0.0888153017294)
    bdt_tt = TH1F('bdt_tt', '', 100, -1, 1)
    tt_tree.Draw('BDTScore>>bdt_tt')
    bdt_tt.Scale(0.0888153017294)    
    bdt_ttLF.Add(bdt_ttLF, bdt_tt, 1.0, 1.0)

    keras_ttLF = TH1F('keras_ttLF', '', 100, 0, 1)
    ttLF_tree.Draw('KerasScore>>keras_ttLF')
    keras_ttLF.Scale(0.0888153017294)
    keras_tt = TH1F('keras_tt', '', 100, 0, 1)
    tt_tree.Draw('KerasScore>>keras_tt')
    keras_tt.Scale(0.0888153017294)
    keras_ttLF.Add(keras_ttLF, keras_tt, 1.0, 1.0)

    target.cd()
    bdt_ttLF.Write()
    keras_ttLF.Write()
 
  elif scores == 'output_'+ch+'_tmva_ttcc.root':
    ttcc = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+scores)
    ttcc_tree = ttcc.Get('tree')

    bdt_ttcc = TH1F('bdt_ttcc', '', 100, -1, 1)
    ttcc_tree.Draw('BDTScore>>bdt_ttcc')
    bdt_ttcc.Scale(0.0888153017294)

    keras_ttcc = TH1F('keras_ttcc', '', 100, 0, 1)
    ttcc_tree.Draw('KerasScore>>keras_ttcc')
    keras_ttcc.Scale(0.0888153017294)

    target.cd()
    bdt_ttcc.Write()
    keras_ttcc.Write()
 
  elif scores == 'output_'+ch+'_tmva_ttbj.root':
    ttbj = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+scores)
    ttbj_tree = ttbj.Get('tree')

    bdt_ttbj = TH1F('bdt_ttbj', '', 100, -1, 1)
    ttbj_tree.Draw('BDTScore>>bdt_ttbj')
    bdt_ttbj.Scale(0.0888153017294)

    keras_ttbj = TH1F('keras_ttbj', '', 100, 0, 1)
    ttbj_tree.Draw('KerasScore>>keras_ttbj')
    keras_ttbj.Scale(0.0888153017294)

    target.cd()
    bdt_ttbj.Write()
    keras_ttbj.Write()

  #no W+Jets and QCD
  elif scores == 'output_'+ch+'_tmva_zjets.root':
    zjets = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+scores)
    zjets_tree = zjets.Get('tree')
    zjets10to50 = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_zjets10to50.root')
    zjets10to50_tree = zjets10to50.Get('tree')

    bdt_others = TH1F('bdt_others', '', 100, -1, 1)
    zjets_tree.Draw('BDTScore>>bdt_others')
    bdt_others.Scale(2.65549294802)
    bdt_zjets10to50 = TH1F('bdt_zjets10to50', '', 100, -1, 1)
    zjets10to50_tree.Draw('BDTScore>>bdt_zjets10to50')
    bdt_zjets10to50.Scale(22.7143748772)
    bdt_others.Add(bdt_others, bdt_zjets10to50, 1.0, 1.0)


    keras_others = TH1F('keras_others', '', 100, 0, 1)
    zjets_tree.Draw('KerasScore>>keras_others')
    keras_others.Scale(2.65549294802)
    keras_zjets10to50 = TH1F('keras_zjets10to50', '', 100, 0, 1)
    zjets10to50_tree.Draw('KerasScore>>keras_zjets10to50')
    keras_zjets10to50.Scale(22.7143748772)
    keras_others.Add(keras_others, keras_zjets10to50, 1.0, 1.0)

    target.cd()
    bdt_others.Write()
    keras_others.Write()

  elif scores == 'output_'+ch+'_tmva_tchannel.root':
    st = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+scores)
    st_tree = st.Get('tree')
    stbar = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_tbarchannel.root')
    stbar_tree = stbar.Get('tree')
    tw = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_tWchannel.root')
    tw_tree = stbar.Get('tree')
    tbarw = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_tbarWchannel.root')
    tbarw_tree = stbar.Get('tree')

    bdt_singletop = TH1F('bdt_singletop', '', 100, -1, 1)
    st_tree.Draw('BDTScore>>bdt_singletop')
    bdt_singletop.Scale(0.023844899675)
    bdt_stbar = TH1F('bdt_stbar', '', 100, -1, 1)
    stbar_tree.Draw('BDTScore>>bdt_stbar')
    bdt_stbar.Scale(0.024575262909)
    bdt_singletop.Add(bdt_singletop, bdt_stbar, 1.0, 1.0)
    bdt_tw = TH1F('bdt_tw', '', 100, -1, 1)
    tw_tree.Draw('BDTScore>>bdt_tw')
    bdt_tw.Scale(0.190335714074)
    bdt_singletop.Add(bdt_singletop, bdt_tw, 1.0, 1.0)
    bdt_tbarw = TH1F('bdt_tbarw', '', 100, -1, 1)
    tbarw_tree.Draw('BDTScore>>bdt_tbarw')
    bdt_tbarw.Scale(0.193026936331)
    bdt_singletop.Add(bdt_singletop, bdt_tbarw, 1.0, 1.0)

    keras_singletop = TH1F('keras_singletop', '', 100, 0, 1)
    st_tree.Draw('KerasScore>>keras_singletop')
    keras_singletop.Scale(0.023844899675)
    keras_stbar = TH1F('keras_stbar', '', 100, 0, 1)
    stbar_tree.Draw('KerasScore>>keras_stbar')
    keras_stbar.Scale(0.024575262909)
    keras_singletop.Add(keras_singletop, keras_stbar, 1.0, 1.0)
    keras_tw = TH1F('keras_tw', '', 100, 0, 1)
    tw_tree.Draw('KerasScore>>keras_tw')
    keras_tw.Scale(0.190335714074)
    keras_singletop.Add(keras_singletop, keras_tw, 1.0, 1.0)
    keras_tbarw = TH1F('keras_tbarw', '', 100, 0, 1)
    tbarw_tree.Draw('KerasScore>>keras_tbarw')
    keras_tbarw.Scale(0.193026936331)
    keras_singletop.Add(keras_singletop, keras_tbarw, 1.0, 1.0)

    target.cd()
    bdt_singletop.Write()
    keras_singletop.Write()

  elif scores == 'output_'+ch+'_tmva_SingleLepton_Run2016.root':
    rd = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_SingleLepton_Run2016.root')
    rd_tree = rd.Get('tree')

    bdt_data_obs = TH1F('bdt_data_obs', '', 100, -1, 1)
    rd_tree.Draw('BDTScore>>bdt_data_obs')

    keras_data_obs = TH1F('keras_data_obs', '', 100, 0, 1)
    rd_tree.Draw('KerasScore>>keras_data_obs')

    target.cd()
    bdt_data_obs.Write()
    keras_data_obs.Write()

  elif scores == 'output_'+ch+'_tmva_Top_Hct.root' and ch == 'Hct3':
    tch = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_Top_Hct.root')
    tch_tree = tch.Get('tree')
    tbarch = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_AntiTop_Hct.root')
    tbarch_tree = tch.Get('tree')

    bdt_sig = TH1F('bdt_sig', '', 100, -1, 1)
    tch_tree.Draw('BDTScore>>bdt_sig')
    bdt_sig.Scale(0.113541253338/2)
    bdt_tbarch = TH1F('bdt_tbarch', '', 100, -1, 1)
    tbarch_tree.Draw('BDTScore>>bdt_tbarch')
    bdt_tbarch.Scale(0.113541253338/2)
    bdt_sig.Add(bdt_sig, bdt_tbarch, 1.0, 1.0)

    keras_sig = TH1F('keras_sig', '', 100, 0, 1)
    tch_tree.Draw('KerasScore>>keras_sig')
    keras_sig.Scale(0.113541253338/2)
    keras_tbarch = TH1F('keras_tbarch', '', 100, 0, 1)
    tbarch_tree.Draw('KerasScore>>keras_tbarch')
    keras_tbarch.Scale(0.113541253338/2)
    keras_sig.Add(keras_sig, keras_tbarch, 1.0, 1.0)

    target.cd()
    bdt_sig.Write()
    keras_sig.Write()

  elif scores == 'output_'+ch+'_tmva_Top_Hut.root' and ch == 'Hut3':
    tuh = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_Top_Hut.root')
    tuh_tree = tuh.Get('tree')
    tbaruh = TFile.Open('/home/minerva1993/fcnc/analysis_jw/tmva/v3/score_mva/'+ch+'/'+'output_'+ch+'_tmva_AntiTop_Hut.root')
    tbaruh_tree = tuh.Get('tree')

    bdt_sig = TH1F('bdt_sig', '', 100, -1, 1)
    tuh_tree.Draw('BDTScore>>bdt_sig')
    bdt_sig.Scale(0.15613733157/2)
    bdt_tbaruh = TH1F('bdt_tbaruh', '', 100, -1, 1)
    tbaruh_tree.Draw('BDTScore>>bdt_tbaruh')
    bdt_tbaruh.Scale(0.15613733157/2)
    bdt_sig.Add(bdt_sig, bdt_tbaruh, 1.0, 1.0)

    keras_sig = TH1F('keras_sig', '', 100, 0, 1)
    tuh_tree.Draw('KerasScore>>keras_sig')
    keras_sig.Scale(0.15613733157/2)
    keras_tbaruh = TH1F('keras_tbaruh', '', 100, 0, 1)
    tbaruh_tree.Draw('KerasScore>>keras_tbaruh')
    keras_tbaruh.Scale(0.15613733157/2)
    keras_sig.Add(keras_sig, keras_tbaruh, 1.0, 1.0)

    target.cd()
    bdt_sig.Write()
    keras_sig.Write()

  else: continue
  
target.Close()

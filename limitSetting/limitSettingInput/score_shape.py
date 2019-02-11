import os
from ROOT import *

tmva_version = 'v1'

##declare cutflows here. j3b2, j3b3, j4b2, j4b3, j4b4
nttbb          = [698.2, 126.3, 1597.8, 685.8, 62.6]
nttbj          = [2251.2, 243.3, 4900.3,  1193.3, 34]
nttcc          = [619.1, 34.9, 1727, 196.1, 6.9]
nttlf          = [45733.4, 1217.1, 99403.2, 4857.8, 45.9]
nttother       = [282743.7, 10976.8, 241836.1, 16421.4, 189.3]
nsinglet       = [12746.9, 314.6, 4198.6, 213.8, 7.2]
nsingletbar    = [8456.9, 231.3, 2512.4, 137.1, 1.3]
nsingletW      = [7231.1, 224.4, 6056, 376, 3.8]
nsingletbarW   = [7203.2, 234.8, 6002.1, 382.2, 4.9]
nw1jet50to150  = [309.6, 15.4, 100, 9.7, 0]
nw1jet150to250 = [102.9, 1.5, 16.6, 0.7, 0]
nw1jet250to400 = [2.3, 0, 0.3, 0, 0]
nw1jet400toinf = [0.3, 0, 0.1, 0, 0]
nw2jet250to400 = [1131.3, 25.6, 702.3, 32.6,  0.4]
nw2jet400toinf = [103.3, 2, 80.4, 3.5, 0.1]
nw3jet         = [4896.4, 100.1, 250.2, 11.9, 0]
nw4jet         = [6353, 149, 5310.3, 224.3, 6.2]
nzjet          = [3804.8, 66.8, 1246.3, 71, 6.9]
nzjet10to50    = [413.7, 22.4, 120.8, 21.2, 0]
nww            = [179, 3.9, 106.9, 4.1, 0]
nwz            = [504.1, 14.2, 139.8, 3.8, 0]
nzz            = [85.8, 1.8, 29.1, 1.4, 0]
nsthct         = [1239.4, 602.3, 565.9, 293.6, 12.8]
nsthut         = [140.3, 57.6,  65.3, 29.8, 0.8]

for ch in [ 'STHct_j3b3_01', 'STHct_j4b2_01', 'STHct_j4b3_01','STHut_j3b3_01', 'STHut_j4b2_01', 'STHut_j4b3_01']:
  if   'j3b2' in ch: idx = 0
  elif 'j3b3' in ch: idx = 1
  elif 'j4b2' in ch: idx = 2
  elif 'j4b3' in ch: idx = 3
  elif 'j4b4' in ch: idx = 4
  else: print "wrong channel"

  if   'Hct' in ch: tag = "hct"
  elif 'Hut' in ch: tag = "hut"

  target = TFile('shape_'+tmva_version+'_'+ch+'.root','RECREATE')

  for scores in os.listdir('/home/minerva1993/fcnc/analysis_2017/tmva/'+tmva_version+'/score_mva/'+ch):
    path = '/home/minerva1993/fcnc/analysis_2017/tmva/'+tmva_version+'/score_mva/'+ch+'/'

    if scores == 'shape_'+ch+'_ttbb.root':
      ttbb = TFile.Open(path+scores)

      bdt_ttbb = ttbb.Get('h_scoreBDT_'+ch+'_ttbb')
      bdt_ttbb.SetName('bdt_ttbb')
      if bdt_ttbb.Integral() != 0: bdt_ttbb.Scale(nttbb[idx]/bdt_ttbb.Integral())

      keras_ttbb = ttbb.Get('h_scoreKeras_'+ch+'_ttbb')
      keras_ttbb.SetName('keras_ttbb')
      if keras_ttbb.Integral() != 0: keras_ttbb.Scale(nttbb[idx]/keras_ttbb.Integral())

      target.cd()
      bdt_ttbb.Write()
      keras_ttbb.Write()
   
    elif scores == 'shape_'+ch+'_ttLF.root':
      ttLF = TFile.Open(path+scores)
      tt = TFile.Open(path+'shape_'+ch+'_ttother.root')
      
      bdt_ttLF = ttLF.Get('h_scoreBDT_'+ch+'_ttLF')
      if bdt_ttLF.Integral() != 0: bdt_ttLF.Scale(nttlf[idx]/bdt_ttLF.Integral())
      bdt_tt = tt.Get('h_scoreBDT_'+ch+'_ttother')
      if bdt_tt.Integral() != 0: bdt_tt.Scale(nttother[idx]/bdt_tt.Integral()) 
      bdt_ttLF.Add(bdt_ttLF, bdt_tt, 1.0, 1.0)
      bdt_ttLF.SetName('bdt_ttLF')

      keras_ttLF = ttLF.Get('h_scoreKeras_'+ch+'_ttLF')
      if keras_ttLF.Integral() != 0: keras_ttLF.Scale(nttlf[idx]/keras_ttLF.Integral())
      keras_tt = tt.Get('h_scoreKeras_'+ch+'_ttother')
      if keras_tt.Integral() != 0: keras_tt.Scale(nttother[idx]/keras_tt.Integral())
      keras_ttLF.Add(keras_ttLF, keras_tt, 1.0, 1.0)
      keras_ttLF.SetName('keras_ttLF')

      target.cd()
      bdt_ttLF.Write()
      keras_ttLF.Write()
   
    elif scores == 'shape_'+ch+'_ttcc.root':
      ttcc = TFile.Open(path+scores)

      bdt_ttcc = ttcc.Get('h_scoreBDT_'+ch+'_ttcc')
      bdt_ttcc.SetName('bdt_ttcc')
      if bdt_ttcc.Integral() != 0: bdt_ttcc.Scale(nttcc[idx]/bdt_ttcc.Integral())

      keras_ttcc = ttcc.Get('h_scoreKeras_'+ch+'_ttcc')
      keras_ttcc.SetName('keras_ttcc')
      if keras_ttcc.Integral() != 0: keras_ttcc.Scale(nttcc[idx]/keras_ttcc.Integral())

      target.cd()
      bdt_ttcc.Write()
      keras_ttcc.Write()
   
    elif scores == 'shape_'+ch+'_ttbj.root':
      ttbj = TFile.Open(path+scores)

      bdt_ttbj = ttbj.Get('h_scoreBDT_'+ch+'_ttbj')
      bdt_ttbj.SetName('bdt_ttbj')
      if bdt_ttbj.Integral() != 0: bdt_ttbj.Scale(nttbj[idx]/bdt_ttbj.Integral())

      keras_ttbj = ttbj.Get('h_scoreKeras_'+ch+'_ttbj')
      keras_ttbj.SetName('keras_ttbj')
      if keras_ttbj.Integral() != 0: keras_ttbj.Scale(nttbj[idx]/keras_ttbj.Integral())

      target.cd()
      bdt_ttbj.Write()
      keras_ttbj.Write()

    elif scores == 'shape_'+ch+'_tchannel.root':
      st = TFile.Open(path+scores)
      stbar = TFile.Open(path+'shape_'+ch+'_tbarchannel.root')
      tw = TFile.Open(path+'shape_'+ch+'_tWchannel.root')
      tbarw = TFile.Open(path+'shape_'+ch+'_tbarWchannel.root')

      bdt_singletop = st.Get('h_scoreBDT_'+ch+'_tchannel')
      if bdt_singletop.Integral() != 0: bdt_singletop.Scale(nsinglet[idx]/bdt_singletop.Integral())
      bdt_stbar = stbar.Get('h_scoreBDT_'+ch+'_tbarchannel')
      if bdt_stbar.Integral() != 0: bdt_stbar.Scale(nsingletbar[idx]/bdt_stbar.Integral())
      bdt_singletop.Add(bdt_singletop, bdt_stbar, 1.0, 1.0)
      bdt_tw = tw.Get('h_scoreBDT_'+ch+'_tWchannel')
      if bdt_tw.Integral() != 0: bdt_tw.Scale(nsingletW[idx]/bdt_tw.Integral())
      bdt_singletop.Add(bdt_singletop, bdt_tw, 1.0, 1.0)
      bdt_tbarw = tbarw.Get('h_scoreBDT_'+ch+'_tbarWchannel')
      if bdt_tbarw.Integral() != 0: bdt_tbarw.Scale(nsingletbarW[idx]/bdt_tbarw.Integral())
      bdt_singletop.Add(bdt_singletop, bdt_tbarw, 1.0, 1.0)
      bdt_singletop.SetName('bdt_singletop')

      keras_singletop = st.Get('h_scoreKeras_'+ch+'_tchannel')
      if keras_singletop.Integral() != 0: keras_singletop.Scale(nsinglet[idx]/keras_singletop.Integral())
      keras_stbar = stbar.Get('h_scoreKeras_'+ch+'_tbarchannel')
      if keras_stbar.Integral() != 0: keras_stbar.Scale(nsingletbar[idx]/keras_stbar.Integral())
      keras_singletop.Add(keras_singletop, keras_stbar, 1.0, 1.0)
      keras_tw = tw.Get('h_scoreKeras_'+ch+'_tWchannel')
      if keras_tw.Integral() != 0: keras_tw.Scale(nsingletW[idx]/keras_tw.Integral())
      keras_singletop.Add(keras_singletop, keras_tw, 1.0, 1.0)
      keras_tbarw = tbarw.Get('h_scoreKeras_'+ch+'_tbarWchannel')
      if keras_tbarw.Integral() != 0: keras_tbarw.Scale(nsingletbarW[idx]/keras_tbarw.Integral())
      keras_singletop.Add(keras_singletop, keras_tbarw, 1.0, 1.0)
      keras_singletop.SetName('keras_singletop')

      target.cd()
      bdt_singletop.Write()
      keras_singletop.Write()

    elif scores == 'shape_'+ch+'_zjets.root':
      zjets = TFile.Open(path+scores)
      zjets10to50 = TFile.Open(path+'shape_'+ch+'_zjets10to50.root')
      w1jets50to150 = TFile.Open(path+'shape_'+ch+'_w1jets50to150.root')
      w1jets150to250 = TFile.Open(path+'shape_'+ch+'_w1jets150to250.root')
      #w1jets250to400 = TFile.Open(path+'shape_'+ch+'_w1jets250to400.root')
      #w1jets400toinf = TFile.Open(path+'shape_'+ch+'_w1jets400toinf.root')
      w2jets250to400 = TFile.Open(path+'shape_'+ch+'_w2jets250to400.root')
      w2jets400toinf = TFile.Open(path+'shape_'+ch+'_w2jets400toinf.root')
      w3jets = TFile.Open(path+'shape_'+ch+'_w3jets.root')
      w4jets = TFile.Open(path+'shape_'+ch+'_w4jets.root')
      ww = TFile.Open(path+'shape_'+ch+'_ww.root')
      wz = TFile.Open(path+'shape_'+ch+'_wz.root')
      zz = TFile.Open(path+'shape_'+ch+'_zz.root')

      bdt_others = zjets.Get('h_scoreBDT_'+ch+'_zjets')
      if bdt_others.Integral() != 0: bdt_others.Scale(nzjet[idx]/bdt_others.Integral())
      bdt_zjets10to50 = zjets10to50.Get('h_scoreBDT_'+ch+'_zjets10to50')
      if bdt_zjets10to50.Integral() != 0: bdt_zjets10to50.Scale(nzjet10to50[idx]/bdt_zjets10to50.Integral())
      bdt_others.Add(bdt_others, bdt_zjets10to50, 1.0, 1.0)

      bdt_w1jets50to150 = w1jets50to150.Get('h_scoreBDT_'+ch+'_w1jets50to150')
      if bdt_w1jets50to150.Integral() != 0: bdt_w1jets50to150.Scale(nw1jet50to150[idx]/bdt_w1jets50to150.Integral())
      bdt_others.Add(bdt_others, bdt_w1jets50to150, 1.0, 1.0)
      bdt_w1jets150to250 = w1jets150to250.Get('h_scoreBDT_'+ch+'_w1jets150to250')
      if bdt_w1jets150to250.Integral() != 0: bdt_w1jets150to250.Scale(nw1jet150to250[idx]/bdt_w1jets150to250.Integral())
      bdt_others.Add(bdt_others, bdt_w1jets150to250, 1.0, 1.0)
      #bdt_w1jets250to400 = w1jets250to400.Get('h_scoreBDT_'+ch+'_w1jets250to400')
      #if bdt_w1jets250to400.Integral() != 0: bdt_w1jets250to400.Scale(nw1jet250to400[idx]/bdt_w1jets250to400.Integral())
      #bdt_others.Add(bdt_others, bdt_w1jets250to400, 1.0, 1.0)
      #bdt_w1jets400toinf = w1jets400toinf.Get('h_scoreBDT_'+ch+'_w1jets400toinf')
      #if bdt_w1jets400toinf.Integral() != 0: bdt_w1jets400toinf.Scale(nw1jet400toinf[idx]/bdt_w1jets400toinf.Integral())
      #bdt_others.Add(bdt_others, bdt_w1jets400toinf, 1.0, 1.0)

      bdt_w2jets250to400 = w2jets250to400.Get('h_scoreBDT_'+ch+'_w2jets250to400')
      if bdt_w2jets250to400.Integral() != 0: bdt_w2jets250to400.Scale(nw2jet250to400[idx]/bdt_w2jets250to400.Integral())
      bdt_others.Add(bdt_others, bdt_w2jets250to400, 1.0, 1.0)
      bdt_w2jets400toinf = w2jets400toinf.Get('h_scoreBDT_'+ch+'_w2jets400toinf')
      if bdt_w2jets400toinf.Integral() != 0: bdt_w2jets400toinf.Scale(nw2jet400toinf[idx]/bdt_w2jets400toinf.Integral())
      bdt_others.Add(bdt_others, bdt_w2jets400toinf, 1.0, 1.0)

      bdt_w3jets = w3jets.Get('h_scoreBDT_'+ch+'_w3jets')
      if bdt_w3jets.Integral() != 0: bdt_w3jets.Scale(nw3jet[idx]/bdt_w3jets.Integral())
      bdt_others.Add(bdt_others, bdt_w3jets, 1.0, 1.0)
      bdt_w4jets = w4jets.Get('h_scoreBDT_'+ch+'_w4jets')
      if bdt_w4jets.Integral() != 0: bdt_w4jets.Scale(nw4jet[idx]/bdt_w4jets.Integral())
      bdt_others.Add(bdt_others, bdt_w4jets, 1.0, 1.0)

      bdt_ww = ww.Get('h_scoreBDT_'+ch+'_ww')
      if bdt_ww.Integral() != 0: bdt_ww.Scale(nww[idx]/bdt_ww.Integral())
      bdt_others.Add(bdt_others, bdt_ww, 1.0, 1.0)
      bdt_wz = wz.Get('h_scoreBDT_'+ch+'_wz')
      if bdt_wz.Integral() != 0: bdt_wz.Scale(nwz[idx]/bdt_wz.Integral())
      bdt_others.Add(bdt_others, bdt_wz, 1.0, 1.0)
      bdt_zz = zz.Get('h_scoreBDT_'+ch+'_zz')
      if bdt_zz.Integral() != 0: bdt_zz.Scale(nzz[idx]/bdt_zz.Integral())
      bdt_others.Add(bdt_others, bdt_zz, 1.0, 1.0)
      bdt_others.SetName('bdt_others')


      keras_others = zjets.Get('h_scoreKeras_'+ch+'_zjets')
      if keras_others.Integral() != 0: keras_others.Scale(nzjet[idx]/keras_others.Integral())
      keras_zjets10to50 = zjets10to50.Get('h_scoreKeras_'+ch+'_zjets10to50')
      if keras_zjets10to50.Integral() != 0: keras_zjets10to50.Scale(nzjet10to50[idx]/keras_zjets10to50.Integral())
      keras_others.Add(keras_others, keras_zjets10to50, 1.0, 1.0)

      keras_w1jets50to150 = w1jets50to150.Get('h_scoreKeras_'+ch+'_w1jets50to150')
      if keras_w1jets50to150.Integral() != 0: keras_w1jets50to150.Scale(nw1jet50to150[idx]/keras_w1jets50to150.Integral())
      keras_others.Add(keras_others, keras_w1jets50to150, 1.0, 1.0)
      keras_w1jets150to250 = w1jets150to250.Get('h_scoreKeras_'+ch+'_w1jets150to250')
      if keras_w1jets150to250.Integral() != 0: keras_w1jets150to250.Scale(nw1jet150to250[idx]/keras_w1jets150to250.Integral())
      keras_others.Add(keras_others, keras_w1jets150to250, 1.0, 1.0)
      #keras_w1jets250to400 = w1jets250to400.Get('h_scoreKeras_'+ch+'_w1jets250to400')
      #if keras_w1jets250to400.Integral() != 0: keras_w1jets250to400.Scale(nw1jet250to400[idx]/keras_w1jets250to400.Integral())
      #keras_others.Add(keras_others, keras_w1jets250to400, 1.0, 1.0)
      #keras_w1jets400toinf = w1jets400toinf.Get('h_scoreKeras_'+ch+'_w1jets400toinf')
      #if keras_w1jets400toinf.Integral() != 0: keras_w1jets400toinf.Scale(nw1jet400toinf[idx]/keras_w1jets400toinf.Integral())
      #keras_others.Add(keras_others, keras_w1jets400toinf, 1.0, 1.0)

      keras_w2jets250to400 = w2jets250to400.Get('h_scoreKeras_'+ch+'_w2jets250to400')
      if keras_w2jets250to400.Integral() != 0: keras_w2jets250to400.Scale(nw2jet250to400[idx]/keras_w2jets250to400.Integral())
      keras_others.Add(keras_others, keras_w2jets250to400, 1.0, 1.0)
      keras_w2jets400toinf = w2jets400toinf.Get('h_scoreKeras_'+ch+'_w2jets400toinf')
      if keras_w2jets400toinf.Integral() != 0: keras_w2jets400toinf.Scale(nw2jet400toinf[idx]/keras_w2jets400toinf.Integral())
      keras_others.Add(keras_others, keras_w2jets400toinf, 1.0, 1.0)

      keras_w3jets = w3jets.Get('h_scoreKeras_'+ch+'_w3jets')
      if keras_w3jets.Integral() != 0: keras_w3jets.Scale(nw3jet[idx]/keras_w3jets.Integral())
      keras_others.Add(keras_others, keras_w3jets, 1.0, 1.0)
      keras_w4jets = w4jets.Get('h_scoreKeras_'+ch+'_w4jets')
      if keras_w4jets.Integral() != 0: keras_w4jets.Scale(nw4jet[idx]/keras_w4jets.Integral())
      keras_others.Add(keras_others, keras_w4jets, 1.0, 1.0)

      keras_ww = ww.Get('h_scoreKeras_'+ch+'_ww')
      if keras_ww.Integral() != 0: keras_ww.Scale(nww[idx]/keras_ww.Integral())
      keras_others.Add(keras_others, keras_ww, 1.0, 1.0)
      keras_wz = wz.Get('h_scoreKeras_'+ch+'_wz')
      if keras_wz.Integral() != 0: keras_wz.Scale(nwz[idx]/keras_wz.Integral())
      keras_others.Add(keras_others, keras_wz, 1.0, 1.0)
      keras_zz = zz.Get('h_scoreKeras_'+ch+'_zz')
      if keras_zz.Integral() != 0: keras_zz.Scale(nzz[idx]/keras_zz.Integral())
      keras_others.Add(keras_others, keras_zz, 1.0, 1.0)
      keras_others.SetName('keras_others')

      target.cd()
      bdt_others.Write()
      keras_others.Write()

    elif scores == 'shape_'+ch+'_SingleLepton_Run2017.root':
      rd = TFile.Open(path+'shape_'+ch+'_SingleLepton_Run2017.root')
      
      bdt_data_obs = rd.Get('h_scoreBDT_'+ch+'_SingleLepton_Run2017')
      bdt_data_obs.SetName('bdt_data_obs')

      keras_data_obs = rd.Get('h_scoreKeras_'+ch+'_SingleLepton_Run2017')
      keras_data_obs.SetName('keras_data_obs')

      target.cd()
      bdt_data_obs.Write()
      keras_data_obs.Write()
     
    elif scores == 'shape_'+ch+'_singleTopHct.root' and tag == "hct":
      tch = TFile.Open(path+'shape_'+ch+'_singleTopHct.root')

      bdt_sig = tch.Get('h_scoreBDT_'+ch+'_singleTopHct')
      bdt_scale1 = nsthct[idx]/bdt_sig.Integral()
      bdt_sig.Scale(bdt_scale1)
      bdt_sig.SetName('bdt_sig')

      keras_sig = tch.Get('h_scoreKeras_'+ch+'_singleTopHct')
      keras_scale1 = nsthct[idx]/keras_sig.Integral()
      keras_sig.Scale(keras_scale1)
      keras_sig.SetName('keras_sig')

      bdt_sig_gen = tch.Get('h_scoreBDTGen_'+ch+'_singleTopHct')
      bdt_sig_gen.Scale(bdt_scale1)
      bdt_sig_gen.SetName('bdt_sig_gen')

      keras_sig_gen = tch.Get('h_scoreKerasGen_'+ch+'_singleTopHct')
      keras_sig_gen.Scale(keras_scale1)
      keras_sig_gen.SetName('keras_sig_gen')

      target.cd()
      bdt_sig.Write()
      keras_sig.Write()
      bdt_sig_gen.Write()
      keras_sig_gen.Write()

    elif scores == 'shape_'+ch+'_singleTopHut.root' and tag == "hut":
      tch = TFile.Open(path+'shape_'+ch+'_singleTopHut.root')

      bdt_sig = tch.Get('h_scoreBDT_'+ch+'_singleTopHut')
      bdt_scale1 = nsthut[idx]/bdt_sig.Integral()
      bdt_sig.Scale(bdt_scale1)
      bdt_sig.SetName('bdt_sig')

      keras_sig = tch.Get('h_scoreKeras_'+ch+'_singleTopHut')
      keras_scale1 = nsthut[idx]/keras_sig.Integral()
      keras_sig.Scale(keras_scale1)
      keras_sig.SetName('keras_sig')

      bdt_sig_gen = tch.Get('h_scoreBDTGen_'+ch+'_singleTopHut')
      bdt_sig_gen.Scale(bdt_scale1)
      bdt_sig_gen.SetName('bdt_sig_gen')

      keras_sig_gen = tch.Get('h_scoreKerasGen_'+ch+'_singleTopHut')
      keras_sig_gen.Scale(keras_scale1)
      keras_sig_gen.SetName('keras_sig_gen')

      target.cd()
      bdt_sig.Write()
      keras_sig.Write()
      bdt_sig_gen.Write()
      keras_sig_gen.Write()

    else: continue
 
  target.Close()

from ROOT import *
import pandas as pd
import os

base_path17 = "/data/users/minerva1993/ntuple_Run2017/V9_6/190702/production"
base_path18 = "/data/users/minerva1993/ntuple_Run2018/V10_2/190702/production"

def input_variables_bdt(jetcat): #Order Does Matter!!
 
  #var_list = ['STTT', 'channel', 'MET', 'lepDPhi', 'lepWpt', 'lepWm']
  var_list = []

  var_list.extend(['stfcnc_jet0pt', 'stfcnc_jet0eta', 'stfcnc_jet0m', 'stfcnc_jet0csv',
                  'stfcnc_jet1pt', 'stfcnc_jet1eta', 'stfcnc_jet1m', 'stfcnc_jet1csv',
                  'stfcnc_jet2pt', 'stfcnc_jet2eta', 'stfcnc_jet2m', 'stfcnc_jet2csv',
                  'stfcnc_jet12pt', 'stfcnc_jet12eta', 'stfcnc_jet12deta',
                  'stfcnc_jet12dphi', 'stfcnc_jet12dR', 'stfcnc_jet12m',
                  'stfcnc_lepTdphi', 'stfcnc_lepTm',
                  'stfcnc_jet0lepdR', 'stfcnc_jet1lepdR', 'stfcnc_jet2lepdR',
                  'stfcnc_jet01dR', 'stfcnc_jet02dR',
                  'stfcnc_jet12_lepdR', 'stfcnc_jet12_0dR',
                  'stfcnc_lepTjet12dphi'])
  if "j4" in jetcat:
    var_list.extend(['ttfcnc_jet0pt', 'ttfcnc_jet0eta', 'ttfcnc_jet0m', 'ttfcnc_jet0csv',
                    'ttfcnc_jet1pt', 'ttfcnc_jet1eta', 'ttfcnc_jet1m',  'ttfcnc_jet1csv',
                    'ttfcnc_jet2pt', 'ttfcnc_jet2eta', 'ttfcnc_jet2m',  'ttfcnc_jet2csv',
                    'ttfcnc_jet3pt', 'ttfcnc_jet3eta', 'ttfcnc_jet3m',  'ttfcnc_jet3csv',
                    'ttfcnc_jet12pt', 'ttfcnc_jet12eta', 'ttfcnc_jet12deta',
                    'ttfcnc_jet12dphi', 'ttfcnc_jet12dR', 'ttfcnc_jet12m',
                    'ttfcnc_jet23pt', 'ttfcnc_jet23eta','ttfcnc_jet23deta',
                    'ttfcnc_jet23dphi', 'ttfcnc_jet23dR', 'ttfcnc_jet23m',
                    'ttfcnc_jet31pt', 'ttfcnc_jet31eta', 'ttfcnc_jet31deta',
                    'ttfcnc_jet31dphi', 'ttfcnc_jet31dR', 'ttfcnc_jet31m',
                    'ttfcnc_lepTpt', 'ttfcnc_lepTdphi', 'ttfcnc_lepTm',
                    'ttfcnc_hadTpt', 'ttfcnc_hadTeta',
                    'ttfcnc_hadT12_3deta', 'ttfcnc_hadT23_1deta', 'ttfcnc_hadT31_2deta',
                    'ttfcnc_hadT12_3dphi', 'ttfcnc_hadT23_1dphi', 'ttfcnc_hadT31_2dphi',
                    'ttfcnc_hadT12_3dR', 'ttfcnc_hadT23_1dR', 'ttfcnc_hadT31_2dR','ttfcnc_hadTm',
                    'ttfcnc_jet0lepdR', 'ttfcnc_jet1lepdR', 'ttfcnc_jet2lepdR', 'ttfcnc_jet3lepdR',
                    'ttfcnc_jet01dR', 'ttfcnc_jet02dR', 'ttfcnc_jet03dR',
                    'ttfcnc_jet12_lepdR', 'ttfcnc_jet23_lepdR', 'ttfcnc_jet31_lepdR',
                    'ttfcnc_jet12_0dR', 'ttfcnc_jet23_0dR', 'ttfcnc_jet31_0dR',
                    'ttfcnc_lepTjet12dphi', 'ttfcnc_lepTjet23dphi', 'ttfcnc_lepTjet31dphi', 'ttfcnc_hadT_jet0dR',])
    var_list.extend(['ttbkg_jet0pt', 'ttbkg_jet0eta', 'ttbkg_jet0m', 'ttbkg_jet0csv',
                    'ttbkg_jet1pt', 'ttbkg_jet1eta', 'ttbkg_jet1m', 'ttbkg_jet1csv',
                    'ttbkg_jet2pt', 'ttbkg_jet2eta', 'ttbkg_jet2m', 'ttbkg_jet2csv',
                    'ttbkg_jet3pt', 'ttbkg_jet3eta', 'ttbkg_jet3m', 'ttbkg_jet3csv',
                    'ttbkg_jet12pt', 'ttbkg_jet12eta', 'ttbkg_jet12deta',
                    'ttbkg_jet12dphi', 'ttbkg_jet12dR', 'ttbkg_jet12m',
                    'ttbkg_jet23pt', 'ttbkg_jet23eta','ttbkg_jet23deta', 
                    'ttbkg_jet23dphi', 'ttbkg_jet23dR', 'ttbkg_jet23m',
                    'ttbkg_jet31pt', 'ttbkg_jet31eta', 'ttbkg_jet31deta',
                    'ttbkg_jet31dphi', 'ttbkg_jet31dR', 'ttbkg_jet31m',
                    'ttbkg_lepTpt', 'ttbkg_lepTdphi', 'ttbkg_lepTm',
                    'ttbkg_hadTpt', 'ttbkg_hadTeta', 
                    'ttbkg_hadT12_3deta', 'ttbkg_hadT23_1deta', 'ttbkg_hadT31_2deta',
                    'ttbkg_hadT12_3dphi', 'ttbkg_hadT23_1dphi', 'ttbkg_hadT31_2dphi',
                    'ttbkg_hadT12_3dR', 'ttbkg_hadT23_1dR', 'ttbkg_hadT31_2dR','ttbkg_hadTm',
                    'ttbkg_jet0lepdR', 'ttbkg_jet1lepdR', 'ttbkg_jet2lepdR', 'ttbkg_jet3lepdR',
                    'ttbkg_jet01dR', 'ttbkg_jet02dR', 'ttbkg_jet03dR',
                    'ttbkg_jet12_lepdR', 'ttbkg_jet23_lepdR', 'ttbkg_jet31_lepdR',
                    'ttbkg_jet12_0dR', 'ttbkg_jet23_0dR', 'ttbkg_jet31_0dR',
                    'ttbkg_lepTjet12dphi', 'ttbkg_lepTjet23dphi', 'ttbkg_lepTjet31dphi', 'ttbkg_hadT_jet0dR',])

    print("Adding variables for 4 jet category")

  return var_list


def input_variables(jetcat):
 
  #var_list = ['STTT', 'channel', 'MET', 'lepDPhi', 'lepWpt', 'lepWm']
  var_list = []

  var_list.extend(['stfcnc_jet0pt', 'stfcnc_jet0eta', 'stfcnc_jet0m', 'stfcnc_jet0csv',
                  'stfcnc_jet1pt', 'stfcnc_jet1eta', 'stfcnc_jet1m', 'stfcnc_jet1csv',
                  'stfcnc_jet2pt', 'stfcnc_jet2eta', 'stfcnc_jet2m', 'stfcnc_jet2csv',
                  'stfcnc_jet12pt', 'stfcnc_jet12eta', 'stfcnc_jet12m',
                  'stfcnc_jet12deta', 'stfcnc_jet12dphi', 'stfcnc_jet12dR',
                  'stfcnc_lepTpt', 'stfcnc_lepTdphi', 'stfcnc_lepTm',
                  'stfcnc_jet0lepdR', 'stfcnc_jet1lepdR', 'stfcnc_jet2lepdR',
                  'stfcnc_jet01dR', 'stfcnc_jet02dR',
                  'stfcnc_jet12_lepdR', 'stfcnc_jet12_0dR',
                  'stfcnc_lepTjet12dphi'])
  if "j4" in jetcat:
    var_list.extend(['ttfcnc_jet0pt', 'ttfcnc_jet0eta', 'ttfcnc_jet0m', 'ttfcnc_jet0csv',
                    'ttfcnc_jet1pt', 'ttfcnc_jet1eta', 'ttfcnc_jet1m', 'ttfcnc_jet1csv',
                    'ttfcnc_jet2pt', 'ttfcnc_jet2eta', 'ttfcnc_jet2m', 'ttfcnc_jet2csv',
                    'ttfcnc_jet3pt', 'ttfcnc_jet3eta', 'ttfcnc_jet3m', 'ttfcnc_jet3csv',
                    'ttfcnc_jet12pt', 'ttfcnc_jet12eta', 'ttfcnc_jet12m',
                    'ttfcnc_jet12deta', 'ttfcnc_jet12dphi', 'ttfcnc_jet12dR',
                    'ttfcnc_jet23pt', 'ttfcnc_jet23eta','ttfcnc_jet23m',
                    'ttfcnc_jet23deta', 'ttfcnc_jet23dphi', 'ttfcnc_jet23dR',
                    'ttfcnc_jet31pt', 'ttfcnc_jet31eta', 'ttfcnc_jet31m',
                    'ttfcnc_jet31deta', 'ttfcnc_jet31dphi', 'ttfcnc_jet31dR',
                    'ttfcnc_lepTpt', 'ttfcnc_lepTdphi', 'ttfcnc_lepTm',
                    'ttfcnc_hadTpt', 'ttfcnc_hadTeta', 'ttfcnc_hadTm',
                    'ttfcnc_hadT12_3deta', 'ttfcnc_hadT23_1deta', 'ttfcnc_hadT31_2deta',
                    'ttfcnc_hadT12_3dphi', 'ttfcnc_hadT23_1dphi', 'ttfcnc_hadT31_2dphi',
                    'ttfcnc_hadT12_3dR', 'ttfcnc_hadT23_1dR', 'ttfcnc_hadT31_2dR',
                    'ttfcnc_jet0lepdR', 'ttfcnc_jet1lepdR', 'ttfcnc_jet2lepdR', 'ttfcnc_jet3lepdR',
                    'ttfcnc_jet01dR', 'ttfcnc_jet02dR', 'ttfcnc_jet03dR',
                    'ttfcnc_jet12_lepdR', 'ttfcnc_jet23_lepdR', 'ttfcnc_jet31_lepdR',
                    'ttfcnc_jet12_0dR', 'ttfcnc_jet23_0dR', 'ttfcnc_jet31_0dR',
                    'ttfcnc_lepTjet12dphi', 'ttfcnc_lepTjet23dphi', 'ttfcnc_lepTjet31dphi', 'ttfcnc_hadT_jet0dR',])
    var_list.extend(['ttbkg_jet0pt', 'ttbkg_jet0eta', 'ttbkg_jet0m', 'ttbkg_jet0csv',
                    'ttbkg_jet1pt', 'ttbkg_jet1eta', 'ttbkg_jet1m', 'ttbkg_jet1csv',
                    'ttbkg_jet2pt', 'ttbkg_jet2eta', 'ttbkg_jet2m', 'ttbkg_jet2csv',
                    'ttbkg_jet3pt', 'ttbkg_jet3eta', 'ttbkg_jet3m', 'ttbkg_jet3csv',
                    'ttbkg_jet12pt', 'ttbkg_jet12eta', 'ttbkg_jet12m',
                    'ttbkg_jet12deta', 'ttbkg_jet12dphi', 'ttbkg_jet12dR',
                    'ttbkg_jet23pt', 'ttbkg_jet23eta','ttbkg_jet23m',
                    'ttbkg_jet23deta', 'ttbkg_jet23dphi', 'ttbkg_jet23dR',
                    'ttbkg_jet31pt', 'ttbkg_jet31eta', 'ttbkg_jet31m',
                    'ttbkg_jet31deta', 'ttbkg_jet31dphi', 'ttbkg_jet31dR',
                    'ttbkg_lepTpt', 'ttbkg_lepTdphi', 'ttbkg_lepTm',
                    'ttbkg_hadTpt', 'ttbkg_hadTeta', 'ttbkg_hadTm',
                    'ttbkg_hadT12_3deta', 'ttbkg_hadT23_1deta', 'ttbkg_hadT31_2deta',
                    'ttbkg_hadT12_3dphi', 'ttbkg_hadT23_1dphi', 'ttbkg_hadT31_2dphi',
                    'ttbkg_hadT12_3dR', 'ttbkg_hadT23_1dR', 'ttbkg_hadT31_2dR',
                    'ttbkg_jet0lepdR', 'ttbkg_jet1lepdR', 'ttbkg_jet2lepdR', 'ttbkg_jet3lepdR',
                    'ttbkg_jet01dR', 'ttbkg_jet02dR', 'ttbkg_jet03dR',
                    'ttbkg_jet12_lepdR', 'ttbkg_jet23_lepdR', 'ttbkg_jet31_lepdR',
                    'ttbkg_jet12_0dR', 'ttbkg_jet23_0dR', 'ttbkg_jet31_0dR',
                    'ttbkg_lepTjet12dphi', 'ttbkg_lepTjet23dphi', 'ttbkg_lepTjet31dphi', 'ttbkg_hadT_jet0dR',])

    print("Adding variables for 4 jet category")

  return var_list


def train_files(ch, era):

  if era == "2017":
    if ch == "Hct":
      sig = ['finalMVA_STTH1L3BHct_000.h5', 'finalMVA_STTH1L3BHct_001.h5',
             'finalMVA_STTH1L3BHct_002.h5', 'finalMVA_STTH1L3BHct_003.h5',
             'finalMVA_TTTH1L3BaTLepHct_000.h5', 'finalMVA_TTTH1L3BTLepHct_000.h5',
             'finalMVA_TTTH1L3BaTLepHct_001.h5', 'finalMVA_TTTH1L3BTLepHct_001.h5',
            ]
    elif ch == "Hut":
      sig = ['finalMVA_STTH1L3BHut_000.h5', 'finalMVA_STTH1L3BHut_001.h5',
             'finalMVA_STTH1L3BHut_002.h5', 'finalMVA_STTH1L3BHut_003.h5',
             'finalMVA_TTTH1L3BaTLepHut_000.h5', 'finalMVA_TTTH1L3BTLepHut_000.h5',
             'finalMVA_TTTH1L3BaTLepHut_001.h5', 'finalMVA_TTTH1L3BTLepHut_001.h5',
            ]
    else:
      print("Check channel: Hct or Hut")
      sig_files = []

    bkg = []
    for i in range(0,5): #upto 5
      bkg.append('finalMVA_TTpowhegttbb_00'+str(i)+'.h5')
      bkg.append('finalMVA_TTpowhegttcc_00'+str(i)+'.h5')
    for i in range(0,10):
      bkg.append('finalMVA_TTpowhegttlf_00'+str(i)+'.h5')
    for i in range(10,25): #upto 24, *5
      bkg.append('finalMVA_TTpowhegttlf_0'+str(i)+'.h5')

    for i in range(0,3): #upto 2
      bkg.append('finalMVA_TTLLpowhegttbb_00'+str(i)+'.h5')
      bkg.append('finalMVA_TTLLpowhegttcc_00'+str(i)+'.h5')
    for i in range(0,9): #upto 8, *4
      bkg.append('finalMVA_TTLLpowhegttlf_00'+str(i)+'.h5')
    #for i in range(10,16): #upto 15, *4
    #  bkg.append('finalMVA_TTLLpowhegttlf_0'+str(i)+'.h5')

  elif era == "2018":
    if ch == "Hct":
      sig = ['finalMVA_STTH1L3BHct_000.h5', 'finalMVA_STTH1L3BHct_001.h5',
             'finalMVA_STTH1L3BHct_002.h5', 'finalMVA_STTH1L3BHct_003.h5',
             'finalMVA_STTH1L3BHct_004.h5', 'finalMVA_STTH1L3BHct_005.h5',
             'finalMVA_TTTH1L3BaTLepHct_000.h5', 'finalMVA_TTTH1L3BTLepHct_000.h5',
             'finalMVA_TTTH1L3BaTLepHct_001.h5', 'finalMVA_TTTH1L3BTLepHct_001.h5',
            ]
    elif ch == "Hut":
      sig = ['finalMVA_STTH1L3BHut_000.h5', 'finalMVA_STTH1L3BHut_001.h5',
             'finalMVA_STTH1L3BHut_002.h5', 'finalMVA_STTH1L3BHut_003.h5',
             'finalMVA_STTH1L3BHut_004.h5', 'finalMVA_STTH1L3BHut_005.h5',
             'finalMVA_TTTH1L3BaTLepHut_000.h5', 'finalMVA_TTTH1L3BTLepHut_000.h5',
             'finalMVA_TTTH1L3BaTLepHut_001.h5', 'finalMVA_TTTH1L3BTLepHut_001.h5',
            ]
    else:
      print("Check channel: Hct or Hut")
      sig_files = []

    bkg = []
    for i in range(0,8): #upto 7, ~10%
      bkg.append('finalMVA_TTpowhegttbb_00'+str(i)+'.h5')
      bkg.append('finalMVA_TTpowhegttcc_00'+str(i)+'.h5')
    for i in range(0,10):
      bkg.append('finalMVA_TTpowhegttlf_00'+str(i)+'.h5')
    for i in range(10,40): #upto 39, *5
      bkg.append('finalMVA_TTpowhegttlf_0'+str(i)+'.h5')

    for i in range(0,4): #upto 3
      bkg.append('finalMVA_TTLLpowhegttbb_00'+str(i)+'.h5')
      bkg.append('finalMVA_TTLLpowhegttcc_00'+str(i)+'.h5')
    for i in range(0,10):
      bkg.append('finalMVA_TTLLpowhegttlf_00'+str(i)+'.h5')
    for i in range(10,16): #upto 15, *4
      bkg.append('finalMVA_TTLLpowhegttlf_0'+str(i)+'.h5')

  return sig, bkg

  
def evalScale(ch, era, sig, bkg):

  if   era == "2017": base_path = base_path17
  elif era == "2018": base_path = base_path18

  ST, TT, TTLJ, TTLL = (0,0,0,0)
  for tmp_file in sig:
    if "STTH" in tmp_file:
      ftmp = TFile.Open(os.path.join(base_path, "ST_TH_1L3B_" + ch,
                        "Tree_fcncLepJets_" + tmp_file.replace("h5","root").split("_")[-1]), "READ")
      info_tmp = ftmp.Get("fcncLepJets/EventInfo")
      ST += info_tmp.GetBinContent(2)
      ftmp.Close()

    elif "TTTH" in tmp_file:
      if "aTLep" in tmp_file: which_top = "aTLep"
      elif "TLep"  in tmp_file: which_top = "TLep"
      else: print("Wrong TT FCNC file name!")

      ftmp = TFile.Open(os.path.join(base_path, "TT_TH_1L3B_" + which_top + '_'  + ch,
                        "Tree_fcncLepJets_" + tmp_file.replace("h5","root").split("_")[-1]), "READ")
      info_tmp = ftmp.Get("fcncLepJets/EventInfo")
      TT += info_tmp.GetBinContent(2)
      ftmp.Close()

  for tmp_file in bkg:
    if "TTpowheg" in tmp_file:
      if "ttbb" not in tmp_file: continue #sum of weight is same for ttbb, ttcc, ttlf
      ftmp = TFile.Open(os.path.join(base_path, "TT_powheg_ttbb",
                        "Tree_fcncLepJets_" + tmp_file.replace("h5","root").split("_")[-1]), "READ")
      info_tmp = ftmp.Get("fcncLepJets/EventInfo")
      TTLJ += info_tmp.GetBinContent(2)
      ftmp.Close()

    elif "TTLL" in tmp_file:
      if "ttbb" not in tmp_file: continue #sum of weight is same for ttbb, ttcc, ttlf
      ftmp = TFile.Open(os.path.join(base_path, "TTLL_powheg_ttbb",
                        "Tree_fcncLepJets_" + tmp_file.replace("h5","root").split("_")[-1]), "READ")
      info_tmp = ftmp.Get("fcncLepJets/EventInfo")
      TTLL += info_tmp.GetBinContent(2)
      ftmp.Close()

  #normalize by cross section
  if   ch == 'Hct': STxsec = 1.9
  elif ch == 'Hut': STxsec = 13.84
  ST_ = STxsec / ST
  TT_ = 46.5 / TT
  TTLJ_ = 365.4 / TTLJ
  TTLL_ = 88.29 / TTLL
  frac_sig = ST_ / TT_
  frac_bkg = TTLL_ / TTLJ_
  frac_sig = round(frac_sig,4)
  frac_bkg = round(frac_bkg,4)

  return float(ST), float(TT), float(TTLJ), float(TTLL), frac_sig, frac_bkg
  #return frac_sig, frac_bkg


def evalFrac(ch, era, sig, nj, nbj):

  if   era == "2017": base_path = base_path17
  elif era == "2018": base_path = base_path18

  ST, TT = (0,0)
  for tmp_file in sig:
    if "STTH" in tmp_file:
      ftmp = TFile.Open(os.path.join(base_path, "ST_TH_1L3B_" + ch,
                        "Tree_fcncLepJets_" + tmp_file.replace("h5","root").split("_")[-1]), "READ")
      info_tmp = ftmp.Get("fcncLepJets/EventInfo")
      ST += info_tmp.GetBinContent(2)
      ftmp.Close()

    elif "TTTH" in tmp_file:
      if "aTLep" in tmp_file: which_top = "aTLep"
      elif "TLep"  in tmp_file: which_top = "TLep"
      else: print("Wrong TT FCNC file name!")

      ftmp = TFile.Open(os.path.join(base_path, "TT_TH_1L3B_" + which_top + '_'  + ch,
                        "Tree_fcncLepJets_" + tmp_file.replace("h5","root").split("_")[-1]), "READ")
      info_tmp = ftmp.Get("fcncLepJets/EventInfo")
      TT += info_tmp.GetBinContent(2)
      ftmp.Close()

  nST, nTT = (0,0)
  #Signal first
  for files in sig:
    data_temp = pd.read_hdf('./mkNtuple/' + era + '/hdf_/' + files)
    data_temp = data_temp[data_temp['njets'] ==  nj]
    if nbj != 0:
      data_temp = data_temp[data_temp['nbjets_m'] == nbj]

    if "STTH" in files:
      nST += len(data_temp.index)
    elif "TTTH" in files:
      nTT += len(data_temp.index)
    else: print("Wrong signal sample!")

  ST = 1.9 / ST
  TT = 46.5 / TT

  frac_list = [(nST * ST)/(nST * ST + nTT * TT), (nTT * TT)/(nST * ST + nTT * TT)]

  return frac_list

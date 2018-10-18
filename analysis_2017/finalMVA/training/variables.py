def input_variables(jetcat):
 
  var_list = ['channel', 'nbjets_m', 'lepton_pt', 'lepton_eta', 'MET', 'lepDPhi', 'lepWpt', 'lepWm']

  var_list.extend(['stfcnc_jet0pt', 'stfcnc_jet0eta', 'stfcnc_jet0m', 'b_stfcnc_jet0csv',
                  'stfcnc_jet1pt', 'stfcnc_jet1eta', 'stfcnc_jet1m', 'b_stfcnc_jet1csv',
                  'stfcnc_jet2pt', 'stfcnc_jet2eta', 'stfcnc_jet2m', 'b_stfcnc_jet2csv',
                  'stfcnc_jet12pt', 'stfcnc_jet12eta', 'stfcnc_jet12m',
                  'stfcnc_jet12deta', 'stfcnc_jet12dphi', 'stfcnc_jet12dR',
                  'stfcnc_lepTdphi', 'stfcnc_lepTm',
                  'stfcnc_jet0lepdR', 'stfcnc_jet1lepdR', 'stfcnc_jet2lepdR',
                  'stfcnc_jet01dR', 'stfcnc_jet02dR',
                  'stfcnc_jet12_lepdR', 'stfcnc_jet12_0dR',
                  'stfcnc_lepTjet12dphi'])
  if "j4" in jetcat:
    var_list.extend(['ttfcnc_jet0pt', 'ttfcnc_jet0eta', 'ttfcnc_jet0m',
                    'ttfcnc_jet1pt', 'ttfcnc_jet1eta', 'ttfcnc_jet1m',
                    'ttfcnc_jet2pt', 'ttfcnc_jet2eta', 'ttfcnc_jet2m',
                    'ttfcnc_jet3pt', 'ttfcnc_jet3eta', 'ttfcnc_jet3m',
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
                    'ttfcnc_jet12_0R', 'ttfcnc_jet23_0dR', 'ttfcnc_jet31_0dR'
                    'ttfcnc_lepTjet12dphi', 'ttfcnc_lepTjet23dphi', 'ttfcnc_lepTjet31dphi', 'ttfcnc_hadT_jet0dR',])
    var_list.extend(['ttbkg_jet0pt', 'ttbkg_jet0eta', 'ttbkg_jet0m',
                    'ttbkg_jet1pt', 'ttbkg_jet1eta', 'ttbkg_jet1m',
                    'ttbkg_jet2pt', 'ttbkg_jet2eta', 'ttbkg_jet2m',
                    'ttbkg_jet3pt', 'ttbkg_jet3eta', 'ttbkg_jet3m',
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
                    'ttbkg_jet12_0R', 'ttbkg_jet23_0dR', 'ttbkg_jet31_0dR'
                    'ttbkg_lepTjet12dphi', 'ttbkg_lepTjet23dphi', 'ttbkg_lepTjet31dphi', 'ttbkg_hadT_jet0dR',])

    print("Adding variables for4 jet category")

  return var_list


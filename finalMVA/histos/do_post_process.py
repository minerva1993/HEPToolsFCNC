from ROOT import *
import ROOT
import os

base_path = "./"
if not os.path.exists( base_path + "post_process" ):
  os.makedirs( base_path + "post_process" )

def write_envelope(syst, nhists, new_sumW):

  if (histos + "__" + syst + "0")  in histo_list:
    var_list = []
    for x in range(0,nhists):
      h = f.Get(histos + "__" + syst + str(x))
      if any(x in syst for x in ['scale', 'ps']):
        pass
      elif 'pdf' in syst:
        if x == 0: continue
        h.Scale(EventInfo.GetBinContent(2) / new_sumW.GetBinContent(1))
      else: h.Scale(EventInfo.GetBinContent(2) / new_sumW.GetBinContent(x+1))
      var_list.append(h)

    nominal = f.Get(histos)
    n_bins = nominal.GetNcells()
    up = nominal.Clone()
    up.SetDirectory(ROOT.nullptr)
    up.Reset()
    down = nominal.Clone()
    down.SetDirectory(ROOT.nullptr)
    down.Reset()

    for i in range(0, n_bins+1):
      minimum = float("inf")
      maximum = float("-inf")

      for v in var_list:
        c = v.GetBinContent(i)
        minimum = min(minimum, c)
        maximum = max(maximum, c)

      up.SetBinContent(i, maximum)
      down.SetBinContent(i, minimum)

    up = bSFNorm(up, bSFInfo)
    down = bSFNorm(down, bSFInfo)
    up.SetName(histos + "__" + syst + "up")
    down.SetName(histos + "__" + syst + "down")
    up.Write()
    down.Write()


def rescale(binNum, new_sumW): # rescale up/dn histos

  #binNum = [up_num, down_num] or [bin]
  if   len(binNum) == 2 : mode = 0 #up, down in one root file
  elif len(binNum) == 0 : mode = 1 #use dedicated sample
  else: mode == 99

  if mode == 0: #FIXME
    if (histos + "__" + syst_name + "up")  in histo_list:
      for x in binNum:
        if x == binNum[0]:
          up = f.Get(histos + "__" + syst_name + "up")
          up.Scale(EventInfo.GetBinContent(2) / sumW_hist.GetBinContent(x))
        elif x == binNum[1]:
          down = f.Get(histos + "__" + syst_name + "down")
          down.Scale(EventInfo.GetBinContent(2) / sumW_hist.GetBinContent(x))

      up.Write()
      down.Write()

  elif mode == 1:
    if syst_name in files:
      h = f.Get(histos)
      if not any(i in h.GetName() for i in ['Info', 'Weight']):
        h.Scale(nom_EventInfo.GetBinContent(2) / EventInfo.GetBinContent(2))

        if any(low_stat in syst_name for low_stat in ['Tune', 'hdamp']):
          bSFInfo_nom = fill_bSFInfo(nom_f)
          h_nom = nom_f.Get(histos)
          h_nom = bSFNorm(h_nom, bSFInfo_nom)

          if 'down' in files:
            f_opp = TFile.Open( os.path.join(pre_path, files.replace('down','up')), "READ")
          elif 'up' in files:
            f_opp = TFile.Open( os.path.join(pre_path, files.replace('up','down')), "READ")

          opp_EventInfo = f_opp.Get('EventInfo')
          bSFInfo_opp = fill_bSFInfo(f_opp)
          h_opp = f_opp.Get(histos)
          h_opp = bSFNorm(h_opp, bSFInfo_opp)
          h_opp.Scale(nom_EventInfo.GetBinContent(2) / opp_EventInfo.GetBinContent(2))

          for xbin in xrange(h.GetNbinsX()):
            if h_nom.GetBinContent(xbin+1) == 0: h.SetBinContent(xbin+1, 0.)
            else:
              ratio = h.GetBinContent(xbin+1) / h_nom.GetBinContent(xbin+1)
              diff = abs(h_nom.GetBinContent(xbin+1)-h.GetBinContent(xbin+1)) + abs(h_nom.GetBinContent(xbin+1)-h_opp.GetBinContent(xbin+1))
              if ratio > 1.: h.SetBinContent(xbin+1, h_nom.GetBinContent(xbin+1) + diff/2.)
              else: h.SetBinContent(xbin+1, h_nom.GetBinContent(xbin+1) - diff/2.)

#          for xbin in xrange(h.GetNbinsX()):
#            if h_nom.GetBinContent(xbin+1) == 0: h.SetBinContent(xbin+1, 0.)
#            else:
#              ratio = h.GetBinContent(xbin+1) / h_nom.GetBinContent(xbin+1)
#              if ratio > 1.2: h.SetBinContent(xbin+1, 1.2 * h_nom.GetBinContent(xbin+1))
#              elif ratio < 0.8: h.SetBinContent(xbin+1, 0.8 * h_nom.GetBinContent(xbin+1))

      f_new.cd()
      h.Write()


def fill_bSFInfo(f_in):
  bSFInfo_tmp = {}
  bSFInfo_tmp['Ch0_J0'] = f_in.Get("bSFInfo_Ch0_J0")
  bSFInfo_tmp['Ch0_J1'] = f_in.Get("bSFInfo_Ch0_J1")
  bSFInfo_tmp['Ch0_J2'] = f_in.Get("bSFInfo_Ch0_J2")
  bSFInfo_tmp['Ch1_J0'] = f_in.Get("bSFInfo_Ch1_J0")
  bSFInfo_tmp['Ch1_J1'] = f_in.Get("bSFInfo_Ch1_J1")
  bSFInfo_tmp['Ch1_J2'] = f_in.Get("bSFInfo_Ch1_J2")
  bSFInfo_tmp['Ch2_J0'] = f_in.Get("bSFInfo_Ch2_J0")
  bSFInfo_tmp['Ch2_J1'] = f_in.Get("bSFInfo_Ch2_J1")
  bSFInfo_tmp['Ch2_J2'] = f_in.Get("bSFInfo_Ch2_J2")
  if bSFInfo_tmp['Ch2_J0'].Integral() == 0 and bSFInfo_tmp['Ch2_J2'].Integral() == 0:
    bSFInfo_tmp.clear() #If sum is 0, remove items: for data, J0 = 0 in MVA

  return bSFInfo_tmp


def bSFNorm(htmp, infos):
  if any(infos):
    hname = htmp.GetName()
    keystr = ''
    if   'Ch0' in hname: keystr += 'Ch0_'
    elif 'Ch1' in hname: keystr += 'Ch1_'
    elif 'Ch2' in hname: keystr += 'Ch2_'

    if 'h_DNN' in hname:
      #case1: MVA -> j3b2, j3b3, j4b2, j4b3, j4b4
      if   'j3' in hname: keystr += 'J1'
      elif 'j4' in hname: keystr += 'J2'
      infotmp = infos[keystr].Clone()
    else:
      #case2: fullAna -> S1,2,3,5,6,7,8
      if   'S0'in hname: keystr += 'J0'
      elif any(i in hname for i in ['S1','S2','S3','S4']): keystr += 'J1'
      elif any(i in hname for i in ['S5','S6','S7','S8']): keystr += 'J2'
      infotmp = infos[keystr].Clone()
      if 'S4' in hname: infotmp.Add(infos[keystr.replace('J1','J2')], 1.0)

    binnum = 2 #nominal = 1
    if any(i in hname for i in ['__lf', '__hf', '__cferr']):
      binnum = infotmp.GetXaxis().FindBin(str(hname.split('__')[-1]))
    if infotmp.GetBinContent(binnum) > 0:
      htmp.Scale(infotmp.GetBinContent(1)/infotmp.GetBinContent(binnum))

  return htmp


#Starts loop over histogram root files
file_list = os.listdir( os.path.join(base_path, "pre_process") )
if "systamatics" in file_list: file_list.remove("systematics")

pre_path = os.path.join(base_path, "pre_process")

if not os.path.exists( pre_path ):
  os.makedirs( pre_path )

for files in file_list:

  print files

  #Prepare root file
  f = TFile.Open( os.path.join(pre_path, files), "READ")

  histo_list = []
  histo_list = [x.GetName() for x in f.GetListOfKeys()]

  EventInfo = f.Get("EventInfo")
  ScaleWeights = f.Get("ScaleWeights")
  PSWeights = f.Get("PSWeights")
  PDFWeights = f.Get("PDFWeights")
  nScaleWeight = ScaleWeights.Integral()
  nPSWeight = PSWeights.Integral()
  nPDFWeight = PDFWeights.Integral()

  #bSF - 6 histos
  bSFInfo = fill_bSFInfo(f)

  #Prepare nominal file for rescaling
  syst_name = ""
  if "__" in files:
    run_on_syst = True
    syst_name = files.split("__")[1].replace(".root","")
    nom_f = TFile.Open( os.path.join(pre_path, files.replace("__" + syst_name,"")), "READ")
    nom_EventInfo = nom_f.Get("EventInfo")
  else: run_on_syst = False

  #Creat output file, in post_process folder
  post_path = os.path.join(base_path, "post_process")
  if os.path.exists(os.path.join(post_path, files)):
    print files + " exists!"
    continue
  f_new = TFile.Open( os.path.join(post_path, files), "RECREATE")

  #Store nominal names, drop scale vars.
  nominal_list = []
  for histos in histo_list:
    if "__" not in histos: nominal_list.append(histos)
    if "scale" in histos: continue
    if "ps" in histos: continue
    if "pdf" in histos: continue
    h = f.Get(histos)
    if not any(i in h.GetName() for i in ['Info', 'Weight']):
      h = bSFNorm(h, bSFInfo)
    else: pass
    h.Write()

  #Store envelope, rescale histos
  for histos in nominal_list:

    if nScaleWeight > 0: write_envelope("scale", 6, ScaleWeights)
    if nPSWeight > 0: write_envelope("ps", 4, PSWeights)
    if nPDFWeight > 0: write_envelope("pdf", 103, PDFWeights)
    if run_on_syst: rescale([], nom_EventInfo)

  f_new.Write()
  f_new.Close()
  f.Close()

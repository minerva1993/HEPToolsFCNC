from ROOT import *
import ROOT
import os

base_path = "./"
if not os.path.exists( base_path + "post_process" ):
  os.makedirs( base_path + "post_process" )

nrebin = 1

def symmetrize(var, var_opp, nom):

  for xbin in xrange(var.GetNbinsX()):
    if nom.GetBinContent(xbin+1) == 0: ratio = 1.
    else:
      ratio = var.GetBinContent(xbin+1) / nom.GetBinContent(xbin+1)
      ratio_opp = 1.
      if var_opp.GetBinContent(xbin+1) > 0: ratio_opp = var.GetBinContent(xbin+1) / var_opp.GetBinContent(xbin+1)

      diff = abs(nom.GetBinContent(xbin+1)-var.GetBinContent(xbin+1)) + abs(nom.GetBinContent(xbin+1)-var_opp.GetBinContent(xbin+1))
      if ratio_opp > 1.:
        var.SetBinContent(xbin+1, nom.GetBinContent(xbin+1) + diff/2.)
        if ratio > 1.2: var.SetBinContent(xbin+1, 1.2 * nom.GetBinContent(xbin+1))
      else:
        var.SetBinContent(xbin+1, nom.GetBinContent(xbin+1) - diff/2.)
        if ratio < 0.8: var.SetBinContent(xbin+1, 0.8 * nom.GetBinContent(xbin+1))

  return var


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
    nominal.SetDirectory(ROOT.nullptr)
    n_bins = nominal.GetNcells()
    up = nominal.Clone()
    up.SetDirectory(ROOT.nullptr)
    up.Reset()
    dn = nominal.Clone()
    dn.SetDirectory(ROOT.nullptr)
    dn.Reset()

    for i in range(0, n_bins+2):
      minimum = float("inf")
      maximum = float("-inf")

      for v in var_list:
        c = v.GetBinContent(i)
        minimum = min(minimum, c)
        maximum = max(maximum, c)

      up.SetBinContent(i, maximum)
      dn.SetBinContent(i, minimum)

    up = bSFNorm(up, bSFInfo)
    dn = bSFNorm(dn, bSFInfo)
    up.Rebin(nrebin)
    dn.Rebin(nrebin)
    up.SetName(histos + "__" + syst + "up")
    dn.SetName(histos + "__" + syst + "down")

    #We don't draw pdf in full ana due to computing resources
    #if syst == 'pdf' and any(sname in h.GetName() for sname in ['j4b2', 'S7', 'j4b4', 'S9']): #For 2017
    if syst == 'pdf' and any(sname in h.GetName() for sname in ['j4b4', 'S9']): #For 2018
      nominal = bSFNorm(nominal, bSFInfo)

      for xbin in xrange(up.GetNbinsX()):
        if nominal.GetBinContent(xbin+1) == 0:
          ratio_up = 1.
          ratio_dn = 1.
        else:
          ratio_up = up.GetBinContent(xbin+1) / nominal.GetBinContent(xbin+1)
          ratio_dn = dn.GetBinContent(xbin+1) / nominal.GetBinContent(xbin+1)

          #By construction, up shape is always above the down shape
          diff = abs(nominal.GetBinContent(xbin+1)-up.GetBinContent(xbin+1)) + abs(nominal.GetBinContent(xbin+1)-dn.GetBinContent(xbin+1))
          up.SetBinContent(xbin+1, nominal.GetBinContent(xbin+1) + diff/2.)
          dn.SetBinContent(xbin+1, nominal.GetBinContent(xbin+1) - diff/2.)
          if dn.GetBinContent(xbin+1) < 0: dn.SetBinContent(xbin+1, 0)
          if ratio_up > 1.2:
            up.SetBinContent(xbin+1, 1.2 * nominal.GetBinContent(xbin+1))
          if ratio_dn < 0.8:
            dn.SetBinContent(xbin+1, 0.8 * nominal.GetBinContent(xbin+1))

    up.Write()
    dn.Write()


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
          dn = f.Get(histos + "__" + syst_name + "down")
          dn.Scale(EventInfo.GetBinContent(2) / sumW_hist.GetBinContent(x))

      up.Write()
      dn.Write()

  elif mode == 1:
    if syst_name in files:
      h = f.Get(histos)
      if not any(i in h.GetName() for i in ['Info', 'Weight']):
        h.Scale(nom_EventInfo.GetBinContent(2) / EventInfo.GetBinContent(2))
        h.Rebin(nrebin)

#        if ( ('Tune' in syst_name and any(sname in h.GetName() for sname in ['j3b2', 'S2', 'j4b2', 'S7'])) #2017
#          or ('hdamp' in syst_name and any(sname in h.GetName() for sname in ['j3b2', 'S2', 'j4b3', 'S8']))
#          or ('jer' in f.GetName() and any(fname not in f.GetName() for fname in ['TTLL', 'TTpowheg','TTHad','TTTH','STTH']) and any(sname in h.GetName() for sname in ['j3b2', 'S2']))
#          or ('jec' in f.GetName() and any(fname not in f.GetName() for fname in ['TTLL', 'TTpowheg','TTHad','TTTH','STTH']) and any(sname in h.GetName() for sname in ['j4b4', 'S9'])) ):
#          bSFInfo_nom = fill_bSFInfo(nom_f)
#          h_nom = nom_f.Get(histos)
#          h_nom = bSFNorm(h_nom, bSFInfo_nom)
#          h_nom.Rebin(nrebin)
#
#          if 'down' in files:
#            f_opp = TFile.Open( os.path.join(pre_path, files.replace('down','up')), "READ")
#          elif 'up' in files:
#            f_opp = TFile.Open( os.path.join(pre_path, files.replace('up','down')), "READ")
#
#          opp_EventInfo = f_opp.Get('EventInfo')
#          bSFInfo_opp = fill_bSFInfo(f_opp)
#          h_opp = f_opp.Get(histos)
#          h_opp = bSFNorm(h_opp, bSFInfo_opp)
#          h_opp.Rebin(nrebin)
#          h_opp.Scale(nom_EventInfo.GetBinContent(2) / opp_EventInfo.GetBinContent(2))
#          h = symmetrize(h, h_opp, h_nom)

      f_new.cd()
      h.Write()


def fill_bSFInfo(f_in):

  bSFInfo_tmp = {}
  bSFInfo_tmp['Ch0_J1'] = f_in.Get("bSFInfo_Ch0_J1")
  bSFInfo_tmp['Ch0_J2'] = f_in.Get("bSFInfo_Ch0_J2")
  bSFInfo_tmp['Ch1_J1'] = f_in.Get("bSFInfo_Ch1_J1")
  bSFInfo_tmp['Ch1_J2'] = f_in.Get("bSFInfo_Ch1_J2")
  bSFInfo_tmp['Ch2_J1'] = f_in.Get("bSFInfo_Ch2_J1")
  bSFInfo_tmp['Ch2_J2'] = f_in.Get("bSFInfo_Ch2_J2")
  if bSFInfo_tmp['Ch2_J1'].Integral() == 0 and bSFInfo_tmp['Ch2_J2'].Integral() == 0:
    bSFInfo_tmp.clear() #If sum is 0, remove items: for data, J0 = 0 in MVA

  return bSFInfo_tmp


def bSFNorm(htmp, infos):

  if any(infos):
    hname = htmp.GetName()
    ch = ''
    if   'Ch0' in hname: ch = 'Ch0_'
    elif 'Ch1' in hname: ch = 'Ch1_'
    elif 'Ch2' in hname: ch = 'Ch2_'

    binnum = 2 #nominal = 1

    if any(i in hname for i in ['__lf', '__hf', '__cferr']):
      binnum = infos[ch+'J1'].GetXaxis().FindBin(str(hname.split('__')[-1]))
    for ibin in xrange(1,6):
      if (ibin == 1 or ibin == 3) and infos[ch+'J1'].GetBinContent(binnum) > 0:
        htmp.SetBinContent(ibin, htmp.GetBinContent(ibin) * (infos[ch+'J1'].GetBinContent(1)/infos[ch+'J1'].GetBinContent(binnum)))
      elif (ibin == 2 or ibin == 4 or ibin == 5) and infos[ch+'J2'].GetBinContent(binnum) > 0:
        htmp.SetBinContent(ibin, htmp.GetBinContent(ibin) * (infos[ch+'J2'].GetBinContent(1)/infos[ch+'J2'].GetBinContent(binnum)))

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
    h.SetDirectory(ROOT.nullptr)
    if not any(i in h.GetName() for i in ['Info', 'Weight']):
      h = bSFNorm(h, bSFInfo)
      h.Rebin(nrebin)
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

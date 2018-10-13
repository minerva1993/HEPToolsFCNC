from ROOT import *
import ROOT
import os

base_path = "/home/minerva1993/HEPToolsFCNC/analysis_2017/fullAna/doReco/"
ch = "STFCNC01"

def write_envelope(syst, nhists, new_sumW):

  if (histos + "__" + syst + "0")  in histo_list:
    var_list = []
    for x in range(0,nhists):
      h = f.Get(histos + "__" + syst + str(x))
#      h.Scale(EventInfo.GetBinContent(2) / new_sumW.GetBinContent(x+1))
#      h.Scale(0.5)
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

    up.SetName(histos + "__" + syst + "up")
    down.SetName(histos + "__" + syst + "down")
    up.Write()
    down.Write()


def rescale(binNum, new_sumW): # rescale up/dn histos

  #binNum = [up_num, down_num] or [bin]
  if   len(binNum) == 2 : mode = 0 #up, down in one root file
  elif len(binNum) == 0 : mode = 1 #use dedicated sample
  else: mode == 99

  if mode == 0:
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
      h.Scale(nom_EventInfo.GetBinContent(2) / EventInfo.GetBinContent(2))
      f_new.cd()
      h.Write()

#Starts loop over histogram root files
file_list = os.listdir( os.path.join(base_path, ch, "pre_process") )
file_list.remove("systematics")

pre_path = os.path.join(base_path, ch, "pre_process")

if not os.path.exists( pre_path ):
  os.makedirs( pre_path )

for files in file_list:

  print files

  #Prepare root file
  f = TFile.Open( os.path.join(pre_path, files), "READ")

  histo_list = []
  histo_list = [x.GetName() for x in f.GetListOfKeys()]

  EventInfo = f.Get("EventInfo")
#  ScaleWeights = f.Get("ScaleWeights")
#  PSWeights = f.Get("PSWeights")
#  PDFWeights = f.Get("PDFWeights")

  #Prepare nominal file for rescaling
  syst_name = ""
  if "__" in files:
    syst_name = files.split("__")[1].replace(".root","")
    nom_f = TFile.Open( os.path.join(pre_path, files.replace("__" + syst_name,"")), "READ")
    nom_EventInfo = nom_f.Get("EventInfo")

  #Creat output file, in post_process folder
  post_path = os.path.join(base_path, ch, "post_process")
  f_new = TFile.Open( os.path.join(post_path, files), "RECREATE")

  #Store nominal names, drop scale vars.
  nominal_list = []
  for histos in histo_list:
    if "__" not in histos: nominal_list.append(histos)
    if "scale" in histos: continue
    h = f.Get(histos)
    h.Write()

  #Store envelope, rescale histos
  for histos in nominal_list:

    write_envelope("scale", 6, EventInfo)#ScaleWeights)
#    write_envelope("pdf", histos, 103, ScaleWeights)
    rescale([], nom_EventInfo)

  f_new.Write()
  f_new.Close()
  f.Close()

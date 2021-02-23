from ROOT import *
import ROOT
import os, sys
import multiprocessing
import numpy as np

if len(sys.argv) < 3:
  print "specify year and ch to fit"
  sys.exit()
year = sys.argv[1]
ch = sys.argv[2]
if year not in ['2017', '2018']:
  print "wrong year"
  sys.exit()

def postProcess(files):

  def scale_postfit(hin, proc_, year_, ch_):

    hname = hin.GetName()
    if   'S2' in hname: step = 'S2'
    elif 'S3' in hname: step = 'S3'
    elif 'S6' in hname: step = 'S6'
    elif 'S7' in hname: step = 'S7'
    elif 'S8' in hname: step = 'S8'
    else              : step = ''

    scale = 1.
    scale_dict = {
        'ttbb': {
            'S2': { '2017': {'Hut':1.3765, 'Hct':1.0745}, '2018': {'Hut':1.0215, 'Hct':0.8947} },
            'S3': { '2017': {'Hut':0.8612, 'Hct':0.9613}, '2018': {'Hut':1.1568, 'Hct':1.3632} },
            'S6': { '2017': {'Hut':1.3974, 'Hct':1.0655}, '2018': {'Hut':0.9525, 'Hct':0.7937} },
            'S7': { '2017': {'Hut':0.8751, 'Hct':0.9455}, '2018': {'Hut':1.1004, 'Hct':1.2514} },
            'S8': { '2017': {'Hut':1.2332, 'Hct':0.9821}, '2018': {'Hut':0.7894, 'Hct':0.924 } } },
        'ttcc': {
            'S2': { '2017': {'Hut':0.9032, 'Hct':1.708 }, '2018': {'Hut':2.014 , 'Hct':2.4249} },
            'S3': { '2017': {'Hut':1.4342, 'Hct':1.7682}, '2018': {'Hut':1.429 , 'Hct':1.249 } },
            'S6': { '2017': {'Hut':0.9397, 'Hct':1.6929}, '2018': {'Hut':1.9329, 'Hct':2.2317} },
            'S7': { '2017': {'Hut':1.5186, 'Hct':1.7194}, '2018': {'Hut':1.3881, 'Hct':1.1633} },
            'S8': { '2017': {'Hut':1.5151, 'Hct':2.8815}, '2018': {'Hut':2.9279, 'Hct':2.4522} } },
        'ttlf': {
            'S2': { '2017': {'Hut':1.0412, 'Hct':0.9661}, '2018': {'Hut':0.9483, 'Hct':0.8707} },
            'S3': { '2017': {'Hut':1.0004, 'Hct':0.9499}, '2018': {'Hut':1.0168, 'Hct':0.9691} },
            'S6': { '2017': {'Hut':1.0294, 'Hct':0.9472}, '2018': {'Hut':0.8779, 'Hct':0.8234} },
            'S7': { '2017': {'Hut':1.0005, 'Hct':0.9191}, '2018': {'Hut':0.9472, 'Hct':0.9269} },
            'S8': { '2017': {'Hut':1.0417, 'Hct':0.5664}, '2018': {'Hut':0.8743, 'Hct':0.7961} } },
        'other': {
            'S2': { '2017': {'Hut':1.1939, 'Hct':1.2613}, '2018': {'Hut':1.1699, 'Hct':1.463 } },
            'S3': { '2017': {'Hut':1.1599, 'Hct':1.1958}, '2018': {'Hut':1.2418, 'Hct':1.5851} },
            'S6': { '2017': {'Hut':1.192 , 'Hct':1.2677}, '2018': {'Hut':1.1089, 'Hct':1.402 } },
            'S7': { '2017': {'Hut':1.1768, 'Hct':1.2316}, '2018': {'Hut':1.1788, 'Hct':1.5116} },
            'S8': { '2017': {'Hut':1.1656, 'Hct':1.2266}, '2018': {'Hut':1.2715, 'Hct':1.6466} } },
        'qcd': {
            'S2': { '2017': {'Hut':0.4312, 'Hct':0.7414}, '2018': {'Hut':0.9075, 'Hct':0.9847} },
            'S3': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S6': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S7': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S8': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } } },
        'sig': {
            'S2': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S3': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S6': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S7': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } },
            'S8': { '2017': {'Hut':1.0   , 'Hct':1.0   }, '2018': {'Hut':1.0   , 'Hct':1.0   } } },
      }

    if step in ['S2','S3','S6','S7','S8']:
      scale = scale_dict[proc][step][year][ch_]

    hin.Scale(scale)

    return hin


  print files

  #Prepare root file
  f = TFile.Open( os.path.join(post_path, files), "READ")

  histo_list = []
  histo_list = [x.GetName() for x in f.GetListOfKeys()]

  #Creat output file, in scaled_postfit folder
  scaled_path = os.path.join(base_path, "scaled_postfit_" + ch)
  if os.path.exists(os.path.join(scaled_path, files)):
    print files + " exists!"

  f_new = TFile.Open( os.path.join(scaled_path, files), "RECREATE")

  #Store nominal names, drop scale vars.
  nominal_list = []
  for histos in histo_list:
    if "__" not in histos: nominal_list.append(histos)
    if "scale" in histos: continue
    if "ps" in histos: continue
    #if "pdf" in histos: continue
    h = f.Get(histos)
    h.SetDirectory(ROOT.nullptr)
    h.Write()

  if not any(x in files for x in ['Run201', '__']):
    if   'ttbb' in files: proc = 'ttbb'
    elif 'ttcc' in files: proc = 'ttcc'
    elif 'ttlf' in files: proc = 'ttlf'
    elif 'QCD'  in files: proc = 'qcd'
    elif any(x in files for x in ['Hut', 'Hut']): proc = 'sig'
    else: proc = 'other'

    #Store envelope, rescale histos
    for histos in nominal_list:
      h = f.Get(histos)
      h.SetDirectory(ROOT.nullptr)
      hscaled = scale_postfit(h, proc, year, ch)
      hscaled.Write()

  f_new.Write()
  f_new.Close()
  f.Close()

if __name__ == '__main__':

  base_path = "./"

  #Starts loop over histogram root files
  file_list = os.listdir( os.path.join(base_path, "post_process") )
  if "systamatics" in file_list: file_list.remove("systematics")

  post_path = os.path.join(base_path, "post_process")
  if not os.path.exists( post_path ):
    os.makedirs( post_path )

  if not os.path.exists( base_path + "scaled_postfit_" + ch ):
    os.makedirs( base_path + "scaled_postfit_" + ch )

  pool = multiprocessing.Pool(54)
  pool.map(postProcess, file_list)
  pool.close()
  pool.join()

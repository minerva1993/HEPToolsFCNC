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
    scale_dict = {#201215 v2_mergeHighBins_3binsb3, newmet)
        'ttbb': {
            'S2': { '2017': {'Hut':0.971 , 'Hct':1.1151}, '2018': {'Hut':1.193 , 'Hct':1.8342} },
            'S3': { '2017': {'Hut':0.9424, 'Hct':0.9778}, '2018': {'Hut':1.2344, 'Hct':1.3472} },
            'S6': { '2017': {'Hut':0.965 , 'Hct':1.1055}, '2018': {'Hut':1.111 , 'Hct':1.5607} },
            'S7': { '2017': {'Hut':0.9315, 'Hct':0.9764}, '2018': {'Hut':1.1589, 'Hct':1.1794} },
            'S8': { '2017': {'Hut':1.3025, 'Hct':1.4004}, '2018': {'Hut':1.0597, 'Hct':1.2818} } },
        'ttcc': {
            'S2': { '2017': {'Hut':1.3458, 'Hct':1.7178}, '2018': {'Hut':2.1591, 'Hct':2.6012} },
            'S3': { '2017': {'Hut':1.3523, 'Hct':1.5153}, '2018': {'Hut':1.4555, 'Hct':2.0079} },
            'S6': { '2017': {'Hut':1.3716, 'Hct':1.7239}, '2018': {'Hut':2.0107, 'Hct':2.1607} },
            'S7': { '2017': {'Hut':1.3908, 'Hct':1.5226}, '2018': {'Hut':1.3551, 'Hct':1.629 } },
            'S8': { '2017': {'Hut':1.3273, 'Hct':1.1453}, '2018': {'Hut':2.1594, 'Hct':1.6264} } },
        'ttlf': {
            'S2': { '2017': {'Hut':0.9971, 'Hct':0.9681}, '2018': {'Hut':0.9306, 'Hct':0.9212} },
            'S3': { '2017': {'Hut':1.005 , 'Hct':0.9808}, '2018': {'Hut':1.0055, 'Hct':0.9583} },
            'S6': { '2017': {'Hut':1.0077, 'Hct':0.9486}, '2018': {'Hut':0.8619, 'Hct':0.8204} },
            'S7': { '2017': {'Hut':1.0284, 'Hct':0.9635}, '2018': {'Hut':0.9408, 'Hct':0.8654} },
            'S8': { '2017': {'Hut':1.0441, 'Hct':0.817 }, '2018': {'Hut':0.8988, 'Hct':0.7963} } },
        'other': {
            'S2': { '2017': {'Hut':1.0585, 'Hct':1.192 }, '2018': {'Hut':1.1651, 'Hct':1.2305} },
            'S3': { '2017': {'Hut':1.0732, 'Hct':1.1781}, '2018': {'Hut':1.2527, 'Hct':1.2422} },
            'S6': { '2017': {'Hut':1.0577, 'Hct':1.1835}, '2018': {'Hut':1.1058, 'Hct':1.1549} },
            'S7': { '2017': {'Hut':1.0655, 'Hct':1.1932}, '2018': {'Hut':1.1633, 'Hct':1.1924} },
            'S8': { '2017': {'Hut':1.0824, 'Hct':1.1931}, '2018': {'Hut':1.1948, 'Hct':1.2896} } },
        'qcd': {
            'S2': { '2017': {'Hut':1.0207, 'Hct':0.8365}, '2018': {'Hut':0.9954, 'Hct':0.5466} },
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
    if 'Run201' in files:
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
  file_list = [ x for x in file_list if "__" not in x ]

  post_path = os.path.join(base_path, "post_process")
  if not os.path.exists( post_path ):
    os.makedirs( post_path )

  if not os.path.exists( base_path + "scaled_postfit_" + ch ):
    os.makedirs( base_path + "scaled_postfit_" + ch )

  pool = multiprocessing.Pool(50)
  pool.map(postProcess, file_list)
  pool.close()
  pool.join()

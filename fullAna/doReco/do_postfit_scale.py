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
            'S2': { '2017': {'Hut':1.0964, 'Hct':1.1568}, '2018': {'Hut':1.1786, 'Hct':1.2413} },
            'S3': { '2017': {'Hut':0.8839, 'Hct':1.0038}, '2018': {'Hut':1.1379, 'Hct':1.3748} },
            'S6': { '2017': {'Hut':1.0882, 'Hct':1.1536}, '2018': {'Hut':1.1536, 'Hct':1.0967} },
            'S7': { '2017': {'Hut':0.8814, 'Hct':1.0068}, '2018': {'Hut':1.1525, 'Hct':1.2462} },
            'S8': { '2017': {'Hut':1.1548, 'Hct':1.3171}, '2018': {'Hut':1.0152, 'Hct':0.9833} } },
        'ttcc': {
            'S2': { '2017': {'Hut':1.3125, 'Hct':1.5653}, '2018': {'Hut':1.1194, 'Hct':2.8532} },
            'S3': { '2017': {'Hut':1.5974, 'Hct':1.5226}, '2018': {'Hut':0.9477, 'Hct':1.666 } },
            'S6': { '2017': {'Hut':1.3272, 'Hct':1.5754}, '2018': {'Hut':1.1423, 'Hct':2.5484} },
            'S7': { '2017': {'Hut':1.6288, 'Hct':1.5444}, '2018': {'Hut':0.9681, 'Hct':1.4887} },
            'S8': { '2017': {'Hut':2.0814, 'Hct':1.5808}, '2018': {'Hut':2.1012, 'Hct':2.4896} } },
        'ttlf': {
            'S2': { '2017': {'Hut':1.0284, 'Hct':1.0012}, '2018': {'Hut':0.975 , 'Hct':0.8703} },
            'S3': { '2017': {'Hut':0.9971, 'Hct':0.9879}, '2018': {'Hut':1.0301, 'Hct':0.9539} },
            'S6': { '2017': {'Hut':1.0079, 'Hct':0.9733}, '2018': {'Hut':0.9446, 'Hct':0.7822} },
            'S7': { '2017': {'Hut':0.9845, 'Hct':0.9618}, '2018': {'Hut':1.0128, 'Hct':0.8634} },
            'S8': { '2017': {'Hut':0.8516, 'Hct':0.7631}, '2018': {'Hut':0.9479, 'Hct':0.7093} } },
        'other': {
            'S2': { '2017': {'Hut':1.0594, 'Hct':1.0828}, '2018': {'Hut':1.3807, 'Hct':1.3673} },
            'S3': { '2017': {'Hut':1.027 , 'Hct':1.0598}, '2018': {'Hut':1.4489, 'Hct':1.4576} },
            'S6': { '2017': {'Hut':1.0536, 'Hct':1.0746}, '2018': {'Hut':1.339 , 'Hct':1.2867} },
            'S7': { '2017': {'Hut':1.037 , 'Hct':1.0692}, '2018': {'Hut':1.3895, 'Hct':1.3727} },
            'S8': { '2017': {'Hut':1.0462, 'Hct':1.0837}, '2018': {'Hut':1.4142, 'Hct':1.4864} } },
        'qcd': {
            'S2': { '2017': {'Hut':0.6514, 'Hct':0.751 }, '2018': {'Hut':0.692 , 'Hct':0.838} },
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

from ROOT import *
import ROOT
import sys, os
from subprocess import call

if len(sys.argv) < 3:
  print("Ver of 2017 and 2018")
  sys.exit()
ver17 = sys.argv[1]
ver18 = sys.argv[2]

lumi = {'2017': 41529, '2018':59741}
files = []
for ch in ['Hct', 'Hut']:
  for proc in ['ST', 'TT']:
    tmp = 'YEAR' + '/RECOVER/post_process/' + 'hist_' + proc + 'TH1L3B'
    if proc == 'TT':
      tmp2 = tmp + 'LEP' + ch + 'SYST.root'
      files.append(tmp2)
    else:
      tmp += ch + 'SYST.root'
      files.append(tmp)

for f in files:
  print f
  for reco in ['STFCNC', 'TTFCNC', 'TTBKG']:
    for syst in ['', '__jecup', '__jecdown', '__jerup', '__jerdown']:

      samples_to_merge = []

      f17 = f.replace('YEAR', '2017')
      f17 = f17.replace('RECO', reco)
      f17 = f17.replace('VER', ver17)
      f17 = f17.replace('SYST', syst)
      f18 = f.replace('YEAR', '2018')
      f18 = f18.replace('RECO', reco)
      f18 = f18.replace('VER', ver18)
      f18 = f18.replace('SYST', syst)

      if 'LEP' in f:
        f17_1 = f17.replace('LEP','TLep')
        f17_2 = f17.replace('LEP','aTLep')
        f18_1 = f18.replace('LEP','TLep')
        f18_2 = f18.replace('LEP','aTLep')
        samples_to_merge.extend([f17_1, f17_2, f18_1, f18_2])
      else: samples_to_merge.extend([f17, f18])

      print samples_to_merge

      nevt_dict = {}
      file_dict = {}
      hist_names = []
      for tmp in samples_to_merge:
        ftmp = TFile.Open(tmp, 'READ')
        if len(hist_names) == 0 and '2018' in tmp:
          hist_names = [x.GetName() for x in ftmp.GetListOfKeys()]
          hist_names = list(dict.fromkeys(hist_names)) #remove duplicates from more than one instances
          hist_names[:] = [item for item in hist_names if item not in ['EventInfo', 'PSWeights', 'ScaleWeights']]
          hist_names.sort()
        ntmp = ftmp.Get("EventInfo").GetBinContent(2)
        nevt_dict[tmp] = ntmp
        file_dict[tmp] = ftmp #Is this the best way?

      target_name = f17.split('/')[3].replace('LEP','')
      dest = TFile.Open('./full1718/' + reco + '/' + target_name, 'RECREATE')

      for hist in hist_names:
        print hist
        for path, fi in file_dict.iteritems(): #f is used
          htmp = fi.Get(hist)
          htmp.Scale(lumi[path.split('/')[0]]/nevt_dict[path])
          try:
            hnew #check if it exists
          except NameError:
            hnew = htmp.Clone('hnew')
            hnew.SetDirectory(0)
          else:
            hnew.Add(htmp, 1.0)
          dest.cd()
          hnew.SetName(hist)
          hnew.Write()
        del hnew
      dest.Write()
      dest.Close()

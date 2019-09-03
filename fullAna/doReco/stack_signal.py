from ROOT import *
import ROOT
import sys, os
from subprocess import check_call
from os import listdir, path
import collections
import glob

if len(sys.argv) < 3:
  print("Ver of 2017 and 2018")
  sys.exit()
ver17 = sys.argv[1]
ver18 = sys.argv[2]

ver_dict = {'2017': ver17, '2018':ver18}
lumi_dict = {'2017': 41529, '2018':59741}
file_names = collections.OrderedDict()

for era in ['2017','2018']:
  for reco in ['STFCNC', 'TTFCNC', 'TTBKG']:
    dir_path = os.path.join(era, reco + ver_dict[era], 'post_process')
    dirs = os.listdir(dir_path)
    dirs[:] = [item for item in dirs if any(i in item for i in ['STTH1L3B','TTTH1L3BaTLep','TTTH1L3BTLep'])] #avoid TTTH merged
    file_names[dir_path] = dirs
"""
for path, file_name in file_names.iteritems():
  for f in file_name:
    ftmp = TFile.Open(os.path.join(path, f), 'READ')

    hist_names = [x.GetName() for x in ftmp.GetListOfKeys()] 
    hist_names = list(dict.fromkeys(hist_names)) #remove duplicates from more than one instances 
    hist_names[:] = [item for item in hist_names if item not in ['EventInfo', 'PSWeights', 'ScaleWeights']] 
    hist_names.sort() 
   
    ntmp = ftmp.Get("EventInfo").GetBinContent(2) 

    dest_name = './full1718/' + path.split('/')[1].rstrip(ver_dict[era]) + '/' + f[:-5] + path.split('/')[0] + '.root'
    dest = TFile.Open(dest_name, 'RECREATE')

    print 'Writing scaled histogram to ' + dest_name

    for hist in hist_names: 
      htmp = ftmp.Get(hist) 
      htmp.Scale(lumi_dict[path.split('/')[0]]/ntmp)
      dest.cd()
      htmp.Write()
    dest.Write()
    dest.Close()
"""

for reco in ['STFCNC', 'TTFCNC', 'TTBKG']:
  for syst in ['', '__jecup', '__jecdown', '__jerup', '__jerdown']:
    for ch in ['Hct', 'Hut']:
      print './full1718/' + reco + '/hist_STTH1L3B' + ch + syst + '.root'
      #check_call(['hadd','-f', './full1718/' + reco + '/hist_STTH1L3B' + ch + syst + '.root'] +  glob.glob('./full1718/' + reco + '/hist_STTH1L3B' + ch + syst + '201*.root'))
      check_call(['hadd','-f', './full1718/' + reco + '/hist_TTTH1L3B' + ch + syst + '.root'] +  glob.glob('./full1718/' + reco + '/hist_TTTH1L3B*TLep' + ch + syst + '201*.root'))

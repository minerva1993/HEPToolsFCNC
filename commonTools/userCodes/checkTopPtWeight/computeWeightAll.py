from __future__ import print_function
from ROOT import *
import ROOT
import sys, os
import ast, array
from collections import OrderedDict
gROOT.SetBatch(True)
hostname = os.environ["HOSTNAME"]

datasets = OrderedDict()
filelists = OrderedDict()

h = TH1F('h', 'h', 40, 0, 2000)
arr = array.array('d',[0,50,100,150,200,250,300,350,400,450,500,550,
            600,800,1000,2000])
#            600,650,700,800,900,1000,1100,1200,1400,1600,1800,2000])
h = h.Rebin(len(arr)-1, h.GetName(), arr)
h.GetXaxis().SetTitle('top quark p_{T} (GeV)')
h.SetStats(0)

hbkg17 = h.Clone('hbkg17')
hhut17 = h.Clone('hhut17')
hhct17 = h.Clone('hhct17')
hbkg18 = h.Clone('hbkg18')
hhut18 = h.Clone('hhut18')
hhct18 = h.Clone('hhct18')
hbkg17.SetDirectory(0)
hhut17.SetDirectory(0)
hhct17.SetDirectory(0)
hbkg18.SetDirectory(0)
hhut18.SetDirectory(0)
hhct18.SetDirectory(0)
datasets['ttbkg17'] = [hbkg17, 831.76]
datasets['tthut17'] = [hhut17, 496.1]
datasets['tthct17'] = [hhct17, 496.1]
datasets['ttbkg18'] = [hbkg18, 831.76]
datasets['tthut18'] = [hhut18, 496.1]
datasets['tthct18'] = [hhct18, 496.1]

c = TCanvas('c','c',400,400)

for ver in ['V9_7', 'V10_3']:
  for name, data in datasets.items():
    if 'V9'  in ver and not '17' in name: continue
    if 'V10' in ver and not '18' in name: continue

    entries = 0
    with open(ver+'_'+name[:-2]+'.txt', 'r') as f:
      bins = f.readline(); #Should be first line
      bins = ast.literal_eval(bins)
      for i in xrange(len(bins)):
        data[0].SetBinContent(i+1, bins[i])
      entries = f.readline();
    data[0].Scale(data[1]/float(entries))

hbkg_hut17 = hbkg17.Clone('hbkg_hut17')
hbkg_hct17 = hbkg17.Clone('hbkg_hct17')
hbkg_hut18 = hbkg18.Clone('hbkg_hut18')
hbkg_hct18 = hbkg18.Clone('hbkg_hct18')
hbkg_hut17.SetDirectory(0)
hbkg_hct17.SetDirectory(0)
hbkg_hut18.SetDirectory(0)
hbkg_hct18.SetDirectory(0)

hbkg_hut17.Divide(hhut17)
hbkg_hct17.Divide(hhct17)
hbkg_hut18.Divide(hhut18)
hbkg_hct18.Divide(hhct18)

l = TLegend(0.3,0.71,0.7,0.87)
l.SetTextSize(0.03)
l.SetLineColor(0)
l.SetFillColor(0)
l.AddEntry(hbkg_hut17, 'Powheg+P8/Hut (2017)', 'l')
l.AddEntry(hbkg_hct17, 'Powheg+P8/Hct (2017)', 'l')
l.AddEntry(hbkg_hut18, 'Powheg+P8/Hut (2018)', 'l')
l.AddEntry(hbkg_hct18, 'Powheg+P8/Hct (2018)', 'l')

hbkg_hut17.SetTitle('Ratio between MG5 LO signal and Powheg NLO tt+LJ')
hbkg_hut17.SetLineColor(2)
hbkg_hut17.SetLineWidth(2)
hbkg_hut17.GetYaxis().SetRangeUser(0.9,1.9)
hbkg_hut17.Draw('ep')
hbkg_hct17.SetLineColor(4)
hbkg_hct17.SetLineWidth(2)
hbkg_hct17.Draw('ep same')
hbkg_hut18.SetLineColor(6)
hbkg_hut18.SetLineWidth(2)
hbkg_hut18.SetLineStyle(2)
hbkg_hut18.Draw('ep same')
hbkg_hct18.SetLineColor(9)
hbkg_hct18.SetLineWidth(2)
hbkg_hct18.SetLineStyle(2)
hbkg_hct18.Draw('ep same')
l.Draw('same')
c.Print('all_sig_ratios.pdf')

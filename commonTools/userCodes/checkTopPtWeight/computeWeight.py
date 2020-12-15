from __future__ import print_function
from ROOT import *
import ROOT
import sys, os
import ast, array
from collections import OrderedDict
gROOT.SetBatch(True)
hostname = os.environ["HOSTNAME"]

ver="V10_3"
datasets = OrderedDict()
filelists = OrderedDict()

hbkg = TH1F('hbkg', 'hbkg', 40, 0, 2000)
hhut = TH1F('hhut', 'hhut', 40, 0, 2000)
hhct = TH1F('hhct', 'hhct', 40, 0, 2000)
arr = array.array('d',[0,50,100,150,200,250,300,350,400,450,500,550,
            600,650,700,800,900,1000,1100,1200,1400,1600,1800,2000])
hbkg = hbkg.Rebin(len(arr)-1, hbkg.GetName(), arr)
hhut = hhut.Rebin(len(arr)-1, hhut.GetName(), arr)
hhct = hhct.Rebin(len(arr)-1, hhct.GetName(), arr)
hbkg.GetXaxis().SetTitle('top quark p_{T} (GeV)')
hhut.GetXaxis().SetTitle('top quark p_{T} (GeV)')
hhct.GetXaxis().SetTitle('top quark p_{T} (GeV)')
hbkg.SetStats(0)
hhut.SetStats(0)
hhct.SetStats(0)
datasets['ttbkg'] = [hbkg, 831.76]
datasets['tthut'] = [hhut, 496.1] #From xsdb
datasets['tthct'] = [hhct, 496.1]
#datasets['tthut'] = [hhut, 831.76] #From xsdb
#datasets['tthct'] = [hhct, 831.76]

c = TCanvas('c','c',400,400)

for name, data in datasets.items():
  entries = 0
  with open(ver+'_'+name+'.txt', 'r') as f:
    bins = f.readline(); #Should be first line
    bins = ast.literal_eval(bins)
    for i in xrange(len(bins)):
      h = data[0]
      h.SetBinContent(i+1, bins[i])
    entries = f.readline();
  h.Scale(data[1]/float(entries))
  h.Draw('ep')
  c.Print(ver+'_'+name+'.pdf')

hbkg_hut = hbkg.Clone('hbkg_hut')
hbkg_hct = hbkg.Clone('hbkg_hct')
hbkg_hut.SetDirectory(0)
hbkg_hct.SetDirectory(0)

hbkg_hut.Divide(hhut)
hbkg_hct.Divide(hhct)

c.Clear()
hbkg_hut.Draw('ep')
c.Print(ver+'_hut_ratio.pdf')
print('Integral of Powheg+P8 dividev by Hut LO', hbkg_hut.Integral()/hbkg_hut.GetNbinsX())
tmp_string = ''
for i in xrange(1,hbkg_hut.GetNbinsX()+1):
  tmp_string += '{'+str(hbkg_hut.GetBinCenter(i))+','+str(hbkg_hut.GetBinContent(i))+'},'
with open(ver+'_hut_ratio.txt', 'w') as f: f.write(tmp_string)

c.Clear()
hbkg_hct.Draw('ep')
c.Print(ver+'_hct_ratio.pdf')
print('Inegral of Powheg+P8 dividev by Hct LO', hbkg_hct.Integral()/hbkg_hct.GetNbinsX())
tmp_string = ''
for i in xrange(1,hbkg_hct.GetNbinsX()+1):
  tmp_string += '{'+str(hbkg_hct.GetBinCenter(i))+','+str(hbkg_hct.GetBinContent(i))+'},'
with open(ver+'_hct_ratio.txt', 'w') as f: f.write(tmp_string)

c.Clear()
l = TLegend(0.3,0.71,0.7,0.87)
l.SetTextSize(0.03)
l.SetLineColor(0)
l.SetFillColor(0)
l.AddEntry(hbkg_hut, 'Powheg+P8/Hut', 'l')
l.AddEntry(hbkg_hct, 'Powheg+P8/Hct', 'l')

hbkg_hut.SetTitle('Ratio between MG5 LO signal and Powheg NLO tt+LJ')
hbkg_hut.SetLineColor(2)
#hbkg_hut.Fit('pol4')
hbkg_hut.Draw('ep')
hbkg_hct.SetLineColor(4)
hbkg_hct.Draw('ep same')
l.Draw('same')
c.Print(ver+'_sig_ratios.pdf')

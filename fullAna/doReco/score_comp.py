import ROOT
from ROOT import *
import os

gROOT.SetBatch()

year = '2018'
dir_org = '/home2/minerva1993/HEPToolsFCNC/fullAna/doReco/'+year
dir_new = '/home2/minerva1993/HEPToolsFCNC/fullAna/doReco/'+year + '/reco_20epoch'

sigs = {'hist_STTH1L3BHct':['S2', 'S3', 'S6', 'S7', 'S8'], 'hist_STTH1L3BHut':['S2', 'S3', 'S6', 'S7', 'S8'],
        'hist_TTTH1L3BHct':['S6', 'S7', 'S8'], 'hist_TTTH1L3BHut':['S6', 'S7', 'S8']}
folders = ['STFCNC01', 'TTFCNC01', 'TTBKG01']

c = TCanvas('c', 'c', 400, 400)

l = TLegend(0.3,0.8,0.7,0.87)
l.SetNColumns(1);
l.SetTextSize(0.03);
l.SetLineColor(0);
l.SetFillColor(0);

label2 = TPaveText()
label2.SetX1NDC(0.3)
label2.SetY1NDC(0.7)
label2.SetX2NDC(0.7)
label2.SetY2NDC(0.8)
label2.SetTextFont(42)
label2.SetFillStyle(0)
label2.SetBorderSize(0)
label2.SetTextSize(0.03)
label2.SetTextAlign(32)

for fd in folders:
  for sig, steps in sigs.items():
    f_org = TFile.Open(os.path.join(dir_org, fd + '/post_process/' + sig + '.root'))
    f_new = TFile.Open(os.path.join(dir_new, fd + '/post_process/' + sig + '.root'))

    for step in steps:
      if 'TT' in fd and step in ['S2', 'S3']: continue

      h_org = f_org.Get('h_FCNHkinScore_Ch2_' + step)
      h_new = f_new.Get('h_FCNHkinScore_Ch2_' + step)

      h_new.SetLineColor(2)
      h_org.SetStats(0)
      h_org.DrawNormalized()
      h_new.DrawNormalized('same')

      l.AddEntry(h_org, 'Best Reco Eff.', 'l')
      l.AddEntry(h_new, 'Epoch 20', 'l')
      l.Draw('same')
      label2.AddText(fd + ', ' + sig + ' ' + step)
      label2.Draw('same')

      c.Print('comp_' + fd + '_' + sig + '_' + step + '.png')

      c.Clear()
      l.Clear()
      label2.Clear()

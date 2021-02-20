from ROOT import *
import ast, array
gROOT.SetBatch(True)

c = TCanvas('c', 'c', 400, 400)
c.SetLogy()
h1 = TH1F('h1','h1', 40, 0, 1)
h2 = TH1F('h2','h2', 40, 0, 1)
with open('recoSTFCNC12/list_tpr1_30_0.9702.txt', 'r') as f: #sig test
  bins = f.readline()
  bins = ast.literal_eval(bins)
  for i in xrange(len(bins)):
    h1.Fill(bins[i])

with open('recoSTFCNC12/list_tpr2_30_0.9702.txt', 'r') as f:
  bins = f.readline()
  bins = ast.literal_eval(bins)
  for i in xrange(len(bins)):
    h2.Fill(bins[i])
h2.SetFillColor(3)
h2.SetStats(0)
h2.GetYaxis().SetRangeUser(10, h2.GetMaximum()*1.2)

h2.DrawNormalized()
h1.DrawNormalized("ep same")

c.Print("test.png")

from ROOT import *
gROOT.SetBatch()

path = '/data/users/minerva1993/ntuple/V9_6/190702/'

c = TCanvas('c','',1)
h = TH1F('h', 'TTLJ mean prefire weight per jet bin', 5, 0, 5)
h.GetXaxis().SetBinLabel(1, 'b2j3')
h.GetXaxis().SetBinLabel(2, 'b2j4')
h.GetXaxis().SetBinLabel(3, 'b3j3')
h.GetXaxis().SetBinLabel(4, 'b3j4')
h.GetXaxis().SetBinLabel(5, 'b4j4')

i = 0
for b in ['2', '3', '4']:
  for j in ['== 3', '>= 4']:
    if int(b[-1]) > int(j[-1]): continue
    i += 1

    f1 = TFile.Open(path + 'TT_powheg_ttbb.root', 'READ')
    f2 = TFile.Open(path + 'TT_powheg_ttcc.root', 'READ')
    f3 = TFile.Open(path + 'TT_powheg_ttlf.root', 'READ')
    t1 = f1.Get('fcncLepJets/tree')
    t2 = f2.Get('fcncLepJets/tree')
    t3 = f3.Get('fcncLepJets/tree')

    h1 = TH1F('h1','h1',50,0.5,1)
    h2 = TH1F('h2','h2',50,0.5,1)
    h3 = TH1F('h3','h3',50,0.5,1)

    t1.Draw("prefireweight>>h1","jet_njet " + j + " && jet_nbjetm ==" + b)
    t2.Draw("prefireweight>>h2","jet_njet " + j + " && jet_nbjetm ==" + b)
    t3.Draw("prefireweight>>h3","jet_njet " + j + " && jet_nbjetm ==" + b)

    h1.Add(h2)
    h1.Add(h3)

    mean = h1.GetMean()
    h.SetBinContent(i, mean)

    del h1, h2, h3

h.SetStats(0)
h.SetMinimum(0.9)
h.SetMaximum(1.0)
h.SetMarkerSize(1.8)
h.Draw('text hist')
c.Print('h_TTLJ_Prefire.pdf')

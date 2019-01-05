import os
from ROOT import TStyle, TF1, TFile, TCanvas, gDirectory, TTree, TH1D, TH1F, THStack, TLegend, gROOT, TColor
import ROOT
from style import *
from variables import input_variables
gROOT.SetBatch()

c1 = TCanvas( 'c1', 'c1', 450, 450 ) 
#c1.SetLogy()

label = TPaveText()
label.SetX1NDC(gStyle.GetPadLeftMargin())
label.SetY1NDC(1.0-gStyle.GetPadTopMargin())
label.SetX2NDC(1.0-gStyle.GetPadRightMargin()+0.03)
label.SetY2NDC(1.0)
label.SetTextFont(62)
label.AddText("Work in Progress        CMS, 41.5 fb^{-1} at #sqrt{s} = 13 TeV")
label.SetFillStyle(0)
label.SetBorderSize(0)
label.SetTextSize(0.035)
label.SetTextAlign(32)

label2 = TPaveText()
label2.SetX1NDC(gStyle.GetPadLeftMargin()+0.6)
label2.SetY1NDC(1.0-gStyle.GetPadTopMargin()-0.35)
label2.SetX2NDC(1.0-gStyle.GetPadRightMargin()-0.03)
label2.SetY2NDC(1.0)
label2.SetTextFont(62)
label2.SetFillStyle(0)
label2.SetBorderSize(0)
label2.SetTextSize(0.03)
label2.SetTextAlign(32)

l = TLegend(0.15,0.8,0.89,0.87)
l.SetNColumns(2);
l.SetTextSize(0.03);
l.SetLineColor(0);
l.SetFillColor(0);

##all histos with event selection!
###########################################################
for ch in ['Hct','Hut']:
  for jetcat in ['j3b2','j3b3','j4b2','j4b3','j4b4']:
    for i in ['01']:
      ver = str(i)
      out = TFile.Open('./final_'+ch+'_'+jetcat+'_'+ver+'/output_'+ch+'_'+jetcat+'.root')
      for method in ['BDT']:
        trainS = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/Method_'+method+'/'+method+'/MVA_'+method+'_Train_S')
        trainB = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/Method_'+method+'/'+method+'/MVA_'+method+'_Train_B')
        testS = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/Method_'+method+'/'+method+'/MVA_'+method+'_S')
        testB = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/Method_'+method+'/'+method+'/MVA_'+method+'_B')
        corrS = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/CorrelationMatrixS')
        corrB = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/CorrelationMatrixB')
        if method == 'BDT': rocBDT = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/Method_'+method+'/'+method+'/MVA_BDT_rejBvsS')
        elif method == 'Keras_TF': rocKeras = out.Get('final_'+ch+'_'+jetcat+'_'+ver+'/Method_'+method+'/'+method+'/MVA_Keras_TF_rejBvsS')

        if jetcat == 'j4b4':
          trainS.Rebin(2)
          trainB.Rebin(2)
          testS.Rebin(2)
          testB.Rebin(2)

        testS.SetStats(0)
        testS.SetTitle('')

        testS.GetXaxis().SetTitle('MVA score')
        testS.GetYaxis().SetTitle('A.U. (Normalized)')
        testB.SetFillStyle(3554)
        testS.SetFillColor(38)
        testB.SetFillColor(2)
        testS.SetLineColor(4)
        testB.SetLineColor(2)
        trainS.SetMarkerStyle(20)
        trainB.SetMarkerStyle(20)
        trainS.SetMarkerSize(0.7)
        trainB.SetMarkerSize(0.7)
        trainS.SetMarkerColor(4)
        trainB.SetMarkerColor(2)

        histoMax = [trainS.GetMaximum(), trainB.GetMaximum(), testS.GetMaximum(), testB.GetMaximum()]
        testS.SetMaximum(max(histoMax)*1.25)

        l.AddEntry(trainS, 'Signal (training)', 'p')
        l.AddEntry(trainB, 'Background (training)', 'p')
        l.AddEntry(testS, 'Signal (testing)', 'f')
        l.AddEntry(testB, 'Background (testing)', 'f')
        label2.AddText(ch+', '+method)

        testS.Draw('hist')
        testB.Draw('hist same')
        trainS.Draw('ep same')
        trainB.Draw('ep same')
        label.Draw('same')
        label2.Draw('same')
        l.Draw('same')
        testS.Draw('axis same')
        c1.Print('bdt_plots/overtrain_'+ch+'_'+jetcat+'_'+method+'.pdf')
        l.Clear()
        label2.Clear()
        c1.Clear()

      #Correlation part
      axislabel = []
      axislabel.extend(input_variables(jetcat))

      corrS.SetTitle('')
      for i in range(0,len(axislabel)):
        corrS.GetXaxis().LabelsOption("v");
      corrS.GetXaxis().SetLabelSize(0.02)
      corrS.GetYaxis().SetLabelSize(0.02)
      corrS.GetZaxis().SetLabelSize(0.02)
      corrS.GetXaxis().SetLabelOffset(0.002)
      corrS.GetYaxis().SetLabelOffset(0.002)
      c1.SetGrid(1,1)
      corrS.Draw('colz')
      label.Draw('same')
      #c1.Print('bdt_plots/corr_'+ch+'_'+jetcat+'_sig.pdf')
      c1.Clear()

      corrB.SetTitle('')
      for i in range(0,len(axislabel)):
        corrB.GetXaxis().LabelsOption("v");
      corrB.GetXaxis().SetLabelSize(0.02)
      corrB.GetYaxis().SetLabelSize(0.02)
      corrB.GetZaxis().SetLabelSize(0.02)
      corrB.GetXaxis().SetLabelOffset(0.002)
      corrB.GetYaxis().SetLabelOffset(0.002)
      c1.SetGrid(1,1)
      corrS.Draw('colz')
      label.Draw('same')
      #c1.Print('bdt_plots/corr_'+ch+'_'+jetcat+'_bkg.pdf')
      c1.Clear()



      #ROC part
      aucBDT = str(round(rocBDT.Integral()/100, 3))
      #aucKeras = str(round(rocKeras.Integral()/100, 3))

      l2 = TLegend(0.16,0.17,0.5,0.25)
      l2.SetTextSize(0.03);
      l2.SetLineColor(0);
      l2.SetFillColor(0);
      #l2.AddEntry(rocKeras, 'Keras+TF, AUC = '+aucKeras, 'l')
      l2.AddEntry(rocBDT, 'BDT, AUC = '+aucBDT, 'l')

      rocBDT.SetTitle('')
      rocBDT.GetXaxis().SetTitle('Signal Efficiency')
      rocBDT.GetYaxis().SetTitle('Background Rejection')
      rocBDT.SetLineColor(1)
      #rocKeras.SetLineColor(2)
      rocBDT.SetLineWidth(3)
      #rocKeras.SetLineWidth(3)
      c1.SetGrid(0,0)
      rocBDT.Draw('hist l')
      #rocKeras.Draw('hist same l')
      label.Draw('same')
      l2.Draw('same')
      #c1.Print('bdt_plots/roc_'+ch+'_'+jetcat+'.pdf')

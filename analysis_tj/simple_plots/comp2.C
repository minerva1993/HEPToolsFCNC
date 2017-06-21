#include <TStyle.h>

void comp2(TString var = "bjmDPhiDEta", TString step = "11" ){

  TFile * f_Top_Hct = new TFile("hist_Top_Hct.root");
  TFile * f_Top_Hut = new TFile("hist_Top_Hut.root");
  TFile * f_AntiTop_Hct = new TFile("hist_AntiTop_Hct.root");
  TFile * f_AntiTop_Hut = new TFile("hist_AntiTop_Hut.root");

  TFile * f_ttbb = new TFile("hist_ttbb.root");

  TH2F * h_Top_Hct = (TH2F *) f_Top_Hct->Get(Form("h_%s_S%s_Top_Hct",var.Data(),step.Data()));
  TH2F * h_Top_Hut = (TH2F *) f_Top_Hut->Get(Form("h_%s_S%s_Top_Hut",var.Data(),step.Data()));
  TH2F * h_AntiTop_Hct = (TH2F *) f_AntiTop_Hct->Get(Form("h_%s_S%s_AntiTop_Hct",var.Data(),step.Data()));
  TH2F * h_AntiTop_Hut = (TH2F *) f_AntiTop_Hut->Get(Form("h_%s_S%s_AntiTop_Hut",var.Data(),step.Data()));
  TH2F * h_ttbb = (TH2F *) f_ttbb->Get(Form("h_%s_S%s_ttbb",var.Data(),step.Data()));

  h_Top_Hct->Scale(1.0/h_Top_Hct->Integral());
  h_Top_Hut->Scale(1.0/h_Top_Hut->Integral());
  h_AntiTop_Hct->Scale(1.0/h_AntiTop_Hct->Integral());
  h_AntiTop_Hut->Scale(1.0/h_AntiTop_Hut->Integral());
  h_ttbb->Scale(1.0/h_ttbb->Integral());
 
  TCanvas * c = new TCanvas("c","c",700,600);
  h_ttbb->Draw("colz");
  h_ttbb->SetStats(0000);
  c->SetRightMargin(0.15);
  c->Print(Form("c_%s_S%s.pdf(",var.Data(),step.Data()));

  h_Top_Hct->Draw("colz");
  h_Top_Hct->SetStats(0000);
  c->Print(Form("c_%s_S%s.pdf",var.Data(),step.Data()));

  h_Top_Hut->Draw("colz");
  h_Top_Hut->SetStats(0000);
  c->Print(Form("c_%s_S%s.pdf",var.Data(),step.Data()));

  h_AntiTop_Hct->Draw("colz");
  h_AntiTop_Hct->SetStats(0000);
  c->Print(Form("c_%s_S%s.pdf",var.Data(),step.Data()));

  h_AntiTop_Hut->Draw("colz");
  h_AntiTop_Hut->SetStats(0000);
  c->Print(Form("c_%s_S%s.pdf)",var.Data(),step.Data()));

/*
  TLegend * l = new TLegend(0.75,0.65,0.9,0.9);
  l->AddEntry(h_NJet_Ch0_S1_Top_Hct,"Top Hct", "L");
  l->AddEntry(h_NJet_Ch0_S1_Top_Hut,"Top Hut", "L");
  l->AddEntry(h_NJet_Ch0_S1_AntiTop_Hct,"AntiTop Hct", "L");
  l->AddEntry(h_NJet_Ch0_S1_AntiTop_Hut,"AntiTop Hut", "L");
  l->AddEntry(h_NJet_Ch0_S1_ttbb,"ttbb", "L");
  l->SetTextSize(.03);
  l->Draw();
*/
}

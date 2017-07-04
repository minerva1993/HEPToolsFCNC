void compare(TString var, TString ch, TString step );

void comp_fcnc(TString ch = "0", TString step = "0"){

  compare("NJet", ch, step);
  compare("NBJetCSVv2M", ch, step);
  compare("NBJetCSVv2T", ch, step);
  compare("NCJetM", ch, step);
  compare("MET", ch, step);
  compare("WMass", ch, step);
  compare("HMass_m", ch, step);
  compare("bJetPtHm", ch, step);
  compare("cJetPt", ch, step);
  compare("DPhi", ch, step);
  compare("bjmDPhi", ch, step);
  compare("bjmDEta", ch, step);
  compare("bjmDR", ch, step);
  compare("LepIso", ch, step);

}

void compare(TString var, TString ch, TString step ){
  //TFile * f_Top_Hut = new TFile("hist_Top_Hut.root");
  //TFile * f_AntiTop_Hct = new TFile("hist_AntiTop_Hct.root");
  //TFile * f_AntiTop_Hut = new TFile("hist_AntiTop_Hut.root");

  TFile * f_tch_tj = new TFile("jw/hist_Top_Hct.root");
  TFile * f_tch_jav = new TFile("jav/hist_Top_Hct.root");
  //TH2F * h_Top_Hct = (TH2F *) f_Top_Hct->Get(Form("h_%s_S%s_Top_Hct",var.Data(),step.Data()));
  //TH2F * h_Top_Hut = (TH2F *) f_Top_Hut->Get(Form("h_%s_S%s_Top_Hut",var.Data(),step.Data()));
  //TH2F * h_AntiTop_Hct = (TH2F *) f_AntiTop_Hct->Get(Form("h_%s_S%s_AntiTop_Hct",var.Data(),step.Data()));
  //TH2F * h_AntiTop_Hut = (TH2F *) f_AntiTop_Hut->Get(Form("h_%s_S%s_AntiTop_Hut",var.Data(),step.Data()));
  TH2F * h_tch_tj = (TH2F *) f_tch_tj->Get(Form("h_%s_Ch%s_S%s_Top_Hct",var.Data(),ch.Data(),step.Data()));
  TH2F * h_tch_jav = (TH2F *) f_tch_jav->Get(Form("h_%s_Ch%s_S%s_Top_Hct",var.Data(),ch.Data(),step.Data()));


  h_tch_tj->Scale(1.0/h_tch_tj->Integral());
  h_tch_jav->Scale(1.0/h_tch_jav->Integral());

  TCanvas * c = new TCanvas("c","c",1);
  h_tch_tj->SetStats(0000);
  h_tch_tj->Draw("Hist");
  h_tch_jav->Draw("Histsame");

  double scale_tj = 1.2*(h_tch_tj->GetMaximum());
  double scale_jav = 1.2*(h_tch_jav->GetMaximum());

  if( scale_tj > scale_jav) h_tch_tj->GetYaxis()->SetRangeUser(0,scale_tj);
  else  h_tch_tj->GetYaxis()->SetRangeUser(0,scale_jav);

  h_tch_tj->SetLineColor(2);
  h_tch_jav->SetLineColor(4);

  TLegend * l = new TLegend(0.75,0.65,0.9,0.9);
  l->AddEntry(h_tch_tj,"tch_jw", "L");
  l->AddEntry(h_tch_jav,"tch_jav", "L");
  l->SetTextSize(.03);
  l->Draw();

  c->Print(Form("c_%s_Ch%s_S%s.pdf",var.Data(),ch.Data(),step.Data()));
}

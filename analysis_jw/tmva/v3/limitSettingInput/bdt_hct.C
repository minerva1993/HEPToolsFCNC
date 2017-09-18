void bdt_hct(){

  TFile *ftrain  = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v3/keras_Hct2v2/output_keras.root");
  TFile *frd = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v3/rd_output_Hct.root");
  TTree *tree = (TTree*)frd->Get("tree");

  TH1D *h_sig = (TH1D*)ftrain->Get("keras_Hct2v2/Method_BDT/BDT/MVA_BDT_S");
  TH1D *h_bkg = (TH1D*)ftrain->Get("keras_Hct2v2/Method_BDT/BDT/MVA_BDT_B");

  TCanvas * c1 = new TCanvas("c1","c1",1);
  c1->cd();
  h_sig->Draw();
  h_sig->SetName("h_sig");

  TCanvas * c2 = new TCanvas("c2","c2",1);
  c2->cd();
  h_bkg->Draw();
  h_bkg->SetName("h_bkg");
  
  TH1F *h_data = new TH1F("h_data", "", 40, -0.24,0.27);
  TCanvas * c3 = new TCanvas("c3","c3",1);
  c3->cd();
  tree->Draw("BDTScore>>h_data");

  TFile* file = new TFile("shape_BDT_Hct.root","RECREATE");
  h_sig->Write();
  h_bkg->Write();
  h_data->Write();
  file->Close();
}

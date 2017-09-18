void keras_hut(){

  TFile *ftrain  = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v3/keras_Hut1v2/output_keras.root");
  TFile *frd = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v3/rd_output_Hut.root");
  TTree *tree = (TTree*)frd->Get("tree");

  TH1D *h_sig = (TH1D*)ftrain->Get("keras_Hut1v2/Method_PyKeras/PyKeras/MVA_PyKeras_S");
  TH1D *h_bkg = (TH1D*)ftrain->Get("keras_Hut1v2/Method_PyKeras/PyKeras/MVA_PyKeras_B");

  TCanvas * c1 = new TCanvas("c1","c1",1);
  c1->cd();
  h_sig->Draw();
  h_sig->SetName("h_sig");

  TCanvas * c2 = new TCanvas("c2","c2",1);
  c2->cd();
  h_bkg->Draw();
  h_bkg->SetName("h_bkg");
  
  TH1F *h_data = new TH1F("h_data", "", 40, 0.02, 0.99);
  TCanvas * c3 = new TCanvas("c3","c3",1);
  c3->cd();
  tree->Draw("KerasScore>>h_data");

  TFile* file = new TFile("shape_PyKeras_Hut.root","RECREATE");
  h_sig->Write();
  h_bkg->Write();
  h_data->Write();
  file->Close();
}

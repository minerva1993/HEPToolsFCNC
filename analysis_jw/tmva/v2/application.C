#include "TFile.h"
#include "TString.h"

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"


void application(){

  TFile* data = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016H_v3.root"); 
  TTree* dataTree = (TTree*)(data->Get("tmva_DataSingleEG"));

/*
  TFile* dataMuB = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016B.root"); 
  TTree* dataTreeMuB = (TTree*)(dataMuB->Get("tmva_DataSingleMu"));
  TFile* dataMuC = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016C.root"); 
  TTree* dataTreeMuC = (TTree*)(dataMuC->Get("tmva_DataSingleMu"));
  TFile* dataMuD = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016D.root"); 
  TTree* dataTreeMuD = (TTree*)(dataMuD->Get("tmva_DataSingleMu"));
  TFile* dataMuE = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016E.root"); 
  TTree* dataTreeMuE = (TTree*)(dataMuE->Get("tmva_DataSingleMu"));
  TFile* dataMuF = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016F.root"); 
  TTree* dataTreeMuF = (TTree*)(dataMuF->Get("tmva_DataSingleMu"));
  TFile* dataMuG = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016G.root"); 
  TTree* dataTreeMuG = (TTree*)(dataMuG->Get("tmva_DataSingleMu"));
  TFile* dataMuH2 = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016H_v2.root"); 
  TTree* dataTreeMuH2 = (TTree*)(dataMuH2->Get("tmva_DataSingleMu"));
  TFile* dataMuH3 = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleMuon_Run2016H_v3.root"); 
  TTree* dataTreeMuH3 = (TTree*)(dataMuH3->Get("tmva_DataSingleMu"));


  TFile* dataEGB = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016B.root");
  TTree* dataTreeEGB = (TTree*)(dataEGB->Get("tmva_DataSingleEG"));
  TFile* dataEGC = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016C.root");
  TTree* dataTreeEGC = (TTree*)(dataEGC->Get("tmva_DataSingleEG"));
  TFile* dataEGD = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016D.root");
  TTree* dataTreeEGD = (TTree*)(dataEGD->Get("tmva_DataSingleEG"));
  TFile* dataEGE = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016E.root");
  TTree* dataTreeEGE = (TTree*)(dataEGE->Get("tmva_DataSingleEG"));
  TFile* dataEGF = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016F.root");
  TTree* dataTreeEGF = (TTree*)(dataEGF->Get("tmva_DataSingleEG"));
  TFile* dataEGG = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016G.root");
  TTree* dataTreeEGG = (TTree*)(dataEGG->Get("tmva_DataSingleEG"));
  TFile* dataEGH2 = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016H_v2.root");
  TTree* dataTreeEGH2 = (TTree*)(dataEGH2->Get("tmva_DataSingleEG"));
  TFile* dataEGH3 = new TFile("/home/minerva1993/fcnc/analysis_jw/tmva/v2/mkNtuple_v2/tmva_SingleElectron_Run2016H_v3.root");
  TTree* dataTreeEGH3 = (TTree*)(dataEGH3->Get("tmva_DataSingleEG"));
*/

  TFile *target = new TFile("rd_BDT_output.root","RECREATE" );
  TTree *tree = new TTree("tree","treelibrated tree");

  TMVA::Tools::Instance();
  TMVA::Reader *reader = new TMVA::Reader( "V:Color:!Silent" );  

  float intVar[3];
  float floatVar[22];

  reader->AddVariable("njets", &intVar[0]);
  reader->AddVariable("nbjets_m", &intVar[1]);
  reader->AddVariable("ncjets_m", &intVar[2]);
  reader->AddVariable("lepDPhi", &floatVar[0]);
  reader->AddVariable("missingET", &floatVar[1]);
  reader->AddVariable("bjetmDR", &floatVar[2]);
  reader->AddVariable("bjetmDEta", &floatVar[3]);
  reader->AddVariable("bjetmDPhi", &floatVar[4]);
  reader->AddVariable("dibjetsMass", &floatVar[5]);
  reader->AddVariable("bjetPt_dibjetsm", &floatVar[6]);
  reader->AddVariable("cjetPt", &floatVar[7]);
  reader->AddVariable("jetPt1", &floatVar[8]);
  reader->AddVariable("jetPt2", &floatVar[9]);
  reader->AddVariable("jetPt3", &floatVar[10]);
  reader->AddVariable("jetPt4", &floatVar[11]);
  reader->AddVariable("jetCSV1", &floatVar[12]);
  reader->AddVariable("JetCSV2", &floatVar[13]);
  reader->AddVariable("JetCSV3", &floatVar[14]);
  reader->AddVariable("JetCSV4", &floatVar[15]);
  reader->AddVariable("DRLepWMass", &floatVar[16]);
  reader->AddVariable("DRHadWMass", &floatVar[17]);
  reader->AddVariable("DRHMass", &floatVar[18]);
  reader->AddVariable("DRDR", &floatVar[19]);
  reader->AddVariable("DRLepTopMass", &floatVar[20]);
  reader->AddVariable("DRHucTopMass", &floatVar[21]);

  reader->BookMVA("BDT", "/home/minerva1993/fcnc/analysis_jw/tmva/v2/inputcorr_test/weights/TMVAClassification_BDT.weights.xml");

  int usrintVar[3];
  float usrfloatVar[22];

  dataTree->SetBranchAddress("njets", &usrintVar[0]);
  dataTree->SetBranchAddress("nbjets_m", &usrintVar[1]);
  dataTree->SetBranchAddress("ncjets_m", &usrintVar[2]);
  dataTree->SetBranchAddress("lepDPhi", &usrfloatVar[0]);
  dataTree->SetBranchAddress("missingET", &usrfloatVar[1]);
  dataTree->SetBranchAddress("bjetmDR", &usrfloatVar[2]);
  dataTree->SetBranchAddress("bjetmDEta", &usrfloatVar[3]);
  dataTree->SetBranchAddress("bjetmDPhi", &usrfloatVar[4]);
  dataTree->SetBranchAddress("dibjetsMass", &usrfloatVar[5]);
  dataTree->SetBranchAddress("bjetPt_dibjetsm", &usrfloatVar[6]);
  dataTree->SetBranchAddress("cjetPt", &usrfloatVar[7]);
  dataTree->SetBranchAddress("jetPt1", &usrfloatVar[8]);
  dataTree->SetBranchAddress("jetPt2", &usrfloatVar[9]);
  dataTree->SetBranchAddress("jetPt3", &usrfloatVar[10]);
  dataTree->SetBranchAddress("jetPt4", &usrfloatVar[11]);
  dataTree->SetBranchAddress("jetCSV1", &usrfloatVar[12]);
  dataTree->SetBranchAddress("JetCSV2", &usrfloatVar[13]);
  dataTree->SetBranchAddress("JetCSV3", &usrfloatVar[14]);
  dataTree->SetBranchAddress("JetCSV4", &usrfloatVar[15]);
  dataTree->SetBranchAddress("DRLepWMass", &usrfloatVar[16]);
  dataTree->SetBranchAddress("DRHadWMass", &usrfloatVar[17]);
  dataTree->SetBranchAddress("DRHMass", &usrfloatVar[18]);
  dataTree->SetBranchAddress("DRDR", &usrfloatVar[19]);
  dataTree->SetBranchAddress("DRLepTopMass", &usrfloatVar[20]);
  dataTree->SetBranchAddress("DRHucTopMass", &usrfloatVar[21]);

  tree->Branch("njets", &usrintVar[0]);
  tree->Branch("nbjets_m", &usrintVar[1]);
  tree->Branch("ncjets_m", &usrintVar[2]);
  tree->Branch("lepDPhi", &usrfloatVar[0]);
  tree->Branch("missingET", &usrfloatVar[1]);
  tree->Branch("bjetmDR", &usrfloatVar[2]);
  tree->Branch("bjetmDEta", &usrfloatVar[3]);
  tree->Branch("bjetmDPhi", &usrfloatVar[4]);
  tree->Branch("dibjetsMass", &usrfloatVar[5]);
  tree->Branch("bjetPt_dibjetsm", &usrfloatVar[6]);
  tree->Branch("cjetPt", &usrfloatVar[7]);
  tree->Branch("jetPt1", &usrfloatVar[8]);
  tree->Branch("jetPt2", &usrfloatVar[9]);
  tree->Branch("jetPt3", &usrfloatVar[10]);
  tree->Branch("jetPt4", &usrfloatVar[11]);
  tree->Branch("jetCSV1", &usrfloatVar[12]);
  tree->Branch("JetCSV2", &usrfloatVar[13]);
  tree->Branch("JetCSV3", &usrfloatVar[14]);
  tree->Branch("JetCSV4", &usrfloatVar[15]);
  tree->Branch("DRLepWMass", &usrfloatVar[16]);
  tree->Branch("DRHadWMass", &usrfloatVar[17]);
  tree->Branch("DRHMass", &usrfloatVar[18]);
  tree->Branch("DRDR", &usrfloatVar[19]);
  tree->Branch("DRLepTopMass", &usrfloatVar[20]);
  tree->Branch("DRHucTopMass", &usrfloatVar[21]);

  //tree->Branch

  Float_t BDT_response;
  tree->Branch("BDT_response",&BDT_response);

  for (Long64_t ievt=0; ievt<dataTree->GetEntries();ievt++) {

    if (ievt%100000 == 0) std::cout << "--- ... Processing event: " << ievt <<std::endl;
    
    dataTree->GetEntry(ievt);

    BDT_response=reader->EvaluateMVA("BDT", 0.6);

    tree->Fill();

  }
  
  tree->Write();

}

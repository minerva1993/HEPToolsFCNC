#define analyzer_cxx
#include "analyzer.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <TLorentzVector.h>
#include <iostream>

const double CSVM = 0.5;

void analyzer::Loop(const string outFileName)
{
  int nevt = 0;

  //objects for ntuple
  int b_nevt = 0;

  int b_njets, b_bjets_n, b_genMatch, b_combiCount;
  int b_jet0Idx, b_jet1Idx, b_jet2Idx, b_jet3Idx;
  float b_lepton_pt, b_lepton_eta, b_lepton_phi, b_met, b_met_phi, b_lepdphi;
  float b_lepWpt, b_lepWeta, b_lepWphi, b_lepWdphi, b_lepWm;

  float b_jet0pt, b_jet0eta, b_jet0phi, b_jet0m;
  float b_jet1pt, b_jet1eta, b_jet1phi, b_jet1m;
  float b_jet2pt, b_jet2eta, b_jet2phi, b_jet2m;
  float b_jet3pt, b_jet3eta, b_jet3phi, b_jet3m;
  int b_jet0csv, b_jet1csv, b_jet2csv, b_jet3csv;

  float b_jet12pt, b_jet12eta, b_jet12deta, b_jet12phi, b_jet12dphi, b_jet12m, b_jet12dR;
  float b_jet23pt, b_jet23eta, b_jet23deta, b_jet23phi, b_jet23dphi, b_jet23m, b_jet23dR;
  float b_jet31pt, b_jet31eta, b_jet31deta, b_jet31phi, b_jet31dphi, b_jet31m, b_jet31dR;
  
  float b_lepTpt, b_lepTeta, b_lepTdeta, b_lepTphi, b_lepTdphi, b_lepTdR, b_lepTm;
  float b_hadTpt, b_hadTeta, b_hadT12_3deta, b_hadT23_1deta, b_hadT31_2deta, b_hadTphi, b_hadTm;
  float b_hadT12_3dphi, b_hadT23_1dphi, b_hadT31_2dphi, b_hadT12_3dR, b_hadT23_1dR, b_hadT31_2dR;

  float b_genLep_pt, b_genLep_eta, b_genLep_phi, b_gennu_pt, b_gennu_eta, b_gennu_phi;
  float b_genLepB_pt, b_genLepB_eta, b_genLepB_phi, b_genLepB_m;
  float b_genLepW_pt, b_genLepW_eta, b_genLepW_phi, b_genLepW_m;
  float b_genHadJ1_pt, b_genHadJ1_eta, b_genHadJ1_phi, b_genHadJ1_m;
  float b_genHadJ2_pt, b_genHadJ2_eta, b_genHadJ2_phi, b_genHadJ2_m;
  float b_genHadJ3_pt, b_genHadJ3_eta, b_genHadJ3_phi, b_genHadJ3_m;
  float b_genLepT_pt, b_genLepT_eta, b_genLepT_phi, b_genLepT_m;
  float b_genHadT_pt, b_genHadT_eta, b_genHadT_phi, b_genHadT_m;
  float b_genHadW_pt, b_genHadW_eta, b_genHadW_phi, b_genHadW_m, b_genHadW_dR;

  b_njets = b_bjets_n = b_combiCount = 0;//int
  b_jet0Idx = b_jet1Idx = b_jet2Idx = b_jet3Idx = -1;
  b_lepton_pt = b_met = b_lepdphi = 0;
  b_lepton_eta = b_lepton_phi = b_met_phi = -10;
  b_lepWpt = b_lepWm = 0;
  b_lepWeta = b_lepWphi = b_lepWdphi = 10;
  b_jet0pt = b_jet0m = b_jet1pt = b_jet1m = b_jet2pt = b_jet2m = b_jet3pt = b_jet3m = 0;
  b_jet0eta = b_jet0phi = b_jet1eta = b_jet1phi = b_jet2eta = b_jet2phi = b_jet3eta = b_jet3phi = 10;
  b_jet0csv = b_jet1csv = b_jet2csv = b_jet3csv = -1;

  b_jet12pt = b_jet12m = b_jet12dR = b_jet23pt = b_jet23m = b_jet23dR = b_jet31pt = b_jet31m = b_jet31dR = 0;
  b_jet12eta = b_jet12deta =  b_jet12phi = b_jet12dphi = 10;
  b_jet23eta = b_jet23deta =  b_jet23phi = b_jet23dphi = 10;
  b_jet31eta = b_jet31deta =  b_jet31phi = b_jet31dphi = 10;

  b_lepTpt = b_lepTdR = b_lepTm = 0;
  b_lepTeta = b_lepTdeta = b_lepTphi = b_lepTdphi = 10;
  b_hadTpt = b_hadT12_3dR = b_hadT23_1dR = b_hadT31_2dR = b_hadTm = 0;
  b_hadTeta = b_hadT12_3deta = b_hadT23_1deta = b_hadT31_2deta = b_hadTphi = 10;
  b_hadT12_3dphi = b_hadT23_1dphi = b_hadT31_2dphi = 10; 

  b_genLep_pt = b_genLep_eta = b_genLep_phi = b_gennu_pt = b_gennu_eta = b_gennu_phi =0;
  b_genLepB_pt = b_genLepB_eta = b_genLepB_phi = b_genLepB_m = 0;
  b_genLepW_pt = b_genLepW_eta = b_genLepW_phi = b_genLepW_m = 0;
  b_genHadJ1_pt = b_genHadJ1_eta = b_genHadJ1_phi = b_genHadJ1_m = 0;
  b_genHadJ2_pt = b_genHadJ2_eta = b_genHadJ2_phi = b_genHadJ2_m = 0;
  b_genHadJ3_pt = b_genHadJ3_eta = b_genHadJ3_phi = b_genHadJ3_m = 0;
  b_genLepT_pt = b_genLepT_eta = b_genLepT_phi = b_genLepT_m = 0;
  b_genHadT_pt = b_genHadT_eta = b_genHadT_phi = b_genHadT_m = 0;
  b_genHadW_pt = b_genHadW_eta = b_genHadW_phi = b_genHadW_m = b_genHadW_dR = 0;

  //////////////////////////////////////////////////////////////////
  TFile* fout = new TFile(outFileName.c_str(), "recreate");
  TTree* sigTree = new TTree("sig_tree","signal tree");
  sigTree->SetDirectory(fout);

  sigTree->Branch("nevt"      , &b_nevt      , "nevt/I");
  sigTree->Branch("combiCount", &b_combiCount, "combiCount/I");
  sigTree->Branch("genMatch"  , &b_genMatch  , "genMatch/I");

  sigTree->Branch("njets"     , &b_njets     , "njets/I");
  sigTree->Branch("nbjets_m"  , &b_bjets_n   , "nbjets_m/I");
  sigTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
  sigTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  sigTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  sigTree->Branch("MET"       , &b_met       , "MET/F");
  sigTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
  sigTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

  //jet assignment
  sigTree->Branch("jet0pt" , &b_jet0pt , "jet0pt/F");
  sigTree->Branch("jet0eta", &b_jet0eta, "jet0eta/F");
  sigTree->Branch("jet0phi", &b_jet0phi, "jet0phi/F");
  sigTree->Branch("jet0m"  , &b_jet0m  , "jet0m/F");
  sigTree->Branch("jet0csv", &b_jet0csv, "jet0csv/I");
  sigTree->Branch("jet0Idx", &b_jet0Idx, "jet0Idx/I");

  sigTree->Branch("jet1pt" , &b_jet1pt , "jet1pt/F");
  sigTree->Branch("jet1eta", &b_jet1eta, "jet1eta/F");
  sigTree->Branch("jet1phi", &b_jet1phi, "jet1phi/F");
  sigTree->Branch("jet1m"  , &b_jet1m  , "jet1m/F");
  sigTree->Branch("jet1csv", &b_jet1csv, "jet1csv/I");
  sigTree->Branch("jet1Idx", &b_jet1Idx, "jet1Idx/I");

  sigTree->Branch("jet2pt" , &b_jet2pt , "jet2pt/F");
  sigTree->Branch("jet2eta", &b_jet2eta, "jet2eta/F");
  sigTree->Branch("jet2phi", &b_jet2phi, "jet2phi/F");
  sigTree->Branch("jet2m"  , &b_jet2m  , "jet2m/F");
  sigTree->Branch("jet2csv", &b_jet2csv, "jet2csv/I");
  sigTree->Branch("jet2Idx", &b_jet2Idx, "jet2Idx/I");

  sigTree->Branch("jet3pt" , &b_jet3pt , "jet3pt/F");
  sigTree->Branch("jet3eta", &b_jet3eta, "jet3eta/F");
  sigTree->Branch("jet3phi", &b_jet3phi, "jet3phi/F");
  sigTree->Branch("jet3m"  , &b_jet3m  , "jet3m/F");
  sigTree->Branch("jet3csv", &b_jet3csv, "jet3csv/I");
  sigTree->Branch("jet3Idx", &b_jet3Idx, "jet3Idx/I");

  sigTree->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
  sigTree->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
  sigTree->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
  sigTree->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");
  sigTree->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
  sigTree->Branch("jet12dR"  , &b_jet12dR  , "jet12dR/F");
  sigTree->Branch("jet12m"   , &b_jet12m   , "jet12m/F");

  sigTree->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
  sigTree->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
  sigTree->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
  sigTree->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
  sigTree->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
  sigTree->Branch("jet23dR"  , &b_jet23dR  , "jet23dR/F");
  sigTree->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

  sigTree->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
  sigTree->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
  sigTree->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
  sigTree->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
  sigTree->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
  sigTree->Branch("jet31dR"  , &b_jet31dR  , "jet31dR/F");
  sigTree->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

  sigTree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
  sigTree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
  sigTree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
  sigTree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
  sigTree->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

  sigTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
  sigTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
  sigTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
  sigTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
  sigTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
  sigTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
  sigTree->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

  sigTree->Branch("hadTpt"      , &b_hadTpt      , "hadTpt/F");
  sigTree->Branch("hadTeta"     , &b_hadTeta     , "hadTeta/F");
  sigTree->Branch("hadT12_3deta", &b_hadT12_3deta, "hadT12_3deta/F");
  sigTree->Branch("hadT23_1deta", &b_hadT23_1deta, "hadT23_1deta/F");
  sigTree->Branch("hadT31_2deta", &b_hadT31_2deta, "hadT31_2deta/F");
  sigTree->Branch("hadTphi"     , &b_hadTphi     , "hadTphi/F");
  sigTree->Branch("hadT12_3dphi", &b_hadT12_3dphi, "hadT12_3dphi/F");
  sigTree->Branch("hadT23_1dphi", &b_hadT23_1dphi, "hadT23_1dphi/F");
  sigTree->Branch("hadT31_2dphi", &b_hadT31_2dphi, "hadT31_2dphi/F");
  sigTree->Branch("hadT12_3dR"  , &b_hadT12_3dR  , "hadT12_3dR/F");
  sigTree->Branch("hadT23_1dR"  , &b_hadT23_1dR  , "hadT23_1dR/F");
  sigTree->Branch("hadT31_2dR"  , &b_hadT31_2dR  , "hadT31_2dR/F");
  sigTree->Branch("hadTm"       , &b_hadTm       , "hadTm/F");

  sigTree->Branch("genLep_pt" , &b_genLep_pt , "genLep_pt/F");
  sigTree->Branch("genLep_eta", &b_genLep_eta, "genLep_eta/F");
  sigTree->Branch("genLep_phi", &b_genLep_phi, "genLep_phi/F");

  sigTree->Branch("gennu_pt" , &b_gennu_pt , "gennu_pt/F");
  sigTree->Branch("gennu_eta", &b_gennu_eta, "gennu_eta/F");
  sigTree->Branch("gennu_phi", &b_gennu_phi, "gennu_phi/F");

  sigTree->Branch("genLepB_pt" , &b_genLepB_pt , "genLepB_pt/F");
  sigTree->Branch("genLepB_eta", &b_genLepB_eta, "genLepB_eta/F");
  sigTree->Branch("genLepB_phi", &b_genLepB_phi, "genLepB_phi/F");
  sigTree->Branch("genLepB_m"  , &b_genLepB_m  , "genLepB_m/F");

  sigTree->Branch("genLepW_pt" , &b_genLepW_pt , "genLepW_pt/F");
  sigTree->Branch("genLepW_eta", &b_genLepW_eta, "genLepW_eta/F");
  sigTree->Branch("genLepW_phi", &b_genLepW_phi, "genLepW_phi/F");
  sigTree->Branch("genLepW_m"  , &b_genLepW_m  , "genLepW_m/F");

  sigTree->Branch("genLepT_pt" , &b_genLepT_pt , "genLepT_pt/F" );
  sigTree->Branch("genLepT_eta", &b_genLepT_eta, "genLepT_eta/F");
  sigTree->Branch("genLepT_phi", &b_genLepT_phi, "genLepT_phi/F");
  sigTree->Branch("genLepT_m"  , &b_genLepT_m  , "genLepT_m/F"  );

  sigTree->Branch("genHadJ1_pt" , &b_genHadJ1_pt , "genHadJ1_pt/F");
  sigTree->Branch("genHadJ1_eta", &b_genHadJ1_eta, "genHadJ1_eta/F");
  sigTree->Branch("genHadJ1_phi", &b_genHadJ1_phi, "genHadJ1_phi/F");
  sigTree->Branch("genHadJ1_m"  , &b_genHadJ1_m  , "genHadJ1_m/F");

  sigTree->Branch("genHadJ2_pt" , &b_genHadJ2_pt , "genHadJ2_pt/F");
  sigTree->Branch("genHadJ2_eta", &b_genHadJ2_eta, "genHadJ2_eta/F");
  sigTree->Branch("genHadJ2_phi", &b_genHadJ2_phi, "genHadJ2_phi/F");
  sigTree->Branch("genHadJ2_m"  , &b_genHadJ2_m  , "genHadJ2_m/F");

  sigTree->Branch("genHadJ3_pt" , &b_genHadJ3_pt , "genHadJ3_pt/F");
  sigTree->Branch("genHadJ3_eta", &b_genHadJ3_eta, "genHadJ3_eta/F");
  sigTree->Branch("genHadJ3_phi", &b_genHadJ3_phi, "genHadJ3_phi/F");
  sigTree->Branch("genHadJ3_m"  , &b_genHadJ3_m  , "genHadJ3_m/F");

  sigTree->Branch("genHadW_pt" , &b_genHadW_pt , "genHadW_pt/F" );
  sigTree->Branch("genHadW_eta", &b_genHadW_eta, "genHadW_eta/F");
  sigTree->Branch("genHadW_phi", &b_genHadW_phi, "genHadW_phi/F");
  sigTree->Branch("genHadW_m"  , &b_genHadW_m  , "genHadW_m/F"  );
  sigTree->Branch("genHadW_dR" , &b_genHadW_dR , "genHadW_dR/F" );

  sigTree->Branch("genHadT_pt" , &b_genHadT_pt , "genHadT_pt/F" );
  sigTree->Branch("genHadT_eta", &b_genHadT_eta, "genHadT_eta/F");
  sigTree->Branch("genHadT_phi", &b_genHadT_phi, "genHadT_phi/F");
  sigTree->Branch("genHadT_m"  , &b_genHadT_m  , "genHadT_m/F"  );

  ///////////////////////////////////////////////////////////////////
  TTree* bkgTree = new TTree("bkg_tree","background tree");
  bkgTree->SetDirectory(fout);

  bkgTree->Branch("nevt"      , &b_nevt      , "nevt/I");
  bkgTree->Branch("combiCount", &b_combiCount, "combiCount/I");
  bkgTree->Branch("genMatch"  , &b_genMatch  , "genMatch/I");

  bkgTree->Branch("njets"     , &b_njets     , "njets/I");
  bkgTree->Branch("nbjets_m"  , &b_bjets_n   , "nbjets_m/I");
  bkgTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
  bkgTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  bkgTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  bkgTree->Branch("MET"       , &b_met       , "MET/F");
  bkgTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
  bkgTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

  //jet assignment
  bkgTree->Branch("jet0pt" , &b_jet0pt , "jet0pt/F");
  bkgTree->Branch("jet0eta", &b_jet0eta, "jet0eta/F");
  bkgTree->Branch("jet0phi", &b_jet0phi, "jet0phi/F");
  bkgTree->Branch("jet0m"  , &b_jet0m  , "jet0m/F");
  bkgTree->Branch("jet0csv", &b_jet0csv, "jet0csv/I");
  bkgTree->Branch("jet0Idx", &b_jet0Idx, "jet0Idx/I");

  bkgTree->Branch("jet1pt" , &b_jet1pt , "jet1pt/F");
  bkgTree->Branch("jet1eta", &b_jet1eta, "jet1eta/F");
  bkgTree->Branch("jet1phi", &b_jet1phi, "jet1phi/F");
  bkgTree->Branch("jet1m"  , &b_jet1m  , "jet1m/F");
  bkgTree->Branch("jet1csv", &b_jet1csv, "jet1csv/I");
  bkgTree->Branch("jet1Idx", &b_jet1Idx, "jet1Idx/I");

  bkgTree->Branch("jet2pt" , &b_jet2pt , "jet2pt/F");
  bkgTree->Branch("jet2eta", &b_jet2eta, "jet2eta/F");
  bkgTree->Branch("jet2phi", &b_jet2phi, "jet2phi/F");
  bkgTree->Branch("jet2m"  , &b_jet2m  , "jet2m/F");
  bkgTree->Branch("jet2csv", &b_jet2csv, "jet2csv/I");
  bkgTree->Branch("jet2Idx", &b_jet2Idx, "jet2Idx/I");

  bkgTree->Branch("jet3pt" , &b_jet3pt , "jet3pt/F");
  bkgTree->Branch("jet3eta", &b_jet3eta, "jet3eta/F");
  bkgTree->Branch("jet3phi", &b_jet3phi, "jet3phi/F");
  bkgTree->Branch("jet3m"  , &b_jet3m  , "jet3m/F");
  bkgTree->Branch("jet3csv", &b_jet3csv, "jet3csv/I");
  bkgTree->Branch("jet3Idx", &b_jet3Idx, "jet3Idx/I");

  bkgTree->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
  bkgTree->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
  bkgTree->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
  bkgTree->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");
  bkgTree->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
  bkgTree->Branch("jet12dR"  , &b_jet12dR  , "jet12dR/F");
  bkgTree->Branch("jet12m"   , &b_jet12m   , "jet12m/F");

  bkgTree->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
  bkgTree->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
  bkgTree->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
  bkgTree->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
  bkgTree->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
  bkgTree->Branch("jet23dR"  , &b_jet23dR  , "jet23dR/F");
  bkgTree->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

  bkgTree->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
  bkgTree->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
  bkgTree->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
  bkgTree->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
  bkgTree->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
  bkgTree->Branch("jet31dR"  , &b_jet31dR  , "jet31dR/F");
  bkgTree->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

  bkgTree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
  bkgTree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
  bkgTree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
  bkgTree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
  bkgTree->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

  bkgTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
  bkgTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
  bkgTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
  bkgTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
  bkgTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
  bkgTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
  bkgTree->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

  bkgTree->Branch("hadTpt"      , &b_hadTpt      , "hadTpt/F");
  bkgTree->Branch("hadTeta"     , &b_hadTeta     , "hadTeta/F");
  bkgTree->Branch("hadT12_3deta", &b_hadT12_3deta, "hadT12_3deta/F");
  bkgTree->Branch("hadT23_1deta", &b_hadT23_1deta, "hadT23_1deta/F");
  bkgTree->Branch("hadT31_2deta", &b_hadT31_2deta, "hadT31_2deta/F");
  bkgTree->Branch("hadTphi"     , &b_hadTphi     , "hadTphi/F");
  bkgTree->Branch("hadT12_3dphi", &b_hadT12_3dphi, "hadT12_3dphi/F");
  bkgTree->Branch("hadT23_1dphi", &b_hadT23_1dphi, "hadT23_1dphi/F");
  bkgTree->Branch("hadT31_2dphi", &b_hadT31_2dphi, "hadT31_2dphi/F");
  bkgTree->Branch("hadT12_3dR"  , &b_hadT12_3dR  , "hadT12_3dR/F");
  bkgTree->Branch("hadT23_1dR"  , &b_hadT23_1dR  , "hadT23_1dR/F");
  bkgTree->Branch("hadT31_2dR"  , &b_hadT31_2dR  , "hadT31_2dR/F");
  bkgTree->Branch("hadTm"       , &b_hadTm       , "hadTm/F");

  bkgTree->Branch("genLep_pt" , &b_genLep_pt , "genLep_pt/F");
  bkgTree->Branch("genLep_eta", &b_genLep_eta, "genLep_eta/F");
  bkgTree->Branch("genLep_phi", &b_genLep_phi, "genLep_phi/F");

  bkgTree->Branch("gennu_pt" , &b_gennu_pt , "gennu_pt/F");
  bkgTree->Branch("gennu_eta", &b_gennu_eta, "gennu_eta/F");
  bkgTree->Branch("gennu_phi", &b_gennu_phi, "gennu_phi/F");

  bkgTree->Branch("genLepB_pt" , &b_genLepB_pt , "genLepB_pt/F");
  bkgTree->Branch("genLepB_eta", &b_genLepB_eta, "genLepB_eta/F");
  bkgTree->Branch("genLepB_phi", &b_genLepB_phi, "genLepB_phi/F");
  bkgTree->Branch("genLepB_m"  , &b_genLepB_m  , "genLepB_m/F");

  bkgTree->Branch("genLepW_pt" , &b_genLepW_pt , "genLepW_pt/F");
  bkgTree->Branch("genLepW_eta", &b_genLepW_eta, "genLepW_eta/F");
  bkgTree->Branch("genLepW_phi", &b_genLepW_phi, "genLepW_phi/F");
  bkgTree->Branch("genLepW_m"  , &b_genLepW_m  , "genLepW_m/F");

  bkgTree->Branch("genLepT_pt" , &b_genLepT_pt , "genLepT_pt/F" );
  bkgTree->Branch("genLepT_eta", &b_genLepT_eta, "genLepT_eta/F");
  bkgTree->Branch("genLepT_phi", &b_genLepT_phi, "genLepT_phi/F");
  bkgTree->Branch("genLepT_m"  , &b_genLepT_m  , "genLepT_m/F"  );

  bkgTree->Branch("genHadJ1_pt" , &b_genHadJ1_pt , "genHadJ1_pt/F");
  bkgTree->Branch("genHadJ1_eta", &b_genHadJ1_eta, "genHadJ1_eta/F");
  bkgTree->Branch("genHadJ1_phi", &b_genHadJ1_phi, "genHadJ1_phi/F");
  bkgTree->Branch("genHadJ1_m"  , &b_genHadJ1_m  , "genHadJ1_m/F");

  bkgTree->Branch("genHadJ2_pt" , &b_genHadJ2_pt , "genHadJ2_pt/F");
  bkgTree->Branch("genHadJ2_eta", &b_genHadJ2_eta, "genHadJ2_eta/F");
  bkgTree->Branch("genHadJ2_phi", &b_genHadJ2_phi, "genHadJ2_phi/F");
  bkgTree->Branch("genHadJ2_m"  , &b_genHadJ2_m  , "genHadJ2_m/F");

  bkgTree->Branch("genHadJ3_pt" , &b_genHadJ3_pt , "genHadJ3_pt/F");
  bkgTree->Branch("genHadJ3_eta", &b_genHadJ3_eta, "genHadJ3_eta/F");
  bkgTree->Branch("genHadJ3_phi", &b_genHadJ3_phi, "genHadJ3_phi/F");
  bkgTree->Branch("genHadJ3_m"  , &b_genHadJ3_m  , "genHadJ3_m/F");

  bkgTree->Branch("genHadW_pt" , &b_genHadW_pt , "genHadW_pt/F" );
  bkgTree->Branch("genHadW_eta", &b_genHadW_eta, "genHadW_eta/F");
  bkgTree->Branch("genHadW_phi", &b_genHadW_phi, "genHadW_phi/F");
  bkgTree->Branch("genHadW_m"  , &b_genHadW_m  , "genHadW_m/F"  );
  bkgTree->Branch("genHadW_dR" , &b_genHadW_dR , "genHadW_dR/F" );

  bkgTree->Branch("genHadT_pt" , &b_genHadT_pt , "genHadT_pt/F" );
  bkgTree->Branch("genHadT_eta", &b_genHadT_eta, "genHadT_eta/F");
  bkgTree->Branch("genHadT_phi", &b_genHadT_phi, "genHadT_phi/F");
  bkgTree->Branch("genHadT_m"  , &b_genHadT_m  , "genHadT_m/F"  );


  ///////////////////////////////////////////////////////////////////
  TTree* testTree = new TTree("test_tree","test tree combined");
  testTree->SetDirectory(fout);

  testTree->Branch("nevt"      , &b_nevt      , "nevt/I");
  testTree->Branch("combiCount", &b_combiCount, "combiCount/I");
  testTree->Branch("genMatch"  , &b_genMatch  , "genMatch/I");

  testTree->Branch("njets"     , &b_njets     , "njets/I");
  testTree->Branch("nbjets_m"  , &b_bjets_n   , "nbjets_m/I");
  testTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
  testTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  testTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  testTree->Branch("MET"       , &b_met       , "MET/F");
  testTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
  testTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

  //jet assignment
  testTree->Branch("jet0pt" , &b_jet0pt , "jet0pt/F");
  testTree->Branch("jet0eta", &b_jet0eta, "jet0eta/F");
  testTree->Branch("jet0phi", &b_jet0phi, "jet0phi/F");
  testTree->Branch("jet0m"  , &b_jet0m  , "jet0m/F");
  testTree->Branch("jet0csv", &b_jet0csv, "jet0csv/I");
  testTree->Branch("jet0Idx", &b_jet0Idx, "jet0Idx/I");

  testTree->Branch("jet1pt" , &b_jet1pt , "jet1pt/F");
  testTree->Branch("jet1eta", &b_jet1eta, "jet1eta/F");
  testTree->Branch("jet1phi", &b_jet1phi, "jet1phi/F");
  testTree->Branch("jet1m"  , &b_jet1m  , "jet1m/F");
  testTree->Branch("jet1csv", &b_jet1csv, "jet1csv/I");
  testTree->Branch("jet1Idx", &b_jet1Idx, "jet1Idx/I");

  testTree->Branch("jet2pt" , &b_jet2pt , "jet2pt/F");
  testTree->Branch("jet2eta", &b_jet2eta, "jet2eta/F");
  testTree->Branch("jet2phi", &b_jet2phi, "jet2phi/F");
  testTree->Branch("jet2m"  , &b_jet2m  , "jet2m/F");
  testTree->Branch("jet2csv", &b_jet2csv, "jet2csv/I");
  testTree->Branch("jet2Idx", &b_jet2Idx, "jet2Idx/I");

  testTree->Branch("jet3pt" , &b_jet3pt , "jet3pt/F");
  testTree->Branch("jet3eta", &b_jet3eta, "jet3eta/F");
  testTree->Branch("jet3phi", &b_jet3phi, "jet3phi/F");
  testTree->Branch("jet3m"  , &b_jet3m  , "jet3m/F");
  testTree->Branch("jet3csv", &b_jet3csv, "jet3csv/I");
  testTree->Branch("jet3Idx", &b_jet3Idx, "jet3Idx/I");

  testTree->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
  testTree->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
  testTree->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
  testTree->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");
  testTree->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
  testTree->Branch("jet12dR"  , &b_jet12dR  , "jet12dR/F");
  testTree->Branch("jet12m"   , &b_jet12m   , "jet12m/F");

  testTree->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
  testTree->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
  testTree->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
  testTree->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
  testTree->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
  testTree->Branch("jet23dR"  , &b_jet23dR  , "jet23dR/F");
  testTree->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

  testTree->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
  testTree->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
  testTree->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
  testTree->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
  testTree->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
  testTree->Branch("jet31dR"  , &b_jet31dR  , "jet31dR/F");
  testTree->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

  testTree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
  testTree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
  testTree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
  testTree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
  testTree->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

  testTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
  testTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
  testTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
  testTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
  testTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
  testTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
  testTree->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

  testTree->Branch("hadTpt"      , &b_hadTpt      , "hadTpt/F");
  testTree->Branch("hadTeta"     , &b_hadTeta     , "hadTeta/F");
  testTree->Branch("hadT12_3deta", &b_hadT12_3deta, "hadT12_3deta/F");
  testTree->Branch("hadT23_1deta", &b_hadT23_1deta, "hadT23_1deta/F");
  testTree->Branch("hadT31_2deta", &b_hadT31_2deta, "hadT31_2deta/F");
  testTree->Branch("hadTphi"     , &b_hadTphi     , "hadTphi/F");
  testTree->Branch("hadT12_3dphi", &b_hadT12_3dphi, "hadT12_3dphi/F");
  testTree->Branch("hadT23_1dphi", &b_hadT23_1dphi, "hadT23_1dphi/F");
  testTree->Branch("hadT31_2dphi", &b_hadT31_2dphi, "hadT31_2dphi/F");
  testTree->Branch("hadT12_3dR"  , &b_hadT12_3dR  , "hadT12_3dR/F");
  testTree->Branch("hadT23_1dR"  , &b_hadT23_1dR  , "hadT23_1dR/F");
  testTree->Branch("hadT31_2dR"  , &b_hadT31_2dR  , "hadT31_2dR/F");
  testTree->Branch("hadTm"       , &b_hadTm       , "hadTm/F");

/*
  //channel
  bool fcnc;
  if ( outFileName.Contains("ttLJ") ) fcnc = false;
  else if ( outFileName.Contains("tch") ) fcnc = true;
  else break
*/

  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    cout << jentry << '/' << 493549 << '\r';
    b_nevt = nevt;

    //Lepton
    TLorentzVector leptonP4;
    if ( nMuon == 1 and Muon_pt[0] > 30 and std::abs(Muon_eta[0]) < 2.4 ) {
      leptonP4.SetPtEtaPhiM(Muon_pt[0], Muon_eta[0], Muon_phi[0], Muon_m[0]);
    }
    else if ( nElectron == 1 and Electron_pt[0] > 35 and std::abs(Electron_eta[0]) < 2.4 ) {
      leptonP4.SetPtEtaPhiM(Electron_pt[0], Electron_eta[0], Electron_phi[0], Electron_m[0]);
    }
    else continue;

    b_lepton_pt = leptonP4.Pt();
    b_lepton_eta = leptonP4.Eta();
    b_lepton_phi = leptonP4.Phi();

    TLorentzVector metP4;
    metP4.SetPtEtaPhiM(MET_pt, 0, MET_phi, 0);
    b_met = MET_pt;
    b_met_phi = MET_phi;
    b_lepdphi = leptonP4.DeltaPhi(metP4);


    //Jet
    std::vector<size_t> jetIdxs;
    b_bjets_n = 0;
    for ( size_t j=0; j<nJet; ++j ) {
      if ( Jet_pt[j] < 30 or std::abs(Jet_eta[j]) > 2.4 ) continue;
      jetIdxs.push_back(j);
      TLorentzVector jetP4;
      jetP4.SetPtEtaPhiM(Jet_pt[j], Jet_eta[j], Jet_phi[j], Jet_m[j]);
      if ( jetP4.DeltaR(leptonP4) < 0.3 ) continue;
      if ( Jet_bTag[j] > 0.5 ) ++b_bjets_n;
    }
    b_njets = jetIdxs.size();
    if ( b_njets < 4 ) continue;
    if ( b_bjets_n < 2 ) continue;
    const int nCombi = (b_njets * (b_njets -1) * (b_njets -2) *(b_njets -3)) / 2;
    if ( nCombi < 12 ) continue;



    // Construct decay tree
    int gen_lepId = 0, gen_nuId = 0, gen_hadJ1Id = 0, gen_hadJ2Id = 0, gen_hadJ3Id = 0;
    TLorentzVector gen_lepW, gen_lepT, gen_hadW, gen_hadT;
    TLorentzVector gen_lep, gen_nu, gen_lepB, gen_hadJ1, gen_hadJ2, gen_hadJ3;
    for ( size_t i=0; i<nGenParticle; ++i ) {
      const int absId = abs(GenParticle_pdgId[i]);
      if ( absId != 24 and absId != 25 ) continue;

      const int mother = GenParticle_mother[i];
      if ( mother < 0 or abs(GenParticle_pdgId[mother]) != 6 ) continue; // Should be from top quark decay

      const int sibling1 = GenParticle_dau1[mother], sibling2 = GenParticle_dau2[mother];
      if ( abs(sibling2-sibling1) != 1 or sibling2 < 0 or sibling1 < 0 ) continue; // two siblingings (including itself)
      const int sibling = (sibling1 == int(i)) ? sibling2 : sibling1;

      int dau1 = GenParticle_dau1[i], dau2 = GenParticle_dau2[i];
      if ( dau2-dau1 != 1 or dau1 < 0 or dau2 < 0 ) continue; // should have two daughters only
      int dau1Aid = abs(GenParticle_pdgId[dau1]), dau2Aid = abs(GenParticle_pdgId[dau2]);
      if ( dau1Aid > dau2Aid ) {
        swap(dau1Aid, dau2Aid);
        swap(dau1, dau2);
      }
      if ( dau1Aid <= 5 and dau2Aid <= 5 ) { // W->jj
        gen_hadT.SetPtEtaPhiM(GenParticle_pt[mother], GenParticle_eta[mother], GenParticle_phi[mother], GenParticle_m[mother]);
        gen_hadJ1.SetPtEtaPhiM(GenParticle_pt[dau1], GenParticle_eta[dau1], GenParticle_phi[dau1], GenParticle_m[dau1]);
        gen_hadJ2.SetPtEtaPhiM(GenParticle_pt[dau2], GenParticle_eta[dau2], GenParticle_phi[dau2], GenParticle_m[dau2]);
        //gen_hadJ3.SetPtEtaPhiM(GenParticle_pt[sibling], GenParticle_eta[sibling], GenParticle_phi[sibling], GenParticle_m[sibling]);
        gen_hadJ3 = gen_hadT - gen_hadJ1 - gen_hadJ2;
        //gen_hadT = gen_hadJ1+gen_hadJ2+gen_hadJ3;
        gen_hadJ1Id = GenParticle_pdgId[dau1];
        gen_hadJ2Id = GenParticle_pdgId[dau2];
        gen_hadJ3Id = GenParticle_pdgId[sibling];
      }
      else if ( dau1Aid >= 11 and dau1Aid <= 15 ) {
        if ( dau1Aid == 15 ) {
          for ( int j=GenParticle_dau1[dau1]; j<=GenParticle_dau2[dau1]; ++j ) {
            if ( j < 0 ) continue;
            const int aid = abs(GenParticle_pdgId[j]);
            if ( aid == 11 or aid == 13 ) { dau1 = j; break; }
          }
        }
        gen_lepT.SetPtEtaPhiM(GenParticle_pt[mother], GenParticle_eta[mother], GenParticle_phi[mother], GenParticle_m[mother]);
        gen_lep.SetPtEtaPhiM(GenParticle_pt[dau1], GenParticle_eta[dau1], GenParticle_phi[dau1], GenParticle_m[dau1]);
        gen_nu.SetPtEtaPhiM(GenParticle_pt[dau2], GenParticle_eta[dau2], GenParticle_phi[dau2], GenParticle_m[dau2]);
        //gen_lepB.SetPtEtaPhiM(GenParticle_pt[sibling], GenParticle_eta[sibling], GenParticle_phi[sibling], GenParticle_m[sibling]);
        gen_lepW = gen_lep + gen_nu;
        gen_lepB = gen_lepT - gen_lep - gen_nu;
        //gen_lepT = gen_lep + gen_lepB + gen_nu;
        gen_lepId = GenParticle_pdgId[dau1];
        gen_nuId = GenParticle_pdgId[dau2];
      }
    }
    if ( gen_lepId != 0 and gen_hadJ3Id != 0 ) { // If we could build full decay tree
      b_genLep_pt = gen_lep.Pt(); b_genLep_eta = gen_lep.Eta(); b_genLep_phi = gen_lep.Phi();
      b_gennu_pt = gen_nu.Pt(); b_gennu_eta = gen_nu.Eta(); b_gennu_phi = gen_nu.Phi();
      b_genLepB_pt = gen_lepB.Pt(); b_genLepB_eta = gen_lepB.Eta(); b_genLepB_phi = gen_lepB.Phi(); b_genLepB_m = gen_lepB.M();
      b_genLepW_pt = gen_lepW.Pt(); b_genLepW_eta = gen_lepW.Eta(); b_genLepW_phi = gen_lepW.Phi(); b_genLepW_m = gen_lepW.M();
      b_genLepT_pt = gen_lepT.Pt(); b_genLepT_eta = gen_lepT.Eta(); b_genLepT_phi = gen_lepT.Phi(); b_genLepT_m = gen_lepT.M();
      b_genHadJ1_pt = gen_hadJ1.Pt(); b_genHadJ1_eta = gen_hadJ1.Eta(); b_genHadJ1_phi = gen_hadJ1.Phi(); b_genHadJ1_m = gen_hadJ1.M();
      b_genHadJ2_pt = gen_hadJ2.Pt(); b_genHadJ2_eta = gen_hadJ2.Eta(); b_genHadJ2_phi = gen_hadJ2.Phi(); b_genHadJ2_m = gen_hadJ2.M();
      b_genHadJ3_pt = gen_hadJ3.Pt(); b_genHadJ3_eta = gen_hadJ3.Eta(); b_genHadJ3_phi = gen_hadJ3.Phi(); b_genHadJ3_m = gen_hadJ3.M();
      b_genHadT_pt = gen_hadT.Pt(); b_genHadT_eta = gen_hadT.Eta(); b_genHadT_phi = gen_hadT.Phi(); b_genHadT_m = gen_hadT.M();

      gen_hadW = gen_hadJ1+gen_hadJ2;
      b_genHadW_pt = gen_hadW.Pt(); b_genHadW_eta = gen_hadW.Eta(); b_genHadW_phi = gen_hadW.Phi(); b_genHadW_m = gen_hadW.M();
      b_genHadW_dR = gen_hadJ1.DeltaR(gen_hadJ2);
    }


    //jet assignments
    TLorentzVector lepW = leptonP4 + metP4;
    b_lepWpt    = lepW.Pt();
    b_lepWeta   = lepW.Eta();
    b_lepWphi   = lepW.Phi();
    b_lepWdphi  = leptonP4.DeltaPhi(metP4);
    b_lepWm     = lepW.M();

    int count = 0;
    TLorentzVector jetP4[4];
    for ( auto ii0 = jetIdxs.begin(); ii0 != jetIdxs.end(); ii0++ ){
      jetP4[0].SetPtEtaPhiM(Jet_pt[*ii0], Jet_eta[*ii0], Jet_phi[*ii0], Jet_m[*ii0]);

      for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ii1++ ) {
        if ( *ii1 == *ii0 ) continue;
        jetP4[3].SetPtEtaPhiM(Jet_pt[*ii1], Jet_eta[*ii1], Jet_phi[*ii1], Jet_m[*ii1]);

        for ( auto ii2 = jetIdxs.begin(); ii2 != jetIdxs.end(); ii2++ ) {
          if ( *ii2 == *ii0 or *ii2 == *ii1 ) continue;
          jetP4[2].SetPtEtaPhiM(Jet_pt[*ii2], Jet_eta[*ii2], Jet_phi[*ii2], Jet_m[*ii2]);

          for ( auto ii3 = ii2+1; ii3 != jetIdxs.end(); ii3++ ) {
            if ( *ii3 == *ii0 or *ii3 == *ii1 or *ii3 == *ii2 ) continue;
            jetP4[1].SetPtEtaPhiM(Jet_pt[*ii3], Jet_eta[*ii3], Jet_phi[*ii3], Jet_m[*ii3]);

            //construct particles: lepB = j0, hadB = j3, hadW = j1+j2
            b_jet0pt = jetP4[0].Pt(); b_jet0eta = jetP4[0].Eta(); b_jet0phi = jetP4[0].Phi(); b_jet0m = jetP4[0].M();
            b_jet1pt = jetP4[1].Pt(); b_jet1eta = jetP4[1].Eta(); b_jet1phi = jetP4[1].Phi(); b_jet1m = jetP4[1].M();
            b_jet2pt = jetP4[2].Pt(); b_jet2eta = jetP4[2].Eta(); b_jet2phi = jetP4[2].Phi(); b_jet2m = jetP4[2].M();
            b_jet3pt = jetP4[3].Pt(); b_jet3eta = jetP4[3].Eta(); b_jet3phi = jetP4[3].Phi(); b_jet3m = jetP4[3].M();
            b_jet0csv = Jet_bTag[*ii0]; b_jet1csv = Jet_bTag[*ii3]; b_jet2csv = Jet_bTag[*ii2]; b_jet3csv = Jet_bTag[*ii1];//index caution!
            b_jet0Idx = *ii0; b_jet1Idx = *ii3; b_jet2Idx = *ii2; b_jet3Idx = *ii1;

            const auto lepT = lepW + jetP4[0];
            const auto had12 = jetP4[1] + jetP4[2];//This is W or H
            const auto had23 = jetP4[2] + jetP4[3];
            const auto had31 = jetP4[3] + jetP4[1];
            const auto hadT = jetP4[1] + jetP4[2] + jetP4[3];

            b_jet12pt = had12.Pt(); b_jet12eta = had12.Eta(); b_jet12phi = had12.Phi(); b_jet12m = had12.M();
            b_jet23pt = had23.Pt(); b_jet23eta = had23.Eta(); b_jet23phi = had23.Phi(); b_jet23m = had23.M();
            b_jet31pt = had31.Pt(); b_jet31eta = had31.Eta(); b_jet31phi = had31.Phi(); b_jet31m = had31.M();
            b_jet12deta = (jetP4[1]-jetP4[2]).Eta(); b_jet12dphi = jetP4[1].DeltaPhi(jetP4[2]);
            b_jet23deta = (jetP4[2]-jetP4[3]).Eta(); b_jet23dphi = jetP4[2].DeltaPhi(jetP4[3]);
            b_jet31deta = (jetP4[3]-jetP4[1]).Eta(); b_jet31dphi = jetP4[3].DeltaPhi(jetP4[1]);
            b_jet12dR = jetP4[1].DeltaR(jetP4[2]); b_jet23dR = jetP4[2].DeltaR(jetP4[3]); b_jet31dR = jetP4[3].DeltaR(jetP4[1]);

            b_lepTpt = lepT.Pt(); b_lepTeta = lepT.Eta(); b_lepTphi = lepT.Phi(); b_lepTm = lepT.M();
            b_lepTdeta = (lepW-jetP4[0]).Eta(); b_lepTdphi = lepW.DeltaPhi(jetP4[0]); b_lepTdR = lepW.DeltaR(jetP4[0]);

            b_hadTpt = hadT.Pt(); b_hadTeta = hadT.Eta(); b_hadTphi = hadT.Phi(); b_hadTm = hadT.M();
            b_hadT12_3deta = (had12-jetP4[3]).Eta(); b_hadT23_1deta = (had23-jetP4[1]).Eta(); b_hadT31_2deta = (had31-jetP4[2]).Eta();
            b_hadT12_3dphi = had12.DeltaPhi(jetP4[3]); b_hadT23_1dphi = had23.DeltaPhi(jetP4[1]); b_hadT31_2dphi = had31.DeltaPhi(jetP4[2]);
            b_hadT12_3dR = had12.DeltaR(jetP4[3]); b_hadT23_1dR = had23.DeltaR(jetP4[1]); b_hadT31_2dR = had31.DeltaR(jetP4[2]);
            b_combiCount = count;


            //some genmatch here
            b_genMatch = 0;
            if ( gen_lep.Pt()   > 0 and gen_lep.DeltaR(leptonP4)     < 0.4 ) b_genMatch += 100000;
            //if ( gen_nu.Pt()    > 0 and gen_nu.DeltaPhi(metP4)       < 0.4 ) b_genMatch += 10000;
            if ( gen_lepB.Pt()  > 0 and gen_lepB.DeltaR(jetP4[0])   < 0.4 ) b_genMatch += 1000;
            if ( gen_hadJ3.Pt() > 0 and gen_hadJ3.DeltaR(jetP4[3])  < 0.4 ) b_genMatch += 100;
            if ( gen_hadJ2.Pt() > 0 and (gen_hadJ2.DeltaR(jetP4[1]) < 0.4 or gen_hadJ2.DeltaR(jetP4[2]) < 0.4 )) b_genMatch += 10;
            if ( gen_hadJ1.Pt() > 0 and (gen_hadJ1.DeltaR(jetP4[1]) < 0.4 or gen_hadJ1.DeltaR(jetP4[2]) < 0.4 )) b_genMatch += 1;

            if ( b_genMatch < 0 ) continue;
            testTree->Fill();
            if ( b_genMatch == 101111 ) sigTree->Fill();
            else bkgTree->Fill();
            //else continue;
            count++;
          }
        }
      }
    }

    nevt++;
  }
  fout->Write();

}




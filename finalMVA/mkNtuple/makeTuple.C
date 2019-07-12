#define makeTuple_cxx

#include "makeTuple.h"
#include <TH2.h>
#include <TStyle.h>

void makeTuple::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void makeTuple::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();
  string syst_str = option.Data();
  syst_str.erase(syst_str.find_first_of("_"),string::npos);
  string ver = syst_str.substr(4,2);
  string era = syst_str.substr(0,4);
  syst_str = syst_str.erase(0,6);
  string sample = option.Data();
  sample.erase(0,sample.find_first_of("_")+1);

  if( syst_str.find("jec") != string::npos or syst_str.find("jer") != string::npos ){
    if     ( syst_str.find("jecup") != string::npos )       syst_ext = "jecup";
    else if( syst_str.find("jecdown") != string::npos )     syst_ext = "jecdown";
    else if( syst_str.find("jerup") != string::npos )       syst_ext = "jerup";
    else if( syst_str.find("jerdown") != string::npos )     syst_ext = "jerdown";
    else if( syst_str.find("TuneCP5up") != string::npos )   syst_ext = "TuneCP5up";
    else if( syst_str.find("TuneCP5down") != string::npos ) syst_ext = "TuneCP5down";
    else if( syst_str.find("hdampup") != string::npos )     syst_ext = "hdampup";
    else if( syst_str.find("hdampdown") != string::npos )   syst_ext = "hdampdown";
  }

  const char* stfcnc_file = "";
  const char* ttfcnc_file = "";
  const char* ttbkg_file = "";
  //Load ST fcnc assign files
  stfcnc_file = Form("../../reco/%s/assignSTFCNC%s%s/assign_deepReco_%s.root",
                    era.c_str(),ver.c_str(),syst_str.c_str(),sample.c_str());
  string stfcnc_file_tmp_path = stfcnc_file;
  ifstream stfcnc_file_tmp(stfcnc_file_tmp_path);

  if( stfcnc_file_tmp.is_open() ){
    stfcnc_File = TFile::Open(stfcnc_file, "READ");
    stfcnc_Tree = (TTree*) stfcnc_File->Get("tree");
    int nevt = stfcnc_Tree->GetEntries();
    if( nevt > 0){
      for(int i = 0; i < nevt; i++){
        stfcnc_Tree->GetEntry(i);
        float pt = stfcnc_Tree->GetLeaf("leptonPt")->GetValue(0);
        float met = stfcnc_Tree->GetLeaf("missingET")->GetValue(0);
        stfcnc_lepPt.push_back(pt);
        stfcnc_missET.push_back(met);
      }
    }
  }
  //else cout << "STFCNC: " << stfcnc_file << endl;

  //Load TT fcnc assign files
  ttfcnc_file = Form("../../reco/%s/assignTTFCNC%s%s/assign_deepReco_%s.root",
                    era.c_str(),ver.c_str(),syst_str.c_str(),sample.c_str());
  string ttfcnc_file_tmp_path = ttfcnc_file;
  ifstream ttfcnc_file_tmp(ttfcnc_file_tmp_path);

  if( ttfcnc_file_tmp.is_open() ){
    ttfcnc_File = TFile::Open(ttfcnc_file, "READ");
    ttfcnc_Tree = (TTree*) ttfcnc_File->Get("tree");
    int nevt = ttfcnc_Tree->GetEntries();
    if( nevt > 0){
      for(int i = 0; i < nevt; i++){
        ttfcnc_Tree->GetEntry(i);
        float pt = ttfcnc_Tree->GetLeaf("leptonPt")->GetValue(0);
        float met = ttfcnc_Tree->GetLeaf("missingET")->GetValue(0);
        ttfcnc_lepPt.push_back(pt);
        ttfcnc_missET.push_back(met);
      }
    }
  }
  //else cout << "TTFCNC: " << ttfcnc_file << endl;

  //Load ttbkg assign files
  ttbkg_file = Form("../../reco/%s/assignTTBKG%s%s/assign_deepReco_%s.root",
                    era.c_str(),ver.c_str(),syst_str.c_str(),sample.c_str());
  string ttbkg_file_tmp_path = ttbkg_file;
  ifstream ttbkg_file_tmp(ttbkg_file_tmp_path);

  if( ttbkg_file_tmp.is_open() ){
    ttbkg_File = TFile::Open(ttbkg_file, "READ");
    ttbkg_Tree = (TTree*) ttbkg_File->Get("tree");
    int nevt = ttbkg_Tree->GetEntries();
    if( nevt > 0){
      for(int i = 0; i < nevt; i++){
        ttbkg_Tree->GetEntry(i);
        float pt = ttbkg_Tree->GetLeaf("leptonPt")->GetValue(0);
        float met = ttbkg_Tree->GetLeaf("missingET")->GetValue(0);
        ttbkg_lepPt.push_back(pt);
        ttbkg_missET.push_back(met);
      }
    }
  }
  //else cout << "TTBKG: " << ttbkg_file << endl;

  //tree
  tree = new TTree("tree","tree for tmva");

  tree->Branch("nevt"         , &b_nevt         , "nevt/I");
  tree->Branch("GoodPV"       , &b_GoodPV       , "GoodPV/I");
  tree->Branch("TruePV"       , &b_TruePV       , "TruePV/I");
  tree->Branch("EventCategory", &b_EventCategory, "EventCategory/I");
  tree->Branch("trigger"      , &b_trigger      , "trigger/I");
  tree->Branch("EventWeight"  , &b_EventWeight  , "EventWeight/F");

  tree->Branch("njets"     , &b_njets     , "njets/I");
  tree->Branch("nbjets_m"  , &b_nbjets_m  , "nbjets_m/I");
  tree->Branch("channel"   , &b_channel   , "channel/I");
  tree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
  tree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  tree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  tree->Branch("lepton_m"  , &b_lepton_m  , "lepton_m/F");
  tree->Branch("MET"       , &b_met       , "MET/F");
  tree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
  tree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

  tree->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
  tree->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
  tree->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
  tree->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
  tree->Branch("lepWm"   , &b_lepWmt  , "lepWm/F");

  //STFCNC part
  tree->Branch("stfcnc_genMatch", &b_stfcnc_genMatch, "stfcnc_genMatch/I");
  tree->Branch("stfcnc_score"   , &b_stfcnc_score   , "stfcnc_score/F");
  tree->Branch("stfcnc_jet0pt"  , &b_stfcnc_jet0pt  , "stfcnc_jet0pt/F");
  tree->Branch("stfcnc_jet0eta" , &b_stfcnc_jet0eta , "stfcnc_jet0eta/F");
  tree->Branch("stfcnc_jet0phi" , &b_stfcnc_jet0phi , "stfcnc_jet0phi/F");
  tree->Branch("stfcnc_jet0m"   , &b_stfcnc_jet0m   , "stfcnc_jet0m/F");
  tree->Branch("stfcnc_jet0csv" , &b_stfcnc_jet0csv , "stfcnc_jet0csv/F");
  tree->Branch("stfcnc_jet0cvsl", &b_stfcnc_jet0cvsl, "stfcnc_jet0cvsl/F");
  tree->Branch("stfcnc_jet0cvsb", &b_stfcnc_jet0cvsb, "stfcnc_jet0cvsb/F");
  tree->Branch("stfcnc_jet0Idx" , &b_stfcnc_jet0Idx , "stfcnc_jet0Idx/I");

  tree->Branch("stfcnc_jet1pt"  , &b_stfcnc_jet1pt  , "stfcnc_jet1pt/F");
  tree->Branch("stfcnc_jet1eta" , &b_stfcnc_jet1eta , "stfcnc_jet1eta/F");
  tree->Branch("stfcnc_jet1phi" , &b_stfcnc_jet1phi , "stfcnc_jet1phi/F");
  tree->Branch("stfcnc_jet1m"   , &b_stfcnc_jet1m   , "stfcnc_jet1m/F");
  tree->Branch("stfcnc_jet1csv" , &b_stfcnc_jet1csv , "stfcnc_jet1csv/F");
  tree->Branch("stfcnc_jet1cvsl", &b_stfcnc_jet1cvsl, "stfcnc_jet1cvsl/F");
  tree->Branch("stfcnc_jet1cvsb", &b_stfcnc_jet1cvsb, "stfcnc_jet1cvsb/F");
  tree->Branch("stfcnc_jet1Idx" , &b_stfcnc_jet1Idx , "stfcnc_jet1Idx/I");

  tree->Branch("stfcnc_jet2pt"  , &b_stfcnc_jet2pt  , "stfcnc_jet2pt/F");
  tree->Branch("stfcnc_jet2eta" , &b_stfcnc_jet2eta , "stfcnc_jet2eta/F");
  tree->Branch("stfcnc_jet2phi" , &b_stfcnc_jet2phi , "stfcnc_jet2phi/F");
  tree->Branch("stfcnc_jet2m"   , &b_stfcnc_jet2m   , "stfcnc_jet2m/F");
  tree->Branch("stfcnc_jet2csv" , &b_stfcnc_jet2csv , "stfcnc_jet2csv/F");
  tree->Branch("stfcnc_jet2cvsl", &b_stfcnc_jet2cvsl, "stfcnc_jet2cvsl/F");
  tree->Branch("stfcnc_jet2cvsb", &b_stfcnc_jet2cvsb, "stfcnc_jet2cvsb/F");
  tree->Branch("stfcnc_jet2Idx" , &b_stfcnc_jet2Idx , "stfcnc_jet2Idx/I");

  tree->Branch("stfcnc_jet3pt"  , &b_stfcnc_jet3pt  , "stfcnc_jet3pt/F");
  tree->Branch("stfcnc_jet3eta" , &b_stfcnc_jet3eta , "stfcnc_jet3eta/F");
  tree->Branch("stfcnc_jet3phi" , &b_stfcnc_jet3phi , "stfcnc_jet3phi/F");
  tree->Branch("stfcnc_jet3m"   , &b_stfcnc_jet3m   , "stfcnc_jet3m/F");
  tree->Branch("stfcnc_jet3csv" , &b_stfcnc_jet3csv , "stfcnc_jet3csv/F");
  tree->Branch("stfcnc_jet3cvsl", &b_stfcnc_jet3cvsl, "stfcnc_jet3cvsl/F");
  tree->Branch("stfcnc_jet3cvsb", &b_stfcnc_jet3cvsb, "stfcnc_jet3cvsb/F");
  tree->Branch("stfcnc_jet3Idx" , &b_stfcnc_jet3Idx , "stfcnc_jet3Idx/I");

  tree->Branch("stfcnc_jet12pt"  , &b_stfcnc_jet12pt  , "stfcnc_jet12pt/F");
  tree->Branch("stfcnc_jet12eta" , &b_stfcnc_jet12eta , "stfcnc_jet12eta/F");
  tree->Branch("stfcnc_jet12deta", &b_stfcnc_jet12deta, "stfcnc_jet12deta/F");
  tree->Branch("stfcnc_jet12phi" , &b_stfcnc_jet12phi , "stfcnc_jet12phi/F");
  tree->Branch("stfcnc_jet12dphi", &b_stfcnc_jet12dphi, "stfcnc_jet12dphi/F");
  tree->Branch("stfcnc_jet12dR"  , &b_stfcnc_jet12dR  , "stfcnc_jet12dR/F");
  tree->Branch("stfcnc_jet12m"   , &b_stfcnc_jet12m   , "stfcnc_jet12m/F");

  tree->Branch("stfcnc_jet23pt"  , &b_stfcnc_jet23pt  , "stfcnc_jet23pt/F");
  tree->Branch("stfcnc_jet23eta" , &b_stfcnc_jet23eta , "stfcnc_jet23eta/F");
  tree->Branch("stfcnc_jet23deta", &b_stfcnc_jet23deta, "stfcnc_jet23deta/F");
  tree->Branch("stfcnc_jet23phi" , &b_stfcnc_jet23phi , "stfcnc_jet23phi/F");
  tree->Branch("stfcnc_jet23dphi", &b_stfcnc_jet23dphi, "stfcnc_jet23dphi/F");
  tree->Branch("stfcnc_jet23dR"  , &b_stfcnc_jet23dR  , "stfcnc_jet23dR/F");
  tree->Branch("stfcnc_jet23m"   , &b_stfcnc_jet23m   , "stfcnc_jet23m/F");

  tree->Branch("stfcnc_jet31pt"  , &b_stfcnc_jet31pt  , "stfcnc_jet31pt/F");
  tree->Branch("stfcnc_jet31eta" , &b_stfcnc_jet31eta , "stfcnc_jet31eta/F");
  tree->Branch("stfcnc_jet31deta", &b_stfcnc_jet31deta, "stfcnc_jet31deta/F");
  tree->Branch("stfcnc_jet31phi" , &b_stfcnc_jet31phi , "stfcnc_jet31phi/F");
  tree->Branch("stfcnc_jet31dphi", &b_stfcnc_jet31dphi, "stfcnc_jet31dphi/F");
  tree->Branch("stfcnc_jet31dR"  , &b_stfcnc_jet31dR  , "stfcnc_jet31dR/F");
  tree->Branch("stfcnc_jet31m"   , &b_stfcnc_jet31m   , "stfcnc_jet31m/F");

  tree->Branch("stfcnc_lepTpt"  , &b_stfcnc_lepTpt  , "stfcnc_lepTpt/F");
  tree->Branch("stfcnc_lepTeta" , &b_stfcnc_lepTeta , "stfcnc_lepTeta/F");
  tree->Branch("stfcnc_lepTdeta", &b_stfcnc_lepTdeta, "stfcnc_lepTdeta/F");
  tree->Branch("stfcnc_lepTphi" , &b_stfcnc_lepTphi , "stfcnc_lepTphi/F");
  tree->Branch("stfcnc_lepTdphi", &b_stfcnc_lepTdphi, "stfcnc_lepTdphi/F");//W and b
  tree->Branch("stfcnc_lepTdR"  , &b_stfcnc_lepTdR  , "stfcnc_lepTdR/F");
  tree->Branch("stfcnc_lepTm"   , &b_stfcnc_lepTmt  , "stfcnc_lepTm/F");

  tree->Branch("stfcnc_hadTpt"      , &b_stfcnc_hadTpt      , "stfcnc_hadTpt/F");
  tree->Branch("stfcnc_hadTeta"     , &b_stfcnc_hadTeta     , "stfcnc_hadTeta/F");
  tree->Branch("stfcnc_hadT12_3deta", &b_stfcnc_hadT12_3deta, "stfcnc_hadT12_3deta/F");
  tree->Branch("stfcnc_hadT23_1deta", &b_stfcnc_hadT23_1deta, "stfcnc_hadT23_1deta/F");
  tree->Branch("stfcnc_hadT31_2deta", &b_stfcnc_hadT31_2deta, "stfcnc_hadT31_2deta/F");
  tree->Branch("stfcnc_hadTphi"     , &b_stfcnc_hadTphi     , "stfcnc_hadTphi/F");
  tree->Branch("stfcnc_hadT12_3dphi", &b_stfcnc_hadT12_3dphi, "stfcnc_hadT12_3dphi/F");
  tree->Branch("stfcnc_hadT23_1dphi", &b_stfcnc_hadT23_1dphi, "stfcnc_hadT23_1dphi/F");
  tree->Branch("stfcnc_hadT31_2dphi", &b_stfcnc_hadT31_2dphi, "stfcnc_hadT31_2dphi/F");
  tree->Branch("stfcnc_hadT12_3dR"  , &b_stfcnc_hadT12_3dR  , "stfcnc_hadT12_3dR/F");
  tree->Branch("stfcnc_hadT23_1dR"  , &b_stfcnc_hadT23_1dR  , "stfcnc_hadT23_1dR/F");
  tree->Branch("stfcnc_hadT31_2dR"  , &b_stfcnc_hadT31_2dR  , "stfcnc_hadT31_2dR/F");
  tree->Branch("stfcnc_hadTm"       , &b_stfcnc_hadTm       , "stfcnc_hadTm/F");

  tree->Branch("stfcnc_jet0lepdR"    , &b_stfcnc_jet0lepdR    , "stfcnc_jet0lepdR/F");
  tree->Branch("stfcnc_jet1lepdR"    , &b_stfcnc_jet1lepdR    , "stfcnc_jet1lepdR/F");
  tree->Branch("stfcnc_jet2lepdR"    , &b_stfcnc_jet2lepdR    , "stfcnc_jet2lepdR/F");
  tree->Branch("stfcnc_jet3lepdR"    , &b_stfcnc_jet3lepdR    , "stfcnc_jet3lepdR/F");
  tree->Branch("stfcnc_jet01dR"      , &b_stfcnc_jet01dR      , "stfcnc_jet01dR/F");
  tree->Branch("stfcnc_jet02dR"      , &b_stfcnc_jet02dR      , "stfcnc_jet02dR/F");
  tree->Branch("stfcnc_jet03dR"      , &b_stfcnc_jet03dR      , "stfcnc_jet03dR/F");
  tree->Branch("stfcnc_jet12_lepdR"  , &b_stfcnc_jet12_lepdR  , "stfcnc_jet12_lepdR/F");
  tree->Branch("stfcnc_jet23_lepdR"  , &b_stfcnc_jet23_lepdR  , "stfcnc_jet23_lepdR/F");
  tree->Branch("stfcnc_jet31_lepdR"  , &b_stfcnc_jet31_lepdR  , "stfcnc_jet31_lepdR/F");
  tree->Branch("stfcnc_jet12_0dR"    , &b_stfcnc_jet12_0dR    , "stfcnc_jet12_0dR/F");
  tree->Branch("stfcnc_jet23_0dR"    , &b_stfcnc_jet23_0dR    , "stfcnc_jet23_0dR/F");
  tree->Branch("stfcnc_jet31_0dR"    , &b_stfcnc_jet31_0dR    , "stfcnc_jet31_0dR/F");
  tree->Branch("stfcnc_lepTjet12dphi", &b_stfcnc_lepTjet12dphi, "stfcnc_lepTjet12dphi/F");
  tree->Branch("stfcnc_lepTjet23dphi", &b_stfcnc_lepTjet23dphi, "stfcnc_lepTjet23dphi/F");
  tree->Branch("stfcnc_lepTjet31dphi", &b_stfcnc_lepTjet31dphi, "stfcnc_lepTjet31dphi/F");
  tree->Branch("stfcnc_hadT_jet0dR"  , &b_stfcnc_hadT_jet0dR  , "stfcnc_hadT_jet0dR/F");

  //TTFCNC part
  tree->Branch("ttfcnc_genMatch", &b_ttfcnc_genMatch, "ttfcnc_genMatch/I");
  tree->Branch("ttfcnc_score"   , &b_ttfcnc_score   , "ttfcnc_score/F");
  tree->Branch("ttfcnc_jet0pt"  , &b_ttfcnc_jet0pt  , "ttfcnc_jet0pt/F");
  tree->Branch("ttfcnc_jet0eta" , &b_ttfcnc_jet0eta , "ttfcnc_jet0eta/F");
  tree->Branch("ttfcnc_jet0phi" , &b_ttfcnc_jet0phi , "ttfcnc_jet0phi/F");
  tree->Branch("ttfcnc_jet0m"   , &b_ttfcnc_jet0m   , "ttfcnc_jet0m/F");
  tree->Branch("ttfcnc_jet0csv" , &b_ttfcnc_jet0csv , "ttfcnc_jet0csv/F");
  tree->Branch("ttfcnc_jet0cvsl", &b_ttfcnc_jet0cvsl, "ttfcnc_jet0cvsl/F");
  tree->Branch("ttfcnc_jet0cvsb", &b_ttfcnc_jet0cvsb, "ttfcnc_jet0cvsb/F");
  tree->Branch("ttfcnc_jet0Idx" , &b_ttfcnc_jet0Idx , "ttfcnc_jet0Idx/I");

  tree->Branch("ttfcnc_jet1pt"  , &b_ttfcnc_jet1pt  , "ttfcnc_jet1pt/F");
  tree->Branch("ttfcnc_jet1eta" , &b_ttfcnc_jet1eta , "ttfcnc_jet1eta/F");
  tree->Branch("ttfcnc_jet1phi" , &b_ttfcnc_jet1phi , "ttfcnc_jet1phi/F");
  tree->Branch("ttfcnc_jet1m"   , &b_ttfcnc_jet1m   , "ttfcnc_jet1m/F");
  tree->Branch("ttfcnc_jet1csv" , &b_ttfcnc_jet1csv , "ttfcnc_jet1csv/F");
  tree->Branch("ttfcnc_jet1cvsl", &b_ttfcnc_jet1cvsl, "ttfcnc_jet1cvsl/F");
  tree->Branch("ttfcnc_jet1cvsb", &b_ttfcnc_jet1cvsb, "ttfcnc_jet1cvsb/F");
  tree->Branch("ttfcnc_jet1Idx" , &b_ttfcnc_jet1Idx , "ttfcnc_jet1Idx/I");

  tree->Branch("ttfcnc_jet2pt"  , &b_ttfcnc_jet2pt  , "ttfcnc_jet2pt/F");
  tree->Branch("ttfcnc_jet2eta" , &b_ttfcnc_jet2eta , "ttfcnc_jet2eta/F");
  tree->Branch("ttfcnc_jet2phi" , &b_ttfcnc_jet2phi , "ttfcnc_jet2phi/F");
  tree->Branch("ttfcnc_jet2m"   , &b_ttfcnc_jet2m   , "ttfcnc_jet2m/F");
  tree->Branch("ttfcnc_jet2csv" , &b_ttfcnc_jet2csv , "ttfcnc_jet2csv/F");
  tree->Branch("ttfcnc_jet2cvsl", &b_ttfcnc_jet2cvsl, "ttfcnc_jet2cvsl/F");
  tree->Branch("ttfcnc_jet2cvsb", &b_ttfcnc_jet2cvsb, "ttfcnc_jet2cvsb/F");
  tree->Branch("ttfcnc_jet2Idx" , &b_ttfcnc_jet2Idx , "ttfcnc_jet2Idx/I");

  tree->Branch("ttfcnc_jet3pt"  , &b_ttfcnc_jet3pt  , "ttfcnc_jet3pt/F");
  tree->Branch("ttfcnc_jet3eta" , &b_ttfcnc_jet3eta , "ttfcnc_jet3eta/F");
  tree->Branch("ttfcnc_jet3phi" , &b_ttfcnc_jet3phi , "ttfcnc_jet3phi/F");
  tree->Branch("ttfcnc_jet3m"   , &b_ttfcnc_jet3m   , "ttfcnc_jet3m/F");
  tree->Branch("ttfcnc_jet3csv" , &b_ttfcnc_jet3csv , "ttfcnc_jet3csv/F");
  tree->Branch("ttfcnc_jet3cvsl", &b_ttfcnc_jet3cvsl, "ttfcnc_jet3cvsl/F");
  tree->Branch("ttfcnc_jet3cvsb", &b_ttfcnc_jet3cvsb, "ttfcnc_jet3cvsb/F");
  tree->Branch("ttfcnc_jet3Idx" , &b_ttfcnc_jet3Idx , "ttfcnc_jet3Idx/I");

  tree->Branch("ttfcnc_jet12pt"  , &b_ttfcnc_jet12pt  , "ttfcnc_jet12pt/F");
  tree->Branch("ttfcnc_jet12eta" , &b_ttfcnc_jet12eta , "ttfcnc_jet12eta/F");
  tree->Branch("ttfcnc_jet12deta", &b_ttfcnc_jet12deta, "ttfcnc_jet12deta/F");
  tree->Branch("ttfcnc_jet12phi" , &b_ttfcnc_jet12phi , "ttfcnc_jet12phi/F");
  tree->Branch("ttfcnc_jet12dphi", &b_ttfcnc_jet12dphi, "ttfcnc_jet12dphi/F");
  tree->Branch("ttfcnc_jet12dR"  , &b_ttfcnc_jet12dR  , "ttfcnc_jet12dR/F");
  tree->Branch("ttfcnc_jet12m"   , &b_ttfcnc_jet12m   , "ttfcnc_jet12m/F");

  tree->Branch("ttfcnc_jet23pt"  , &b_ttfcnc_jet23pt  , "ttfcnc_jet23pt/F");
  tree->Branch("ttfcnc_jet23eta" , &b_ttfcnc_jet23eta , "ttfcnc_jet23eta/F");
  tree->Branch("ttfcnc_jet23deta", &b_ttfcnc_jet23deta, "ttfcnc_jet23deta/F");
  tree->Branch("ttfcnc_jet23phi" , &b_ttfcnc_jet23phi , "ttfcnc_jet23phi/F");
  tree->Branch("ttfcnc_jet23dphi", &b_ttfcnc_jet23dphi, "ttfcnc_jet23dphi/F");
  tree->Branch("ttfcnc_jet23dR"  , &b_ttfcnc_jet23dR  , "ttfcnc_jet23dR/F");
  tree->Branch("ttfcnc_jet23m"   , &b_ttfcnc_jet23m   , "ttfcnc_jet23m/F");

  tree->Branch("ttfcnc_jet31pt"  , &b_ttfcnc_jet31pt  , "ttfcnc_jet31pt/F");
  tree->Branch("ttfcnc_jet31eta" , &b_ttfcnc_jet31eta , "ttfcnc_jet31eta/F");
  tree->Branch("ttfcnc_jet31deta", &b_ttfcnc_jet31deta, "ttfcnc_jet31deta/F");
  tree->Branch("ttfcnc_jet31phi" , &b_ttfcnc_jet31phi , "ttfcnc_jet31phi/F");
  tree->Branch("ttfcnc_jet31dphi", &b_ttfcnc_jet31dphi, "ttfcnc_jet31dphi/F");
  tree->Branch("ttfcnc_jet31dR"  , &b_ttfcnc_jet31dR  , "ttfcnc_jet31dR/F");
  tree->Branch("ttfcnc_jet31m"   , &b_ttfcnc_jet31m   , "ttfcnc_jet31m/F");

  tree->Branch("ttfcnc_lepTpt"  , &b_ttfcnc_lepTpt  , "ttfcnc_lepTpt/F");
  tree->Branch("ttfcnc_lepTeta" , &b_ttfcnc_lepTeta , "ttfcnc_lepTeta/F");
  tree->Branch("ttfcnc_lepTdeta", &b_ttfcnc_lepTdeta, "ttfcnc_lepTdeta/F");
  tree->Branch("ttfcnc_lepTphi" , &b_ttfcnc_lepTphi , "ttfcnc_lepTphi/F");
  tree->Branch("ttfcnc_lepTdphi", &b_ttfcnc_lepTdphi, "ttfcnc_lepTdphi/F");//W and b
  tree->Branch("ttfcnc_lepTdR"  , &b_ttfcnc_lepTdR  , "ttfcnc_lepTdR/F");
  tree->Branch("ttfcnc_lepTm"   , &b_ttfcnc_lepTmt  , "ttfcnc_lepTm/F");

  tree->Branch("ttfcnc_hadTpt"      , &b_ttfcnc_hadTpt      , "ttfcnc_hadTpt/F");
  tree->Branch("ttfcnc_hadTeta"     , &b_ttfcnc_hadTeta     , "ttfcnc_hadTeta/F");
  tree->Branch("ttfcnc_hadT12_3deta", &b_ttfcnc_hadT12_3deta, "ttfcnc_hadT12_3deta/F");
  tree->Branch("ttfcnc_hadT23_1deta", &b_ttfcnc_hadT23_1deta, "ttfcnc_hadT23_1deta/F");
  tree->Branch("ttfcnc_hadT31_2deta", &b_ttfcnc_hadT31_2deta, "ttfcnc_hadT31_2deta/F");
  tree->Branch("ttfcnc_hadTphi"     , &b_ttfcnc_hadTphi     , "ttfcnc_hadTphi/F");
  tree->Branch("ttfcnc_hadT12_3dphi", &b_ttfcnc_hadT12_3dphi, "ttfcnc_hadT12_3dphi/F");
  tree->Branch("ttfcnc_hadT23_1dphi", &b_ttfcnc_hadT23_1dphi, "ttfcnc_hadT23_1dphi/F");
  tree->Branch("ttfcnc_hadT31_2dphi", &b_ttfcnc_hadT31_2dphi, "ttfcnc_hadT31_2dphi/F");
  tree->Branch("ttfcnc_hadT12_3dR"  , &b_ttfcnc_hadT12_3dR  , "ttfcnc_hadT12_3dR/F");
  tree->Branch("ttfcnc_hadT23_1dR"  , &b_ttfcnc_hadT23_1dR  , "ttfcnc_hadT23_1dR/F");
  tree->Branch("ttfcnc_hadT31_2dR"  , &b_ttfcnc_hadT31_2dR  , "ttfcnc_hadT31_2dR/F");
  tree->Branch("ttfcnc_hadTm"       , &b_ttfcnc_hadTm       , "ttfcnc_hadTm/F");

  tree->Branch("ttfcnc_jet0lepdR"    , &b_ttfcnc_jet0lepdR    , "ttfcnc_jet0lepdR/F");
  tree->Branch("ttfcnc_jet1lepdR"    , &b_ttfcnc_jet1lepdR    , "ttfcnc_jet1lepdR/F");
  tree->Branch("ttfcnc_jet2lepdR"    , &b_ttfcnc_jet2lepdR    , "ttfcnc_jet2lepdR/F");
  tree->Branch("ttfcnc_jet3lepdR"    , &b_ttfcnc_jet3lepdR    , "ttfcnc_jet3lepdR/F");
  tree->Branch("ttfcnc_jet01dR"      , &b_ttfcnc_jet01dR      , "ttfcnc_jet01dR/F");
  tree->Branch("ttfcnc_jet02dR"      , &b_ttfcnc_jet02dR      , "ttfcnc_jet02dR/F");
  tree->Branch("ttfcnc_jet03dR"      , &b_ttfcnc_jet03dR      , "ttfcnc_jet03dR/F");
  tree->Branch("ttfcnc_jet12_lepdR"  , &b_ttfcnc_jet12_lepdR  , "ttfcnc_jet12_lepdR/F");
  tree->Branch("ttfcnc_jet23_lepdR"  , &b_ttfcnc_jet23_lepdR  , "ttfcnc_jet23_lepdR/F");
  tree->Branch("ttfcnc_jet31_lepdR"  , &b_ttfcnc_jet31_lepdR  , "ttfcnc_jet31_lepdR/F");
  tree->Branch("ttfcnc_jet12_0dR"    , &b_ttfcnc_jet12_0dR    , "ttfcnc_jet12_0dR/F");
  tree->Branch("ttfcnc_jet23_0dR"    , &b_ttfcnc_jet23_0dR    , "ttfcnc_jet23_0dR/F");
  tree->Branch("ttfcnc_jet31_0dR"    , &b_ttfcnc_jet31_0dR    , "ttfcnc_jet31_0dR/F");
  tree->Branch("ttfcnc_lepTjet12dphi", &b_ttfcnc_lepTjet12dphi, "ttfcnc_lepTjet12dphi/F");
  tree->Branch("ttfcnc_lepTjet23dphi", &b_ttfcnc_lepTjet23dphi, "ttfcnc_lepTjet23dphi/F");
  tree->Branch("ttfcnc_lepTjet31dphi", &b_ttfcnc_lepTjet31dphi, "ttfcnc_lepTjet31dphi/F");
  tree->Branch("ttfcnc_hadT_jet0dR"  , &b_ttfcnc_hadT_jet0dR  , "ttfcnc_hadT_jet0dR/F");


  //TTBKG part
  tree->Branch("ttbkg_genMatch", &b_ttbkg_genMatch, "ttbkg_genMatch/I");
  tree->Branch("ttbkg_score"   , &b_ttbkg_score   , "ttbkg_score/F");
  tree->Branch("ttbkg_jet0pt"  , &b_ttbkg_jet0pt  , "ttbkg_jet0pt/F");
  tree->Branch("ttbkg_jet0eta" , &b_ttbkg_jet0eta , "ttbkg_jet0eta/F");
  tree->Branch("ttbkg_jet0phi" , &b_ttbkg_jet0phi , "ttbkg_jet0phi/F");
  tree->Branch("ttbkg_jet0m"   , &b_ttbkg_jet0m   , "ttbkg_jet0m/F");
  tree->Branch("ttbkg_jet0csv" , &b_ttbkg_jet0csv , "ttbkg_jet0csv/F");
  tree->Branch("ttbkg_jet0cvsl", &b_ttbkg_jet0cvsl, "ttbkg_jet0cvsl/F");
  tree->Branch("ttbkg_jet0cvsb", &b_ttbkg_jet0cvsb, "ttbkg_jet0cvsb/F");
  tree->Branch("ttbkg_jet0Idx" , &b_ttbkg_jet0Idx , "ttbkg_jet0Idx/I");

  tree->Branch("ttbkg_jet1pt"  , &b_ttbkg_jet1pt  , "ttbkg_jet1pt/F");
  tree->Branch("ttbkg_jet1eta" , &b_ttbkg_jet1eta , "ttbkg_jet1eta/F");
  tree->Branch("ttbkg_jet1phi" , &b_ttbkg_jet1phi , "ttbkg_jet1phi/F");
  tree->Branch("ttbkg_jet1m"   , &b_ttbkg_jet1m   , "ttbkg_jet1m/F");
  tree->Branch("ttbkg_jet1csv" , &b_ttbkg_jet1csv , "ttbkg_jet1csv/F");
  tree->Branch("ttbkg_jet1cvsl", &b_ttbkg_jet1cvsl, "ttbkg_jet1cvsl/F");
  tree->Branch("ttbkg_jet1cvsb", &b_ttbkg_jet1cvsb, "ttbkg_jet1cvsb/F");
  tree->Branch("ttbkg_jet1Idx" , &b_ttbkg_jet1Idx , "ttbkg_jet1Idx/I");

  tree->Branch("ttbkg_jet2pt"  , &b_ttbkg_jet2pt  , "ttbkg_jet2pt/F");
  tree->Branch("ttbkg_jet2eta" , &b_ttbkg_jet2eta , "ttbkg_jet2eta/F");
  tree->Branch("ttbkg_jet2phi" , &b_ttbkg_jet2phi , "ttbkg_jet2phi/F");
  tree->Branch("ttbkg_jet2m"   , &b_ttbkg_jet2m   , "ttbkg_jet2m/F");
  tree->Branch("ttbkg_jet2csv" , &b_ttbkg_jet2csv , "ttbkg_jet2csv/F");
  tree->Branch("ttbkg_jet2cvsl", &b_ttbkg_jet2cvsl, "ttbkg_jet2cvsl/F");
  tree->Branch("ttbkg_jet2cvsb", &b_ttbkg_jet2cvsb, "ttbkg_jet2cvsb/F");
  tree->Branch("ttbkg_jet2Idx" , &b_ttbkg_jet2Idx , "ttbkg_jet2Idx/I");

  tree->Branch("ttbkg_jet3pt"  , &b_ttbkg_jet3pt  , "ttbkg_jet3pt/F");
  tree->Branch("ttbkg_jet3eta" , &b_ttbkg_jet3eta , "ttbkg_jet3eta/F");
  tree->Branch("ttbkg_jet3phi" , &b_ttbkg_jet3phi , "ttbkg_jet3phi/F");
  tree->Branch("ttbkg_jet3m"   , &b_ttbkg_jet3m   , "ttbkg_jet3m/F");
  tree->Branch("ttbkg_jet3csv" , &b_ttbkg_jet3csv , "ttbkg_jet3csv/F");
  tree->Branch("ttbkg_jet3cvsl", &b_ttbkg_jet3cvsl, "ttbkg_jet3cvsl/F");
  tree->Branch("ttbkg_jet3cvsb", &b_ttbkg_jet3cvsb, "ttbkg_jet3cvsb/F");
  tree->Branch("ttbkg_jet3Idx" , &b_ttbkg_jet3Idx , "ttbkg_jet3Idx/I");

  tree->Branch("ttbkg_jet12pt"  , &b_ttbkg_jet12pt  , "ttbkg_jet12pt/F");
  tree->Branch("ttbkg_jet12eta" , &b_ttbkg_jet12eta , "ttbkg_jet12eta/F");
  tree->Branch("ttbkg_jet12deta", &b_ttbkg_jet12deta, "ttbkg_jet12deta/F");
  tree->Branch("ttbkg_jet12phi" , &b_ttbkg_jet12phi , "ttbkg_jet12phi/F");
  tree->Branch("ttbkg_jet12dphi", &b_ttbkg_jet12dphi, "ttbkg_jet12dphi/F");
  tree->Branch("ttbkg_jet12dR"  , &b_ttbkg_jet12dR  , "ttbkg_jet12dR/F");
  tree->Branch("ttbkg_jet12m"   , &b_ttbkg_jet12m   , "ttbkg_jet12m/F");

  tree->Branch("ttbkg_jet23pt"  , &b_ttbkg_jet23pt  , "ttbkg_jet23pt/F");
  tree->Branch("ttbkg_jet23eta" , &b_ttbkg_jet23eta , "ttbkg_jet23eta/F");
  tree->Branch("ttbkg_jet23deta", &b_ttbkg_jet23deta, "ttbkg_jet23deta/F");
  tree->Branch("ttbkg_jet23phi" , &b_ttbkg_jet23phi , "ttbkg_jet23phi/F");
  tree->Branch("ttbkg_jet23dphi", &b_ttbkg_jet23dphi, "ttbkg_jet23dphi/F");
  tree->Branch("ttbkg_jet23dR"  , &b_ttbkg_jet23dR  , "ttbkg_jet23dR/F");
  tree->Branch("ttbkg_jet23m"   , &b_ttbkg_jet23m   , "ttbkg_jet23m/F");

  tree->Branch("ttbkg_jet31pt"  , &b_ttbkg_jet31pt  , "ttbkg_jet31pt/F");
  tree->Branch("ttbkg_jet31eta" , &b_ttbkg_jet31eta , "ttbkg_jet31eta/F");
  tree->Branch("ttbkg_jet31deta", &b_ttbkg_jet31deta, "ttbkg_jet31deta/F");
  tree->Branch("ttbkg_jet31phi" , &b_ttbkg_jet31phi , "ttbkg_jet31phi/F");
  tree->Branch("ttbkg_jet31dphi", &b_ttbkg_jet31dphi, "ttbkg_jet31dphi/F");
  tree->Branch("ttbkg_jet31dR"  , &b_ttbkg_jet31dR  , "ttbkg_jet31dR/F");
  tree->Branch("ttbkg_jet31m"   , &b_ttbkg_jet31m   , "ttbkg_jet31m/F");

  tree->Branch("ttbkg_lepTpt"  , &b_ttbkg_lepTpt  , "ttbkg_lepTpt/F");
  tree->Branch("ttbkg_lepTeta" , &b_ttbkg_lepTeta , "ttbkg_lepTeta/F");
  tree->Branch("ttbkg_lepTdeta", &b_ttbkg_lepTdeta, "ttbkg_lepTdeta/F");
  tree->Branch("ttbkg_lepTphi" , &b_ttbkg_lepTphi , "ttbkg_lepTphi/F");
  tree->Branch("ttbkg_lepTdphi", &b_ttbkg_lepTdphi, "ttbkg_lepTdphi/F");//W and b
  tree->Branch("ttbkg_lepTdR"  , &b_ttbkg_lepTdR  , "ttbkg_lepTdR/F");
  tree->Branch("ttbkg_lepTm"   , &b_ttbkg_lepTmt  , "ttbkg_lepTm/F");

  tree->Branch("ttbkg_hadTpt"      , &b_ttbkg_hadTpt      , "ttbkg_hadTpt/F");
  tree->Branch("ttbkg_hadTeta"     , &b_ttbkg_hadTeta     , "ttbkg_hadTeta/F");
  tree->Branch("ttbkg_hadT12_3deta", &b_ttbkg_hadT12_3deta, "ttbkg_hadT12_3deta/F");
  tree->Branch("ttbkg_hadT23_1deta", &b_ttbkg_hadT23_1deta, "ttbkg_hadT23_1deta/F");
  tree->Branch("ttbkg_hadT31_2deta", &b_ttbkg_hadT31_2deta, "ttbkg_hadT31_2deta/F");
  tree->Branch("ttbkg_hadTphi"     , &b_ttbkg_hadTphi     , "ttbkg_hadTphi/F");
  tree->Branch("ttbkg_hadT12_3dphi", &b_ttbkg_hadT12_3dphi, "ttbkg_hadT12_3dphi/F");
  tree->Branch("ttbkg_hadT23_1dphi", &b_ttbkg_hadT23_1dphi, "ttbkg_hadT23_1dphi/F");
  tree->Branch("ttbkg_hadT31_2dphi", &b_ttbkg_hadT31_2dphi, "ttbkg_hadT31_2dphi/F");
  tree->Branch("ttbkg_hadT12_3dR"  , &b_ttbkg_hadT12_3dR  , "ttbkg_hadT12_3dR/F");
  tree->Branch("ttbkg_hadT23_1dR"  , &b_ttbkg_hadT23_1dR  , "ttbkg_hadT23_1dR/F");
  tree->Branch("ttbkg_hadT31_2dR"  , &b_ttbkg_hadT31_2dR  , "ttbkg_hadT31_2dR/F");
  tree->Branch("ttbkg_hadTm"       , &b_ttbkg_hadTm       , "ttbkg_hadTm/F");

  tree->Branch("ttbkg_jet0lepdR"    , &b_ttbkg_jet0lepdR    , "ttbkg_jet0lepdR/F");
  tree->Branch("ttbkg_jet1lepdR"    , &b_ttbkg_jet1lepdR    , "ttbkg_jet1lepdR/F");
  tree->Branch("ttbkg_jet2lepdR"    , &b_ttbkg_jet2lepdR    , "ttbkg_jet2lepdR/F");
  tree->Branch("ttbkg_jet3lepdR"    , &b_ttbkg_jet3lepdR    , "ttbkg_jet3lepdR/F");
  tree->Branch("ttbkg_jet01dR"      , &b_ttbkg_jet01dR      , "ttbkg_jet01dR/F");
  tree->Branch("ttbkg_jet02dR"      , &b_ttbkg_jet02dR      , "ttbkg_jet02dR/F");
  tree->Branch("ttbkg_jet03dR"      , &b_ttbkg_jet03dR      , "ttbkg_jet03dR/F");
  tree->Branch("ttbkg_jet12_lepdR"  , &b_ttbkg_jet12_lepdR  , "ttbkg_jet12_lepdR/F");
  tree->Branch("ttbkg_jet23_lepdR"  , &b_ttbkg_jet23_lepdR  , "ttbkg_jet23_lepdR/F");
  tree->Branch("ttbkg_jet31_lepdR"  , &b_ttbkg_jet31_lepdR  , "ttbkg_jet31_lepdR/F");
  tree->Branch("ttbkg_jet12_0dR"    , &b_ttbkg_jet12_0dR    , "ttbkg_jet12_0dR/F");
  tree->Branch("ttbkg_jet23_0dR"    , &b_ttbkg_jet23_0dR    , "ttbkg_jet23_0dR/F");
  tree->Branch("ttbkg_jet31_0dR"    , &b_ttbkg_jet31_0dR    , "ttbkg_jet31_0dR/F");
  tree->Branch("ttbkg_lepTjet12dphi", &b_ttbkg_lepTjet12dphi, "ttbkg_lepTjet12dphi/F");
  tree->Branch("ttbkg_lepTjet23dphi", &b_ttbkg_lepTjet23dphi, "ttbkg_lepTjet23dphi/F");
  tree->Branch("ttbkg_lepTjet31dphi", &b_ttbkg_lepTjet31dphi, "ttbkg_lepTjet31dphi/F");
  tree->Branch("ttbkg_hadT_jet0dR"  , &b_ttbkg_hadT_jet0dR  , "ttbkg_hadT_jet0dR/F");


  fOutput->Add(tree);

} 

Bool_t makeTuple::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  int era = 0;
  TString current_file_name = makeTuple::fReader.GetTree()->GetCurrentFile()->GetName();
  if     ( current_file_name.Contains("2017") ) era = 2017;
  else if( current_file_name.Contains("2018") ) era = 2018;
  //cout << era;

  int mode = 999;
  mode = *channel;
  if( mode > 2 ) return kTRUE;

  float wrongPVrate = 1.0;
  if( era == 2017 ){
    if( !option.Contains("Run2017") ){
      if     ( option.Contains("DYJets10to50") ) wrongPVrate = 1.04849;
      else if( option.Contains("QCDEM15to20") ) wrongPVrate = 1.02703;
      else if( option.Contains("QCDEM20to30") ) wrongPVrate = 1.02484;
      else if( option.Contains("QCDEM300toInf") ) wrongPVrate = 1.03165;
      else if( option.Contains("QCDEM30to50") ) wrongPVrate = 1.02575;
      else if( option.Contains("QCDEM50to80") ) wrongPVrate = 1.04114;
      else if( option.Contains("QCDMu120to170") ) wrongPVrate = 1.02968;
      else if( option.Contains("QCDMu170to300") ) wrongPVrate = 1.02596;
      else if( option.Contains("QCDMu20to30") ) wrongPVrate = 1.04353;
      else if( option.Contains("QCDMu30to50") ) wrongPVrate = 1.03696;
      else if( option.Contains("QCDMu470to600") ) wrongPVrate = 1.02922;
      else if( option.Contains("QCDMu50to80") ) wrongPVrate = 1.02786;
      else if( option.Contains("QCDMu80to120") ) wrongPVrate = 1.03184;
      else if( option.Contains("TTLLpowhegttbbhdampup") ) wrongPVrate = 1.03463;
      else if( option.Contains("TTLLpowhegttcchdampup") ) wrongPVrate = 1.03494;
      else if( option.Contains("TTLLpowhegttlfhdampup") ) wrongPVrate = 1.03468;
      else if( option.Contains("TTZToLLNuNu") ) wrongPVrate = 1.04219;
      else if( option.Contains("TTpowhegttbbTuneCP5down") ) wrongPVrate = 1.04743;
      else if( option.Contains("TTpowhegttbbhdampdown") ) wrongPVrate = 1.04677;
      else if( option.Contains("TTpowhegttccTuneCP5down") ) wrongPVrate = 1.04768;
      else if( option.Contains("TTpowhegttcchdampdown") ) wrongPVrate = 1.04708;
      else if( option.Contains("TTpowhegttlfTuneCP5down") ) wrongPVrate = 1.048;
      else if( option.Contains("TTpowhegttlfhdampdown") ) wrongPVrate = 1.04758;
      else if( option.Contains("W3JetsToLNu") ) wrongPVrate = 1.04195;
      else if( option.Contains("WW") ) wrongPVrate = 1.04685;
      else if( option.Contains("WZ") ) wrongPVrate = 1.04381;
      else if( option.Contains("ZZ") ) wrongPVrate = 1.02981;
      else   wrongPVrate = 1.0;
    }
    if( wrongPVrate > 1.01 ){
      if( *TruePV < 10 || *TruePV > 75 ) return kTRUE;
    }
  }

  float relIso = *lepton_relIso;

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 

  TLorentzVector p4met;
  float met = *MET;
  float met_phi = *MET_phi;
  float apt = TMath::Abs(met);
  float met_x =  apt*TMath::Cos(met_phi);
  float met_y =  apt*TMath::Sin(met_phi);
  p4met.SetPxPyPzE( met_x, met_y, 0, met);

  TLorentzVector lepton;
  lepton.SetPtEtaPhiE(*lepton_pt, *lepton_eta, *lepton_phi, *lepton_e);
  lepton = lepton*(lepton_scale[0]);

  float transverseM = transverseMass(lepton, p4met);
  float lepDphi = lepton.DeltaPhi(p4met);

  //Selection Option
  bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);
  bool passelectron = (mode == 1) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);

  if( option.Contains("DataSingleMu") ){
    if( !passmuon ) return kTRUE;//RDMu
    if( passelectron ) return kTRUE;//RDMu
  }
  else if( option.Contains("DataSingleEG") ){
    if( !passelectron ) return kTRUE;//RDelec
    if( passmuon ) return kTRUE;//RDelec
  }
  else{
    if( !passmuon && !passelectron ) return kTRUE;
  }

  float bWP_M, bWP_T, cvsbWP_M, cvslWP_M;
  if      ( era == 2017 ) bWP_M = 0.4941;
  else if ( era == 2018 ) bWP_M = 0.4184;
  else                    bWP_M = 0.0;

  for (unsigned int iJet = 0; iJet < jet_pt.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pt[iJet], jet_eta[iJet], jet_phi[iJet], jet_e[iJet]);

    if( !option.Contains("Run201") ){
      if     ( syst_ext == "jecup" )   jet = jet * jet_JER_Nom[iJet] * jet_JES_Up[iJet];
      else if( syst_ext == "jecdown" ) jet = jet * jet_JER_Nom[iJet] * jet_JES_Down[iJet];
      else if( syst_ext == "jerup" )   jet = jet * jet_JER_Up[iJet];
      else if( syst_ext == "jerdown" ) jet = jet * jet_JER_Down[iJet];
      else                             jet = jet * jet_JER_Nom[iJet];
    }

    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      njets++;
      if( jet_deepCSV[iJet] > bWP_M ) nbjets_m++;
    }
  } 

  bool st_njet = (njets >= 3 and nbjets_m >= 2);
  bool tt_njet = (njets >= 4 and nbjets_m >= 2);
  if( !st_njet ) return kTRUE; // At least 3 jets including 2 b jets

  if( option.Contains("Run201") ) b_EventCategory = -1;
  if( option.Contains("Run201") ) b_EventCategory = -1;
  else if( option.Contains("Hct") || option.Contains("Hut") ) b_EventCategory = 0;
  else if( option.Contains("ttbb") ) b_EventCategory = 1;
  else if( option.Contains("ttcc") ) b_EventCategory = 2;
  else if( option.Contains("ttlf") ) b_EventCategory = 3;
  else if( option.Contains("SingleT") ) b_EventCategory = 4; //singletop
  else if( option.Contains("TTZ") or option.Contains("TTW") or option.Contains("ttH")) b_EventCategory = 5; //ttX
  else if( option.Contains("DY") ) b_EventCategory = 6;
  else if( option.Contains("W1Jets") or option.Contains("W2Jets") or option.Contains("W3Jets") or option.Contains("W4Jets") ) b_EventCategory = 7;
  else if( option.Contains("WW") or option.Contains("WZ") or option.Contains("ZZ") ) b_EventCategory = 8;
  else b_EventCategory = 10;

  b_EventWeight = 1.0;
  if( !option.Contains("Run201") ){
    if( passmuon ) b_EventWeight *= lepton_SF[0] * lepton_SF[3] * lepton_SF[6];
    else if( passelectron) b_EventWeight *= lepton_SF[0] * lepton_SF[3] * lepton_SF[6] * lepton_SF[9];
    b_EventWeight *= PUWeight[0] * wrongPVrate * jet_SF_deepCSV_30[0] * (*genweight);
    if( era == 2017 ) b_EventWeight *= prefireweight[0];
  }

  b_channel = mode;
  b_TruePV = *TruePV;
  b_GoodPV = *GoodPV;
  b_nevt = nevt;

  b_njets = njets;
  b_nbjets_m = nbjets_m;

  b_met = met;
  b_met_phi = met_phi;
  b_lepdphi = lepDphi;

  TLorentzVector lepW = lepton + p4met;

  b_lepton_pt = lepton.Pt(); b_lepton_phi = lepton.Phi(); b_lepton_eta = lepton.Eta(); b_lepton_m = lepton.M();
  b_lepWpt = (lepW).Pt(); b_lepWeta = (lepW).Eta(); b_lepWphi = (lepW).Phi(); b_lepWmt = transverseM;
  b_lepWdphi = lepton.DeltaPhi(p4met);

  //Jet Assignment
  if( st_njet ){
    int jetIdx[4];
    TLorentzVector jetP4[4];
    vector<double>::iterator iter;
    int evtIdx = 0;
    //find combination
    if( !stfcnc_lepPt.empty() ){
      for( iter = stfcnc_lepPt.begin(); iter != stfcnc_lepPt.end(); iter++ ){
        if( *iter == static_cast<float>(lepton.Pt()) ){
          int tmpIdx = distance(stfcnc_lepPt.begin(), iter);
          if( stfcnc_missET[tmpIdx] == met ) evtIdx = tmpIdx;
          else continue;
        }
      }
      stfcnc_dupCheck.push_back(evtIdx);
      //cout << evtIdx << endl;

      stfcnc_Tree->GetEntry(evtIdx);
      b_stfcnc_genMatch = stfcnc_Tree->GetLeaf("match")->GetValue(0);
      b_stfcnc_score = stfcnc_Tree->GetLeaf("score")->GetValue(0);
      int i0 = stfcnc_Tree->GetLeaf("idx0")->GetValue(0);
      int i1 = stfcnc_Tree->GetLeaf("idx1")->GetValue(0);
      int i2 = stfcnc_Tree->GetLeaf("idx2")->GetValue(0);
      int i3 = stfcnc_Tree->GetLeaf("idx3")->GetValue(0);
      jetIdx[0] = i0; jetIdx[1] = i1; jetIdx[2] = i2; jetIdx[3] = i3;
      b_stfcnc_jet0Idx = jetIdx[0]; b_stfcnc_jet1Idx = jetIdx[1];
      b_stfcnc_jet2Idx = jetIdx[2]; b_stfcnc_jet3Idx = jetIdx[3];
      //cout << i0 << endl;

      for( int i=0; i < 4; i++) jetP4[i].SetPtEtaPhiE(jet_pt[jetIdx[i]], jet_eta[jetIdx[i]], jet_phi[jetIdx[i]], jet_e[jetIdx[i]]);
    }

    TLorentzVector jetP4cor[4];
    if( !option.Contains("Run201") ){
      if     ( syst_ext == "jecup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Up[jetIdx[i]];
      else if( syst_ext == "jecdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Down[jetIdx[i]];
      else if( syst_ext == "jerup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Up[jetIdx[i]];
      else if( syst_ext == "jerdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Down[jetIdx[i]];
      else                             for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]];
    }
    else if( option.Contains("Run201") ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i];

    b_stfcnc_jet0pt = jetP4cor[0].Pt(); b_stfcnc_jet0eta = jetP4cor[0].Eta();
    b_stfcnc_jet0phi = jetP4cor[0].Phi(); b_stfcnc_jet0m = jetP4cor[0].M();
    b_stfcnc_jet0csv = jet_deepCSV[jetIdx[0]];
    b_stfcnc_jet0cvsl = jet_deepCvsL[jetIdx[0]]; b_stfcnc_jet0cvsb = jet_deepCvsB[jetIdx[0]];

    b_stfcnc_jet1pt = jetP4cor[1].Pt(); b_stfcnc_jet1eta = jetP4cor[1].Eta();
    b_stfcnc_jet1phi = jetP4cor[1].Phi(); b_stfcnc_jet1m = jetP4cor[1].M();
    b_stfcnc_jet1csv = jet_deepCSV[jetIdx[1]];
    b_stfcnc_jet1cvsl = jet_deepCvsL[jetIdx[1]]; b_stfcnc_jet1cvsb = jet_deepCvsB[jetIdx[1]];

    b_stfcnc_jet2pt = jetP4cor[2].Pt(); b_stfcnc_jet2eta = jetP4cor[2].Eta();
    b_stfcnc_jet2phi = jetP4cor[2].Phi(); b_stfcnc_jet2m = jetP4cor[2].M();
    b_stfcnc_jet2csv = jet_deepCSV[jetIdx[2]];
    b_stfcnc_jet2cvsl = jet_deepCvsL[jetIdx[2]]; b_stfcnc_jet2cvsb = jet_deepCvsB[jetIdx[2]];

    b_stfcnc_jet3pt = jetP4cor[3].Pt(); b_stfcnc_jet3eta = jetP4cor[3].Eta();
    b_stfcnc_jet3phi = jetP4cor[3].Phi(); b_stfcnc_jet3m = jetP4cor[3].M();
    b_stfcnc_jet3csv = jet_deepCSV[jetIdx[3]];
    b_stfcnc_jet3cvsl = jet_deepCvsL[jetIdx[3]]; b_stfcnc_jet3cvsb = jet_deepCvsB[jetIdx[3]];

    if( b_stfcnc_jet0csv < 0 ) b_stfcnc_jet0csv = 0; if( b_stfcnc_jet1csv < 0 ) b_stfcnc_jet1csv = 0;
    if( b_stfcnc_jet2csv < 0 ) b_stfcnc_jet2csv = 0; if( b_stfcnc_jet3csv < 0 ) b_stfcnc_jet3csv = 0;

    const auto lepT = lepW + jetP4cor[0];
    const auto had12 = jetP4cor[1] + jetP4cor[2];//This is W or H
    const auto had23 = jetP4cor[2] + jetP4cor[3];
    const auto had31 = jetP4cor[3] + jetP4cor[1];
    const auto hadT = jetP4cor[1] + jetP4cor[2] + jetP4cor[3];

    b_stfcnc_jet12pt = had12.Pt(); b_stfcnc_jet12eta = had12.Eta();
    b_stfcnc_jet12phi = had12.Phi(); b_stfcnc_jet12m = had12.M();
    b_stfcnc_jet12deta = (jetP4cor[1]-jetP4cor[2]).Eta();
    b_stfcnc_jet12dphi = jetP4cor[1].DeltaPhi(jetP4cor[2]);
    b_stfcnc_jet12dR = jetP4cor[1].DeltaR(jetP4cor[2]);

    b_stfcnc_jet23pt = had23.Pt(); b_stfcnc_jet23eta = had23.Eta();
    b_stfcnc_jet23phi = had23.Phi(); b_stfcnc_jet23m = had23.M();
    b_stfcnc_jet23deta = (jetP4cor[2]-jetP4cor[3]).Eta();
    b_stfcnc_jet23dphi = jetP4cor[2].DeltaPhi(jetP4cor[3]);
    b_stfcnc_jet23dR = jetP4cor[2].DeltaR(jetP4cor[3]);

    b_stfcnc_jet31pt = had31.Pt(); b_stfcnc_jet31eta = had31.Eta();
    b_stfcnc_jet31phi = had31.Phi(); b_stfcnc_jet31m = had31.M();
    b_stfcnc_jet31deta = (jetP4cor[3]-jetP4cor[1]).Eta();
    b_stfcnc_jet31dphi = jetP4cor[3].DeltaPhi(jetP4cor[1]);
    b_stfcnc_jet31dR = jetP4cor[3].DeltaR(jetP4cor[1]);

    b_stfcnc_lepTpt = lepT.Pt(); b_stfcnc_lepTeta = lepT.Eta();
    b_stfcnc_lepTphi = lepT.Phi(); b_stfcnc_lepTmt = transverseMass(lepton+jetP4cor[0],p4met);
    b_stfcnc_lepTdeta = (lepW-jetP4cor[0]).Eta();
    b_stfcnc_lepTdphi = lepW.DeltaPhi(jetP4cor[0]);
    b_stfcnc_lepTdR = lepW.DeltaR(jetP4cor[0]);

    b_stfcnc_hadTpt = hadT.Pt(); b_stfcnc_hadTeta = hadT.Eta();
    b_stfcnc_hadTphi = hadT.Phi(); b_stfcnc_hadTm = hadT.M();
    b_stfcnc_hadT12_3deta = (had12-jetP4cor[3]).Eta();
    b_stfcnc_hadT23_1deta = (had23-jetP4cor[1]).Eta();
    b_stfcnc_hadT31_2deta = (had31-jetP4cor[2]).Eta();
    b_stfcnc_hadT12_3dphi = had12.DeltaPhi(jetP4cor[3]);
    b_stfcnc_hadT23_1dphi = had23.DeltaPhi(jetP4cor[1]);
    b_stfcnc_hadT31_2dphi = had31.DeltaPhi(jetP4cor[2]);
    b_stfcnc_hadT12_3dR = had12.DeltaR(jetP4cor[3]);
    b_stfcnc_hadT23_1dR = had23.DeltaR(jetP4cor[1]);
    b_stfcnc_hadT31_2dR = had31.DeltaR(jetP4cor[2]);

    b_stfcnc_jet0lepdR = jetP4cor[0].DeltaR(lepton);
    b_stfcnc_jet1lepdR = jetP4cor[1].DeltaR(lepton);
    b_stfcnc_jet2lepdR = jetP4cor[2].DeltaR(lepton);
    b_stfcnc_jet3lepdR = jetP4cor[3].DeltaR(lepton);
    b_stfcnc_jet01dR = jetP4cor[0].DeltaR(jetP4cor[1]);
    b_stfcnc_jet02dR = jetP4cor[0].DeltaR(jetP4cor[2]);
    b_stfcnc_jet03dR = jetP4cor[0].DeltaR(jetP4cor[3]);
    b_stfcnc_jet12_lepdR = lepton.DeltaR(had12);
    b_stfcnc_jet23_lepdR = lepton.DeltaR(had23);
    b_stfcnc_jet31_lepdR = lepton.DeltaR(had31);
    b_stfcnc_jet12_0dR = jetP4cor[0].DeltaR(had12);
    b_stfcnc_jet23_0dR = jetP4cor[0].DeltaR(had23);
    b_stfcnc_jet31_0dR = jetP4cor[0].DeltaR(had31);
    b_stfcnc_lepTjet12dphi = lepT.DeltaPhi(had12);
    b_stfcnc_lepTjet23dphi = lepT.DeltaPhi(had23);
    b_stfcnc_lepTjet31dphi = lepT.DeltaPhi(had31);
    b_stfcnc_hadT_jet0dR = hadT.DeltaR(jetP4cor[0]);

  }

  //Jet Assignment
  if( tt_njet ){
    int jetIdx[4];
    TLorentzVector jetP4[4];
    vector<double>::iterator iter;
    int evtIdx = 0;
    //find combination
    if( !ttfcnc_lepPt.empty() ){
      for( iter = ttfcnc_lepPt.begin(); iter != ttfcnc_lepPt.end(); iter++ ){
        if( *iter == static_cast<float>(lepton.Pt()) ){
          int tmpIdx = distance(ttfcnc_lepPt.begin(), iter);
          if( ttfcnc_missET[tmpIdx] == met ) evtIdx = tmpIdx;
          else continue;
        }
      }
      ttfcnc_dupCheck.push_back(evtIdx);
      //cout << evtIdx << endl;

      ttfcnc_Tree->GetEntry(evtIdx);
      b_ttfcnc_genMatch = ttfcnc_Tree->GetLeaf("match")->GetValue(0);
      b_ttfcnc_score = ttfcnc_Tree->GetLeaf("score")->GetValue(0);
      int i0 = ttfcnc_Tree->GetLeaf("idx0")->GetValue(0);
      int i1 = ttfcnc_Tree->GetLeaf("idx1")->GetValue(0);
      int i2 = ttfcnc_Tree->GetLeaf("idx2")->GetValue(0);
      int i3 = ttfcnc_Tree->GetLeaf("idx3")->GetValue(0);
      jetIdx[0] = i0; jetIdx[1] = i1; jetIdx[2] = i2; jetIdx[3] = i3;
      b_ttfcnc_jet0Idx = jetIdx[0]; b_ttfcnc_jet1Idx = jetIdx[1];
      b_ttfcnc_jet2Idx = jetIdx[2]; b_ttfcnc_jet3Idx = jetIdx[3];
      //cout << i0 << endl;

      for( int i=0; i < 4; i++) jetP4[i].SetPtEtaPhiE(jet_pt[jetIdx[i]], jet_eta[jetIdx[i]], jet_phi[jetIdx[i]], jet_e[jetIdx[i]]);
    }

    TLorentzVector jetP4cor[4];
    if( !option.Contains("Run201") ){
      if     ( syst_ext == "jecup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Up[jetIdx[i]];
      else if( syst_ext == "jecdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Down[jetIdx[i]];
      else if( syst_ext == "jerup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Up[jetIdx[i]];
      else if( syst_ext == "jerdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Down[jetIdx[i]];
      else                             for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]];
    }
    else if( option.Contains("Run201") ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i];

    b_ttfcnc_jet0pt = jetP4cor[0].Pt(); b_ttfcnc_jet0eta = jetP4cor[0].Eta();
    b_ttfcnc_jet0phi = jetP4cor[0].Phi(); b_ttfcnc_jet0m = jetP4cor[0].M();
    b_ttfcnc_jet0csv = jet_deepCSV[jetIdx[0]];
    b_ttfcnc_jet0cvsl = jet_deepCvsL[jetIdx[0]]; b_ttfcnc_jet0cvsb = jet_deepCvsB[jetIdx[0]];

    b_ttfcnc_jet1pt = jetP4cor[1].Pt(); b_ttfcnc_jet1eta = jetP4cor[1].Eta();
    b_ttfcnc_jet1phi = jetP4cor[1].Phi(); b_ttfcnc_jet1m = jetP4cor[1].M();
    b_ttfcnc_jet1csv = jet_deepCSV[jetIdx[1]];
    b_ttfcnc_jet1cvsl = jet_deepCvsL[jetIdx[1]]; b_ttfcnc_jet1cvsb = jet_deepCvsB[jetIdx[1]];

    b_ttfcnc_jet2pt = jetP4cor[2].Pt(); b_ttfcnc_jet2eta = jetP4cor[2].Eta();
    b_ttfcnc_jet2phi = jetP4cor[2].Phi(); b_ttfcnc_jet2m = jetP4cor[2].M();
    b_ttfcnc_jet2csv = jet_deepCSV[jetIdx[2]];
    b_ttfcnc_jet2cvsl = jet_deepCvsL[jetIdx[2]]; b_ttfcnc_jet2cvsb = jet_deepCvsB[jetIdx[2]];

    b_ttfcnc_jet3pt = jetP4cor[3].Pt(); b_ttfcnc_jet3eta = jetP4cor[3].Eta();
    b_ttfcnc_jet3phi = jetP4cor[3].Phi(); b_ttfcnc_jet3m = jetP4cor[3].M();
    b_ttfcnc_jet3csv = jet_deepCSV[jetIdx[3]];
    b_ttfcnc_jet3cvsl = jet_deepCvsL[jetIdx[3]]; b_ttfcnc_jet3cvsb = jet_deepCvsB[jetIdx[3]];

    if( b_ttfcnc_jet0csv < 0 ) b_ttfcnc_jet0csv = 0; if( b_ttfcnc_jet1csv < 0 ) b_ttfcnc_jet1csv = 0;
    if( b_ttfcnc_jet2csv < 0 ) b_ttfcnc_jet2csv = 0; if( b_ttfcnc_jet3csv < 0 ) b_ttfcnc_jet3csv = 0;

    const auto lepT = lepW + jetP4cor[0];
    const auto had12 = jetP4cor[1] + jetP4cor[2];//This is W or H
    const auto had23 = jetP4cor[2] + jetP4cor[3];
    const auto had31 = jetP4cor[3] + jetP4cor[1];
    const auto hadT = jetP4cor[1] + jetP4cor[2] + jetP4cor[3];

    b_ttfcnc_jet12pt = had12.Pt(); b_ttfcnc_jet12eta = had12.Eta();
    b_ttfcnc_jet12phi = had12.Phi(); b_ttfcnc_jet12m = had12.M();
    b_ttfcnc_jet12deta = (jetP4cor[1]-jetP4cor[2]).Eta();
    b_ttfcnc_jet12dphi = jetP4cor[1].DeltaPhi(jetP4cor[2]);
    b_ttfcnc_jet12dR = jetP4cor[1].DeltaR(jetP4cor[2]);

    b_ttfcnc_jet23pt = had23.Pt(); b_ttfcnc_jet23eta = had23.Eta();
    b_ttfcnc_jet23phi = had23.Phi(); b_ttfcnc_jet23m = had23.M();
    b_ttfcnc_jet23deta = (jetP4cor[2]-jetP4cor[3]).Eta();
    b_ttfcnc_jet23dphi = jetP4cor[2].DeltaPhi(jetP4cor[3]);
    b_ttfcnc_jet23dR = jetP4cor[2].DeltaR(jetP4cor[3]);

    b_ttfcnc_jet31pt = had31.Pt(); b_ttfcnc_jet31eta = had31.Eta();
    b_ttfcnc_jet31phi = had31.Phi(); b_ttfcnc_jet31m = had31.M();
    b_ttfcnc_jet31deta = (jetP4cor[3]-jetP4cor[1]).Eta();
    b_ttfcnc_jet31dphi = jetP4cor[3].DeltaPhi(jetP4cor[1]);
    b_ttfcnc_jet31dR = jetP4cor[3].DeltaR(jetP4cor[1]);

    b_ttfcnc_lepTpt = lepT.Pt(); b_ttfcnc_lepTeta = lepT.Eta();
    b_ttfcnc_lepTphi = lepT.Phi(); b_ttfcnc_lepTmt = transverseMass(lepton+jetP4cor[0],p4met);
    b_ttfcnc_lepTdeta = (lepW-jetP4cor[0]).Eta();
    b_ttfcnc_lepTdphi = lepW.DeltaPhi(jetP4cor[0]);
    b_ttfcnc_lepTdR = lepW.DeltaR(jetP4cor[0]);

    b_ttfcnc_hadTpt = hadT.Pt(); b_ttfcnc_hadTeta = hadT.Eta();
    b_ttfcnc_hadTphi = hadT.Phi(); b_ttfcnc_hadTm = hadT.M();
    b_ttfcnc_hadT12_3deta = (had12-jetP4cor[3]).Eta();
    b_ttfcnc_hadT23_1deta = (had23-jetP4cor[1]).Eta();
    b_ttfcnc_hadT31_2deta = (had31-jetP4cor[2]).Eta();
    b_ttfcnc_hadT12_3dphi = had12.DeltaPhi(jetP4cor[3]);
    b_ttfcnc_hadT23_1dphi = had23.DeltaPhi(jetP4cor[1]);
    b_ttfcnc_hadT31_2dphi = had31.DeltaPhi(jetP4cor[2]);
    b_ttfcnc_hadT12_3dR = had12.DeltaR(jetP4cor[3]);
    b_ttfcnc_hadT23_1dR = had23.DeltaR(jetP4cor[1]);
    b_ttfcnc_hadT31_2dR = had31.DeltaR(jetP4cor[2]);

    b_ttfcnc_jet0lepdR = jetP4cor[0].DeltaR(lepton);
    b_ttfcnc_jet1lepdR = jetP4cor[1].DeltaR(lepton);
    b_ttfcnc_jet2lepdR = jetP4cor[2].DeltaR(lepton);
    b_ttfcnc_jet3lepdR = jetP4cor[3].DeltaR(lepton);
    b_ttfcnc_jet01dR = jetP4cor[0].DeltaR(jetP4cor[1]);
    b_ttfcnc_jet02dR = jetP4cor[0].DeltaR(jetP4cor[2]);
    b_ttfcnc_jet03dR = jetP4cor[0].DeltaR(jetP4cor[3]);
    b_ttfcnc_jet12_lepdR = lepton.DeltaR(had12);
    b_ttfcnc_jet23_lepdR = lepton.DeltaR(had23);
    b_ttfcnc_jet31_lepdR = lepton.DeltaR(had31);
    b_ttfcnc_jet12_0dR = jetP4cor[0].DeltaR(had12);
    b_ttfcnc_jet23_0dR = jetP4cor[0].DeltaR(had23);
    b_ttfcnc_jet31_0dR = jetP4cor[0].DeltaR(had31);
    b_ttfcnc_lepTjet12dphi = lepT.DeltaPhi(had12);
    b_ttfcnc_lepTjet23dphi = lepT.DeltaPhi(had23);
    b_ttfcnc_lepTjet31dphi = lepT.DeltaPhi(had31);
    b_ttfcnc_hadT_jet0dR = hadT.DeltaR(jetP4cor[0]);

  }

  //Jet Assignment
  if( tt_njet ){
    int jetIdx[4];
    TLorentzVector jetP4[4];
    vector<double>::iterator iter;
    int evtIdx = 0;
    //find combination
    if( !ttbkg_lepPt.empty() ){
      for( iter = ttbkg_lepPt.begin(); iter != ttbkg_lepPt.end(); iter++ ){
        if( *iter == static_cast<float>(lepton.Pt()) ){
          int tmpIdx = distance(ttbkg_lepPt.begin(), iter);
          if( ttbkg_missET[tmpIdx] == met ) evtIdx = tmpIdx;
          else continue;
        }
      }
      ttbkg_dupCheck.push_back(evtIdx);
      //cout << evtIdx << endl;

      ttbkg_Tree->GetEntry(evtIdx);
      b_ttbkg_genMatch = ttbkg_Tree->GetLeaf("match")->GetValue(0);
      b_ttbkg_score = ttbkg_Tree->GetLeaf("score")->GetValue(0);
      int i0 = ttbkg_Tree->GetLeaf("idx0")->GetValue(0);
      int i1 = ttbkg_Tree->GetLeaf("idx1")->GetValue(0);
      int i2 = ttbkg_Tree->GetLeaf("idx2")->GetValue(0);
      int i3 = ttbkg_Tree->GetLeaf("idx3")->GetValue(0);
      jetIdx[0] = i0; jetIdx[1] = i1; jetIdx[2] = i2; jetIdx[3] = i3;
      b_ttbkg_jet0Idx = jetIdx[0]; b_ttbkg_jet1Idx = jetIdx[1];
      b_ttbkg_jet2Idx = jetIdx[2]; b_ttbkg_jet3Idx = jetIdx[3];
      //cout << i0 << endl;

      for( int i=0; i < 4; i++) jetP4[i].SetPtEtaPhiE(jet_pt[jetIdx[i]], jet_eta[jetIdx[i]], jet_phi[jetIdx[i]], jet_e[jetIdx[i]]);
    }

    TLorentzVector jetP4cor[4];
    if( !option.Contains("Run201") ){
      if     ( syst_ext == "jecup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Up[jetIdx[i]];
      else if( syst_ext == "jecdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Down[jetIdx[i]];
      else if( syst_ext == "jerup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Up[jetIdx[i]];
      else if( syst_ext == "jerdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Down[jetIdx[i]];
      else                             for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[jetIdx[i]];
    }
    else if( option.Contains("Run201") ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i];

    b_ttbkg_jet0pt = jetP4cor[0].Pt(); b_ttbkg_jet0eta = jetP4cor[0].Eta();
    b_ttbkg_jet0phi = jetP4cor[0].Phi(); b_ttbkg_jet0m = jetP4cor[0].M();
    b_ttbkg_jet0csv = jet_deepCSV[jetIdx[0]];
    b_ttbkg_jet0cvsl = jet_deepCvsL[jetIdx[0]]; b_ttbkg_jet0cvsb = jet_deepCvsB[jetIdx[0]];

    b_ttbkg_jet1pt = jetP4cor[1].Pt(); b_ttbkg_jet1eta = jetP4cor[1].Eta();
    b_ttbkg_jet1phi = jetP4cor[1].Phi(); b_ttbkg_jet1m = jetP4cor[1].M();
    b_ttbkg_jet1csv = jet_deepCSV[jetIdx[1]];
    b_ttbkg_jet1cvsl = jet_deepCvsL[jetIdx[1]]; b_ttbkg_jet1cvsb = jet_deepCvsB[jetIdx[1]];

    b_ttbkg_jet2pt = jetP4cor[2].Pt(); b_ttbkg_jet2eta = jetP4cor[2].Eta();
    b_ttbkg_jet2phi = jetP4cor[2].Phi(); b_ttbkg_jet2m = jetP4cor[2].M();
    b_ttbkg_jet2csv = jet_deepCSV[jetIdx[2]];
    b_ttbkg_jet2cvsl = jet_deepCvsL[jetIdx[2]]; b_ttbkg_jet2cvsb = jet_deepCvsB[jetIdx[2]];

    b_ttbkg_jet3pt = jetP4cor[3].Pt(); b_ttbkg_jet3eta = jetP4cor[3].Eta();
    b_ttbkg_jet3phi = jetP4cor[3].Phi(); b_ttbkg_jet3m = jetP4cor[3].M();
    b_ttbkg_jet3csv = jet_deepCSV[jetIdx[3]];
    b_ttbkg_jet3cvsl = jet_deepCvsL[jetIdx[3]]; b_ttbkg_jet3cvsb = jet_deepCvsB[jetIdx[3]];

    if( b_ttbkg_jet0csv < 0 ) b_ttbkg_jet0csv = 0; if( b_ttbkg_jet1csv < 0 ) b_ttbkg_jet1csv = 0;
    if( b_ttbkg_jet2csv < 0 ) b_ttbkg_jet2csv = 0; if( b_ttbkg_jet3csv < 0 ) b_ttbkg_jet3csv = 0;

    const auto lepT = lepW + jetP4cor[0];
    const auto had12 = jetP4cor[1] + jetP4cor[2];//This is W or H
    const auto had23 = jetP4cor[2] + jetP4cor[3];
    const auto had31 = jetP4cor[3] + jetP4cor[1];
    const auto hadT = jetP4cor[1] + jetP4cor[2] + jetP4cor[3];

    b_ttbkg_jet12pt = had12.Pt(); b_ttbkg_jet12eta = had12.Eta();
    b_ttbkg_jet12phi = had12.Phi(); b_ttbkg_jet12m = had12.M();
    b_ttbkg_jet12deta = (jetP4cor[1]-jetP4cor[2]).Eta();
    b_ttbkg_jet12dphi = jetP4cor[1].DeltaPhi(jetP4cor[2]);
    b_ttbkg_jet12dR = jetP4cor[1].DeltaR(jetP4cor[2]);

    b_ttbkg_jet23pt = had23.Pt(); b_ttbkg_jet23eta = had23.Eta();
    b_ttbkg_jet23phi = had23.Phi(); b_ttbkg_jet23m = had23.M();
    b_ttbkg_jet23deta = (jetP4cor[2]-jetP4cor[3]).Eta();
    b_ttbkg_jet23dphi = jetP4cor[2].DeltaPhi(jetP4cor[3]);
    b_ttbkg_jet23dR = jetP4cor[2].DeltaR(jetP4cor[3]);

    b_ttbkg_jet31pt = had31.Pt(); b_ttbkg_jet31eta = had31.Eta();
    b_ttbkg_jet31phi = had31.Phi(); b_ttbkg_jet31m = had31.M();
    b_ttbkg_jet31deta = (jetP4cor[3]-jetP4cor[1]).Eta();
    b_ttbkg_jet31dphi = jetP4cor[3].DeltaPhi(jetP4cor[1]);
    b_ttbkg_jet31dR = jetP4cor[3].DeltaR(jetP4cor[1]);

    b_ttbkg_lepTpt = lepT.Pt(); b_ttbkg_lepTeta = lepT.Eta();
    b_ttbkg_lepTphi = lepT.Phi(); b_ttbkg_lepTmt = transverseMass(lepton+jetP4cor[0],p4met);
    b_ttbkg_lepTdeta = (lepW-jetP4cor[0]).Eta();
    b_ttbkg_lepTdphi = lepW.DeltaPhi(jetP4cor[0]);
    b_ttbkg_lepTdR = lepW.DeltaR(jetP4cor[0]);

    b_ttbkg_hadTpt = hadT.Pt(); b_ttbkg_hadTeta = hadT.Eta();
    b_ttbkg_hadTphi = hadT.Phi(); b_ttbkg_hadTm = hadT.M();
    b_ttbkg_hadT12_3deta = (had12-jetP4cor[3]).Eta();
    b_ttbkg_hadT23_1deta = (had23-jetP4cor[1]).Eta();
    b_ttbkg_hadT31_2deta = (had31-jetP4cor[2]).Eta();
    b_ttbkg_hadT12_3dphi = had12.DeltaPhi(jetP4cor[3]);
    b_ttbkg_hadT23_1dphi = had23.DeltaPhi(jetP4cor[1]);
    b_ttbkg_hadT31_2dphi = had31.DeltaPhi(jetP4cor[2]);
    b_ttbkg_hadT12_3dR = had12.DeltaR(jetP4cor[3]);
    b_ttbkg_hadT23_1dR = had23.DeltaR(jetP4cor[1]);
    b_ttbkg_hadT31_2dR = had31.DeltaR(jetP4cor[2]);

    b_ttbkg_jet0lepdR = jetP4cor[0].DeltaR(lepton);
    b_ttbkg_jet1lepdR = jetP4cor[1].DeltaR(lepton);
    b_ttbkg_jet2lepdR = jetP4cor[2].DeltaR(lepton);
    b_ttbkg_jet3lepdR = jetP4cor[3].DeltaR(lepton);
    b_ttbkg_jet01dR = jetP4cor[0].DeltaR(jetP4cor[1]);
    b_ttbkg_jet02dR = jetP4cor[0].DeltaR(jetP4cor[2]);
    b_ttbkg_jet03dR = jetP4cor[0].DeltaR(jetP4cor[3]);
    b_ttbkg_jet12_lepdR = lepton.DeltaR(had12);
    b_ttbkg_jet23_lepdR = lepton.DeltaR(had23);
    b_ttbkg_jet31_lepdR = lepton.DeltaR(had31);
    b_ttbkg_jet12_0dR = jetP4cor[0].DeltaR(had12);
    b_ttbkg_jet23_0dR = jetP4cor[0].DeltaR(had23);
    b_ttbkg_jet31_0dR = jetP4cor[0].DeltaR(had31);
    b_ttbkg_lepTjet12dphi = lepT.DeltaPhi(had12);
    b_ttbkg_lepTjet23dphi = lepT.DeltaPhi(had23);
    b_ttbkg_lepTjet31dphi = lepT.DeltaPhi(had31);
    b_ttbkg_hadT_jet0dR = hadT.DeltaR(jetP4cor[0]);

  }

  tree->Fill();

  nevt++;

  return kTRUE;
}

void makeTuple::SlaveTerminate()
{
  TString option = GetOption();
}
   

void makeTuple::Terminate()
{
  TString option = GetOption();
  string syst_str = option.Data();
  syst_str.erase(syst_str.find_first_of("_"),string::npos);
  string ver = syst_str.substr(4,6);
  string era = syst_str.substr(0,4);
  syst_str = syst_str.erase(0,6);
  string sample = option.Data();  
  sample.erase(0,sample.find_first_of("_")+1);

  TFile *hfile = TFile::Open(Form("%s/root_%s/finalMVA_%s.root",era.c_str(),syst_str.c_str(),sample.c_str()), "RECREATE");

  fOutput->FindObject("tree")->Write();

  hfile->Write();
  hfile->Close();

}

float makeTuple::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  float out = sumT.M();

  return out;

}

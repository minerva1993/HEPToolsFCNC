#define MyAnalysis_cxx

#include "MyAnalysis.h"
#include <TH2.h>
#include <TStyle.h>

void MyAnalysis::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void MyAnalysis::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();
  string sample = option.Data();
  string train_scheme = sample.substr(sample.find_first_of("-")+1,string::npos);
  sample.erase(sample.find_first_of("-"),string::npos);
  
  dosyst = true;
  if( option.Contains("Run2017") ) dosyst = false;

  //Delete ntuple number and data era so that we can merge histos w.r.t. dataset, prepare assign ntuple
  const char* score_file = "";
  if( train_scheme.find("jec") != string::npos or train_scheme.find("jer") != string::npos 
    or train_scheme.find("TuneCP5") != string::npos or train_scheme.find("hdamp") != string::npos ){
    dosyst = false;//make another hist file
    if     ( train_scheme.find("jecup") != string::npos )       syst_ext = "jecup";
    else if( train_scheme.find("jecdown") != string::npos )     syst_ext = "jecdown";
    else if( train_scheme.find("jerup") != string::npos )       syst_ext = "jerup";
    else if( train_scheme.find("jerdown") != string::npos )     syst_ext = "jerdown";
    else if( train_scheme.find("TuneCP5up") != string::npos )   syst_ext = "TuneCP5up";
    else if( train_scheme.find("TuneCP5down") != string::npos ) syst_ext = "TuneCP5down";
    else if( train_scheme.find("hdampup") != string::npos )     syst_ext = "hdampup";
    else if( train_scheme.find("hdampdown") != string::npos )   syst_ext = "hdampdown";
  }
  string syst_str = "-" + train_scheme.substr(train_scheme.find_last_of("_")+1,string::npos );
  if( syst_ext == "" ) syst_str.erase(0,1);
  train_scheme.erase( train_scheme.find_last_of("_"),string::npos );

  score_file = Form("/home/minerva1993/HEPToolsFCNC/analysis_2017/finalMVA/scores/%s%s/score_finalMVA_%s.root",
                     train_scheme.c_str(), syst_str.c_str(), sample.c_str());

  string file_tmp_path = score_file;
  ifstream file_tmp(file_tmp_path);
  if( file_tmp.is_open() ){
    scoreF = TFile::Open(score_file, "READ");
    scoreT = (TTree*) scoreF->Get("tree");
    int nevt = scoreT->GetEntries();
    if( nevt > 0){
      for(int i = 0; i < nevt; i++){
        scoreT->GetEntry(i);
        float pt = scoreT->GetLeaf("lepPt")->GetValue(0);
        float met = scoreT->GetLeaf("missinget")->GetValue(0);
        lepPt.push_back(pt);
        missET.push_back(met);
      }
    }
  }
  else cout << option.Data() << endl;

  if( !dosyst and syst_ext.length() < 5 and !option.Contains("Run2017") ) cout << sample.c_str() << " " << "No external systematic option!" << endl;


  //cout << "SlaveBegin" << endl;
  for( int ich=0; ich < 3; ich++ ){
    for( int syst = 0; syst != syst_num; ++syst ){
      if( syst > 0 and !dosyst ) continue;

      h_MVA[ich][syst] = new TH1D(Form("h_DNN_Ch%i%s",ich,syst_name[syst]), "Final MVA", 40, 0, 1);
      h_MVA[ich][syst]->SetXTitle("DNN Score");
      h_MVA[ich][syst]->Sumw2();
      fOutput->Add(h_MVA[ich][syst]);

    }
  }
} 

Bool_t MyAnalysis::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();
  string sample = option.Data();

  int njet_cut = 0;
  int nbjet_cut = 0;
  if     ( isPartOf("j3", sample) ) njet_cut = 3;
  else if( isPartOf("j4", sample) ) njet_cut = 4;
  else{
    cout << "Wrong njet cut!" << endl;
    return kTRUE;
  }
  if     ( isPartOf("b2", sample) ) nbjet_cut = 2;
  else if( isPartOf("b3", sample) ) nbjet_cut = 3; 
  else if( isPartOf("b4", sample) ) nbjet_cut = 4;

  int mode = 999; 
  mode = *channel;

  if( mode > 2 ) return kTRUE;

  float wrongPVrate = 1;
  if( !option.Contains("Run2017") ){
    //181013, 10-75
    if      ( option.Contains("DYJets10to50") ) wrongPVrate = 1.04879;
    else if ( option.Contains("DYJetsv2") ) wrongPVrate = 1.04287;
    else if ( option.Contains("DYJets_") or option.Contains("DYJetspart2") ) wrongPVrate = 1.04287;
    else if ( option.Contains("STTH1L3BHct") ) wrongPVrate = 1.04399;
    else if ( option.Contains("STTH1L3BHut") ) wrongPVrate = 1.0311;
    else if ( option.Contains("SingleTbart") ) wrongPVrate = 1.04165;
    else if ( option.Contains("SingleTbartW") ) wrongPVrate = 1.04271;
    else if ( option.Contains("SingleTops") ) wrongPVrate = 1.03979;
    else if ( option.Contains("SingleTopt") ) wrongPVrate = 1.04027;
    else if ( option.Contains("SingleToptW") ) wrongPVrate = 1.04483;
    else if ( option.Contains("TTHadpowheg") ) wrongPVrate = 1.02945;
    else if ( option.Contains("TTHadpowhegTuneCP5down") ) wrongPVrate = 1.02704;
    else if ( option.Contains("TTHadpowhegTuneCP5up") ) wrongPVrate = 1.0248;
    else if ( option.Contains("TTHadpowheghdampup") ) wrongPVrate = 1.02732;
    else if ( option.Contains("TTLLpowheg") ) wrongPVrate = 1.04709;
    else if ( option.Contains("TTLLpowhegTuneCP5down") ) wrongPVrate = 1.04085;
    else if ( option.Contains("TTLLpowheghdampdown") ) wrongPVrate = 1.05645;
    else if ( option.Contains("TTLLpowheghdampup") ) wrongPVrate = 1.0349;
    else if ( option.Contains("TTWJetsToLNuPSweight") ) wrongPVrate = 1.04085;
    else if ( option.Contains("TTWJetsToQQ") ) wrongPVrate = 1.02418;
    else if ( option.Contains("TTZToLLNuNu") ) wrongPVrate = 1.04257;
    else if ( option.Contains("TTZToQQ") ) wrongPVrate = 1.04888;
    else if ( option.Contains("TTpowhegttbb") ) wrongPVrate = 1.0486;
    else if ( option.Contains("TTpowhegttbbTuneCP5down") ) wrongPVrate = 1.04757;
    else if ( option.Contains("TTpowhegttbbTuneCP5up") ) wrongPVrate = 1.03875;
    else if ( option.Contains("TTpowhegttbbhdampdown") ) wrongPVrate = 1.04653;
    else if ( option.Contains("TTpowhegttbbhdampup") ) wrongPVrate = 1.04047;
    else if ( option.Contains("TTpowhegttbj") ) wrongPVrate = 1.04818;
    else if ( option.Contains("TTpowhegttbjTuneCP5down") ) wrongPVrate = 1.04639;
    else if ( option.Contains("TTpowhegttbjTuneCP5up") ) wrongPVrate = 1.03904;
    else if ( option.Contains("TTpowhegttbjhdampdown") ) wrongPVrate = 1.0466;
    else if ( option.Contains("TTpowhegttbjhdampup") ) wrongPVrate = 1.03973;
    else if ( option.Contains("TTpowhegttcc") ) wrongPVrate = 1.04758;
    else if ( option.Contains("TTpowhegttccTuneCP5down") ) wrongPVrate = 1.04788;
    else if ( option.Contains("TTpowhegttccTuneCP5up") ) wrongPVrate = 1.03774;
    else if ( option.Contains("TTpowhegttcchdampdown") ) wrongPVrate = 1.04723;
    else if ( option.Contains("TTpowhegttcchdampup") ) wrongPVrate = 1.03993;
    else if ( option.Contains("TTpowhegttlf") ) wrongPVrate = 1.04759;
    else if ( option.Contains("TTpowhegttlfTuneCP5down") ) wrongPVrate = 1.04765;
    else if ( option.Contains("TTpowhegttlfTuneCP5up") ) wrongPVrate = 1.03818;
    else if ( option.Contains("TTpowhegttlfhdampdown") ) wrongPVrate = 1.04787;
    else if ( option.Contains("TTpowhegttlfhdampup") ) wrongPVrate = 1.03931;
    else if ( option.Contains("TTpowhegttother") ) wrongPVrate = 1.04785;
    else if ( option.Contains("TTpowhegttotherTuneCP5down") ) wrongPVrate = 1.04796;
    else if ( option.Contains("TTpowhegttotherTuneCP5up") ) wrongPVrate = 1.03817;
    else if ( option.Contains("TTpowhegttotherhdampdown") ) wrongPVrate = 1.04797;
    else if ( option.Contains("TTpowhegttotherhdampup") ) wrongPVrate = 1.03942;
    else if ( option.Contains("W3JetsToLNu") ) wrongPVrate = 1.0416;
    else if ( option.Contains("W4JetsToLNu") ) wrongPVrate = 1.04368;
    else if ( option.Contains("WW") ) wrongPVrate = 1.04711;
    else if ( option.Contains("WZ") ) wrongPVrate = 1.04261;
    else if ( option.Contains("ZZ") ) wrongPVrate = 1.02918;
    else if ( option.Contains("ttHToNonbb") ) wrongPVrate = 1.03169;
    else if ( option.Contains("ttHTobb") ) wrongPVrate = 1.0374;
    else    wrongPVrate = 1.0;
  }
  if( wrongPVrate > 1.01 ){
    if( *TruePV < 10 || *TruePV > 75 ) return kTRUE;
  }
  float relIso = *lepton_relIso; 

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 

  TLorentzVector p4met;
  double met = *MET;
  double met_phi = *MET_phi;
  double apt = TMath::Abs(met);
  double met_x =  apt*TMath::Cos(met_phi);
  double met_y =  apt*TMath::Sin(met_phi);
  p4met.SetPxPyPzE( met_x, met_y, 0, met);

  TLorentzVector lepton;
  lepton.SetPtEtaPhiE(*lepton_pt, *lepton_eta, *lepton_phi, *lepton_e);

  double transverseM = transverseMass(lepton, p4met);
  double lepDphi = lepton.DeltaPhi(p4met);

  //Selection Option
  bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);
  bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.4) && (*elec_trg >= 10);
//  bool passelectron;
//  if ( *elec_trg == 10 ) passelectron = (mode == 1) && (lepton.Pt() > 33) && (abs(lepton.Eta()) <= 2.1);
//  else                   passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  //Single Lepton only
  if( option.Contains("SingleMuon") ){
    if( !passmuon ) return kTRUE;//RDMu
    if( passelectron) return kTRUE;//RDMu
  }
  else if( option.Contains("SingleElectron") ){
    if( !passelectron ) return kTRUE;//RDelec
    if( passmuon ) return kTRUE;//RDelec
  }
  else{
    if( !passmuon && !passelectron ) return kTRUE;
  }
  //if( passmuon || passelectron ){

  int jetIdx[4];
  TLorentzVector jetP4s[4];

  for (unsigned int iJet = 0; iJet < jet_pt.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pt[iJet], jet_eta[iJet], jet_phi[iJet], jet_e[iJet]);

    if( !option.Contains("Run2017") ){
      if     ( syst_ext == "jecup" )   jet = jet * jet_JER_Nom[iJet] * jet_JES_Up[iJet];
      else if( syst_ext == "jecdown" ) jet = jet * jet_JER_Nom[iJet] * jet_JES_Down[iJet];
      else if( syst_ext == "jerup" )   jet = jet * jet_JER_Up[iJet];
      else if( syst_ext == "jerdown" ) jet = jet * jet_JER_Down[iJet];
      else                             jet = jet * jet_JER_Nom[iJet];
    }

    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
//      if( passelectron and  *elec_trg == 10 and njets == 0 and jet_pt[iJet] < 38 ) continue;
      njets++;

      if( jet_deepCSV[iJet] > 0.4941 ) nbjets_m++;
    }
  }

  //Event selection is done here!
  if     ( njet_cut == 3 and njets != njet_cut) return kTRUE;
  else if( njet_cut == 4 and njets < 4 ) return kTRUE;
  if( nbjet_cut != 0 ){
    if( nbjets_m != nbjet_cut ) return kTRUE;
  }
  else{
    if( nbjets_m < 2 or nbjets_m > 4 ) return kTRUE;
  }

  float tmp_score = -1;
  vector<double>::iterator iter;
  int evtIdx = 0;
  if( !lepPt.empty() ){
    for( iter = lepPt.begin(); iter != lepPt.end(); iter++ ){
      if( *iter == static_cast<float>(lepton.Pt()) ){
        int tmpIdx = distance(lepPt.begin(), iter);
        if( missET[tmpIdx] == met ) evtIdx = tmpIdx;
        else continue;
      }
    }
    dupCheck.push_back(evtIdx);
    //cout << evtIdx << endl;

    scoreT->GetEntry(evtIdx);
    tmp_score = scoreT->GetLeaf("MLScore")->GetValue(0);
  }

  /////Fill histograms
  int modeArray[2] = {mode, 2};

  for( int MODE : modeArray ){
    for( int syst = 0; syst != syst_num; ++syst ){
      if( syst > 0 and !dosyst ) continue;

      float EventWeight = 1;
      //Multiply syst. to event weight
      if( !option.Contains("Run2017") ){
        EventWeight *= wrongPVrate;
        EventWeight *= *genweight;
        //PUWight
        if      ( isPartOf("puup", std::string(syst_name[syst])) )   EventWeight *= PUWeight[1];
        else if ( isPartOf("pudown", std::string(syst_name[syst])) ) EventWeight *= PUWeight[2];
        else    EventWeight *= PUWeight[0];
        //leptonSF
        if( passmuon ){
          //mu ID: 0, mu Iso: 1, mu Trg: 2
          if      ( isPartOf("__muidup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[1];
          else if ( isPartOf("__muiddown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[2];
          else    EventWeight *= lepton_SF[0];
          if      ( isPartOf("__muisoup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[4];
          else if ( isPartOf("__muisodown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[5];
          else    EventWeight *= lepton_SF[3];
          if      ( isPartOf("__mutrgup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[7];
          else if ( isPartOf("__mutrgdown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[8];
          else    EventWeight *= lepton_SF[6];
          if      ( isPartOf("__elidup", std::string(syst_name[syst])) )   EventWeight *= 1.0;
          else if ( isPartOf("__eliddown", std::string(syst_name[syst])) ) EventWeight *= 1.0;
          else    EventWeight *= 1.0;
          if      ( isPartOf("__elrecoup", std::string(syst_name[syst])) )   EventWeight *= 1.0;
          else if ( isPartOf("__elrecodown", std::string(syst_name[syst])) ) EventWeight *= 1.0;
          else    EventWeight *= 1.0;
          if      ( isPartOf("__elzvtxup", std::string(syst_name[syst])) )   EventWeight *= 1.0;
          else if ( isPartOf("__elzvtxdown", std::string(syst_name[syst])) ) EventWeight *= 1.0;
          else    EventWeight *= 1.0;
        }
        else if( passelectron ){
          //elec ID: 0, elec Reco: 1, elec Zvtx: 2
          if      ( isPartOf("__muidup", std::string(syst_name[syst])) )   EventWeight *= 1.0;
          else if ( isPartOf("__muiddown", std::string(syst_name[syst])) ) EventWeight *= 1.0;
          else    EventWeight *= 1.0;
          if      ( isPartOf("__muisoup", std::string(syst_name[syst])) )   EventWeight *= 1.0;
          else if ( isPartOf("__muisodown", std::string(syst_name[syst])) ) EventWeight *= 1.0;
          else    EventWeight *= 1.0;
          if      ( isPartOf("__mutrgup", std::string(syst_name[syst])) )   EventWeight *= 1.0;
          else if ( isPartOf("__mutrgdown", std::string(syst_name[syst])) ) EventWeight *= 1.0;
          else    EventWeight *= 1.0;
          if      ( isPartOf("__elidup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[1];
          else if ( isPartOf("__eliddown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[2];
          else    EventWeight *= lepton_SF[0];
          if      ( isPartOf("__elrecoup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[4];
          else if ( isPartOf("__elrecodown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[5];
          else    EventWeight *= lepton_SF[3];
          if      ( isPartOf("__elzvtxup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[7];
          else if ( isPartOf("__elzvtxdown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[8];
          else    EventWeight *= lepton_SF[6];
        }
        //ME&PS
        //[0] = muF up , [1] = muF down, [2] = muR up, [3] = muR up && muF up, [4] = muR down, [5] = muF down && muF down
        if( option.Contains("TTpowheg") or option.Contains("TTLL") or option.Contains("TTHad") ){
          //Scale weight
          if      ( isPartOf("scale0", std::string(syst_name[syst])) ) EventWeight *= scaleweight[0];
          else if ( isPartOf("scale1", std::string(syst_name[syst])) ) EventWeight *= scaleweight[1];
          else if ( isPartOf("scale2", std::string(syst_name[syst])) ) EventWeight *= scaleweight[2];
          else if ( isPartOf("scale3", std::string(syst_name[syst])) ) EventWeight *= scaleweight[3];
          else if ( isPartOf("scale4", std::string(syst_name[syst])) ) EventWeight *= scaleweight[4];
          else if ( isPartOf("scale5", std::string(syst_name[syst])) ) EventWeight *= scaleweight[5];
          else    EventWeight *= 1;
          //PS weight
          if      ( isPartOf("ps0", std::string(syst_name[syst])) ) EventWeight *= psweight[0];//isr down
          else if ( isPartOf("ps1", std::string(syst_name[syst])) ) EventWeight *= psweight[1];//fsr down
          else if ( isPartOf("ps2", std::string(syst_name[syst])) ) EventWeight *= psweight[2];//isr up
          else if ( isPartOf("ps3", std::string(syst_name[syst])) ) EventWeight *= psweight[3];//fsr up
          else    EventWeight *= 1;
        }
        else EventWeight *= 1;
        //Deep CSV shape 
        if      ( isPartOf("lfup",        std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[3];
        else if ( isPartOf("lfdown",      std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[4];
        else if ( isPartOf("hfup",        std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[5];
        else if ( isPartOf("hfdown",      std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[6];
        else if ( isPartOf("hfstat1up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[7];
        else if ( isPartOf("hfstat1down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[8];
        else if ( isPartOf("hfstat2up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[9];
        else if ( isPartOf("hfstat2down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[10];
        else if ( isPartOf("lfstat1up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[11];
        else if ( isPartOf("lfstat1down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[12];
        else if ( isPartOf("lfstat2up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[13];
        else if ( isPartOf("lfstat2down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[14];
        else if ( isPartOf("cferr1up",    std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[15];
        else if ( isPartOf("cferr1down",  std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[16];
        else if ( isPartOf("cferr2up",    std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[17];
        else if ( isPartOf("cferr2down",  std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[18];
        else                                                              EventWeight *= jet_SF_deepCSV_30[0];
      }

      h_MVA[MODE][syst]->Fill(tmp_score,EventWeight);

    }//syst loop
  }//mode loop : e or mu + emu
  evtNum++;
  //cout << evtNum << '\r';

  return kTRUE;
}

void MyAnalysis::SlaveTerminate()
{
  TString option = GetOption();
}
   

void MyAnalysis::Terminate()
{
  TString option = GetOption();
  string sample = option.Data();

  dosyst = true;
  if( option.Contains("Run2017") ) dosyst = false;

  const char* score_file = "";
  string train_scheme = sample.substr(sample.find_first_of("-")+1,string::npos);
  syst_ext = "";
  if( train_scheme.find("jec") != string::npos or train_scheme.find("jer") != string::npos
    or train_scheme.find("TuneCP5") != string::npos or train_scheme.find("hdamp") != string::npos ){
    if     ( train_scheme.find("jecup") != string::npos )       syst_ext = "__jecup";
    else if( train_scheme.find("jecdown") != string::npos )     syst_ext = "__jecdown";
    else if( train_scheme.find("jerup") != string::npos )       syst_ext = "__jerup";
    else if( train_scheme.find("jerdown") != string::npos )     syst_ext = "__jerdown";
    else if( train_scheme.find("TuneCP5up") != string::npos )   syst_ext = "__TuneCP5up";
    else if( train_scheme.find("TuneCP5down") != string::npos ) syst_ext = "__TuneCP5down";
    else if( train_scheme.find("hdampup") != string::npos )     syst_ext = "__hdampup";
    else if( train_scheme.find("hdampdown") != string::npos )   syst_ext = "__hdampdown";
  }
  sample.erase( sample.find_first_of("-"),string::npos );
  string syst_str = "-" + train_scheme.substr(train_scheme.find_last_of("_")+1,string::npos );
  if( syst_ext == "" ) syst_str.erase(0,1);
  train_scheme.erase( train_scheme.find_last_of("_"),string::npos );

  score_file = Form("/home/minerva1993/HEPToolsFCNC/analysis_2017/finalMVA/scores/%s%s/score_finalMVA_%s.root",
                     train_scheme.c_str(), syst_str.c_str(), sample.c_str());
  string file_tmp_path = score_file;
  ifstream file_tmp(file_tmp_path);

  if(file_tmp.is_open()){
    //cout << lepcount << endl;
    auto it = unique( dupCheck.begin(), dupCheck.end() );
    if( it != dupCheck.end() ) cout << string(option.Data()) + string(": Duplicate(s)") << endl;
  }
  else cout << "file " + file_tmp_path + " not opened!" << endl;
  //out = TFile::Open(Form("./histos/%s/hist_%s%s.root", train_scheme.c_str(), sample.c_str(), syst_ext.c_str()),"RECREATE");
  out = TFile::Open(Form("./histos/temp/hist_%s%s.root", sample.c_str(), syst_ext.c_str()),"RECREATE");

  TList * l = GetOutputList();
  TIter next(l);
  TObject *object = 0;
  while( ( object = next()) ){
    const char * name = object->GetName();
    std::string str(name);
    if (str.find("h_") !=std::string::npos ){
      object->Write();
    }
  }

  out->Write();
  out->Close();
}

double MyAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

bool MyAnalysis::isPartOf(const std::string& word, const std::string& sentence) {
    return sentence.find(word)    // this returns the index of the first instance
                                  // word
           != std::string::npos;  // which will take this value if it's not found
}

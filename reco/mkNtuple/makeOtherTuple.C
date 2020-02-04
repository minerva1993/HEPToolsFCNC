#define makeOtherTuple_cxx

#include "makeOtherTuple.h"
#include <TH2.h>
#include <TStyle.h>
#include <string>

void makeOtherTuple::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void makeOtherTuple::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();
  string ch = option.Data();
  ch.erase(ch.find_first_of("_"),string::npos);
  string sample = option.Data();
  sample.erase(0,sample.find_first_of("_")+1);

  if( ch.find("jec") != string::npos or ch.find("jer") != string::npos ){
    if     ( ch.find("jecup") != string::npos )       syst_ext = "jecup";
    else if( ch.find("jecdown") != string::npos )     syst_ext = "jecdown";
    else if( ch.find("jerup") != string::npos )       syst_ext = "jerup";
    else if( ch.find("jerdown") != string::npos )     syst_ext = "jerdown";
    else if( ch.find("TuneCP5up") != string::npos )   syst_ext = "Tuneup";
    else if( ch.find("TuneCP5down") != string::npos ) syst_ext = "TuneCP5down";
    else if( ch.find("hdampup") != string::npos )     syst_ext = "hdampup";
    else if( ch.find("hdampdown") != string::npos )   syst_ext = "hdampdown";
  }

  if     ( ch.find("STFCNC") != string::npos ) chBit = 1;
  else if( ch.find("TTFCNC") != string::npos ) chBit = 2;
  else if( ch.find("TTBKG") != string::npos )  chBit = 3;
  else   cout << "Wrong channel name!" << endl;

  //////////////////////////////////////////////////////////////////
  testTree = new TTree("test_tree","background tree");

  testTree->Branch("nevt"         , &b_nevt         , "nevt/I");
  testTree->Branch("file"         , &b_file         , "file/I");
  testTree->Branch("GoodPV"       , &b_GoodPV       , "GoodPV/I");
  testTree->Branch("EventCategory", &b_EventCategory, "EventCategory/I");
  testTree->Branch("genMatch"     , &b_genMatch     , "genMatch/I");
  testTree->Branch("EventWeight"  , &b_EventWeight  , "EventWeight/F");
  
  testTree->Branch("njets"     , &b_njets     , "njets/I");
  testTree->Branch("nbjets_m"  , &b_nbjets_m  , "nbjets_m/I");
  testTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
  testTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  testTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  testTree->Branch("MET"       , &b_met       , "MET/F");
  testTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");

  //jet assignment
  testTree->Branch("jet0pt"  , &b_jet0pt  , "jet0pt/F");
  testTree->Branch("jet0eta" , &b_jet0eta , "jet0eta/F");
  testTree->Branch("jet0phi" , &b_jet0phi , "jet0phi/F");
  testTree->Branch("jet0m"   , &b_jet0m   , "jet0m/F");
  testTree->Branch("jet0csv" , &b_jet0csv , "jet0csv/F");
  testTree->Branch("jet0cvsl", &b_jet0cvsl, "jet0cvsl/F");
  testTree->Branch("jet0cvsb", &b_jet0cvsb, "jet0cvsb/F");
  testTree->Branch("jet0Idx" , &b_jet0Idx , "jet0Idx/I");

  testTree->Branch("jet1pt"  , &b_jet1pt  , "jet1pt/F");
  testTree->Branch("jet1eta" , &b_jet1eta , "jet1eta/F");
  testTree->Branch("jet1phi" , &b_jet1phi , "jet1phi/F");
  testTree->Branch("jet1m"   , &b_jet1m   , "jet1m/F");
  testTree->Branch("jet1csv" , &b_jet1csv , "jet1csv/F");
  testTree->Branch("jet1cvsl", &b_jet1cvsl, "jet1cvsl/F");
  testTree->Branch("jet1cvsb", &b_jet1cvsb, "jet1cvsb/F");
  testTree->Branch("jet1Idx" , &b_jet1Idx , "jet1Idx/I");

  testTree->Branch("jet2pt"  , &b_jet2pt  , "jet2pt/F");
  testTree->Branch("jet2eta" , &b_jet2eta , "jet2eta/F");
  testTree->Branch("jet2phi" , &b_jet2phi , "jet2phi/F");
  testTree->Branch("jet2m"   , &b_jet2m   , "jet2m/F");
  testTree->Branch("jet2csv" , &b_jet2csv , "jet2csv/F");
  testTree->Branch("jet2cvsl", &b_jet2cvsl, "jet2cvsl/F");
  testTree->Branch("jet2cvsb", &b_jet2cvsb, "jet2cvsb/F");
  testTree->Branch("jet2Idx" , &b_jet2Idx , "jet2Idx/I");

  testTree->Branch("jet3pt"  , &b_jet3pt  , "jet3pt/F");
  testTree->Branch("jet3eta" , &b_jet3eta , "jet3eta/F");
  testTree->Branch("jet3phi" , &b_jet3phi , "jet3phi/F");
  testTree->Branch("jet3m"   , &b_jet3m   , "jet3m/F");
  testTree->Branch("jet3csv" , &b_jet3csv , "jet3csv/F");
  testTree->Branch("jet3cvsl", &b_jet3cvsl, "jet3cvsl/F");
  testTree->Branch("jet3cvsb", &b_jet3cvsb, "jet3cvsb/F");
  testTree->Branch("jet3Idx" , &b_jet3Idx , "jet3Idx/I");

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
  testTree->Branch("lepWm"   , &b_lepWmt  , "lepWm/F");//Should be transverse M

  testTree->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
  testTree->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
  testTree->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
  testTree->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
  testTree->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
  testTree->Branch("lepTdR"  , &b_lepTdR  , "lepTdR/F");
  testTree->Branch("lepTm"   , &b_lepTmt  , "lepTm/F");//Should be transverse M

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

  testTree->Branch("jet0lepdR"    , &b_jet0lepdR    , "jet0lepdR/F");
  testTree->Branch("jet1lepdR"    , &b_jet1lepdR    , "jet1lepdR/F");
  testTree->Branch("jet2lepdR"    , &b_jet2lepdR    , "jet2lepdR/F");
  testTree->Branch("jet3lepdR"    , &b_jet3lepdR    , "jet3lepdR/F");
  testTree->Branch("jet01dR"      , &b_jet01dR      , "jet01dR/F");
  testTree->Branch("jet02dR"      , &b_jet02dR      , "jet02dR/F");
  testTree->Branch("jet03dR"      , &b_jet03dR      , "jet03dR/F");
  testTree->Branch("jet12_lepdR"  , &b_jet12_lepdR  , "jet12_lepdR/F");
  testTree->Branch("jet23_lepdR"  , &b_jet23_lepdR  , "jet23_lepdR/F");
  testTree->Branch("jet31_lepdR"  , &b_jet31_lepdR  , "jet31_lepdR/F");
  testTree->Branch("jet12_0dR"    , &b_jet12_0dR    , "jet12_0dR/F");
  testTree->Branch("jet23_0dR"    , &b_jet23_0dR    , "jet23_0dR/F");
  testTree->Branch("jet31_0dR"    , &b_jet31_0dR    , "jet31_0dR/F");
  testTree->Branch("lepTjet12dphi", &b_lepTjet12dphi, "lepTjet12dphi/F");
  testTree->Branch("lepTjet23dphi", &b_lepTjet23dphi, "lepTjet23dphi/F");
  testTree->Branch("lepTjet31dphi", &b_lepTjet31dphi, "lepTjet31dphi/F");
  testTree->Branch("hadT_jet0dR"  , &b_hadT_jet0dR  , "hadT_jet0dR/F");

  fOutput->Add(testTree);
} 

Bool_t makeOtherTuple::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  int era = 0;
  TString current_file_name = makeOtherTuple::fReader.GetTree()->GetCurrentFile()->GetName();
  if     ( current_file_name.Contains("V9") ) era = 2017;
  else if( current_file_name.Contains("V10") ) era = 2018;

  //Ntuple for ST, TT respectively
  string ch = option.Data();
  ch.erase(ch.find_first_of("_"),string::npos);

  int mode = 999; 
  mode = *channel;

  if( mode > 2 ) return kTRUE;

  float EventWeight = 1.0;
  float wrongPVrate = 1;
  if( !option.Contains("Run201") ){
    EventWeight *= lepton_SF[0]*lepton_SF[3]*lepton_SF[6];
    if( mode == 1 ) EventWeight *= lepton_SF[9];
    EventWeight *= *genweight;
    EventWeight *= PUWeight[0];
    EventWeight *= jet_SF_deepCSV_30[0];
  }
  if( era == 2017 ){
    if( !option.Contains("Run2017") ){
      if     ( option.Contains("DYJets10to50") ) wrongPVrate = 1.02921;
      else if( option.Contains("QCDEM15to20") ) wrongPVrate = 1.01333;
      else if( option.Contains("QCDEM20to30") ) wrongPVrate = 1.01227;
      else if( option.Contains("QCDEM300toInf") ) wrongPVrate = 1.01194;
      else if( option.Contains("QCDEM50to80") ) wrongPVrate = 1.02226;
      else if( option.Contains("QCDMu120to170") ) wrongPVrate = 1.01289;
      else if( option.Contains("QCDMu170to300") ) wrongPVrate = 1.01181;
      else if( option.Contains("QCDMu20to30") ) wrongPVrate = 1.0253;
      else if( option.Contains("QCDMu30to50") ) wrongPVrate = 1.02105;
      else if( option.Contains("QCDMu470to600") ) wrongPVrate = 1.0141;
      else if( option.Contains("QCDMu50to80") ) wrongPVrate = 1.01149;
      else if( option.Contains("QCDMu80to120") ) wrongPVrate = 1.01278;
      else if( option.Contains("TTLLpowhegttbbhdampup") ) wrongPVrate = 1.01807;
      else if( option.Contains("TTLLpowhegttcchdampup") ) wrongPVrate = 1.01978;
      else if( option.Contains("TTLLpowhegttlfhdampup") ) wrongPVrate = 1.01938;
      else if( option.Contains("TTZToLLNuNu") ) wrongPVrate = 1.02425;
      else if( option.Contains("TTpowhegttbbTuneCP5down") ) wrongPVrate = 1.02715;
      else if( option.Contains("TTpowhegttbbhdampdown") ) wrongPVrate = 1.02717;
      else if( option.Contains("TTpowhegttccTuneCP5down") ) wrongPVrate = 1.0273;
      else if( option.Contains("TTpowhegttcchdampdown") ) wrongPVrate = 1.02746;
      else if( option.Contains("TTpowhegttlfTuneCP5down") ) wrongPVrate = 1.02742;
      else if( option.Contains("TTpowhegttlfhdampdown") ) wrongPVrate = 1.02774;
      else if( option.Contains("W3JetsToLNu") ) wrongPVrate = 1.02348;
      else if( option.Contains("WW") ) wrongPVrate = 1.0295;
      else if( option.Contains("WZ") ) wrongPVrate = 1.02298;
      else if( option.Contains("ZZ") ) wrongPVrate = 1.01508;
      else   wrongPVrate = 1.0;
    }
    if( wrongPVrate > 1.01 ){
      if( *TruePV <=0 ) return kTRUE;
    }
  }

  float relIso = *lepton_relIso; 

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 
  int ncjets_m = 0; 

  TLorentzVector p4met;
  float met = *MET;
  float met_phi = *MET_phi;
  float apt = TMath::Abs(met);
  float met_x = apt*TMath::Cos(met_phi);
  float met_y = apt*TMath::Sin(met_phi);
  //p4met.SetPxPyPzE( met_x, met_y, 0, met);

  TLorentzVector lepton;
  lepton.SetPtEtaPhiE(*lepton_pt, *lepton_eta, *lepton_phi, *lepton_e);
  //lepton = lepton*(lepton_scale[0]);

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);
  bool passelectron = (mode == 1) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.4);

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

  b_lepton_pt = lepton.Pt(); b_lepton_eta = lepton.Eta(); b_lepton_phi = lepton.Phi();

  float bWP_M, bWP_T, cvsbWP_M, cvslWP_M;
  if      ( era == 2017 ) bWP_M = 0.4941;
  else if ( era == 2018 ) bWP_M = 0.4184;
  else                    bWP_M = 0.0;

  vector<size_t> jetIdxs;
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

    if( jet.Pt() > 30 && abs(jet.Eta()) <= 2.4 ){
      njets++;
      jetIdxs.push_back(iJet);

      if( jet_deepCSV[iJet] > bWP_M ) nbjets_m++;
    }
  }

  if( !option.Contains("Run201") ){
    if( syst_ext == "jecup" ){
      met_x = MET_unc_x[0]; met_y = MET_unc_y[0];
    }
    else if( syst_ext == "jecdown" ){
      met_x = MET_unc_x[1]; met_y = MET_unc_y[1];
    }
    else if( syst_ext == "jerup" ){
      met_x = MET_unc_x[2]; met_y = MET_unc_y[2];
    }
    else if( syst_ext == "jerdown" ){
      met_x = MET_unc_x[3]; met_y = MET_unc_y[3];
    }
  }
  p4met.SetPxPyPzE(met_x, met_y, 0, sqrt(met_x*met_x + met_y*met_y));

  //Selection Option
  float transverseM = transverseMass(lepton, p4met);
  float lepDphi = lepton.DeltaPhi(p4met);
  //bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  //bool makeIso = true;
  //bool isIso = *lepton_isIso;
  //if( makeIso && !isIso ) return kTRUE;
  //if( !makeIso && isIso ) return kTRUE;

  if( (chBit == 2 or chBit == 3 ) and (njets <  4 or nbjets_m < 2) ) return kTRUE;
  else if( chBit == 1 and (njets <  3 or nbjets_m < 2) ) return kTRUE;

  //cout << nevt << endl;
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

  //gen particles
  TLorentzVector gen_lep, gen_nu, gen_lepB, gen_hadJ1, gen_hadJ2, gen_hadJ3;
  b_genHadW = (gen_hadJ2+gen_hadJ3).M();


  //fill trees
  b_EventWeight = EventWeight;
  b_GoodPV = *GoodPV;
  b_nevt = nevt;

  b_njets = njets;
  b_nbjets_m = nbjets_m;

  b_met = p4met.Pt();
  b_met_phi = p4met.Phi();

  //jet assignments
  TLorentzVector lepW = lepton + p4met;
  b_lepWpt    = lepW.Pt();
  b_lepWeta   = lepW.Eta();
  b_lepWphi   = lepW.Phi();
  b_lepWdphi  = lepDphi;
  b_lepWmt    = transverseM;

  //int count = 0;
  TLorentzVector jetP4[4];
  for( auto ii0 = jetIdxs.begin(); ii0 != jetIdxs.end(); ++ii0 ){
    if( chBit == 3 and jet_deepCSV[*ii0] < bWP_M ) continue; //for ttbar reco signal
    jetP4[0].SetPtEtaPhiE(jet_pt[*ii0], jet_eta[*ii0], jet_phi[*ii0], jet_e[*ii0]);

    for( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ){
      if( (chBit == 2 or chBit == 3) and *ii1 == *ii0 ) continue;
      if( chBit == 3 and jet_deepCSV[*ii1] < bWP_M ) continue; //for ttbar reco signal
      if( chBit == 2 or chBit == 3 ) jetP4[3].SetPtEtaPhiE(jet_pt[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_e[*ii1]);

      for( auto ii2 = jetIdxs.begin(); ii2 != jetIdxs.end(); ++ii2 ){
        if( *ii2 == *ii0 ) continue;
        if( (chBit == 2 or chBit == 3) and *ii2 == *ii1 ) continue;
        if( (chBit == 1 or chBit == 2) and jet_deepCSV[*ii2] < bWP_M ) continue;

        for( auto ii3 = ii2+1; ii3 != jetIdxs.end(); ++ii3 ){
          if( *ii3 == *ii0 or *ii3 == *ii2 ) continue;
          if( (chBit == 2 or chBit == 3) and *ii3 == *ii1 ) continue;
          if( (chBit == 1 or chBit == 2) and nbjets_m > 2 and jet_deepCSV[*ii3] < bWP_M ) continue;
          if( jet_deepCSV[*ii2] > jet_deepCSV[*ii3] ){
            jetP4[1].SetPtEtaPhiE(jet_pt[*ii2], jet_eta[*ii2], jet_phi[*ii2], jet_e[*ii2]);
            jetP4[2].SetPtEtaPhiE(jet_pt[*ii3], jet_eta[*ii3], jet_phi[*ii3], jet_e[*ii3]);
          }
          else{
            jetP4[1].SetPtEtaPhiE(jet_pt[*ii3], jet_eta[*ii3], jet_phi[*ii3], jet_e[*ii3]);
            jetP4[2].SetPtEtaPhiE(jet_pt[*ii2], jet_eta[*ii2], jet_phi[*ii2], jet_e[*ii2]);
          }
          //count++;

          if( chBit == 1 and *ii1 != 0 ) continue;

          int tmp_idx[4];
          tmp_idx[0] = *ii0; tmp_idx[3] = *ii1;
          if( jet_deepCSV[*ii2] > jet_deepCSV[*ii3] ){
            tmp_idx[1] = *ii2;
            tmp_idx[2] = *ii3;
          }
          else{
            tmp_idx[1] = *ii3;
            tmp_idx[2] = *ii2;
          }

          //construct particles: lepB = j0, hadB = j3, hadW = j1+j2
          TLorentzVector jetP4cor[4];
          if( !option.Contains("Run201") ){
            if     ( syst_ext == "jecup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[tmp_idx[i]] * jet_JES_Up[tmp_idx[i]];
            else if( syst_ext == "jecdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[tmp_idx[i]] * jet_JES_Down[tmp_idx[i]];
            else if( syst_ext == "jerup" )   for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Up[tmp_idx[i]];
            else if( syst_ext == "jerdown" ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Down[tmp_idx[i]];
            else                             for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i] * jet_JER_Nom[tmp_idx[i]];
          }
          else if( option.Contains("Run201") ) for( int i=0; i < 4; i++) jetP4cor[i] = jetP4[i];

          b_jet0pt = jetP4cor[0].Pt(); b_jet0eta = jetP4cor[0].Eta(); b_jet0phi = jetP4cor[0].Phi(); b_jet0m = jetP4cor[0].M();
          b_jet1pt = jetP4cor[1].Pt(); b_jet1eta = jetP4cor[1].Eta(); b_jet1phi = jetP4cor[1].Phi(); b_jet1m = jetP4cor[1].M();
          b_jet2pt = jetP4cor[2].Pt(); b_jet2eta = jetP4cor[2].Eta(); b_jet2phi = jetP4cor[2].Phi(); b_jet2m = jetP4cor[2].M();
          b_jet3pt = jetP4cor[3].Pt(); b_jet3eta = jetP4cor[3].Eta(); b_jet3phi = jetP4cor[3].Phi(); b_jet3m = jetP4cor[3].M();

          b_jet0csv = jet_deepCSV[tmp_idx[0]]; b_jet0cvsl = jet_deepCvsL[tmp_idx[0]]; b_jet0cvsb = jet_deepCvsB[tmp_idx[0]];
          b_jet1csv = jet_deepCSV[tmp_idx[1]]; b_jet1cvsl = jet_deepCvsL[tmp_idx[1]]; b_jet1cvsb = jet_deepCvsB[tmp_idx[1]];
          b_jet2csv = jet_deepCSV[tmp_idx[2]]; b_jet2cvsl = jet_deepCvsL[tmp_idx[2]]; b_jet2cvsb = jet_deepCvsB[tmp_idx[2]];
          b_jet3csv = jet_deepCSV[tmp_idx[3]]; b_jet3cvsl = jet_deepCvsL[tmp_idx[3]]; b_jet3cvsb = jet_deepCvsB[tmp_idx[3]];
          b_jet0Idx = tmp_idx[0]; b_jet1Idx = tmp_idx[1]; b_jet2Idx = tmp_idx[2]; b_jet3Idx = tmp_idx[3];

          if( b_jet0csv < 0 ) b_jet0csv = 0; if( b_jet1csv < 0 ) b_jet1csv = 0;
          if( b_jet2csv < 0 ) b_jet2csv = 0; if( b_jet3csv < 0 ) b_jet3csv = 0;

          const auto lepT = lepW + jetP4cor[0];
          const auto had12 = jetP4cor[1] + jetP4cor[2];//This is W or H
          const auto had23 = jetP4cor[2] + jetP4cor[3];
          const auto had31 = jetP4cor[3] + jetP4cor[1];
          const auto hadT = jetP4cor[1] + jetP4cor[2] + jetP4cor[3];

          b_jet12pt = had12.Pt(); b_jet12eta = had12.Eta(); b_jet12phi = had12.Phi(); b_jet12m = had12.M();
          b_jet23pt = had23.Pt(); b_jet23eta = had23.Eta(); b_jet23phi = had23.Phi(); b_jet23m = had23.M();
          b_jet31pt = had31.Pt(); b_jet31eta = had31.Eta(); b_jet31phi = had31.Phi(); b_jet31m = had31.M();
          b_jet12deta = (jetP4cor[1]-jetP4cor[2]).Eta(); b_jet12dphi = jetP4cor[1].DeltaPhi(jetP4cor[2]);
          b_jet23deta = (jetP4cor[2]-jetP4cor[3]).Eta(); b_jet23dphi = jetP4cor[2].DeltaPhi(jetP4cor[3]);
          b_jet31deta = (jetP4cor[3]-jetP4cor[1]).Eta(); b_jet31dphi = jetP4cor[3].DeltaPhi(jetP4cor[1]);
          b_jet12dR = jetP4cor[1].DeltaR(jetP4cor[2]); b_jet23dR = jetP4cor[2].DeltaR(jetP4cor[3]); b_jet31dR = jetP4cor[3].DeltaR(jetP4cor[1]);

          b_lepTpt = lepT.Pt(); b_lepTeta = lepT.Eta(); b_lepTphi = lepT.Phi(); b_lepTmt = transverseMass(lepton+jetP4cor[0],p4met);
          b_lepTdeta = (lepW-jetP4cor[0]).Eta(); b_lepTdphi = lepW.DeltaPhi(jetP4cor[0]); b_lepTdR = lepW.DeltaPhi(jetP4cor[0]);

          b_hadTpt = hadT.Pt(); b_hadTeta = hadT.Eta(); b_hadTphi = hadT.Phi(); b_hadTm = hadT.M();
          b_hadT12_3deta = (had12-jetP4cor[3]).Eta(); b_hadT23_1deta = (had23-jetP4cor[1]).Eta(); b_hadT31_2deta = (had31-jetP4cor[2]).Eta();
          b_hadT12_3dphi = had12.DeltaPhi(jetP4cor[3]); b_hadT23_1dphi = had23.DeltaPhi(jetP4cor[1]); b_hadT31_2dphi = had31.DeltaPhi(jetP4cor[2]);
          b_hadT12_3dR = had12.DeltaR(jetP4cor[3]); b_hadT23_1dR = had23.DeltaR(jetP4cor[1]); b_hadT31_2dR = had31.DeltaR(jetP4cor[2]);

          //cross variables
          b_jet0lepdR = jetP4cor[0].DeltaR(lepton); b_jet1lepdR = jetP4cor[1].DeltaR(lepton);
          b_jet2lepdR = jetP4cor[2].DeltaR(lepton); b_jet3lepdR = jetP4cor[3].DeltaR(lepton);
          b_jet01dR = jetP4cor[0].DeltaR(jetP4cor[1]); b_jet02dR = jetP4cor[0].DeltaR(jetP4cor[2]); b_jet03dR = jetP4cor[0].DeltaR(jetP4cor[3]);
          b_jet12_lepdR = lepton.DeltaR(had12); b_jet23_lepdR = lepton.DeltaR(had23); b_jet31_lepdR = lepton.DeltaR(had31);
          b_jet12_0dR = jetP4cor[0].DeltaR(had12); b_jet23_0dR = jetP4cor[0].DeltaR(had23); b_jet31_0dR = jetP4cor[0].DeltaR(had31);
          b_lepTjet12dphi = lepT.DeltaPhi(had12); b_lepTjet23dphi = lepT.DeltaPhi(had23); b_lepTjet31dphi = lepT.DeltaPhi(had31);
          b_hadT_jet0dR = hadT.DeltaR(jetP4cor[0]);

          testTree->Fill();

        }
      }
    }
  }
  nevt++;
  return kTRUE;
}

void makeOtherTuple::SlaveTerminate()
{
  TString option = GetOption();
}
   

void makeOtherTuple::Terminate()
{
  TString option = GetOption();
  string ch = option.Data();
  ch.erase(ch.find_first_of("_"),string::npos);
  string era = ch.substr(0,4);
  ch.erase(0,4);
  string sample = option.Data();
  sample.erase(0,sample.find_first_of("_")+1);

  //TFile *hfile = TFile::Open(Form("%s/root_%s/deepReco_%s.root", era.c_str(), ch.c_str(),sample.c_str()), "RECREATE");
  TFile *hfile = TFile::Open(Form("/data1/users/minerva1993/work/fcnc_RunII%s/reco/current_ver/root_%s/deepReco_%s.root", era.c_str(), ch.c_str(),sample.c_str()), "RECREATE");

  fOutput->FindObject("test_tree")->Write();

  hfile->Write();
  hfile->Close();

}

float makeOtherTuple::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  float out = sumT.M();

  return out;

}

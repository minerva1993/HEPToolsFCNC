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
    if     ( ch.find("jecup") != string::npos )   syst_ext = "jecup";
    else if( ch.find("jecdown") != string::npos ) syst_ext = "jecdown";
    else if( ch.find("jerup") != string::npos )   syst_ext = "jerup";
    else if( ch.find("jerdown") != string::npos ) syst_ext = "jerdown";
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
  testTree->Branch("trigger"      , &b_trigger     , "trigger/I");
  
  testTree->Branch("njets"     , &b_njets     , "njets/I");
  testTree->Branch("nbjets_m"  , &b_nbjets_m  , "nbjets_m/I");
  testTree->Branch("lepton_pt" , &b_lepton_pt , "lepton_pt/F");
  testTree->Branch("lepton_eta", &b_lepton_eta, "lepton_eta/F");
  testTree->Branch("lepton_phi", &b_lepton_phi, "lepton_phi/F");
  testTree->Branch("MET"       , &b_met       , "MET/F");
  testTree->Branch("MET_phi"   , &b_met_phi   , "MET_phi/F");
  testTree->Branch("lepDPhi"   , &b_lepdphi   , "lepDPhi/F");

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

  fOutput->Add(testTree);
} 

Bool_t makeOtherTuple::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  //Ntuple for ST, TT respectively
  string ch = option.Data();
  ch.erase(ch.find_first_of("_"),string::npos);

  int mode = 999; 
  mode = *channel;

  if( mode > 2 ) return kTRUE;

  float EventWeight = 1.0;
  float wrongPVrate = 1;
  if( !option.Contains("Run2017") ){
    EventWeight *= lepton_SF[0];
    EventWeight *= *genweight;
    EventWeight *= PUWeight[0];

    if      ( option.Contains("DYJets4to50HT100to200v2") ) wrongPVrate = 1.04010725093;
    else if ( option.Contains("DYJets4to50HT100to200_") or option.Contains("DYJets4to50HT100to200part2") ) wrongPVrate = 1.04010725093;
    else if ( option.Contains("DYJets4to50HT200to400") ) wrongPVrate = 1.0354286358;
    else if ( option.Contains("DYJets4to50HT400to600") ) wrongPVrate = 1.0395588053;
    else if ( option.Contains("DYJets4to50HT70to100") ) wrongPVrate = 1.04495754249;
    else if ( option.Contains("DYJetsv2") ) wrongPVrate = 1.04239771686;
    else if ( option.Contains("DYJets_") or option.Contains("DYJetspart2") ) wrongPVrate = 1.04239771686;
    else if ( option.Contains("STTH1L3BHct") ) wrongPVrate = 1.04397325747;
    else if ( option.Contains("STTH1L3BHut") ) wrongPVrate = 1.03100541427;
    else if ( option.Contains("SingleTbart") ) wrongPVrate = 1.04171033265;
    else if ( option.Contains("SingleTbartW") ) wrongPVrate = 1.04261195626;
    else if ( option.Contains("SingleTops") ) wrongPVrate = 1.03972787611;
    else if ( option.Contains("SingleTopt") ) wrongPVrate = 1.04028244802;
    else if ( option.Contains("SingleToptW") ) wrongPVrate = 1.04478300674;
    else if ( option.Contains("TTHadpowheg") ) wrongPVrate = 1.02911490683;
    else if ( option.Contains("TTLLpowheg") ) wrongPVrate = 1.04707001305;
    else if ( option.Contains("TTWJetsToLNuPSweight") ) wrongPVrate = 1.04084725366;
    else if ( option.Contains("TTWJetsToQQ") ) wrongPVrate = 1.02412815424;
    else if ( option.Contains("TTZToLLNuNu") ) wrongPVrate = 1.04242525748;
    else if ( option.Contains("TTZToQQ") ) wrongPVrate = 1.04903621775;
    else if ( option.Contains("TTpowhegttbb") ) wrongPVrate = 1.04855161159;
    else if ( option.Contains("TTpowhegttbj") ) wrongPVrate = 1.0481276857;
    else if ( option.Contains("TTpowhegttcc") ) wrongPVrate = 1.04745462065;
    else if ( option.Contains("TTpowhegttlf") ) wrongPVrate = 1.04754742837;
    else if ( option.Contains("TTpowhegttother") ) wrongPVrate = 1.04775924349;
    else if ( option.Contains("W3JetsToLNu") ) wrongPVrate = 1.04151096637;
    else if ( option.Contains("W4JetsToLNu") ) wrongPVrate = 1.04351715127;
    else if ( option.Contains("WW") ) wrongPVrate = 1.04691597274;
    else if ( option.Contains("WZ") ) wrongPVrate = 1.04251110292;
    else if ( option.Contains("ZZ") ) wrongPVrate = 1.02960069789;
    else if ( option.Contains("ttHToNonbb") ) wrongPVrate = 1.03169876149;
    else if ( option.Contains("ttHTobb") ) wrongPVrate = 1.03747398495;
    else    wrongPVrate = 1.0;

    EventWeight *= wrongPVrate;
  }
  if( wrongPVrate > 1.01 ){
    if( *TruePV < 10 || *TruePV > 75 ) return kTRUE;
  }

  float relIso = *lepton_relIso; 

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 
  int ncjets_m = 0; 

  TLorentzVector metP4;
  double met = *MET;
  double met_phi = *MET_phi;
  double apt = TMath::Abs(met);
  double met_x =  apt*TMath::Cos(met_phi);
  double met_y =  apt*TMath::Sin(met_phi);
  metP4.SetPxPyPzE( met_x, met_y, 0, met);

  TLorentzVector lepton;
  lepton.SetPtEtaPhiE(*lepton_pt, *lepton_eta, *lepton_phi, *lepton_e);

  double transverseM = transverseMass(lepton, metP4);
  double lepDphi = lepton.DeltaPhi(metP4);

  //Selection Option
  bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
  bool passelectron;
  if ( *elec_trg == 10 ) passelectron = (mode == 1) && (lepton.Pt() > 33) && (abs(lepton.Eta()) <= 2.1);
  else                   passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

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

  if( passelectron && *elec_trg == 10 ) EventWeight *= jet_SF_deepCSV_38[0];
  else                                  EventWeight *= jet_SF_deepCSV_30[0];

  b_lepton_pt = lepton.Pt(); b_lepton_eta = lepton.Eta(); b_lepton_phi = lepton.Phi();
  b_trigger = *elec_trg;

  vector<size_t> jetIdxs;
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

    if( jet.Pt() > 30 && abs(jet.Eta()) <= 2.4 ){
      if( passelectron and *elec_trg == 10 and njets == 0 and jet_pt[iJet] < 38 ) continue;
      njets++;
      jetIdxs.push_back(iJet);

      if( jet_deepCSV[iJet] > 0.4941 ){
        nbjets_m++;
      }
    }
  }

  if( (chBit == 2 or chBit == 3 ) and (njets <  4 or nbjets_m < 2) ) return kTRUE;
  else if( chBit == 1 and (njets <  3 or nbjets_m < 2) ) return kTRUE;

  //cout << nevt << endl;
  if( option.Contains("Run2017") ) b_EventCategory = -1;
  else if( option.Contains("Hct") || option.Contains("Hut") ) b_EventCategory = 0;
  else if( option.Contains("ttbb") ) b_EventCategory = 1;
  else if( option.Contains("ttbj") ) b_EventCategory = 2;
  else if( option.Contains("ttcc") ) b_EventCategory = 3;
  else if( option.Contains("ttLF") ) b_EventCategory = 4;
  else if( option.Contains("ttother") ) b_EventCategory = 5;
  else if( option.Contains("SingleT") ) b_EventCategory = 6; //singletop
  else if( option.Contains("TTZ") or option.Contains("TTW") or option.Contains("ttH")) b_EventCategory = 7; //VV
  else if( option.Contains("DY") ) b_EventCategory = 8;
  else if( option.Contains("W1Jets") or option.Contains("W2Jets") or option.Contains("W3Jets") or option.Contains("W4Jets") ) b_EventCategory = 9;
  else if( option.Contains("WW") or option.Contains("WZ") or option.Contains("ZZ") ) b_EventCategory = 10;
  else b_EventCategory = 20;

  //gen particles
  TLorentzVector gen_lep, gen_nu, gen_lepB, gen_hadJ1, gen_hadJ2, gen_hadJ3;
  b_genHadW = (gen_hadJ2+gen_hadJ3).M();


  //fill trees
  b_EventWeight = EventWeight;
  b_GoodPV = *GoodPV;
  b_nevt = nevt;

  b_njets = njets;
  b_nbjets_m = nbjets_m;

  b_met = met;
  b_met_phi = met_phi;
  b_lepdphi = lepDphi;
  b_transversem = transverseMass(lepton, metP4);


  //jet assignments
  TLorentzVector lepW = lepton + metP4;
  b_lepWpt    = lepW.Pt();
  b_lepWeta   = lepW.Eta();
  b_lepWphi   = lepW.Phi();
  b_lepWdphi  = lepton.DeltaPhi(metP4);
  b_lepWm     = lepW.M();

  //int count = 0;
  TLorentzVector jetP4[4];
  for( auto ii0 = jetIdxs.begin(); ii0 != jetIdxs.end(); ++ii0 ){
    if( chBit == 3 and jet_deepCSV[*ii0] < 0.4941 ) continue; //for ttbar reco signal
    jetP4[0].SetPtEtaPhiE(jet_pt[*ii0], jet_eta[*ii0], jet_phi[*ii0], jet_e[*ii0]);

    for( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ){
      if( (chBit == 2 or chBit == 3) and *ii1 == *ii0 ) continue;
      if( chBit == 3 and jet_deepCSV[*ii1] < 0.4941 ) continue; //for ttbar reco signal
      if( chBit == 2 or chBit == 3 ) jetP4[3].SetPtEtaPhiE(jet_pt[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_e[*ii1]);

      for( auto ii2 = jetIdxs.begin(); ii2 != jetIdxs.end(); ++ii2 ){
        if( *ii2 == *ii0 ) continue;
        if( (chBit == 2 or chBit == 3) and *ii2 == *ii1 ) continue;
        if( (chBit == 1 or chBit == 2) and jet_deepCSV[*ii2] < 0.4941 ) continue;

        for( auto ii3 = ii2+1; ii3 != jetIdxs.end(); ++ii3 ){
          if( *ii3 == *ii0 or *ii3 == *ii2 ) continue;
          if( (chBit == 2 or chBit == 3) and *ii3 == *ii1 ) continue;
          if( (chBit == 1 or chBit == 2) and jet_deepCSV[*ii3] < 0.4941 ) continue;
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
          if( !option.Contains("Run2017") ){
            if     ( syst_ext == "jecup" )   for( int i=0; i < 4; i++) jetP4[i] = jetP4[i] * jet_JER_Nom[tmp_idx[i]] * jet_JES_Up[tmp_idx[i]];
            else if( syst_ext == "jecdown" ) for( int i=0; i < 4; i++) jetP4[i] = jetP4[i] * jet_JER_Nom[tmp_idx[i]] * jet_JES_Down[tmp_idx[i]];
            else if( syst_ext == "jerup" )   for( int i=0; i < 4; i++) jetP4[i] = jetP4[i] * jet_JER_Up[tmp_idx[i]];
            else if( syst_ext == "jerdown" ) for( int i=0; i < 4; i++) jetP4[i] = jetP4[i] * jet_JER_Down[tmp_idx[i]];
            else                             for( int i=0; i < 4; i++) jetP4[i] = jetP4[i] * jet_JER_Nom[tmp_idx[i]];
          }

          b_jet0pt = jetP4[0].Pt(); b_jet0eta = jetP4[0].Eta(); b_jet0phi = jetP4[0].Phi(); b_jet0m = jetP4[0].M();
          b_jet1pt = jetP4[1].Pt(); b_jet1eta = jetP4[1].Eta(); b_jet1phi = jetP4[1].Phi(); b_jet1m = jetP4[1].M();
          b_jet2pt = jetP4[2].Pt(); b_jet2eta = jetP4[2].Eta(); b_jet2phi = jetP4[2].Phi(); b_jet2m = jetP4[2].M();
          b_jet3pt = jetP4[3].Pt(); b_jet3eta = jetP4[3].Eta(); b_jet3phi = jetP4[3].Phi(); b_jet3m = jetP4[3].M();

          b_jet0csv = jet_deepCSV[tmp_idx[0]]; b_jet0cvsl = jet_deepCvsL[tmp_idx[0]]; b_jet0cvsb = jet_deepCvsB[tmp_idx[0]];
          b_jet1csv = jet_deepCSV[tmp_idx[1]]; b_jet1cvsl = jet_deepCvsL[tmp_idx[1]]; b_jet1cvsb = jet_deepCvsB[tmp_idx[1]];
          b_jet2csv = jet_deepCSV[tmp_idx[2]]; b_jet2cvsl = jet_deepCvsL[tmp_idx[2]]; b_jet2cvsb = jet_deepCvsB[tmp_idx[2]];
          b_jet3csv = jet_deepCSV[tmp_idx[3]]; b_jet3cvsl = jet_deepCvsL[tmp_idx[3]]; b_jet3cvsb = jet_deepCvsB[tmp_idx[3]];
          b_jet0Idx = tmp_idx[0]; b_jet1Idx = tmp_idx[1]; b_jet2Idx = tmp_idx[2]; b_jet3Idx = tmp_idx[3];

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
          b_lepTdeta = (lepW-jetP4[0]).Eta(); b_lepTdphi = lepW.DeltaPhi(jetP4[0]); b_lepTdR = lepW.DeltaPhi(jetP4[0]);

          b_hadTpt = hadT.Pt(); b_hadTeta = hadT.Eta(); b_hadTphi = hadT.Phi(); b_hadTm = hadT.M();
          b_hadT12_3deta = (had12-jetP4[3]).Eta(); b_hadT23_1deta = (had23-jetP4[1]).Eta(); b_hadT31_2deta = (had31-jetP4[2]).Eta();
          b_hadT12_3dphi = had12.DeltaPhi(jetP4[3]); b_hadT23_1dphi = had23.DeltaPhi(jetP4[1]); b_hadT31_2dphi = had31.DeltaPhi(jetP4[2]);
          b_hadT12_3dR = had12.DeltaR(jetP4[3]); b_hadT23_1dR = had23.DeltaR(jetP4[1]); b_hadT31_2dR = had31.DeltaR(jetP4[2]);

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
  string sample = option.Data();
  sample.erase(0,sample.find_first_of("_")+1);

  TFile *hfile = TFile::Open(Form("root_%s/deepReco_%s.root",ch.c_str(),sample.c_str()), "RECREATE");

  fOutput->FindObject("test_tree")->Write();

  hfile->Write();
  hfile->Close();

}

double makeOtherTuple::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

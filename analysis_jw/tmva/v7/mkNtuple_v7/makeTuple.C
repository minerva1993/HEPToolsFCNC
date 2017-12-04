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

    treeTMVA = new TTree("tmva_tree","tree for tmva");

    treeTMVA->Branch("nevt", &b_nevt, "nevt/I");
    treeTMVA->Branch("GoodPV", &b_GoodPV, "b_GoodPV/I");
    treeTMVA->Branch("EventCategory", &b_EventCategory, "b_EventCategory/I");
    treeTMVA->Branch("GenMatch", &b_GenMatch, "b_GenMatch/I");
    treeTMVA->Branch("EventWeight", &b_EventWeight, "b_EventWeight/F");

    treeTMVA->Branch("njets", &b_njets, "b_njets/I");
    treeTMVA->Branch("nbjets_m", &b_nbjets_m, "b_nbjets_m/I");
    treeTMVA->Branch("ncjets_m", &b_ncjets_m, "b_ncjets_m/I");
    treeTMVA->Branch("transverseMass", &b_transversem, "b_transversem/F");
    treeTMVA->Branch("lepDPhi", &b_lepdphi, "b_lepdphi/F");
    treeTMVA->Branch("missingET", &b_met, "b_met/F");
    treeTMVA->Branch("bjetmDR", &b_bjmdr, "b_bjmdr/F");
    treeTMVA->Branch("bjetmDEta", &b_bjmdeta, "b_bjmdeta/F");
    treeTMVA->Branch("bjetmDPhi", &b_bjmdphi, "b_bjmdphi/F");
    treeTMVA->Branch("dibjetsMass", &b_dibjetm, "b_dibjetm/F");
    treeTMVA->Branch("bjetPt_dibjetsm", &b_bjmpt, "b_bjmpt/F");
    treeTMVA->Branch("cjetPt", &b_cjmpt, "b_cjmpt/F");

    treeTMVA->Branch("jet1pt", &b_jet1pt, "b_jet1pt/F");
    treeTMVA->Branch("jet1eta", &b_jet1eta, "b_jet1eta/F");
    treeTMVA->Branch("jet1phi", &b_jet1phi, "b_jet1phi/F");
    treeTMVA->Branch("jet1m", &b_jet1m, "b_jet1m/F");
    treeTMVA->Branch("jet1csv", &b_jet1csv, "b_jet1csv/F");
    treeTMVA->Branch("jet1cvsl", &b_jet1cvsl, "b_jet1cvsl/F");
    treeTMVA->Branch("jet1cvsb", &b_jet1cvsb, "b_jet1cvsb/F");

    treeTMVA->Branch("jet2pt", &b_jet2pt, "b_jet2pt/F");
    treeTMVA->Branch("jet2eta", &b_jet2eta, "b_jet2eta/F");
    treeTMVA->Branch("jet2phi", &b_jet2phi, "b_jet2phi/F");
    treeTMVA->Branch("jet2m", &b_jet2m, "b_jet2m/F");
    treeTMVA->Branch("jet2csv", &b_jet2csv, "b_jet2csv/F");
    treeTMVA->Branch("jet2cvsl", &b_jet2cvsl, "b_jet2cvsl/F");
    treeTMVA->Branch("jet2cvsb", &b_jet2cvsb, "b_jet2cvsb/F");

    treeTMVA->Branch("jet3pt", &b_jet3pt, "b_jet3pt/F");
    treeTMVA->Branch("jet3eta", &b_jet3eta, "b_jet3eta/F");
    treeTMVA->Branch("jet3phi", &b_jet3phi, "b_jet3phi/F");
    treeTMVA->Branch("jet3m", &b_jet3m, "b_jet3m/F");
    treeTMVA->Branch("jet3csv", &b_jet3csv, "b_jet3csv/F");
    treeTMVA->Branch("jet3cvsl", &b_jet3cvsl, "b_jet3cvsl/F");
    treeTMVA->Branch("jet3cvsb", &b_jet3cvsb, "b_jet3cvsb/F");

    treeTMVA->Branch("jet4pt", &b_jet4pt, "b_jet4pt/F");
    treeTMVA->Branch("jet4eta", &b_jet4eta, "b_jet4eta/F");
    treeTMVA->Branch("jet4phi", &b_jet4phi, "b_jet4phi/F");
    treeTMVA->Branch("jet4m", &b_jet4m, "b_jet4m/F");
    treeTMVA->Branch("jet4csv", &b_jet4csv, "b_jet4csv/F");
    treeTMVA->Branch("jet4cvsl", &b_jet4cvsl, "b_jet4cvsl/F");
    treeTMVA->Branch("jet4cvsb", &b_jet4cvsb, "b_jet4cvsb/F");

    //DR jets
    treeTMVA->Branch("DRlepWpt", &b_DRlepWpt, "b_DRlepWpt/F");
    treeTMVA->Branch("DRlepWeta", &b_DRlepWeta, "b_DRlepWeta/F");
    treeTMVA->Branch("DRlepWdeta", &b_DRlepWdeta, "b_DRlepWdeta/F");
    treeTMVA->Branch("DRlepWphi", &b_DRlepWphi, "b_DRlepWphi/F");
    treeTMVA->Branch("DRlepWdphi", &b_DRlepWdphi, "b_DRlepWdphi/F");
    treeTMVA->Branch("DRlepWm", &b_DRlepWm, "b_DRlepWm/F");

    treeTMVA->Branch("DRjet0pt", &b_DRjet0pt, "DRjet0pt/F");
    treeTMVA->Branch("DRjet0eta", &b_DRjet0eta, "DRjet0eta/F");
    treeTMVA->Branch("DRjet0phi", &b_DRjet0phi, "DRjet0phi/F");
    treeTMVA->Branch("DRjet0m", &b_DRjet0m, "DRjet0m/F");
    treeTMVA->Branch("DRjet0csv", &b_DRjet0csv, "DRjet0csv/F");
    treeTMVA->Branch("DRjet0cvsl", &b_DRjet0cvsl, "DRjet0cvsl/F");
    treeTMVA->Branch("DRjet0cvsb", &b_DRjet0cvsb, "DRjet0cvsb/F");

    treeTMVA->Branch("DRjet1pt",&b_DRjet1pt, "DRjet1pt/F");
    treeTMVA->Branch("DRjet1eta",&b_DRjet1eta, "DRjet1eta/F");
    treeTMVA->Branch("DRjet1phi",&b_DRjet1phi, "DRjet1phi/F");
    treeTMVA->Branch("DRjet1m",&b_DRjet1m, "DRjet1m/F");
    treeTMVA->Branch("DRjet1csv", &b_DRjet1csv, "DRjet1csv/F");
    treeTMVA->Branch("DRjet1cvsl", &b_DRjet1cvsl, "DRjet1cvsl/F");
    treeTMVA->Branch("DRjet1cvsb", &b_DRjet1cvsb, "DRjet1cvsb/F");

    treeTMVA->Branch("DRjet2pt",&b_DRjet2pt, "DRjet2pt/F");
    treeTMVA->Branch("DRjet2eta",&b_DRjet2eta, "DRjet2eta/F");
    treeTMVA->Branch("DRjet2phi",&b_DRjet2phi, "DRjet2phi/F");
    treeTMVA->Branch("DRjet2m",&b_DRjet2m, "DRjet2m/F");
    treeTMVA->Branch("DRjet2csv", &b_DRjet2csv, "DRjet2csv/F");
    treeTMVA->Branch("DRjet2cvsl", &b_DRjet2cvsl, "DRjet2cvsl/F");
    treeTMVA->Branch("DRjet2cvsb", &b_DRjet2cvsb, "DRjet2cvsb/F");

    treeTMVA->Branch("DRjet3pt",&b_DRjet3pt, "DRjet3pt/F");
    treeTMVA->Branch("DRjet3eta",&b_DRjet3eta, "DRjet3eta/F");
    treeTMVA->Branch("DRjet3phi",&b_DRjet3phi, "DRjet3phi/F");
    treeTMVA->Branch("DRjet3m",&b_DRjet3m, "DRjet3m/F");
    treeTMVA->Branch("DRjet3csv", &b_DRjet3csv, "DRjet3csv/F");
    treeTMVA->Branch("DRjet3cvsl", &b_DRjet3cvsl, "DRjet3cvsl/F");
    treeTMVA->Branch("DRjet3cvsb", &b_DRjet3cvsb, "DRjet3cvsb/F");

    treeTMVA->Branch("DRjet12pt",&b_DRjet12pt, "DRjet12pt/F");
    treeTMVA->Branch("DRjet12eta",&b_DRjet12eta, "DRjet12eta/F");
    treeTMVA->Branch("DRjet12deta",&b_DRjet12deta, "DRjet12deta/F");
    treeTMVA->Branch("DRjet12phi",&b_DRjet12phi, "DRjet12phi/F");//mass->higgs
    treeTMVA->Branch("DRjet12dphi",&b_DRjet12dphi, "DRjet12dphi/F");
    treeTMVA->Branch("DRjet12m", &b_DRjet12m, "b_DRjet12m/F");//higgs candidate
    treeTMVA->Branch("DRjet12DR", &b_DRjet12DR, "b_DRjet12DR/F");

    treeTMVA->Branch("DRjet23pt",&b_DRjet23pt, "DRjet23pt/F");
    treeTMVA->Branch("DRjet23eta",&b_DRjet23eta, "DRjet23eta/F");
    treeTMVA->Branch("DRjet23deta",&b_DRjet23deta, "DRjet23deta/F");
    treeTMVA->Branch("DRjet23phi",&b_DRjet23phi, "DRjet23phi/F");//mass->had W
    treeTMVA->Branch("DRjet23dphi",&b_DRjet23dphi, "DRjet23dphi/F");
    treeTMVA->Branch("DRjet23m", &b_DRjet23m, "b_DRjet23m/F");//hadW

    treeTMVA->Branch("DRjet31pt",&b_DRjet31pt, "DRjet31pt/F");
    treeTMVA->Branch("DRjet31eta",&b_DRjet31eta, "DRjet31eta/F");
    treeTMVA->Branch("DRjet31deta",&b_DRjet31deta, "DRjet31deta/F");
    treeTMVA->Branch("DRjet31phi",&b_DRjet31phi, "DRjet31phi/F");
    treeTMVA->Branch("DRjet31dphi",&b_DRjet31dphi, "DRjet31dphi/F");
    treeTMVA->Branch("DRjet31m",&b_DRjet31m, "DRjet31m/F");

    treeTMVA->Branch("DRlepTpt", &b_DRlepTpt, "DRlepTpt/F");
    treeTMVA->Branch("DRlepTeta", &b_DRlepTeta, "DRlepTeta/F");
    treeTMVA->Branch("DRlepTdeta", &b_DRlepTdeta, "DRlepTdeta/F");
    treeTMVA->Branch("DRlepTphi", &b_DRlepTphi, "DRlepTphi/F");
    treeTMVA->Branch("DRlepTdphi", &b_DRlepTdphi, "DRlepTdphi/F");//W and b
    treeTMVA->Branch("DRlepTm", &b_DRlepTm, "b_DRlepTm/F");

    treeTMVA->Branch("DRhadTpt", &b_DRhadTpt, "DRhadTpt/F");
    treeTMVA->Branch("DRhadTeta", &b_DRhadTeta, "DRhadTeta/F");
    treeTMVA->Branch("DRhadTHbdeta", &b_DRhadTHbdeta, "DRhadTHbdeta/F");
    treeTMVA->Branch("DRhadTWbdeta", &b_DRhadTWbdeta, "DRhadTWbdeta/F");
    treeTMVA->Branch("DRhadTphi", &b_DRhadTphi, "DRhadTphi/F");
    treeTMVA->Branch("DRhadTHbdphi", &b_DRhadTHbdphi, "DRhadTHbdphi/F");//H and b
    treeTMVA->Branch("DRhadTWbdphi", &b_DRhadTWbdphi, "DRhadTWbdphi/F");//W and b
    treeTMVA->Branch("DRhadTm", &b_DRhadTm, "b_DRhadTm/F");

    fOutput->Add(treeTMVA);
} 

Bool_t makeTuple::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  int mode = 999; 
  mode = *channel;

  if( mode > 2) return kTRUE;

  float lep_SF = 1.0;
  if( !option.Contains("Data") ) lep_SF = lepton_SF[0]; //for datam turn this off
  float genweight = *genWeight;
  float puweight = PUWeight[0];
  float jetsf = jet_SF_CSV_30[0];
  float EventWeight = puweight*genweight*lep_SF*jetsf;

  float relIso = *lepton_relIso; 

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 
  int ncjets_m = 0; 

  TLorentzVector p4met;
  double met = *MET;
  double met_phi = *MET_phi;
  double apt = TMath::Abs(met);
  double met_x =  apt*TMath::Cos(met_phi);
  double met_y =  apt*TMath::Sin(met_phi);
  p4met.SetPxPyPzE( met_x, met_y, 0, met);

  TLorentzVector lepton;
  lepton.SetPtEtaPhiE(*lepton_pT, *lepton_eta, *lepton_phi, *lepton_E);

  double transverseM = transverseMass(lepton, p4met);
  double lepDphi = lepton.DeltaPhi(p4met);

  TLorentzVector hbjet1, hbjet2, genH;
  if(*addHbjet1_pt > 20 && *addHbjet2_pt > 20 && abs(*addHbjet1_eta) < 2.4 && abs(*addHbjet2_eta) < 2.4){
    hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
    hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

    genH = hbjet1 + hbjet2;
  }

  double bjmDPhi = 999;
  double bjmDEta = 999;
  double bjmDR = 999;
  double higgsMass_m = 9999;
  double bJetPtHm = 9999;
  double cjetPt = 0;

  //for Goh's Kin fit
  vector<size_t> jetIdxs;
  int b_kin_bjetcode;
  vector<float> jetCSVsDR;
  vector<float> jetCVSLsDR;
  vector<float> jetCVSBsDR;

  //Selection Option
  bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
  bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( option.Contains("DataSingleMu") ){
    if( !passmuon ) return kTRUE;//RDMu
    if( passelectron) return kTRUE;//RDMu
  }
  else if( option.Contains("DataSingleEG") ){
    if( !passelectron ) return kTRUE;//RDelec
    if( passmuon ) return kTRUE;//RDelec
  }
  else{
    if( !passmuon && !passelectron ) return kTRUE;
  }

  vector<float> v_cjet_m;
  vector<TLorentzVector> v_bjet_m;

  multimap<float /*jet pT*/, int  /*jet idx*/, std::greater<float>> m_jets;
    multimap<float, int, std::greater<float>>::iterator j_itr;

  for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
    if( !option.Contains("Data") ) jet = jet * jet_JER_Nom[iJet];

    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      njets++;
      jetIdxs.push_back(iJet);//Goh's kinfit

      m_jets.insert(pair<float, int>(jet_pT[iJet], iJet));

      if( jet_CSV[iJet] > 0.8484 ){
        nbjets_m++;
        v_bjet_m.push_back(jet);
      }
  
      if( jet_CvsL[iJet] > -0.1 && jet_CvsB[iJet] > 0.08 ){
        ncjets_m++;
        v_cjet_m.push_back(jet.Pt());
      }
    }
  } 

  if( njets <  4 || nbjets_m < 3) return kTRUE;
  
  //cout << nevt << endl;
  if( option.Contains("Data") ) b_EventCategory = -1;
  else if( option.Contains("Hct") || option.Contains("Hut") ) b_EventCategory = 0;
  else if( option.Contains("ttbb") ) b_EventCategory = 1;
  else if( option.Contains("ttbj") ) b_EventCategory = 2;
  else if( option.Contains("ttcc") ) b_EventCategory = 3;
  else if( option.Contains("ttLF") || option.Contains("ttother") ) b_EventCategory = 4;
  else if( option.Contains("channel") ) b_EventCategory = 5; //singletop
  else if( option.Contains("zjets") ) b_EventCategory = 6; //DY
  else b_EventCategory = 7;

  b_EventWeight = EventWeight;
  b_GoodPV = *GoodPV;
  b_nevt = nevt;

  b_njets = njets;
  b_nbjets_m = nbjets_m;
  b_ncjets_m = ncjets_m;

  b_met = met;
  b_lepdphi = lepDphi;
  b_transversem = transverseMass(lepton, p4met);

  float tmp_jetpt[4];
  float tmp_jeteta[4];
  float tmp_jetphi[4];
  float tmp_jetm[4];
  float tmp_jetcsv[4];
  float tmp_jetcvsl[4];
  float tmp_jetcvsb[4];
  int tmp_index = 0;

  for(j_itr = m_jets.begin(); j_itr != m_jets.end(); ++j_itr){
    if(tmp_index < 4){
      tmp_jetpt[tmp_index] = jet_pT[j_itr->second];
      tmp_jeteta[tmp_index] = jet_eta[j_itr->second];
      tmp_jetphi[tmp_index] = jet_phi[j_itr->second];
      tmp_jetm[tmp_index] = jet_E[j_itr->second];
      tmp_jetcsv[tmp_index] = jet_CSV[j_itr->second];
      tmp_jetcvsl[tmp_index] = jet_CvsL[j_itr->second];
      tmp_jetcvsb[tmp_index] = jet_CvsB[j_itr->second];
      ++tmp_index;
    }
    else continue;
  }

  b_jet1pt = tmp_jetpt[0];
  b_jet2pt = tmp_jetpt[1];
  b_jet3pt = tmp_jetpt[2];
  b_jet4pt = tmp_jetpt[3];
  b_jet1eta = tmp_jeteta[0];
  b_jet2eta = tmp_jeteta[1];
  b_jet3eta = tmp_jeteta[2];
  b_jet4eta = tmp_jeteta[3];
  b_jet1phi = tmp_jetphi[0];
  b_jet2phi = tmp_jetphi[1];
  b_jet3phi = tmp_jetphi[2];
  b_jet4phi = tmp_jetphi[3];
  b_jet1m = tmp_jetm[0];
  b_jet2m = tmp_jetm[1];
  b_jet3m = tmp_jetm[2];
  b_jet4m = tmp_jetm[3];
  b_jet1csv = tmp_jetcsv[0];
  b_jet2csv = tmp_jetcsv[1];
  b_jet3csv = tmp_jetcsv[2];
  b_jet4csv = tmp_jetcsv[3];
  b_jet1cvsl = tmp_jetcvsl[0];
  b_jet2cvsl = tmp_jetcvsl[1];
  b_jet3cvsl = tmp_jetcvsl[2];
  b_jet4cvsl = tmp_jetcvsl[3];
  b_jet1cvsb = tmp_jetcvsb[0];
  b_jet2cvsb = tmp_jetcvsb[1];
  b_jet3cvsb = tmp_jetcvsb[2];
  b_jet4cvsb = tmp_jetcvsb[3];

  if( ncjets_m != 0 ) b_cjmpt  = *max_element(v_cjet_m.begin(), v_cjet_m.end());

  double tmp_bjmDR = 999;
  double tmp_higgsMass_m  = 9999;
  double tmp_bjmDEta = 999;
  double tmp_bjmDPhi = 999;
  double tmp_bjmPt1 = 9999;
  double tmp_bjmPt2 = 9999;

  for(int m = 0; m < nbjets_m; m++){
    for(int n = 1; n <  nbjets_m; n++){
      if(m < n){
        tmp_bjmDR = v_bjet_m[m].DeltaR(v_bjet_m[n]);
        tmp_higgsMass_m = (v_bjet_m[m] + v_bjet_m[n]).M();
        tmp_bjmDEta = v_bjet_m[m].Eta()-v_bjet_m[n].Eta();
        tmp_bjmDPhi = v_bjet_m[m].DeltaPhi(v_bjet_m[n]);
        tmp_bjmPt1 = v_bjet_m[m].Pt();
        tmp_bjmPt2 = v_bjet_m[n].Pt();

        if( tmp_bjmDR < bjmDR ){
          bjmDR = tmp_bjmDR;
          higgsMass_m = tmp_higgsMass_m;
          bjmDEta = tmp_bjmDEta;
          bjmDPhi = tmp_bjmDPhi;

          if( tmp_bjmPt1 > tmp_bjmPt2) bJetPtHm = tmp_bjmPt1;
          else                         bJetPtHm = tmp_bjmPt2;

        }
      }
    }
  }

  //DR kin
  std::vector<size_t> bestIdxsDR;
  TLorentzVector jetP4sDR[4];
  size_t lepidx = 0;

  double minDRlep = 1e9;
  for ( auto ii0 = jetIdxs.begin(); ii0 != jetIdxs.end(); ++ii0 ) {
    jetP4sDR[0].SetPtEtaPhiE(jet_pT[*ii0], jet_eta[*ii0], jet_phi[*ii0], jet_E[*ii0]);
    if ( jet_CSV[*ii0] < 0.8484 ) continue;
    const double dRlep = jetP4sDR[0].DeltaR(lepton);
    if ( dRlep < minDRlep ) {
      lepidx = *ii0;
      minDRlep = dRlep;
    }
  }

  double minDR = 1e9;
  for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ) {
    if ( *ii1 == lepidx ) continue;
    jetP4sDR[1].SetPtEtaPhiE(jet_pT[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_E[*ii1]);

    for ( auto ii2 = ii1+1; ii2 != jetIdxs.end(); ++ii2 ) {
      if ( *ii2 == lepidx ) continue;
      int nbjetsInHadW = 0;
      if ( jet_CSV[*ii1] > 0.8484 ) ++nbjetsInHadW;
      if ( jet_CSV[*ii2] > 0.8484 ) ++nbjetsInHadW;
        // FCNC mode: require b jets in the trijet system, t->Hc / H->bb
        // can be used for the charged Higgs case as well, t->H+b / H+ -> cb
        if ( nbjets_m >= 3 and nbjetsInHadW < 2 ) continue; // at least two b jets in hadronic side
        else if ( nbjets_m == 2 and nbjetsInHadW < 1 ) continue; // at least one b jet in hadronic side

      jetP4sDR[2].SetPtEtaPhiE(jet_pT[*ii2], jet_eta[*ii2], jet_phi[*ii2], jet_E[*ii2]);
      const double dR = jetP4sDR[1].DeltaR(jetP4sDR[2]);
      if ( dR < minDR ) {
        bestIdxsDR = { lepidx, *ii1, *ii2, size_t(njets)};
        minDR = dR;
      }
    }
  }
  if ( !bestIdxsDR.empty() ) {
    const auto i1 = bestIdxsDR[1], i2 = bestIdxsDR[2];
    jetP4sDR[1].SetPtEtaPhiE(jet_pT[i1], jet_eta[i1], jet_phi[i1], jet_E[i1]);
    jetP4sDR[2].SetPtEtaPhiE(jet_pT[i2], jet_eta[i2], jet_phi[i2], jet_E[i2]);
    if( !option.Contains("Data") ){
      jetP4sDR[1] = jetP4sDR[1] *  jet_JER_Nom[i1];
      jetP4sDR[2] = jetP4sDR[2] *  jet_JER_Nom[i2];
    }
    const auto wP4 = jetP4sDR[1]+jetP4sDR[2];
    double minmassdiff = 1e9;
    for ( auto i3 : jetIdxs ) {
      if ( i3 == i1 or i3 == i2 ) continue;
      //if ( jet_CSV[i3] > 0.8484 ) continue;
      jetP4sDR[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);
      if( !option.Contains("Data") ) jetP4sDR[3] = jetP4sDR[3] * jet_JER_Nom[i3];

      double mass = (jetP4sDR[3] + wP4).M();
      double massdiff = (mass - 172.5)*(mass - 172.5);
      if ( massdiff < minmassdiff) {
        bestIdxsDR[3] = i3;
        minmassdiff = massdiff;
      }
    }
    if ( minmassdiff == 1e9 ) bestIdxsDR.clear();
  }

  stable_sort(next(bestIdxsDR.begin()), bestIdxsDR.end(),
                   [&](size_t a, size_t b){ return jet_CSV[a] > jet_CSV[b]; });

  for ( size_t i=0; i<4; ++i ) {
    const size_t j = bestIdxsDR[i];
    jetP4sDR[i].SetPtEtaPhiE(jet_pT[j], jet_eta[j], jet_phi[j], jet_E[j]);
    if( !option.Contains("Data") ) jetP4sDR[i] = jetP4sDR[i] * jet_JER_Nom[j];
    jetCSVsDR.push_back(jet_CSV[j]);
    jetCVSLsDR.push_back(jet_CvsL[j]);
    jetCVSBsDR.push_back(jet_CvsB[j]);
  }

  //Gen vs reco higgs->bjet matching
  bool match1 = false;
  bool match2 = false;
  if( option.Contains("Hct") || option.Contains("Hut") ){
    if(hbjet1.DeltaR(jetP4sDR[1]) < 0.4 or hbjet1.DeltaR(jetP4sDR[2]) < 0.4) match1 = true;
    if(hbjet2.DeltaR(jetP4sDR[1]) < 0.4 or hbjet2.DeltaR(jetP4sDR[2]) < 0.4) match2 = true;
      
  }

  if(match1 && match2){
    if(b_EventCategory == 0) b_GenMatch = 2;
    else if(b_EventCategory != 0 && b_EventCategory > -1 ) b_GenMatch = 1;   
  }
  else b_GenMatch = -1;

  //fill trees
  b_bjmdeta = bjmDEta;
  b_bjmdphi = bjmDPhi;
  b_bjmdr = bjmDR;
  b_dibjetm = higgsMass_m;
  b_bjmpt = bJetPtHm;

  b_DRlepWpt = (lepton+p4met).Pt();
  b_DRlepWeta = (lepton+p4met).Eta();
  b_DRlepWdeta = (lepton-p4met).Eta();
  b_DRlepWphi = (lepton+p4met).Phi();
  b_DRlepWdphi = lepton.DeltaPhi(p4met);
  b_DRlepWm = (lepton+p4met).M();

  b_DRjet0pt = jetP4sDR[0].Pt();
  b_DRjet0eta = jetP4sDR[0].Eta();
  b_DRjet0phi = jetP4sDR[0].Phi();
  b_DRjet0m = jetP4sDR[0].M();
  b_DRjet0csv = jetCSVsDR[0];
  b_DRjet0cvsl = jetCVSLsDR[0];
  b_DRjet0cvsb = jetCVSBsDR[0];

  b_DRjet1pt = jetP4sDR[1].Pt();
  b_DRjet1eta = jetP4sDR[1].Eta();
  b_DRjet1phi = jetP4sDR[1].Phi();
  b_DRjet1m = jetP4sDR[1].M();
  b_DRjet1csv = jetCSVsDR[1];
  b_DRjet1cvsl = jetCVSLsDR[1];
  b_DRjet1cvsb = jetCVSBsDR[1];

  b_DRjet2pt = jetP4sDR[2].Pt();
  b_DRjet2eta = jetP4sDR[2].Eta();
  b_DRjet2phi = jetP4sDR[2].Phi();
  b_DRjet2m = jetP4sDR[2].M();
  b_DRjet2csv = jetCSVsDR[2];
  b_DRjet2cvsl = jetCVSLsDR[2];
  b_DRjet2cvsb = jetCVSBsDR[2];

  b_DRjet3pt = jetP4sDR[3].Pt();
  b_DRjet3eta = jetP4sDR[3].Eta();
  b_DRjet3phi = jetP4sDR[3].Phi();
  b_DRjet3m = jetP4sDR[3].M();
  b_DRjet3csv = jetCSVsDR[3];
  b_DRjet3cvsl = jetCVSLsDR[3];
  b_DRjet3cvsb = jetCVSBsDR[3];

  b_DRjet12pt = (jetP4sDR[1]+jetP4sDR[2]).Pt();
  b_DRjet12eta = (jetP4sDR[1]+jetP4sDR[2]).Eta();
  b_DRjet12deta = (jetP4sDR[1]-jetP4sDR[2]).Eta();
  b_DRjet12phi = (jetP4sDR[1]+jetP4sDR[2]).Phi();
  b_DRjet12dphi = jetP4sDR[1].DeltaPhi(jetP4sDR[2]);
  b_DRjet12m = (jetP4sDR[1]+jetP4sDR[2]).M();
  b_DRjet12DR = jetP4sDR[1].DeltaR(jetP4sDR[2]);

  b_DRjet23pt = (jetP4sDR[2]+jetP4sDR[3]).Pt();
  b_DRjet23eta = (jetP4sDR[2]+jetP4sDR[3]).Eta();
  b_DRjet23deta = (jetP4sDR[2]-jetP4sDR[3]).Eta();
  b_DRjet23phi = (jetP4sDR[2]+jetP4sDR[3]).Phi();
  b_DRjet23dphi = jetP4sDR[2].DeltaPhi(jetP4sDR[3]);
  b_DRjet23m = (jetP4sDR[2]+jetP4sDR[3]).M();

  b_DRjet31pt = (jetP4sDR[1]+jetP4sDR[3]).Pt();
  b_DRjet31eta = (jetP4sDR[1]+jetP4sDR[3]).Eta();
  b_DRjet31deta = (jetP4sDR[1]-jetP4sDR[3]).Eta();
  b_DRjet31phi = (jetP4sDR[1]+jetP4sDR[3]).Phi();
  b_DRjet31dphi = jetP4sDR[1].DeltaPhi(jetP4sDR[3]);
  b_DRjet31m = (jetP4sDR[1]+jetP4sDR[3]).M();

  b_DRlepTpt = (lepton+p4met+jetP4sDR[0]).Pt();
  b_DRlepTeta = (lepton+p4met+jetP4sDR[0]).Eta();
  b_DRlepTdeta = (lepton+p4met-jetP4sDR[0]).Eta();
  b_DRlepTphi = (lepton+p4met+jetP4sDR[0]).Phi();
  b_DRlepTdphi = (lepton+p4met).DeltaPhi(jetP4sDR[0]);
  b_DRlepTm = (lepton+p4met+jetP4sDR[0]).M();

  b_DRhadTpt = (jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).Pt();
  b_DRhadTeta = (jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).Eta();
  b_DRhadTHbdeta = (jetP4sDR[1]+jetP4sDR[2]-jetP4sDR[3]).Eta();
  b_DRhadTWbdeta = (-jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).Eta();
  b_DRhadTphi = (jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).Phi();
  b_DRhadTHbdphi = (jetP4sDR[1]+jetP4sDR[2]).DeltaPhi(jetP4sDR[3]);
  b_DRhadTWbdphi = (jetP4sDR[2]+jetP4sDR[3]).DeltaPhi(jetP4sDR[1]);
  b_DRhadTm = (jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M();

  treeTMVA->Fill();

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

    TFile hfile(Form("tmva_%s.root",option.Data()), "RECREATE", "Tree for tmva run");

    fOutput->FindObject("tmva_tree")->Write();

    hfile.Write();
    hfile.Close();

}

double makeTuple::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

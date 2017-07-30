#define tmvaAnalysis_cxx

#include "tmvaAnalysis.h"
#include <TH2.h>
#include <TStyle.h>

void tmvaAnalysis::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void tmvaAnalysis::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();

    //treeTMVA = new TTree(Form("tmva_%s",option.Data()), "tree for tmva");

    //treeTMVA = new TTree("tmva_DataSingleMu", "tree for tmva");
    treeTMVA = new TTree("tmva_DataSingleEG", "tree for tmva");
    //treeTMVA = new TTree("tmva_zjets", "tree for tmva");
    //treeTMVA = new TTree("tmva_ttLF", "tree for tmva");
    //treeTMVA = new TTree("tmva_tt", "tree for tmva");

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
    treeTMVA->Branch("jetPt1", &b_jet1pt, "b_jet1pt/F");
    treeTMVA->Branch("jetPt2", &b_jet2pt, "b_jet2pt/F");
    treeTMVA->Branch("jetPt3", &b_jet3pt, "b_jet3pt/F");
    treeTMVA->Branch("jetPt4", &b_jet4pt, "b_jet4pt/F");
    treeTMVA->Branch("jetCSV1", &b_jet1csv, "b_jet1csv/F");
    treeTMVA->Branch("JetCSV2", &b_jet2csv, "b_jet2csv/F");
    treeTMVA->Branch("JetCSV3", &b_jet3csv, "b_jet3csv/F");
    treeTMVA->Branch("JetCSV4", &b_jet4csv, "b_jet4csv/F");
    //kinfit
    treeTMVA->Branch("KinLepWMass", &b_kinLepW, "b_kinLepW/F");
    treeTMVA->Branch("KinLepTopMass", &b_kinLepT, "b_kinLepT/F");
    treeTMVA->Branch("KinHadWMass", &b_kinHadW, "b_kinHadW/F");
    treeTMVA->Branch("KinHadTopMass", &b_kinHadT, "b_kinHadT/F");
    treeTMVA->Branch("FCNHKinLepWMass", &b_fcnhkinLepW, "b_fcnhkinLepW/F");
    treeTMVA->Branch("FCNHKinLepTopMass", &b_fcnhkinLepT, "b_fcnhkinLepT/F");
    treeTMVA->Branch("FCNHKinHMass", &b_fcnhkinH, "b_fcnhkinH/F");
    treeTMVA->Branch("FCNHKinHucTopMass", &b_fcnhkinHcT, "b_fcnhkinHcT/F");

    treeTMVA->Branch("M3LepWMass", &b_m3LepW, "b_m3LepW/F");
    treeTMVA->Branch("M3HadWMass", &b_m3HadW, "b_m3HadW/F");
    treeTMVA->Branch("M3HMass", &b_m3H, "b_m3H/F");
    treeTMVA->Branch("M3R", &b_m3DR, "b_m3DR/F");
    treeTMVA->Branch("M3LepTopMass", &b_m3LepT, "b_m3LepT/F");
    treeTMVA->Branch("M3HucTopMass", &b_m3HadT, "b_m3HadT/F");

    treeTMVA->Branch("DRLepWMass", &b_DRLepW, "b_DRLepW/F");
    treeTMVA->Branch("DRHadWMass", &b_DRHadW, "b_DRHadW/F");
    treeTMVA->Branch("DRHMass", &b_DRH, "b_DRH/F");
    treeTMVA->Branch("DRDR", &b_DRDR, "b_DRDR/F");
    treeTMVA->Branch("DRLepTopMass", &b_DRLepT, "b_DRLepT/F");
    treeTMVA->Branch("DRHucTopMass", &b_DRHadT, "b_DRHadT/F");

    fOutput->Add(treeTMVA);   
} 

Bool_t tmvaAnalysis::Process(Long64_t entry)
{

    fReader.SetEntry(entry);
    TString option = GetOption();   

    int mode = 999; 
    mode = *channel;

    if( mode > 2) return kTRUE;

    float lep_SF = 1.0;
    if( !option.Contains("Data") ) lep_SF = lepton_SF[0];
    float genweight = *genWeight;
    float puweight = PUWeight[0];
    float EventWeight = puweight*genweight*lep_SF;

    float relIso = *lepton_relIso; 

    //Object selection
    int njets = 0;
    int nbjets_m = 0; 
    int nbjets_t = 0; 
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

    TLorentzVector jet;

    double bjmDPhi = 999;
    double bjmDEta = 999;
    double bjmDR = 999;
    double higgsMass_m = 9999;
    double bJetPtHm = 9999;
    double cjetPt = 0;

    vector<float> jet_cvsl;
    vector<float> bjm_csv;
    vector<float> bjt_csv;

    //for Goh's Kin fit
    vector<size_t> jetIdxs;
    int b_kin_bjetcode;

    //Selection Option
    bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
    bool makeIso = true;
    bool isIso = *lepton_isIso; 
    if( makeIso && !isIso ) return kTRUE;
    if( !makeIso && isIso ) return kTRUE;

    //Event selection 
    bool passmuon = (mode == 0) && (lepton.Pt() > 27) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);


  if( passmuon || passelectron ){

    vector<float> v_cjet_m;
    vector<TLorentzVector> v_bjet_m;

    multimap<float /*jet pT*/, float /*CSV*/, std::greater<float>> m_jets;
      multimap<float, float /*CSV*/, std::greater<float>>::iterator j_itr;

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        jetIdxs.push_back(iJet);//Goh's kinfit
        m_jets.insert(pair<float, float>(jet_pT[iJet], jet_CSV[iJet]));

        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          v_bjet_m.push_back(jet);
          bjm_csv.push_back(jet_CSV[iJet]);
        }

        if( jet_CvsL[iJet] > -0.1 && jet_CvsL[iJet] > 0.08 ){
          ncjets_m++;
          v_cjet_m.push_back(jet.Pt());
        }
      }
    } 

    if( njets >= 4 && nbjets_m >= 3) {
      b_njets = njets;
      b_nbjets_m = nbjets_m;
      b_ncjets_m = ncjets_m;

      b_met = met;
      b_lepdphi = lepDphi;
      b_transversem = transverseMass(lepton, p4met);

      float tmp_jetpt[4];
      float tmp_jetcsv[4];
      int tmp_index = 0;

      for(j_itr = m_jets.begin(); j_itr != m_jets.end(); ++j_itr){
        if(tmp_index < 4){
          tmp_jetpt[tmp_index] = j_itr->first;
          tmp_jetcsv[tmp_index] = j_itr->second;
          ++tmp_index;
        }
        else continue;
      }

      b_jet1pt = tmp_jetpt[0];
      b_jet2pt = tmp_jetpt[1];
      b_jet3pt = tmp_jetpt[2];
      b_jet4pt = tmp_jetpt[3];
      b_jet1csv = tmp_jetcsv[0];
      b_jet2csv = tmp_jetcsv[1];
      b_jet3csv = tmp_jetcsv[2];
      b_jet4csv = tmp_jetcsv[3];

      if( ncjets_m != 0 ) b_cjmpt  = *max_element(v_cjet_m.begin(), v_cjet_m.end());

  ////kinfit values: for kin, lep T->W (lep + kin nu) + kinbl, had T -> (j1 + j2) + bjrefit. for fcnc, lep T -> W (lep + kin nu) + kinbl, fcn T -> H (kinj1 kinj2) + kinbjrefit(c/u jet)

      TLorentzVector kinnu, kinLepb, kinHadb, kinj1, kinj2, fcnhkinnu, fcnhkinLepb, fcnhkinuc, fcnhkinHb1, fcnhkinHb2;

      TLorentzVector kinLepT, kinHadT, kinLepW, kinHadW, fcnhkinH, fcnhkinLepW, fcnhkinLepT, fcnhkinHcT;

      kinnu.SetPtEtaPhiE(*kinnu_pT, *kinnu_eta, *kinnu_phi, *kinnu_E);
      kinLepW = lepton + kinnu;
      kinLepb.SetPtEtaPhiE(kinjet_pT[3], kinjet_eta[3], kinjet_phi[3], kinjet_E[3]);
      kinLepT = kinLepW + kinLepb;

      kinHadb.SetPtEtaPhiE(kinjet_pT[0], kinjet_eta[0], kinjet_phi[0], kinjet_E[0]);
      kinj1.SetPtEtaPhiE(kinjet_pT[1], kinjet_eta[1], kinjet_phi[1], kinjet_E[1]);
      kinj2.SetPtEtaPhiE(kinjet_pT[2], kinjet_eta[2], kinjet_phi[2], kinjet_E[2]);
      kinHadW = kinj1 + kinj2;
      kinHadT = kinHadW + kinHadb;

      fcnhkinnu.SetPtEtaPhiE(*fcnhkinnu_pT, *fcnhkinnu_eta, *fcnhkinnu_phi, *fcnhkinnu_E);
      fcnhkinLepW = fcnhkinnu + lepton;
      fcnhkinLepb.SetPtEtaPhiE(fcnhkinjet_pT[3], fcnhkinjet_eta[3], fcnhkinjet_phi[3], fcnhkinjet_E[3]);
      fcnhkinLepT = fcnhkinLepW + fcnhkinLepb;

      fcnhkinuc.SetPtEtaPhiE(fcnhkinjet_pT[0], fcnhkinjet_eta[0], fcnhkinjet_phi[0], fcnhkinjet_E[0]);
      fcnhkinHb1.SetPtEtaPhiE(fcnhkinjet_pT[1], fcnhkinjet_eta[1], fcnhkinjet_phi[1], fcnhkinjet_E[1]);
      fcnhkinHb2.SetPtEtaPhiE(fcnhkinjet_pT[2], fcnhkinjet_eta[2], fcnhkinjet_phi[2], fcnhkinjet_E[2]);
      if ( fcnhkinjet_pT[1] != 0 and fcnhkinjet_pT[2] != 0 ) fcnhkinH = fcnhkinHb1 + fcnhkinHb2;
      fcnhkinHcT = fcnhkinH + fcnhkinuc;

      if(*kin_chi2 < 10000){
        b_kinLepW = kinLepW.M();
        b_kinLepT = kinLepT.M();
        b_kinHadW = kinHadW.M();
        b_kinHadT = kinHadT.M();
      }
      if(*fcnhkin_chi2 < 10000){
        b_fcnhkinLepW = fcnhkinLepW.M();
        b_fcnhkinLepT = fcnhkinLepT.M();
        b_fcnhkinH = fcnhkinH.M();
        b_fcnhkinHcT = fcnhkinHcT.M();
      }
    }

    if( nbjets_m >1 ){

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
    }

    //M3
    vector<size_t> bestIdxs;
    TLorentzVector jetP4s[4];

    if (njets >= 3){
      double maxM3Pt = 0;
      for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ) {
        jetP4s[1].SetPtEtaPhiE(jet_pT[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_E[*ii1]);

        for ( auto ii2 = ii1+1; ii2 != jetIdxs.end(); ++ii2 ) {
          jetP4s[2].SetPtEtaPhiE(jet_pT[*ii2], jet_eta[*ii2], jet_phi[*ii2], jet_E[*ii2]);

          for ( auto ii3 = ii2+1; ii3 != jetIdxs.end(); ++ii3 ) {
            int nbjetsInHadT = 0;

            if ( jet_CSV[*ii1] > 0.8484 ) ++nbjetsInHadT;
            if ( jet_CSV[*ii2] > 0.8484 ) ++nbjetsInHadT;
            if ( jet_CSV[*ii3] > 0.8484 ) ++nbjetsInHadT;
              if ( nbjets_m >= 3 and nbjetsInHadT < 2 ) continue; // at least two b jets in hadronic side
              else if ( nbjets_m == 2 and nbjetsInHadT < 1 ) continue; // at least one b jet in hadronic side
            jetP4s[3].SetPtEtaPhiE(jet_pT[*ii3], jet_eta[*ii3], jet_phi[*ii3], jet_E[*ii3]);

            const double m3Pt = (jetP4s[1]+jetP4s[2]+jetP4s[3]).Pt();
            if ( m3Pt > maxM3Pt ) {
              maxM3Pt = m3Pt;
              bestIdxs = {size_t(nbjets_m), *ii1, *ii2, *ii3};
            }
          }
        }
      }

      stable_sort(next(bestIdxs.begin()), bestIdxs.end(),
                       [&](size_t a, size_t b){ return jet_CSV[a] > jet_CSV[b]; });


      for ( auto i : jetIdxs ) {
        if ( i == bestIdxs[1] or i == bestIdxs[2] or i == bestIdxs[3] ) continue;
        if ( bestIdxs[0] == size_t(njets) or jet_pT[bestIdxs[0]] < jet_pT[i] ) {
          bestIdxs[0] = i;
        }
      }

      for ( size_t i=0; i<4; ++i ) {
        const size_t j = bestIdxs[i];
        jetP4s[i].SetPtEtaPhiE(jet_pT[j], jet_eta[j], jet_phi[j], jet_E[j]);
      }
    }


    //DR kin
    std::vector<size_t> bestIdxsDR;
    TLorentzVector jetP4sDR[4];

    if( njets >= 3){
      double minDR = 1e9;
      for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ) {
        jetP4sDR[1].SetPtEtaPhiE(jet_pT[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_E[*ii1]);
        for ( auto ii2 = ii1+1; ii2 != jetIdxs.end(); ++ii2 ) {
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
            bestIdxsDR = {size_t(njets), *ii1, *ii2, size_t(njets)};
            minDR = dR;
          }
        }
      }
      if ( !bestIdxsDR.empty() ) {
        const auto i1 = bestIdxsDR[1], i2 = bestIdxsDR[2];
        jetP4sDR[1].SetPtEtaPhiE(jet_pT[i1], jet_eta[i1], jet_phi[i1], jet_E[i1]);
        jetP4sDR[2].SetPtEtaPhiE(jet_pT[i2], jet_eta[i2], jet_phi[i2], jet_E[i2]);
        const auto wP4 = jetP4sDR[1]+jetP4sDR[2];
        double minDR2 = 1e9;
        for ( auto i3 : jetIdxs ) {
          if ( i3 == i1 or i3 == i2 ) continue;
          jetP4sDR[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);

          const double dR = jetP4sDR[3].DeltaR(wP4);
          if ( dR < minDR2 ) {
            bestIdxsDR[3] = i3;
            minDR2 = dR;
          }
        }
        if ( minDR2 == 1e9 ) bestIdxsDR.clear();
      }

      stable_sort(next(bestIdxsDR.begin()), bestIdxsDR.end(),
                       [&](size_t a, size_t b){ return jet_CSV[a] > jet_CSV[b]; });

      for ( auto i : jetIdxs ) {
        if ( i == bestIdxsDR[1] or i == bestIdxsDR[2] or i == bestIdxsDR[3] ) continue;
        if ( bestIdxsDR[0] == size_t(njets) or jet_pT[bestIdxsDR[0]] < jet_pT[i] ) {
          bestIdxsDR[0] = i;
        }
      }

      for ( size_t i=0; i<4; ++i ) {
        const size_t j = bestIdxsDR[i];
        jetP4sDR[i].SetPtEtaPhiE(jet_pT[j], jet_eta[j], jet_phi[j], jet_E[j]);
      }
    }

    if( njets >= 4 && nbjets_m >= 3){
      b_bjmdeta = bjmDEta;
      b_bjmdphi = bjmDPhi;
      b_bjmdr = bjmDR;
      b_dibjetm = higgsMass_m;
      b_bjmpt = bJetPtHm;

      b_m3LepW = (lepton+p4met).M();
      b_m3HadW = (jetP4s[2]+jetP4s[3]).M();
      b_m3LepT = (lepton+p4met+jetP4s[0]).M();
      b_m3H = (jetP4s[1]+jetP4s[2]).M();
      b_m3DR = jetP4s[1].DeltaR(jetP4s[2]);
      b_m3HadT = (jetP4s[1]+jetP4s[2]+jetP4s[3]).M();

      b_DRLepW = (lepton+p4met).M();
      b_DRHadW = (jetP4sDR[2]+jetP4sDR[3]).M();
      b_DRLepT = (lepton+p4met+jetP4sDR[0]).M();
      b_DRH = (jetP4sDR[1]+jetP4sDR[2]).M();
      b_DRDR = jetP4sDR[1].DeltaR(jetP4sDR[2]);
      b_DRHadT = (jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M();
    }

    treeTMVA->Fill();

  }
  return kTRUE;
}

void tmvaAnalysis::SlaveTerminate()
{
  TString option = GetOption();
}
   

void tmvaAnalysis::Terminate()
{
  TString option = GetOption();

    TFile hfile(Form("tmva_%s.root",option.Data()), "RECREATE", "Tree for tmva run");
    //fOutput->FindObject(Form("tmva_%s",option.Data()))->Write();
    //fOutput->FindObject("tmva_DataSingleMu")->Write();
    fOutput->FindObject("tmva_DataSingleEG")->Write();
    //fOutput->FindObject("tmva_zjets")->Write();
    //fOutput->FindObject("tmva_ttLF")->Write();
    //fOutput->FindObject("tmva_tt")->Write();

    hfile.Write();
    hfile.Close();

}

double tmvaAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

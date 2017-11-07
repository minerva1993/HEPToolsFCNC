#define higgsRecoEff_cxx

#include "higgsRecoEff.h"
#include <TH2.h>
#include <TStyle.h>

void higgsRecoEff::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void higgsRecoEff::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
   
   cout << "SlaveBegin" << endl;
   for(int ich=0; ich < 2; ich++){
     for(int i=0; i < 2; i++){
     
      h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()), "Number of jets", 12, 0, 12);
      h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
      h_NJet[ich][i]->Sumw2();
      fOutput->Add(h_NJet[ich][i]);

      h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (medium)", 6, 0, 6);
      h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
      h_NBJetCSVv2M[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2M[ich][i]);

      h_NCJetM[ich][i] = new TH1D(Form("h_NCJetM_Ch%i_S%i_%s",ich,i,option.Data()), "Number of c tagged jets", 6, 0, 6);
      h_NCJetM[ich][i]->SetXTitle("c-tagged Jet Multiplicity (M)");
      h_NCJetM[ich][i]->Sumw2();
      fOutput->Add(h_NCJetM[ich][i]);  

      h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()), "MET", 30,0,300);
      h_MET[ich][i]->SetXTitle("MET (GeV)");
      h_MET[ich][i]->Sumw2();
      fOutput->Add(h_MET[ich][i]);

      h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass", 20 ,0 ,200);
      h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[ich][i]->Sumw2();
      fOutput->Add(h_WMass[ich][i]);

      h_HMass_m[ich][i] = new TH1D(Form("h_HMassM_Ch%i_S%i_%s",ich,i,option.Data()), "Di-bjet Mass (medium) wrt min DR", 30 ,0 ,300);
      h_HMass_m[ich][i]->SetXTitle("Di-bjet (medium) Mass (GeV)");
      h_HMass_m[ich][i]->Sumw2();
      fOutput->Add(h_HMass_m[ich][i]);

      h_bJetPtHm[ich][i] = new TH1D(Form("h_bJetPtHm_Ch%i_S%i_%s",ich,i,option.Data()), "Leading b jet (medium) pT from di-bjets", 27 ,30 ,300);
      h_bJetPtHm[ich][i]->SetXTitle("b Jet (medium) pT from di-bjets (GeV)");
      h_bJetPtHm[ich][i]->Sumw2();
      fOutput->Add(h_bJetPtHm[ich][i]);

      h_cJetPt[ich][i] = new TH1D(Form("h_cJetPt_Ch%i_S%i_%s",ich,i,option.Data()), "leading c jet (medium) pT", 27 ,30 ,300);
      h_cJetPt[ich][i]->SetXTitle("leading c Jet (medium) pT (GeV)");
      h_cJetPt[ich][i]->Sumw2();
      fOutput->Add(h_cJetPt[ich][i]);

      h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()), "lepton DPhi", 32 ,0 ,3.2);
      h_DPhi[ich][i]->SetXTitle("DPhi");
      h_DPhi[ich][i]->Sumw2();
      fOutput->Add(h_DPhi[ich][i]);

      h_bjmDPhi[ich][i] = new TH1D(Form("h_bjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Phi", 32 ,0 ,3.2);
      h_bjmDPhi[ich][i]->SetXTitle("bjet_m Delta Phi");
      h_bjmDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjmDPhi[ich][i]);

      h_bjmDEta[ich][i] = new TH1D(Form("h_bjmDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Eta", 32 ,0 ,3.2);
      h_bjmDEta[ich][i]->SetXTitle("bjet_m Delta Eta");
      h_bjmDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjmDEta[ich][i]);

      h_bjmDR[ich][i] = new TH1D(Form("h_bjmDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta R", 40 ,0 ,4);
      h_bjmDR[ich][i]->SetXTitle("bjet_m Delta R");
      h_bjmDR[ich][i]->Sumw2();
      fOutput->Add(h_bjmDR[ich][i]);

      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 20 ,0 ,0.2);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

      h_DRFCNHkinLepWMass[ich][i] = new TH1D(Form("h_DRFCNHkinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from DRFCNHkinfit (Lep)", 29 ,10 ,300);
      h_DRFCNHkinLepWMass[ich][i]->SetXTitle("W Mass from DRFCNHkinfit (Lep) (GeV)");
      h_DRFCNHkinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinLepWMass[ich][i]);

      h_DRFCNHkinHadWMass[ich][i] = new TH1D(Form("h_DRFCNHkinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from DRFCNHkinfit (Had)", 29 ,10 ,300);
      h_DRFCNHkinHadWMass[ich][i]->SetXTitle("W Mass from DRFCNHkinfit (Had) (GeV)");
      h_DRFCNHkinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHadWMass[ich][i]);

      h_DRFCNHkinHMass[ich][i] = new TH1D(Form("h_DRFCNHkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "H Mass from DRFCNHkinfit (bb)", 29 ,10 ,300);
      h_DRFCNHkinHMass[ich][i]->SetXTitle("H Mass from DRFCNHkinfit (bb)(GeV)");
      h_DRFCNHkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHMass[ich][i]);

      h_DRFCNHkinDR[ich][i] = new TH1D(Form("h_DRFCNHkinDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R from DRFCNHkinfit", 39, 0.1 ,4);
      h_DRFCNHkinDR[ich][i]->SetXTitle("Delta R from DRFCNHkinfit Higgs");
      h_DRFCNHkinDR[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinDR[ich][i]);

      h_DRFCNHkinTopMWb[ich][i] = new TH1D(Form("h_DRFCNHkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from DRFCNHkinfit (Lep)", 35 , 50 , 400);
      h_DRFCNHkinTopMWb[ich][i]->SetXTitle("Top Mass from DRFCNHkinfit (Lep) (GeV)");
      h_DRFCNHkinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMWb[ich][i]);

      h_DRFCNHkinTopMHc[ich][i] = new TH1D(Form("h_DRFCNHkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc/u DRFCNHkinfit", 35 , 50 , 400);
      h_DRFCNHkinTopMHc[ich][i]->SetXTitle("Top Mass from Hc/u DRFCNHkinfit (GeV)");
      h_DRFCNHkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMHc[ich][i]);

      //GenMatching
      h_HbjetsDR[ich][i] = new TH1D(Form("h_HbjetsDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen b jets from Higgs", 40 , 0 , 4);
      h_HbjetsDR[ich][i]->SetXTitle("Higgs->bb delta R");
      h_HbjetsDR[ich][i]->Sumw2();
      fOutput->Add(h_HbjetsDR[ich][i]);

      h_Hbjets1DR[ich][i] = new TH1D(Form("h_Hbjets1DR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen and reco b jets1 from Higg", 30 , 0 , 0.6);
      h_Hbjets1DR[ich][i]->SetXTitle("Gen and reco b jet1 delta R");
      h_Hbjets1DR[ich][i]->Sumw2();
      fOutput->Add(h_Hbjets1DR[ich][i]);

      h_Hbjets2DR[ich][i] = new TH1D(Form("h_Hbjets2DR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen and reco b jets2 from Higg", 30 , 0 , 0.6);
      h_Hbjets2DR[ich][i]->SetXTitle("Gen and reco b jet2 delta R");
      h_Hbjets2DR[ich][i]->Sumw2();
      fOutput->Add(h_Hbjets2DR[ich][i]);

      h_Hpt[ich][i] = new TH1D(Form("h_Hpt_Ch%i_S%i_%s",ich,i,option.Data()), "Gen Higgs pt", 60 , 0 , 300);
      h_Hpt[ich][i]->SetXTitle("Gen Higgs pt (GeV)");
      h_Hpt[ich][i]->Sumw2();
      fOutput->Add(h_Hpt[ich][i]);

      h_HptDR[ich][i] = new TH2D(Form("h_HptDR_Ch%i_S%i_%s",ich,i,option.Data()),"Gen delta R vs Higgs pt", 60, 0, 300, 40, 0, 4);
      h_HptDR[ich][i]->SetXTitle("Gen Higgs pt (GeV)");
      h_HptDR[ich][i]->SetYTitle("Higgs->bb delta R");
      h_HptDR[ich][i]->Sumw2();
      fOutput->Add(h_HptDR[ich][i]);
      }
    }
} 

Bool_t higgsRecoEff::Process(Long64_t entry)
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

    TLorentzVector hbjet1, hbjet2, genH;
    if((*Hbjet1_pt)*(*Hbjet2_pt) != 0){
      hbjet1.SetPtEtaPhiE(*Hbjet1_pt, *Hbjet1_eta, *Hbjet1_phi, *Hbjet1_e);
      hbjet2.SetPtEtaPhiE(*Hbjet2_pt, *Hbjet2_eta, *Hbjet2_phi, *Hbjet2_e);

      genH = hbjet1 + hbjet2;
      n_genH++;
    }

    double transverseM = transverseMass(lepton, p4met);
    double lepDphi = lepton.DeltaPhi(p4met);

    double bjmDPhi = 999;
    double bjmDEta = 999;
    double bjmDR = 999;
    double higgsMass_m = 9999;
    double higgsMass_t = 9999;
    double bJetPtHm = 9999;
    double bJetPtHt = 9999;
    double cjetPt = 0;

    vector<float> jet_cvsl;
    vector<float> bjm_csv;

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
    bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( passmuon || passelectron ){

    vector<float> v_cjet_m;
    vector<TLorentzVector> v_bjet_m;
    vector<TLorentzVector> v_bjet_t;
    vector<TLorentzVector> v_jet;

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        jetIdxs.push_back(iJet);//Goh's kinfit
        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          v_bjet_m.push_back(jet);
          bjm_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsB[iJet] > 0.08 ){
          ncjets_m++;
          v_cjet_m.push_back(jet.Pt());
        }
      }
    } 

    if( ncjets_m != 0 ) cjetPt = *max_element(v_cjet_m.begin(), v_cjet_m.end());

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

    //DR kin
    std::vector<size_t> bestIdxsDR;
    TLorentzVector jetP4sDR[4];

    if( njets >= 3 ){
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

    //Gen vs reco higgs->bjet matching
    double bjet1DR = 9;
    double bjet2DR = 9;
    if(hbjet1.DeltaR(jetP4sDR[1]) < 0.5){
      bjet1DR = hbjet1.DeltaR(jetP4sDR[1]);
      if(hbjet2.DeltaR(jetP4sDR[2]) < 0.5){
        bjet2DR = hbjet2.DeltaR(jetP4sDR[2]);
      }
      matchCount++;
    }
    else if(hbjet1.DeltaR(jetP4sDR[2]) < 0.5){
      bjet1DR = hbjet1.DeltaR(jetP4sDR[2]);
      if(hbjet1.DeltaR(jetP4sDR[1]) < 0.5){
        bjet2DR = hbjet1.DeltaR(jetP4sDR[1]);
      }
      matchCount++;
    }
    else{ 
      bjet1DR = 9;
      bjet2DR = 9;
    }

    /////Fill histograms

    h_NJet[mode][0]->Fill(njets, EventWeight);
    h_NBJetCSVv2M[mode][0]->Fill(nbjets_m, EventWeight);
    h_NCJetM[mode][0]->Fill(ncjets_m, EventWeight);
    h_MET[mode][0]->Fill(*MET, EventWeight);
    h_WMass[mode][0]->Fill(transverseM, EventWeight);
    h_DPhi[mode][0]->Fill(lepDphi, EventWeight);
    h_LepIso[mode][0]->Fill(relIso, EventWeight);

    if( nbjets_m >1 ){
      h_bjmDPhi[mode][0]->Fill(bjmDPhi, EventWeight);
      h_bjmDEta[mode][0]->Fill(bjmDEta, EventWeight);
      h_bjmDR[mode][0]->Fill(bjmDR, EventWeight);
      h_HMass_m[mode][0]->Fill(higgsMass_m, EventWeight);
      h_bJetPtHm[mode][0]->Fill(bJetPtHm, EventWeight);
    }

    if( ncjets_m >0 ){
      h_cJetPt[mode][0]->Fill(cjetPt, EventWeight);
    }

    h_DRFCNHkinLepWMass[mode][0]->Fill((lepton+p4met).M(),EventWeight);
    h_DRFCNHkinHadWMass[mode][0]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
    h_DRFCNHkinTopMWb[mode][0]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
    h_DRFCNHkinHMass[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
    h_DRFCNHkinDR[mode][0]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
    h_DRFCNHkinTopMHc[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

    if(hbjet1.DeltaR(hbjet2) > 0){
      h_HbjetsDR[mode][0]->Fill(hbjet1.DeltaR(hbjet2),EventWeight);
      h_Hpt[mode][0]->Fill(genH.Pt(),EventWeight);
      h_HptDR[mode][0]->Fill(genH.Pt(),hbjet1.DeltaR(hbjet2),EventWeight);
      h_Hbjets1DR[mode][0]->Fill(bjet1DR,EventWeight);
      h_Hbjets2DR[mode][0]->Fill(bjet1DR,EventWeight);
    }

    if( njets >= 4) {
      //STEP10
      if( nbjets_m >= 3 ){
        if(genH.Pt()>0) n_genH2++;

        //Gen vs reco higgs->bjet matching
        double bjet1DR2 = 9;
        double bjet2DR2 = 9;
        if(hbjet1.DeltaR(jetP4sDR[1]) < 0.5){
          bjet1DR2 = hbjet1.DeltaR(jetP4sDR[1]);
          if(hbjet2.DeltaR(jetP4sDR[2]) < 0.5){
            bjet2DR2 = hbjet2.DeltaR(jetP4sDR[2]);
          }
          matchCount2++;
        }
        else if(hbjet1.DeltaR(jetP4sDR[2]) < 0.5){
          bjet1DR2 = hbjet1.DeltaR(jetP4sDR[2]);
          if(hbjet1.DeltaR(jetP4sDR[1]) < 0.5){
            bjet2DR2 = hbjet1.DeltaR(jetP4sDR[1]);
          }
          matchCount2++;
        }
        else{
          bjet1DR2 = 9;
          bjet2DR2 = 9;
        }


        h_NJet[mode][1]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][1]->Fill(nbjets_m, EventWeight);
        h_NCJetM[mode][1]->Fill(ncjets_m, EventWeight);
        h_MET[mode][1]->Fill(*MET, EventWeight);
        h_WMass[mode][1]->Fill(transverseM, EventWeight);
        h_DPhi[mode][1]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][1]->Fill(relIso, EventWeight);
        
        if( nbjets_m >1 ){
          h_bjmDPhi[mode][1]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][1]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][1]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][1]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][1]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][1]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][1]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][1]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][1]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][1]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(hbjet1.DeltaR(hbjet2) > 0){
          h_HbjetsDR[mode][1]->Fill(hbjet1.DeltaR(hbjet2),EventWeight);
          h_Hpt[mode][1]->Fill(genH.Pt(),EventWeight);
          h_HptDR[mode][1]->Fill(genH.Pt(),hbjet1.DeltaR(hbjet2),EventWeight);
          h_Hbjets1DR[mode][1]->Fill(bjet1DR,EventWeight);
          h_Hbjets2DR[mode][1]->Fill(bjet1DR,EventWeight);
        }
      }
    }
  }
  return kTRUE;
}

void higgsRecoEff::SlaveTerminate()
{
  TString option = GetOption();
}
   

void higgsRecoEff::Terminate()
{
  TString option = GetOption();

  TFile * out = TFile::Open(Form("hist_%s.root",option.Data()),"RECREATE");

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

  cout << "Matched Higgs b jets :" << matchCount << endl;
  cout << "Matched Higgs b jets with event selection:" << matchCount2 << endl;
  cout << "number of gen level Higgs :" << n_genH << endl;
  cout << "number of gen level Higgs with event selection:" << n_genH2 << endl;
}

double higgsRecoEff::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

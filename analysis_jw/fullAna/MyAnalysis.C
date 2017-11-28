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
   
   cout << "SlaveBegin" << endl;
   for(int ich=0; ich < 2; ich++){
     for(int i=0; i < 16; i++){
     
      h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()), "Number of jets", 12, 0, 12);
      h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
      h_NJet[ich][i]->Sumw2();
      fOutput->Add(h_NJet[ich][i]);

      h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (medium)", 6, 0, 6);
      h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
      h_NBJetCSVv2M[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2M[ich][i]);

      h_NBJetCSVv2T[ich][i] = new TH1D(Form("h_NBJetCSVv2T_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (tight)", 6, 0, 6);
      h_NBJetCSVv2T[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2T)");
      h_NBJetCSVv2T[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2T[ich][i]);

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

      h_LepIsoQCD[ich][i] = new TH1D(Form("h_LepIsoQCD_Ch%i_S%i_%s",ich,i,option.Data()), "LepIsoQCD", 20 ,0 ,0.2);
      h_LepIsoQCD[ich][i]->SetXTitle("Relative Isolation (QCD)");
      h_LepIsoQCD[ich][i]->Sumw2();
      fOutput->Add(h_LepIsoQCD[ich][i]);

      h_csvv2[ich][i] = new TH1D(Form("h_csvv2_Ch%i_S%i_%s",ich,i,option.Data()), "CSVv2", 20 ,0 ,1);
      h_csvv2[ich][i]->SetXTitle("CSVv2");
      h_csvv2[ich][i]->Sumw2();
      fOutput->Add(h_csvv2[ich][i]);

      h_cvsl[ich][i] = new TH1D(Form("h_cvsl_Ch%i_S%i_%s",ich,i,option.Data()), "CvsL", 20 , -0.1 ,1);
      h_cvsl[ich][i]->SetXTitle("CvsL");
      h_cvsl[ich][i]->Sumw2();
      fOutput->Add(h_cvsl[ich][i]);

      h_cvsb[ich][i] = new TH1D(Form("h_cvsb_Ch%i_S%i_%s",ich,i,option.Data()), "CvsB", 20 , 0.08 ,1);
      h_cvsb[ich][i]->SetXTitle("CvsB");
      h_cvsb[ich][i]->Sumw2();
      fOutput->Add(h_cvsb[ich][i]);

      h_DRFCNHkinLepWMass[ich][i] = new TH1D(Form("h_DRFCNHkinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from DRFCNHkinfit (Lep)", 29 ,10 ,300);
      h_DRFCNHkinLepWMass[ich][i]->SetXTitle("W Mass (Lep) (GeV)");
      h_DRFCNHkinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinLepWMass[ich][i]);

      h_DRFCNHkinHadWMass[ich][i] = new TH1D(Form("h_DRFCNHkinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from DRFCNHkinfit (Had)", 29 ,10 ,300);
      h_DRFCNHkinHadWMass[ich][i]->SetXTitle("W Mass (Had) (GeV)");
      h_DRFCNHkinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHadWMass[ich][i]);

      h_DRFCNHkinHMass[ich][i] = new TH1D(Form("h_DRFCNHkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "H Mass from DRFCNHkinfit (bb)", 29 ,10 ,300);
      h_DRFCNHkinHMass[ich][i]->SetXTitle("Higgs Mass (GeV)");
      h_DRFCNHkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHMass[ich][i]);

      h_DRFCNHkinDR[ich][i] = new TH1D(Form("h_DRFCNHkinDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R from DRFCNHkinfit", 39, 0.1 ,4);
      h_DRFCNHkinDR[ich][i]->SetXTitle("Delta R of b jets from Higgs");
      h_DRFCNHkinDR[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinDR[ich][i]);

      h_DRFCNHkinTopMWb[ich][i] = new TH1D(Form("h_DRFCNHkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from DRFCNHkinfit (Lep)", 35 , 50 , 400);
      h_DRFCNHkinTopMWb[ich][i]->SetXTitle("Top Mass (Lep) (GeV)");
      h_DRFCNHkinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMWb[ich][i]);

      h_DRFCNHkinTopMHc[ich][i] = new TH1D(Form("h_DRFCNHkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc/u DRFCNHkinfit", 35 , 50 , 400);
      h_DRFCNHkinTopMHc[ich][i]->SetXTitle("Top Mass (Had) (GeV)");
      h_DRFCNHkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMHc[ich][i]);

      //GenInfo
      h_genDR[ich][i] = new TH1D(Form("h_genDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen b jets from Higgs", 39 , 0.1 , 4);
      h_genDR[ich][i]->SetXTitle("gen #Delta R");
      h_genDR[ich][i]->Sumw2();
      fOutput->Add(h_genDR[ich][i]);

      h_matchDR[ich][i] = new TH1D(Form("h_matchDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen matched b jets from Higgs", 39 , 0.1 , 4);
      h_matchDR[ich][i]->SetXTitle("gen matched #Delta R");
      h_matchDR[ich][i]->Sumw2();
      fOutput->Add(h_matchDR[ich][i]);

      h_genHm[ich][i] = new TH1D(Form("h_genHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen Higgs mass", 29 , 10 , 300);
      h_genHm[ich][i]->SetXTitle("gen Higgs Mass (GeV)");
      h_genHm[ich][i]->Sumw2();
      fOutput->Add(h_genHm[ich][i]);

      h_matchHm[ich][i] = new TH1D(Form("h_matchHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen matched Higgs mass", 29 , 10 , 300);
      h_matchHm[ich][i]->SetXTitle("gen matched Higgs Mass (GeV)");
      h_matchHm[ich][i]->Sumw2();
      fOutput->Add(h_matchHm[ich][i]);
      }
    }
} 

Bool_t MyAnalysis::Process(Long64_t entry)
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
    float jetsf = jet_SF_CSV_30[0];
    float EventWeight = puweight*genweight*lep_SF*jetsf;

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

    double bjmDPhi = 999;
    double bjmDEta = 999;
    double bjmDR = 999;
    double bjtDPhi = 999;
    double bjtDEta = 999;
    double bjtDR = 999;
    double higgsMass_m = 9999;
    double higgsMass_t = 9999;
    double bJetPtHm = 9999;
    double bJetPtHt = 9999;
    double cjetPt = 0;

    vector<float> bjm_csv;
    vector<float> bjt_csv;
    vector<float> cjm_cvsl;
    vector<float> cjm_cvsb;

    //for Goh's Kin fit
    vector<size_t> jetIdxs;
    int b_kin_bjetcode;
    bool match1 = false;
    bool match2 = false;
    double gendR = -1.0;
    double matchdR = -1.0;
    double genHm = 0;
    double matchHm = 0;

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
          //bjm_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CSV[iJet] > 0.9535 ){
          nbjets_t++;
          v_bjet_t.push_back(jet);
          //bjt_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsB[iJet] > 0.08 ){
          ncjets_m++;
          v_cjet_m.push_back(jet.Pt());
          //cjm_cvsl.push_back(jet_CvsL[iJet]);
          //cjm_cvsb.push_back(jet_CvsB[iJet]);
        }
      }
    }

    //if( (option.Contains("Hct") || option.Contains("Hut")) && (*addHbjet1_pt)*(*addHbjet2_pt) == 0) return kTRUE;

    TLorentzVector hbjet1, hbjet2, genH;
    if(*addHbjet1_pt > 25 && *addHbjet2_pt > 25 && abs(*addHbjet1_eta) < 2.5 && abs(*addHbjet2_eta) < 2.5){
      hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
      hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

      genH = hbjet1 + hbjet2;
      gendR = hbjet1.DeltaR(hbjet2);
      genHm = genH.M();
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
    size_t lepidx = 0;

    if( njets >= 3){
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
        const auto wP4 = jetP4sDR[1]+jetP4sDR[2];
        double minmassdiff = 1e9;
        //double minDR2 = 1e9;
        for ( auto i3 : jetIdxs ) {
          if ( i3 == i1 or i3 == i2 ) continue;
          jetP4sDR[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);
        /*
          const double dR = jetP4sDR[3].DeltaR(wP4);
          if ( dR < minDR2 ) {
            bestIdxsDR[3] = i3;
            minDR2 = dR;
          }
        }
        if ( minDR2 == 1e9 ) bestIdxsDR.clear();
        */
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

      /*
      for ( auto i : jetIdxs ) {
        if ( i == bestIdxsDR[1] or i == bestIdxsDR[2] or i == bestIdxsDR[3] ) continue;
        if ( bestIdxsDR[0] == size_t(njets) or jet_pT[bestIdxsDR[0]] < jet_pT[i] ) {
          bestIdxsDR[0] = i;
        }
      }
      */

      for ( size_t i=0; i<4; ++i ) {
        const size_t j = bestIdxsDR[i];
        jetP4sDR[i].SetPtEtaPhiE(jet_pT[j], jet_eta[j], jet_phi[j], jet_E[j]);
      }

      //Gen vs reco higgs->bjet matching
      if(hbjet1.DeltaR(jetP4sDR[1]) < 0.4 or hbjet1.DeltaR(jetP4sDR[2]) < 0.4) match1 = true;
      if(hbjet2.DeltaR(jetP4sDR[1]) < 0.4 or hbjet2.DeltaR(jetP4sDR[2]) < 0.4) match2 = true;
    }

    /////Fill histograms

    h_NJet[mode][0]->Fill(njets, EventWeight);
    h_NBJetCSVv2M[mode][0]->Fill(nbjets_m, EventWeight);
    h_NBJetCSVv2T[mode][0]->Fill(nbjets_t, EventWeight);
    h_NCJetM[mode][0]->Fill(ncjets_m, EventWeight);
    h_MET[mode][0]->Fill(*MET, EventWeight);
    h_WMass[mode][0]->Fill(transverseM, EventWeight);
    h_DPhi[mode][0]->Fill(lepDphi, EventWeight);
    h_LepIso[mode][0]->Fill(relIso, EventWeight);
    if( isQCD ) h_LepIsoQCD[mode][0]->Fill(relIso, EventWeight);

    if( !bestIdxsDR.empty() ){
      for( int i=0; i<bestIdxsDR.size(); ++i ){
        const size_t j = bestIdxsDR[i];
        h_csvv2[mode][0]->Fill(jet_CSV[j]);
        h_cvsl[mode][0]->Fill(jet_CvsL[j]);
        h_cvsb[mode][0]->Fill(jet_CvsB[j]);
      }
    }

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

    if(genH.Pt() > 0){
      h_genDR[mode][0]->Fill(gendR, EventWeight);
      h_genHm[mode][0]->Fill(genHm, EventWeight);
      if(match1 && match2){
        h_matchDR[mode][0]->Fill(gendR, EventWeight);
        h_matchHm[mode][0]->Fill(genHm, EventWeight);
      }
    }

    //step1
    if( njets >= 3 ) {
      h_NJet[mode][1]->Fill(njets, EventWeight);
      h_NBJetCSVv2M[mode][1]->Fill(nbjets_m, EventWeight);
      h_NBJetCSVv2T[mode][1]->Fill(nbjets_t, EventWeight);
      h_NCJetM[mode][1]->Fill(ncjets_m, EventWeight);
      h_MET[mode][1]->Fill(*MET, EventWeight);
      h_WMass[mode][1]->Fill(transverseM, EventWeight);
      h_DPhi[mode][1]->Fill(lepDphi, EventWeight);
      h_LepIso[mode][1]->Fill(relIso, EventWeight);
      if( isQCD ) h_LepIsoQCD[mode][1]->Fill(relIso, EventWeight);

      if( !bestIdxsDR.empty() ){
        for( int i=0; i<bestIdxsDR.size(); ++i ){
          const size_t j = bestIdxsDR[i];
          h_csvv2[mode][1]->Fill(jet_CSV[j]);
          h_cvsl[mode][1]->Fill(jet_CvsL[j]);
          h_cvsb[mode][1]->Fill(jet_CvsB[j]);
        }
      }

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

      if(genH.Pt() > 0){
        h_genDR[mode][1]->Fill(gendR, EventWeight);
        h_genHm[mode][1]->Fill(genHm, EventWeight);
        if(match1 && match2){
          h_matchDR[mode][1]->Fill(gendR, EventWeight);
          h_matchHm[mode][1]->Fill(genHm, EventWeight);
        }
      }

      //STEP2
      if( nbjets_m == 2 ){
        h_NJet[mode][2]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][2]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][2]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][2]->Fill(ncjets_m, EventWeight);
        h_MET[mode][2]->Fill(*MET, EventWeight);
        h_WMass[mode][2]->Fill(transverseM, EventWeight);
        h_DPhi[mode][2]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][2]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][2]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][2]->Fill(jet_CSV[j]);
            h_cvsl[mode][2]->Fill(jet_CvsL[j]);
            h_cvsb[mode][2]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m > 1 ){
          h_bjmDPhi[mode][2]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][2]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][2]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][2]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][2]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][2]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][2]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][2]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][2]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][2]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][2]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][2]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][2]->Fill(gendR, EventWeight);
          h_genHm[mode][2]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][2]->Fill(gendR, EventWeight);
            h_matchHm[mode][2]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP3
      if( nbjets_m == 3 ){
        h_NJet[mode][3]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][3]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][3]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][3]->Fill(ncjets_m, EventWeight);
        h_MET[mode][3]->Fill(*MET, EventWeight);
        h_WMass[mode][3]->Fill(transverseM, EventWeight);
        h_DPhi[mode][3]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][3]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][3]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][3]->Fill(jet_CSV[j]);
            h_cvsl[mode][3]->Fill(jet_CvsL[j]);
            h_cvsb[mode][3]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m > 1 ){
          h_bjmDPhi[mode][3]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][3]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][3]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][3]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][3]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][3]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][3]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][3]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][3]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][3]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][3]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][3]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][3]->Fill(gendR, EventWeight);
          h_genHm[mode][3]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][3]->Fill(gendR, EventWeight);
            h_matchHm[mode][3]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP4
      if( nbjets_m == 2 || nbjets_m == 3 ){
        h_NJet[mode][4]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][4]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][4]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][4]->Fill(ncjets_m, EventWeight);
        h_MET[mode][4]->Fill(*MET, EventWeight);
        h_WMass[mode][4]->Fill(transverseM, EventWeight);
        h_DPhi[mode][4]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][4]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][4]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][4]->Fill(jet_CSV[j]);
            h_cvsl[mode][4]->Fill(jet_CvsL[j]);
            h_cvsb[mode][4]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m > 1 ){
          h_bjmDPhi[mode][4]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][4]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][4]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][4]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][4]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][4]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][4]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][4]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][4]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][4]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][4]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][4]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][4]->Fill(gendR, EventWeight);
          h_genHm[mode][4]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][4]->Fill(gendR, EventWeight);
            h_matchHm[mode][4]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP5
      if( nbjets_m >= 3 ){
        h_NJet[mode][5]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][5]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][5]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][5]->Fill(ncjets_m, EventWeight);
        h_MET[mode][5]->Fill(*MET, EventWeight);
        h_WMass[mode][5]->Fill(transverseM, EventWeight);
        h_DPhi[mode][5]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][5]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][5]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][5]->Fill(jet_CSV[j]);
            h_cvsl[mode][5]->Fill(jet_CvsL[j]);
            h_cvsb[mode][5]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m > 1 ){
          h_bjmDPhi[mode][5]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][5]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][5]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][5]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][5]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][5]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][5]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][5]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][5]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][5]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][5]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][5]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][5]->Fill(gendR, EventWeight);
          h_genHm[mode][5]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][5]->Fill(gendR, EventWeight);
            h_matchHm[mode][5]->Fill(genHm, EventWeight);
          }
        }
      }
    }

    //STEP6    
    if( njets >= 4) {
      h_NJet[mode][6]->Fill(njets, EventWeight);
      h_NBJetCSVv2M[mode][6]->Fill(nbjets_m, EventWeight);
      h_NBJetCSVv2T[mode][6]->Fill(nbjets_t, EventWeight);
      h_NCJetM[mode][6]->Fill(ncjets_m, EventWeight);
      h_MET[mode][6]->Fill(*MET, EventWeight);
      h_WMass[mode][6]->Fill(transverseM, EventWeight);
      h_DPhi[mode][6]->Fill(lepDphi, EventWeight);
      h_LepIso[mode][6]->Fill(relIso, EventWeight);
      if( isQCD ) h_LepIsoQCD[mode][6]->Fill(relIso, EventWeight);

      if( !bestIdxsDR.empty() ){
        for( int i=0; i<bestIdxsDR.size(); ++i ){
          const size_t j = bestIdxsDR[i];
          h_csvv2[mode][6]->Fill(jet_CSV[j]);
          h_cvsl[mode][6]->Fill(jet_CvsL[j]);
          h_cvsb[mode][6]->Fill(jet_CvsB[j]);
        }
      }

      if( nbjets_m >1 ){
        h_bjmDPhi[mode][6]->Fill(bjmDPhi, EventWeight);
        h_bjmDEta[mode][6]->Fill(bjmDEta, EventWeight);
        h_bjmDR[mode][6]->Fill(bjmDR, EventWeight);
        h_HMass_m[mode][6]->Fill(higgsMass_m, EventWeight);
        h_bJetPtHm[mode][6]->Fill(bJetPtHm, EventWeight);
      }

      if( ncjets_m >0 ){
        h_cJetPt[mode][6]->Fill(cjetPt, EventWeight);
      }

      h_DRFCNHkinLepWMass[mode][6]->Fill((lepton+p4met).M(),EventWeight);
      h_DRFCNHkinHadWMass[mode][6]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
      h_DRFCNHkinTopMWb[mode][6]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
      h_DRFCNHkinHMass[mode][6]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
      h_DRFCNHkinDR[mode][6]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
      h_DRFCNHkinTopMHc[mode][6]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

      if(genH.Pt() > 0){
        h_genDR[mode][6]->Fill(gendR, EventWeight);
        h_genHm[mode][6]->Fill(genHm, EventWeight);
        if(match1 && match2){
          h_matchDR[mode][6]->Fill(gendR, EventWeight);
          h_matchHm[mode][6]->Fill(genHm, EventWeight);
        }
      }

      //STEP7
      if( nbjets_m == 2 ){
        h_NJet[mode][7]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][7]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][7]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][7]->Fill(ncjets_m, EventWeight);
        h_MET[mode][7]->Fill(*MET, EventWeight);
        h_WMass[mode][7]->Fill(transverseM, EventWeight);
        h_DPhi[mode][7]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][7]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][7]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][7]->Fill(jet_CSV[j]);
            h_cvsl[mode][7]->Fill(jet_CvsL[j]);
            h_cvsb[mode][7]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][7]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][7]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][7]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][7]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][7]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][7]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][7]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][7]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][7]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][7]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][7]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][7]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][7]->Fill(gendR, EventWeight);
          h_genHm[mode][7]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][7]->Fill(gendR, EventWeight);
            h_matchHm[mode][7]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP8
      if( nbjets_m == 3 ){
        h_NJet[mode][8]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][8]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][8]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][8]->Fill(ncjets_m, EventWeight);
        h_MET[mode][8]->Fill(*MET, EventWeight);
        h_WMass[mode][8]->Fill(transverseM, EventWeight);
        h_DPhi[mode][8]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][8]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][8]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][8]->Fill(jet_CSV[j]);
            h_cvsl[mode][8]->Fill(jet_CvsL[j]);
            h_cvsb[mode][8]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][8]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][8]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][8]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][8]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][8]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][8]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][8]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][8]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][8]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][8]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][8]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][8]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][8]->Fill(gendR, EventWeight);
          h_genHm[mode][8]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][8]->Fill(gendR, EventWeight);
            h_matchHm[mode][8]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP9
      if( nbjets_m == 4 ){
        h_NJet[mode][9]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][9]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][9]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][9]->Fill(ncjets_m, EventWeight);
        h_MET[mode][9]->Fill(*MET, EventWeight);
        h_WMass[mode][9]->Fill(transverseM, EventWeight);
        h_DPhi[mode][9]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][9]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][9]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][9]->Fill(jet_CSV[j]);
            h_cvsl[mode][9]->Fill(jet_CvsL[j]);
            h_cvsb[mode][9]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][9]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][9]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][9]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][9]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][9]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][9]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][9]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][9]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][9]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][9]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][9]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][9]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][9]->Fill(gendR, EventWeight);
          h_genHm[mode][9]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][9]->Fill(gendR, EventWeight);
            h_matchHm[mode][9]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP10
      if( nbjets_m >= 3 ){
        h_NJet[mode][10]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][10]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][10]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][10]->Fill(ncjets_m, EventWeight);
        h_MET[mode][10]->Fill(*MET, EventWeight);
        h_WMass[mode][10]->Fill(transverseM, EventWeight);
        h_DPhi[mode][10]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][10]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][10]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][10]->Fill(jet_CSV[j]);
            h_cvsl[mode][10]->Fill(jet_CvsL[j]);
            h_cvsb[mode][10]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][10]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][10]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][10]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][10]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][10]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][10]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][10]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][10]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][10]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][10]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][10]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][10]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][10]->Fill(gendR, EventWeight);
          h_genHm[mode][10]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][10]->Fill(gendR, EventWeight);
            h_matchHm[mode][10]->Fill(genHm, EventWeight);
          }
        }
      }


      //STEP11
      if( nbjets_m >= 4 ){
        h_NJet[mode][11]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][11]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][11]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][11]->Fill(ncjets_m, EventWeight);
        h_MET[mode][11]->Fill(*MET, EventWeight);
        h_WMass[mode][11]->Fill(transverseM, EventWeight);
        h_DPhi[mode][11]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][11]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][11]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][11]->Fill(jet_CSV[j]);
            h_cvsl[mode][11]->Fill(jet_CvsL[j]);
            h_cvsb[mode][11]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][11]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][11]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][11]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][11]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][11]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][11]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][11]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][11]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][11]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][11]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][11]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][11]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][11]->Fill(gendR, EventWeight);
          h_genHm[mode][11]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][11]->Fill(gendR, EventWeight);
            h_matchHm[mode][11]->Fill(genHm, EventWeight);
          }
        }
      }
    }

    //STEP12
    if( njets >= 6 ) {
      h_NJet[mode][12]->Fill(njets, EventWeight);
      h_NBJetCSVv2M[mode][12]->Fill(nbjets_m, EventWeight);
      h_NBJetCSVv2T[mode][12]->Fill(nbjets_t, EventWeight);
      h_NCJetM[mode][12]->Fill(ncjets_m, EventWeight);
      h_MET[mode][12]->Fill(*MET, EventWeight);
      h_WMass[mode][12]->Fill(transverseM, EventWeight);
      h_DPhi[mode][12]->Fill(lepDphi, EventWeight);
      h_LepIso[mode][12]->Fill(relIso, EventWeight);
      if( isQCD ) h_LepIsoQCD[mode][12]->Fill(relIso, EventWeight);

      if( !bestIdxsDR.empty() ){
        for( int i=0; i<bestIdxsDR.size(); ++i ){
          const size_t j = bestIdxsDR[i];
          h_csvv2[mode][12]->Fill(jet_CSV[j]);
          h_cvsl[mode][12]->Fill(jet_CvsL[j]);
          h_cvsb[mode][12]->Fill(jet_CvsB[j]);
        }
      }

      if( nbjets_m >1 ){
        h_bjmDPhi[mode][12]->Fill(bjmDPhi, EventWeight);
        h_bjmDEta[mode][12]->Fill(bjmDEta, EventWeight);
        h_bjmDR[mode][12]->Fill(bjmDR, EventWeight);
        h_HMass_m[mode][12]->Fill(higgsMass_m, EventWeight);
        h_bJetPtHm[mode][12]->Fill(bJetPtHm, EventWeight);
      }

      if( ncjets_m >0 ){
        h_cJetPt[mode][12]->Fill(cjetPt, EventWeight);
      }

      h_DRFCNHkinLepWMass[mode][12]->Fill((lepton+p4met).M(),EventWeight);
      h_DRFCNHkinHadWMass[mode][12]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
      h_DRFCNHkinTopMWb[mode][12]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
      h_DRFCNHkinHMass[mode][12]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
      h_DRFCNHkinDR[mode][12]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
      h_DRFCNHkinTopMHc[mode][12]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

      if(genH.Pt() > 0){
        h_genDR[mode][12]->Fill(gendR, EventWeight);
        h_genHm[mode][12]->Fill(genHm, EventWeight);
        if(match1 && match2){
          h_matchDR[mode][12]->Fill(gendR, EventWeight);
          h_matchHm[mode][12]->Fill(genHm, EventWeight);
        }
      }

      //STEP13
      if( nbjets_m == 3 ){
        h_NJet[mode][13]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][13]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][13]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][13]->Fill(ncjets_m, EventWeight);
        h_MET[mode][13]->Fill(*MET, EventWeight);
        h_WMass[mode][13]->Fill(transverseM, EventWeight);
        h_DPhi[mode][13]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][13]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][13]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][13]->Fill(jet_CSV[j]);
            h_cvsl[mode][13]->Fill(jet_CvsL[j]);
            h_cvsb[mode][13]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][13]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][13]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][13]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][13]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][13]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][13]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][13]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][13]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][13]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][13]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][13]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][13]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][13]->Fill(gendR, EventWeight);
          h_genHm[mode][13]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][13]->Fill(gendR, EventWeight);
            h_matchHm[mode][13]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP14
      if( nbjets_m == 2 || nbjets_m == 3 ){
        h_NJet[mode][14]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][14]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][14]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][14]->Fill(ncjets_m, EventWeight);
        h_MET[mode][14]->Fill(*MET, EventWeight);
        h_WMass[mode][14]->Fill(transverseM, EventWeight);
        h_DPhi[mode][14]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][14]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][14]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][14]->Fill(jet_CSV[j]);
            h_cvsl[mode][14]->Fill(jet_CvsL[j]);
            h_cvsb[mode][14]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][14]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][14]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][14]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][14]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][14]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][14]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][14]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][14]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][14]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][14]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][14]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][14]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][14]->Fill(gendR, EventWeight);
          h_genHm[mode][14]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][14]->Fill(gendR, EventWeight);
            h_matchHm[mode][14]->Fill(genHm, EventWeight);
          }
        }
      }

      //STEP15
      if( nbjets_m >= 3 ){
        h_NJet[mode][15]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][15]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][15]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][15]->Fill(ncjets_m, EventWeight);
        h_MET[mode][15]->Fill(*MET, EventWeight);
        h_WMass[mode][15]->Fill(transverseM, EventWeight);
        h_DPhi[mode][15]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][15]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][15]->Fill(relIso, EventWeight);

        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][15]->Fill(jet_CSV[j]);
            h_cvsl[mode][15]->Fill(jet_CvsL[j]);
            h_cvsb[mode][15]->Fill(jet_CvsB[j]);
          }
        }

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][15]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][15]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][15]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][15]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][15]->Fill(bJetPtHm, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][15]->Fill(cjetPt, EventWeight);
        }

        h_DRFCNHkinLepWMass[mode][15]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][15]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][15]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][15]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][15]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][15]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        if(genH.Pt() > 0){
          h_genDR[mode][15]->Fill(gendR, EventWeight);
          h_genHm[mode][15]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][15]->Fill(gendR, EventWeight);
            h_matchHm[mode][15]->Fill(genHm, EventWeight);
          }
        }
      }
    }
  }
  return kTRUE;
}

void MyAnalysis::SlaveTerminate()
{
  TString option = GetOption();
}
   

void MyAnalysis::Terminate()
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
}

double MyAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

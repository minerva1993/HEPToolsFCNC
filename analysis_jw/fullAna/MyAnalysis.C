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
     for(int i=0; i < 21; i++){
     
      h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()), "Number of jets", 12, 0, 12);
      h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
      h_NJet[ich][i]->Sumw2();
      fOutput->Add(h_NJet[ich][i]);

      h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (medium)", 5, 0, 5);
      h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
      h_NBJetCSVv2M[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2M[ich][i]);

      h_NBJetCSVv2T[ich][i] = new TH1D(Form("h_NBJetCSVv2T_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (tight)", 5, 0, 5);
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

      h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "WMass", 80 ,0 ,160);
      h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[ich][i]->Sumw2();
      fOutput->Add(h_WMass[ich][i]);

      h_HMass_m[ich][i] = new TH1D(Form("h_HMass_m_Ch%i_S%i_%s",ich,i,option.Data()), "HMass (2 medium b jets)", 75 ,50 ,200);
      h_HMass_m[ich][i]->SetXTitle("Higg Mass (GeV)");
      h_HMass_m[ich][i]->Sumw2();
      fOutput->Add(h_HMass_m[ich][i]);

      h_bJetPtHm[ich][i] = new TH1D(Form("h_bJetPtHm_Ch%i_S%i_%s",ich,i,option.Data()), "b jet (medium) pT from H", 56 ,20 ,300);
      h_bJetPtHm[ich][i]->SetXTitle("b Jet (medium) pT from Higgs (GeV)");
      h_bJetPtHm[ich][i]->Sumw2();
      fOutput->Add(h_bJetPtHm[ich][i]);

      h_HMass_t[ich][i] = new TH1D(Form("h_HMass_t_Ch%i_S%i_%s",ich,i,option.Data()), "HMass (2 tight b jets)", 75 ,50 ,200);
      h_HMass_t[ich][i]->SetXTitle("Higg Mass (GeV)");
      h_HMass_t[ich][i]->Sumw2();
      fOutput->Add(h_HMass_t[ich][i]);

      h_bJetPtHt[ich][i] = new TH1D(Form("h_bJetPtHt_Ch%i_S%i_%s",ich,i,option.Data()), "b jet (tight) pT from H", 56 ,20 ,300);
      h_bJetPtHt[ich][i]->SetXTitle("b Jet (tight) pT from Higgs (GeV)");
      h_bJetPtHt[ich][i]->Sumw2();
      fOutput->Add(h_bJetPtHt[ich][i]);

      h_cJetPt[ich][i] = new TH1D(Form("h_cJetPt_Ch%i_S%i_%s",ich,i,option.Data()), "leading c jet (medium) pT", 56 ,20 ,300);
      h_cJetPt[ich][i]->SetXTitle("leading c Jet (medium) pT (GeV)");
      h_cJetPt[ich][i]->Sumw2();
      fOutput->Add(h_cJetPt[ich][i]);

      h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()), "lepton DPhi", 64 ,0 ,3.2);
      h_DPhi[ich][i]->SetXTitle("DPhi");
      h_DPhi[ich][i]->Sumw2();
      fOutput->Add(h_DPhi[ich][i]);

      h_bjmDPhi[ich][i] = new TH1D(Form("h_bjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Phi", 64 ,0 ,3.2);
      h_bjmDPhi[ich][i]->SetXTitle("bjet_m Delta Phi");
      h_bjmDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjmDPhi[ich][i]);

      h_bjmDEta[ich][i] = new TH1D(Form("h_bjmDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Eta", 64 ,0 ,3.2);
      h_bjmDEta[ich][i]->SetXTitle("bjet_m Delta Eta");
      h_bjmDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjmDEta[ich][i]);

      h_bjmDR[ich][i] = new TH1D(Form("h_bjmDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta R", 100 ,0 ,5);
      h_bjmDR[ich][i]->SetXTitle("bjet_m Delta R");
      h_bjmDR[ich][i]->Sumw2();
      fOutput->Add(h_bjmDR[ich][i]);

      h_bjtDPhi[ich][i] = new TH1D(Form("h_bjtDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta Phi", 64 ,0 ,3.2);
      h_bjtDPhi[ich][i]->SetXTitle("bjet_t Delta Phi");
      h_bjtDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjtDPhi[ich][i]);

      h_bjtDEta[ich][i] = new TH1D(Form("h_bjtDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta Eta", 64 ,0 ,3.2);
      h_bjtDEta[ich][i]->SetXTitle("bjet_t Delta Eta");
      h_bjtDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjtDEta[ich][i]);

      h_bjtDR[ich][i] = new TH1D(Form("h_bjtDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta R", 100 ,0 ,5);
      h_bjtDR[ich][i]->SetXTitle("bjet_t Delta R");
      h_bjtDR[ich][i]->Sumw2();
      fOutput->Add(h_bjtDR[ich][i]);

      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 100 ,0 ,1);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

      h_LepIsoQCD[ich][i] = new TH1D(Form("h_LepIsoQCD_Ch%i_S%i_%s",ich,i,option.Data()), "LepIsoQCD", 100 ,0 ,1);
      h_LepIsoQCD[ich][i]->SetXTitle("Relative Isolation");
      h_LepIsoQCD[ich][i]->Sumw2();
      fOutput->Add(h_LepIsoQCD[ich][i]);

      h_kinLepWMass[ich][i] = new TH1D(Form("h_kinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from kinfit (Lep)", 79 ,2 ,160);
      h_kinLepWMass[ich][i]->SetXTitle("W Mass (GeV)");
      h_kinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_kinLepWMass[ich][i]);

      h_kinHadWMass[ich][i] = new TH1D(Form("h_kinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from kinfit (qqb)", 79 ,2 ,160);
      h_kinHadWMass[ich][i]->SetXTitle("W Mass (GeV)");
      h_kinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_kinHadWMass[ich][i]);

      h_fcnhkinWMass[ich][i] = new TH1D(Form("h_fcnhkinWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from fcnh kinfit (Lep)", 79 ,2 ,160);
      h_fcnhkinWMass[ich][i]->SetXTitle("W Mass (GeV)");
      h_fcnhkinWMass[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinWMass[ich][i]);

      h_fcnhkinHMass[ich][i] = new TH1D(Form("h_fcnhkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "H Mass fron fcnh kinfit (bb)", 99 ,2 ,200);
      h_fcnhkinHMass[ich][i]->SetXTitle("H Mass (GeV)");
      h_fcnhkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinHMass[ich][i]);

      h_kinTopMqqb[ich][i] = new TH1D(Form("h_kinTopMqqb_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from kinfit (qqb)", 109 ,2 , 220);
      h_kinTopMqqb[ich][i]->SetXTitle("Top Mass (GeV)");
      h_kinTopMqqb[ich][i]->Sumw2();
      fOutput->Add(h_kinTopMqqb[ich][i]);

      h_kinTopMWb[ich][i] = new TH1D(Form("h_kinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from kinfit (Lep)", 109 , 2 , 220);
      h_kinTopMWb[ich][i]->SetXTitle("Top Mass (GeV)");
      h_kinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_kinTopMWb[ich][i]);

      h_fcnhkinTopMHc[ich][i] = new TH1D(Form("h_fcnhkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc/u fcnhkinfit", 109 ,2 , 220);
      h_fcnhkinTopMHc[ich][i]->SetXTitle("Top Mass (GeV)");
      h_fcnhkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinTopMHc[ich][i]);

      h_fcnhkinTopMWb[ich][i] = new TH1D(Form("h_fcnhkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from fcnhkinfit (Lep)", 109 , 2 , 220);
      h_fcnhkinTopMWb[ich][i]->SetXTitle("Top Mass (GeV)");
      h_fcnhkinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinTopMWb[ich][i]);

      h_fcnhkinNuM[ich][i] = new TH1D(Form("h_fcnhkinNuM_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from fcnhkinfit (Lep)", 100 , 0 , 100);
      h_fcnhkinNuM[ich][i]->SetXTitle("Nu Mass (=0) (GeV)");
      h_fcnhkinNuM[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinNuM[ich][i]);

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

    vector<float> jet_cvsl;
    vector<float> bjm_csv;
    vector<float> bjt_csv;

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
    vector<TLorentzVector> v_bjet_t;
    vector<TLorentzVector> v_jet;

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;

        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          v_bjet_m.push_back(jet);
          bjm_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CSV[iJet] > 0.9535 ){
          nbjets_t++;
          v_bjet_t.push_back(jet);
          bjt_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsL[iJet] > 0.08 ){
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

    if( nbjets_t >1 ){

      double tmp_bjtDR = 999;
      double tmp_higgsMass_t  = 9999;
      double tmp_bjtDEta = 999;
      double tmp_bjtDPhi = 999;
      double tmp_bjtPt1 = 9999;
      double tmp_bjtPt2 = 9999;

      for(int m = 0; m < nbjets_t; m++){
        for(int n = 1; n <  nbjets_t; n++){
          if(m < n){
            tmp_bjtDR = v_bjet_t[m].DeltaR(v_bjet_t[n]);
            tmp_higgsMass_t = (v_bjet_t[m] + v_bjet_t[n]).M();
            tmp_bjtDEta = v_bjet_t[m].Eta()-v_bjet_t[n].Eta();
            tmp_bjtDPhi = v_bjet_t[m].DeltaPhi(v_bjet_t[n]);
            tmp_bjtPt1 = v_bjet_t[m].Pt();
            tmp_bjtPt2 = v_bjet_t[n].Pt();

            if( tmp_bjtDR < bjtDR ){
              bjtDR = tmp_bjtDR;
              higgsMass_t = tmp_higgsMass_t;
              bjtDEta = tmp_bjtDEta;
              bjtDPhi = tmp_bjtDPhi;

              if( tmp_bjtPt1 > tmp_bjtPt2) bJetPtHt = tmp_bjtPt1;
              else                         bJetPtHt = tmp_bjtPt2;

            }
          }
        }
      }
    }

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

    if( nbjets_m >1 ){
      h_bjmDPhi[mode][0]->Fill(bjmDPhi, EventWeight);
      h_bjmDEta[mode][0]->Fill(bjmDEta, EventWeight);
      h_bjmDR[mode][0]->Fill(bjmDR, EventWeight);
      h_HMass_m[mode][0]->Fill(higgsMass_m, EventWeight);
      h_bJetPtHm[mode][0]->Fill(bJetPtHm, EventWeight);
    }

    if( nbjets_t >1 ){
      h_bjtDPhi[mode][0]->Fill(bjtDPhi, EventWeight);
      h_bjtDEta[mode][0]->Fill(bjtDEta, EventWeight);
      h_bjtDR[mode][0]->Fill(bjtDR, EventWeight);
      h_HMass_t[mode][0]->Fill(higgsMass_t, EventWeight);
      h_bJetPtHt[mode][0]->Fill(bJetPtHt, EventWeight);
    }

    if( ncjets_m >0 ){
      h_cJetPt[mode][0]->Fill(cjetPt, EventWeight);
    }
    //kinfits
    //if(*kin_chi2 < 10000){
      h_kinLepWMass[mode][0]->Fill(kinLepW.M(),EventWeight);
      h_kinHadWMass[mode][0]->Fill(kinLepW.M(),EventWeight);
      h_kinTopMWb[mode][0]->Fill(kinLepT.M(),EventWeight);
      h_kinTopMqqb[mode][0]->Fill(kinHadT.M(),EventWeight);
    //}
    //if(*fcnhkin_chi2 < 10000){
      h_fcnhkinWMass[mode][0]->Fill(fcnhkinLepW.M(),EventWeight);
      h_fcnhkinHMass[mode][0]->Fill(fcnhkinH.M(),EventWeight);
      h_fcnhkinTopMWb[mode][0]->Fill(fcnhkinLepT.M(),EventWeight);
      h_fcnhkinTopMHc[mode][0]->Fill(fcnhkinHcT.M(),EventWeight);
      h_fcnhkinNuM[mode][0]->Fill(fcnhkinnu.M(),EventWeight);
    //}

//step1

      if( njets == 3 ) {
        h_NJet[mode][1]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][1]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][1]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][1]->Fill(ncjets_m, EventWeight);
        h_MET[mode][1]->Fill(*MET, EventWeight);
        h_WMass[mode][1]->Fill(transverseM, EventWeight);
        h_DPhi[mode][1]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][1]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][1]->Fill(relIso, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][1]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][1]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][1]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][1]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][1]->Fill(bJetPtHm, EventWeight);
        }

        if( nbjets_t >1 ){
          h_bjtDPhi[mode][1]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][1]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][1]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][1]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][1]->Fill(bJetPtHt, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][1]->Fill(cjetPt, EventWeight);
        }
        //kinfits
        if(*kin_chi2 < 10000){
          h_kinLepWMass[mode][1]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][1]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][1]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][1]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 10000){
          h_fcnhkinWMass[mode][1]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][1]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][1]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][1]->Fill(fcnhkinHcT.M(),EventWeight);
        }
/*
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
        }
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
        }
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
        }
        if( nbjets_m >= 2 ){
          h_NJet[mode][5]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][5]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][5]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][5]->Fill(ncjets_m, EventWeight);
          h_MET[mode][5]->Fill(*MET, EventWeight);
          h_WMass[mode][5]->Fill(transverseM, EventWeight);
          h_DPhi[mode][5]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][5]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][5]->Fill(relIso, EventWeight);
        }
*/
    }
     
      if( njets == 4) {
        h_NJet[mode][6]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][6]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][6]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][6]->Fill(ncjets_m, EventWeight);
        h_MET[mode][6]->Fill(*MET, EventWeight);
        h_WMass[mode][6]->Fill(transverseM, EventWeight);
        h_DPhi[mode][6]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][6]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][6]->Fill(relIso, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][6]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][6]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][6]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][6]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][6]->Fill(bJetPtHm, EventWeight);
        }

        if( nbjets_t >1 ){
          h_bjtDPhi[mode][6]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][6]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][6]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][6]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][6]->Fill(bJetPtHt, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][6]->Fill(cjetPt, EventWeight);
        }
        //kinfits
        if(*kin_chi2 < 10000){
          h_kinLepWMass[mode][6]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][6]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][6]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][6]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 10000){
          h_fcnhkinWMass[mode][6]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][6]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][6]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][6]->Fill(fcnhkinHcT.M(),EventWeight);
        }
/*
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
        }
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
        }
        if( nbjets_m == 2 || nbjets_m == 3 ){
          h_NJet[mode][9]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][9]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][9]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][9]->Fill(ncjets_m, EventWeight);
          h_MET[mode][9]->Fill(*MET, EventWeight);
          h_WMass[mode][9]->Fill(transverseM, EventWeight);
          h_DPhi[mode][9]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][9]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][9]->Fill(relIso, EventWeight);
        }
        if( nbjets_m >= 2 ){
          h_NJet[mode][10]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][10]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][10]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][10]->Fill(ncjets_m, EventWeight);
          h_MET[mode][10]->Fill(*MET, EventWeight);
          h_WMass[mode][10]->Fill(transverseM, EventWeight);
          h_DPhi[mode][10]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][10]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][10]->Fill(relIso, EventWeight);
        }
*/
      }

      if( njets >= 3 ) {
        h_NJet[mode][11]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][11]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][11]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][11]->Fill(ncjets_m, EventWeight);
        h_MET[mode][11]->Fill(*MET, EventWeight);
        h_WMass[mode][11]->Fill(transverseM, EventWeight);
        h_DPhi[mode][11]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][11]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][11]->Fill(relIso, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][11]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][11]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][11]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][11]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][11]->Fill(bJetPtHm, EventWeight);
        }

        if( nbjets_t >1 ){
          h_bjtDPhi[mode][11]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][11]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][11]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][11]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][11]->Fill(bJetPtHt, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][11]->Fill(cjetPt, EventWeight);
        }
        //kinfits
        if(*kin_chi2 < 10000){
          h_kinLepWMass[mode][11]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][11]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][11]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][11]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 10000){
          h_fcnhkinWMass[mode][11]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][11]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][11]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][11]->Fill(fcnhkinHcT.M(),EventWeight);
        }
/*
        if( nbjets_m == 2 ){
          h_NJet[mode][12]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][12]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][12]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][12]->Fill(ncjets_m, EventWeight);
          h_MET[mode][12]->Fill(*MET, EventWeight);
          h_WMass[mode][12]->Fill(transverseM, EventWeight);
          h_DPhi[mode][12]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][12]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][12]->Fill(relIso, EventWeight);
        }
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
        }
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
        }
        if( nbjets_m >= 2 ){
          h_NJet[mode][15]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][15]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][15]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][15]->Fill(ncjets_m, EventWeight);
          h_MET[mode][15]->Fill(*MET, EventWeight);
          h_WMass[mode][15]->Fill(transverseM, EventWeight);
          h_DPhi[mode][15]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][15]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][15]->Fill(relIso, EventWeight);
        }
*/
      }

      if( njets >= 4 ) {
        h_NJet[mode][16]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][16]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][16]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][16]->Fill(ncjets_m, EventWeight);
        h_MET[mode][16]->Fill(*MET, EventWeight);
        h_WMass[mode][16]->Fill(transverseM, EventWeight);
        h_DPhi[mode][16]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][16]->Fill(relIso, EventWeight);
        if( isQCD ) h_LepIsoQCD[mode][16]->Fill(relIso, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][16]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][16]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][16]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][16]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][16]->Fill(bJetPtHm, EventWeight);
        }

        if( nbjets_t >1 ){
          h_bjtDPhi[mode][16]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][16]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][16]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][16]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][16]->Fill(bJetPtHt, EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][16]->Fill(cjetPt, EventWeight);
        }
        //kinfits
        if(*kin_chi2 < 10000){
          h_kinLepWMass[mode][16]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][16]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][16]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][16]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 10000){
          h_fcnhkinWMass[mode][16]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][16]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][16]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][16]->Fill(fcnhkinHcT.M(),EventWeight);
        }
/*
        if( nbjets_m == 2 ){
          h_NJet[mode][17]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][17]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][17]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][17]->Fill(ncjets_m, EventWeight);
          h_MET[mode][17]->Fill(*MET, EventWeight);
          h_WMass[mode][17]->Fill(transverseM, EventWeight);
          h_DPhi[mode][17]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][17]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][17]->Fill(relIso, EventWeight);
        }
        if( nbjets_m == 3 ){
          h_NJet[mode][18]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][18]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][18]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][18]->Fill(ncjets_m, EventWeight);
          h_MET[mode][18]->Fill(*MET, EventWeight);
          h_WMass[mode][18]->Fill(transverseM, EventWeight);
          h_DPhi[mode][18]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][18]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][18]->Fill(relIso, EventWeight);
        }
        if( nbjets_m == 2 || nbjets_m == 3 ){
          h_NJet[mode][19]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][19]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][19]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][19]->Fill(ncjets_m, EventWeight);
          h_MET[mode][19]->Fill(*MET, EventWeight);
          h_WMass[mode][19]->Fill(transverseM, EventWeight);
          h_DPhi[mode][19]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][19]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][19]->Fill(relIso, EventWeight);
        }
*/
        if( nbjets_m >= 3 ){
          h_NJet[mode][20]->Fill(njets, EventWeight);
          h_NJet[mode][20]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][20]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][20]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][20]->Fill(ncjets_m, EventWeight);
          h_MET[mode][20]->Fill(*MET, EventWeight);
          h_WMass[mode][20]->Fill(transverseM, EventWeight);
          h_DPhi[mode][20]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][20]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][20]->Fill(relIso, EventWeight);

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][20]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][20]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][20]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][20]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][20]->Fill(bJetPtHm, EventWeight);
          }

          if( nbjets_t >1 ){
            h_bjtDPhi[mode][20]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][20]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][20]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][20]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][20]->Fill(bJetPtHt, EventWeight);
          }

          if( ncjets_m >0 ){
            h_cJetPt[mode][20]->Fill(cjetPt, EventWeight);
          }
          //kinfits
          if(*kin_chi2 < 10000){
            h_kinLepWMass[mode][20]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][20]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][20]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][20]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 10000){
            h_fcnhkinWMass[mode][20]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][20]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][20]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][20]->Fill(fcnhkinHcT.M(),EventWeight);
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

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
/*
      h_HMass_t[ich][i] = new TH1D(Form("h_HMassT_Ch%i_S%i_%s",ich,i,option.Data()), "Di-bjet Mass (tight) wrt min DR", 60 ,0 ,300);
      h_HMass_t[ich][i]->SetXTitle("Di-bjet (tight) Mass (GeV)");
      h_HMass_t[ich][i]->Sumw2();
      fOutput->Add(h_HMass_t[ich][i]);

      h_bJetPtHt[ich][i] = new TH1D(Form("h_bJetPtHt_Ch%i_S%i_%s",ich,i,option.Data()), "Leading b jet (tight) pT from the di-bjets", 54 ,30 ,300);
      h_bJetPtHt[ich][i]->SetXTitle("b Jet (tight) pT from Higgs (GeV)");
      h_bJetPtHt[ich][i]->Sumw2();
      fOutput->Add(h_bJetPtHt[ich][i]);
*/
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
/*
      h_bjtDPhi[ich][i] = new TH1D(Form("h_bjtDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta Phi", 32 ,0 ,3.2);
      h_bjtDPhi[ich][i]->SetXTitle("bjet_t Delta Phi");
      h_bjtDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjtDPhi[ich][i]);

      h_bjtDEta[ich][i] = new TH1D(Form("h_bjtDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta Eta", 32 ,0 ,3.2);
      h_bjtDEta[ich][i]->SetXTitle("bjet_t Delta Eta");
      h_bjtDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjtDEta[ich][i]);

      h_bjtDR[ich][i] = new TH1D(Form("h_bjtDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta R", 50 ,0 ,5);
      h_bjtDR[ich][i]->SetXTitle("bjet_t Delta R");
      h_bjtDR[ich][i]->Sumw2();
      fOutput->Add(h_bjtDR[ich][i]);
*/
      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 20 ,0 ,0.2);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

      h_LepIsoQCD[ich][i] = new TH1D(Form("h_LepIsoQCD_Ch%i_S%i_%s",ich,i,option.Data()), "LepIsoQCD", 20 ,0 ,0.2);
      h_LepIsoQCD[ich][i]->SetXTitle("Relative Isolation (QCD)");
      h_LepIsoQCD[ich][i]->Sumw2();
      fOutput->Add(h_LepIsoQCD[ich][i]);
/*
      h_kinLepWMass[ich][i] = new TH1D(Form("h_kinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from kinfit (Lep)", 20 ,60 ,100);
      h_kinLepWMass[ich][i]->SetXTitle("W Mass from kinfit (Lep) (GeV)");
      h_kinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_kinLepWMass[ich][i]);

      h_kinHadWMass[ich][i] = new TH1D(Form("h_kinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from kinfit (qqb)", 20 ,60 ,100);
      h_kinHadWMass[ich][i]->SetXTitle("W Mass from kinfit (qqb) (GeV)");
      h_kinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_kinHadWMass[ich][i]);

      h_kinTopMqqb[ich][i] = new TH1D(Form("h_kinTopMqqb_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from kinfit (qqb)", 25 , 150 , 200);
      h_kinTopMqqb[ich][i]->SetXTitle("Top Mass from kinfit (qqb) (GeV)");
      h_kinTopMqqb[ich][i]->Sumw2();
      fOutput->Add(h_kinTopMqqb[ich][i]);

      h_kinTopMWb[ich][i] = new TH1D(Form("h_kinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from kinfit (Lep)", 25 , 150 , 200);
      h_kinTopMWb[ich][i]->SetXTitle("Top Mass from kinfit (Lep) (GeV)");
      h_kinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_kinTopMWb[ich][i]);

      h_fcnhkinWMass[ich][i] = new TH1D(Form("h_fcnhkinWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from fcnhkinfit (Lep)", 39 , 5 ,200);
      h_fcnhkinWMass[ich][i]->SetXTitle("W Mass from fcnhkinfit (Lep) (GeV)");
      h_fcnhkinWMass[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinWMass[ich][i]);

      h_fcnhkinHMass[ich][i] = new TH1D(Form("h_fcnhkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "H Mass from fcnhkinfit (bb)", 25 ,100 ,150);
      h_fcnhkinHMass[ich][i]->SetXTitle("H Mass from fcnhkinfit (bb)(GeV)");
      h_fcnhkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinHMass[ich][i]);

      h_fcnhkinTopMHc[ich][i] = new TH1D(Form("h_fcnhkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc/u fcnhkinfit", 26 , 120 , 250);
      h_fcnhkinTopMHc[ich][i]->SetXTitle("Top Mass from Hc/u fcnhkinfit (GeV)");
      h_fcnhkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinTopMHc[ich][i]);

      h_fcnhkinTopMWb[ich][i] = new TH1D(Form("h_fcnhkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from fcnhkinfit (Lep)", 26 , 120 , 250);
      h_fcnhkinTopMWb[ich][i]->SetXTitle("Top Mass from fcnhkinfit (Lep) (GeV)");
      h_fcnhkinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_fcnhkinTopMWb[ich][i]);

      h_m3FCNHkinLepWMass[ich][i] = new TH1D(Form("h_m3FCNHkinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from m3FCNHkinfit (Lep)", 29 ,10 ,300);
      h_m3FCNHkinLepWMass[ich][i]->SetXTitle("W Mass from m3FCNHkinfit (Lep) (GeV)");
      h_m3FCNHkinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_m3FCNHkinLepWMass[ich][i]);

      h_m3FCNHkinHadWMass[ich][i] = new TH1D(Form("h_m3FCNHkinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass from m3FCNHkinfit (Had)", 29 ,10 ,300);
      h_m3FCNHkinHadWMass[ich][i]->SetXTitle("W Mass from m3FCNHkinfit (Had) (GeV)");
      h_m3FCNHkinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_m3FCNHkinHadWMass[ich][i]);

      h_m3FCNHkinHMass[ich][i] = new TH1D(Form("h_m3FCNHkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "H Mass from m3FCNHkinfit (bb)", 29 ,10 ,300);
      h_m3FCNHkinHMass[ich][i]->SetXTitle("H Mass from m3FCNHkinfit (bb)(GeV)");
      h_m3FCNHkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_m3FCNHkinHMass[ich][i]);

      h_m3FCNHkinDR[ich][i] = new TH1D(Form("h_m3FCNHkinDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R from m3FCNHkinfit", 49, 0.1 ,5);
      h_m3FCNHkinDR[ich][i]->SetXTitle("Delta R from m3FCNHkinfit Higgs");
      h_m3FCNHkinDR[ich][i]->Sumw2();
      fOutput->Add(h_m3FCNHkinDR[ich][i]);

      h_m3FCNHkinTopMWb[ich][i] = new TH1D(Form("h_m3FCNHkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from m3FCNHkinfit (Lep)", 29 , 10 , 300);
      h_m3FCNHkinTopMWb[ich][i]->SetXTitle("Top Mass from m3FCNHkinfit (Lep) (GeV)");
      h_m3FCNHkinTopMWb[ich][i]->Sumw2();
       fOutput->Add(h_m3FCNHkinTopMWb[ich][i]);

      h_m3FCNHkinTopMHc[ich][i] = new TH1D(Form("h_m3FCNHkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc/u m3FCNHkinfit", 29 , 10 , 300);
      h_m3FCNHkinTopMHc[ich][i]->SetXTitle("Top Mass from Hc/u m3FCNHkinfit (GeV)");
      h_m3FCNHkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_m3FCNHkinTopMHc[ich][i]);
*/
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
        if( jet_CSV[iJet] > 0.9535 ){
          nbjets_t++;
          v_bjet_t.push_back(jet);
          bjt_csv.push_back(jet_CSV[iJet]);
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
/*
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
*/
/*
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
*/

    //DR kin
    std::vector<size_t> bestIdxsDR;
    TLorentzVector jetP4sDR[4];

    if( njets >= 3){
      double minDR = 1e9;
      for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ) {

        if (jet_pT[*ii1] < 30 || jet_eta[*ii1] > 2.4) continue;

        jetP4sDR[1].SetPtEtaPhiE(jet_pT[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_E[*ii1]);
        for ( auto ii2 = ii1+1; ii2 != jetIdxs.end(); ++ii2 ) {

          if (jet_pT[*ii2] < 30 || jet_eta[*ii2] > 2.4) continue;

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
/*
    if( nbjets_t >1 ){
      h_bjtDPhi[mode][0]->Fill(bjtDPhi, EventWeight);
      h_bjtDEta[mode][0]->Fill(bjtDEta, EventWeight);
      h_bjtDR[mode][0]->Fill(bjtDR, EventWeight);
      h_HMass_t[mode][0]->Fill(higgsMass_t, EventWeight);
      h_bJetPtHt[mode][0]->Fill(bJetPtHt, EventWeight);
    }
*/
    if( ncjets_m >0 ){
      h_cJetPt[mode][0]->Fill(cjetPt, EventWeight);
    }
/*
    //kinfits
    if(*kin_chi2 < 200000){
      h_kinLepWMass[mode][0]->Fill(kinLepW.M(),EventWeight);
      h_kinHadWMass[mode][0]->Fill(kinLepW.M(),EventWeight);
      h_kinTopMWb[mode][0]->Fill(kinLepT.M(),EventWeight);
      h_kinTopMqqb[mode][0]->Fill(kinHadT.M(),EventWeight);
    }
    if(*fcnhkin_chi2 < 200000){
      h_fcnhkinWMass[mode][0]->Fill(fcnhkinLepW.M(),EventWeight);
      h_fcnhkinHMass[mode][0]->Fill(fcnhkinH.M(),EventWeight);
      h_fcnhkinTopMWb[mode][0]->Fill(fcnhkinLepT.M(),EventWeight);
      h_fcnhkinTopMHc[mode][0]->Fill(fcnhkinHcT.M(),EventWeight);
      //h_fcnhkinNuM[mode][0]->Fill(fcnhkinnu.M(),EventWeight);
    }

    h_m3FCNHkinLepWMass[mode][0]->Fill((lepton+p4met).M(),EventWeight);
    h_m3FCNHkinHadWMass[mode][0]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
    h_m3FCNHkinTopMWb[mode][0]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
    h_m3FCNHkinHMass[mode][0]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
    h_m3FCNHkinDR[mode][0]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
    h_m3FCNHkinTopMHc[mode][0]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
    h_DRFCNHkinLepWMass[mode][0]->Fill((lepton+p4met).M(),EventWeight);
    h_DRFCNHkinHadWMass[mode][0]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
    h_DRFCNHkinTopMWb[mode][0]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
    h_DRFCNHkinHMass[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
    h_DRFCNHkinDR[mode][0]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
    h_DRFCNHkinTopMHc[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][1]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][1]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][1]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][1]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][1]->Fill(bJetPtHm, EventWeight);
        }
/*
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][1]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][1]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][1]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][1]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][1]->Fill(bJetPtHt, EventWeight);
        }
*/
        if( ncjets_m >0 ){
          h_cJetPt[mode][1]->Fill(cjetPt, EventWeight);
        }
/*
        //kinfits
        if(*kin_chi2 < 200000){
          h_kinLepWMass[mode][1]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][1]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][1]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][1]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 200000){
          h_fcnhkinWMass[mode][1]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][1]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][1]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][1]->Fill(fcnhkinHcT.M(),EventWeight);
        }

        h_m3FCNHkinLepWMass[mode][1]->Fill((lepton+p4met).M(),EventWeight);
        h_m3FCNHkinHadWMass[mode][1]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
        h_m3FCNHkinTopMWb[mode][1]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
        h_m3FCNHkinHMass[mode][1]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
        h_m3FCNHkinDR[mode][1]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
        h_m3FCNHkinTopMHc[mode][1]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
        h_DRFCNHkinLepWMass[mode][1]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][1]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][1]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][1]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

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

          if( nbjets_m > 1 ){
            h_bjmDPhi[mode][2]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][2]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][2]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][2]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][2]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1  ){
            h_bjtDPhi[mode][2]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][2]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][2]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][2]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][2]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][2]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][2]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][2]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][2]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][2]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][2]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][2]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][2]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][2]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][2]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][2]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][2]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][2]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][2]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][2]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][2]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][2]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][2]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][2]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][2]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][2]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m > 1 ){
            h_bjmDPhi[mode][3]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][3]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][3]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][3]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][3]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t > 1 ){
            h_bjtDPhi[mode][3]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][3]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][3]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][3]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][3]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][3]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][3]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][3]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][3]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][3]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][3]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][3]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][3]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][3]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][3]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][3]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][3]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][3]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][3]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][3]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][3]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][3]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][3]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][3]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][3]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][3]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m > 1 ){
            h_bjmDPhi[mode][4]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][4]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][4]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][4]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][4]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t > 1 ){
            h_bjtDPhi[mode][4]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][4]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][4]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][4]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][4]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][4]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][4]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][4]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][4]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][4]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][4]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][4]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][4]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][4]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][4]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][4]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][4]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][4]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][4]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][4]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][4]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][4]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][4]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][4]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][4]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][4]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m > 1 ){
            h_bjmDPhi[mode][5]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][5]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][5]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][5]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][5]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t > 1 ){
            h_bjtDPhi[mode][5]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][5]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][5]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][5]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][5]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][5]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][5]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][5]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][5]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][5]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][5]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][5]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][5]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][5]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][5]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][5]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][5]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][5]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][5]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][5]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][5]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][5]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][5]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][5]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][5]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][5]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][6]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][6]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][6]->Fill(bjmDR, EventWeight);
          h_HMass_m[mode][6]->Fill(higgsMass_m, EventWeight);
          h_bJetPtHm[mode][6]->Fill(bJetPtHm, EventWeight);
        }
/*
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][6]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][6]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][6]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][6]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][6]->Fill(bJetPtHt, EventWeight);
        }
*/
        if( ncjets_m >0 ){
          h_cJetPt[mode][6]->Fill(cjetPt, EventWeight);
        }
/*
        //kinfits
        if(*kin_chi2 < 200000){
          h_kinLepWMass[mode][6]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][6]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][6]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][6]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 200000){
          h_fcnhkinWMass[mode][6]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][6]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][6]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][6]->Fill(fcnhkinHcT.M(),EventWeight);
        }

        h_m3FCNHkinLepWMass[mode][6]->Fill((lepton+p4met).M(),EventWeight);
        h_m3FCNHkinHadWMass[mode][6]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
        h_m3FCNHkinTopMWb[mode][6]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
        h_m3FCNHkinHMass[mode][6]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
        h_m3FCNHkinDR[mode][6]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
        h_m3FCNHkinTopMHc[mode][6]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
        h_DRFCNHkinLepWMass[mode][6]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][6]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][6]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][6]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][6]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][6]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

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

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][7]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][7]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][7]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][7]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][7]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][7]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][7]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][7]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][7]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][7]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][7]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][7]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][7]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][7]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][7]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][7]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][7]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][7]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][7]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][7]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][7]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][7]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][7]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][7]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][7]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][7]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][7]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][7]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][7]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][7]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][7]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][8]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][8]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][8]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][8]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][8]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][8]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][8]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][8]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][8]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][8]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][8]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][8]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][8]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][8]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][8]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][8]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][8]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][8]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][8]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][8]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][8]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][8]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][8]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][8]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][8]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][8]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][8]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][8]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][8]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][8]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][8]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        }

        //STEP9
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
          
          if( nbjets_m >1 ){
            h_bjmDPhi[mode][9]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][9]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][9]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][9]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][9]->Fill(bJetPtHm, EventWeight);
          }
/*          
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][9]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][9]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][9]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][9]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][9]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][9]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][9]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][9]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][9]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][9]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][9]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][9]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][9]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][9]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][9]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][9]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][9]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][9]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][9]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][9]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][9]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][9]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][9]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][9]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][9]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][9]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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
          
          if( nbjets_m >1 ){
            h_bjmDPhi[mode][10]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][10]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][10]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][10]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][10]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][10]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][10]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][10]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][10]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][10]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][10]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][10]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][10]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][10]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][10]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][10]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][10]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][10]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][10]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][10]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][10]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][10]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][10]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][10]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][10]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][10]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][10]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][10]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][10]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][10]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][10]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        }
      }


      //STEP11
      if( njets >= 6 ) {
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
/*
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][11]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][11]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][11]->Fill(bjtDR, EventWeight);
          h_HMass_t[mode][11]->Fill(higgsMass_t, EventWeight);
          h_bJetPtHt[mode][11]->Fill(bJetPtHt, EventWeight);
        }
*/
        if( ncjets_m >0 ){
          h_cJetPt[mode][11]->Fill(cjetPt, EventWeight);
        }
/*
        //kinfits
        if(*kin_chi2 < 200000){
          h_kinLepWMass[mode][11]->Fill(kinLepW.M(),EventWeight);
          h_kinHadWMass[mode][11]->Fill(kinLepW.M(),EventWeight);
          h_kinTopMWb[mode][11]->Fill(kinLepT.M(),EventWeight);
          h_kinTopMqqb[mode][11]->Fill(kinHadT.M(),EventWeight);
        }
        if(*fcnhkin_chi2 < 200000){
          h_fcnhkinWMass[mode][11]->Fill(fcnhkinLepW.M(),EventWeight);
          h_fcnhkinHMass[mode][11]->Fill(fcnhkinH.M(),EventWeight);
          h_fcnhkinTopMWb[mode][11]->Fill(fcnhkinLepT.M(),EventWeight);
          h_fcnhkinTopMHc[mode][11]->Fill(fcnhkinHcT.M(),EventWeight);
        }

        h_m3FCNHkinLepWMass[mode][11]->Fill((lepton+p4met).M(),EventWeight);
        h_m3FCNHkinHadWMass[mode][11]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
        h_m3FCNHkinTopMWb[mode][11]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
        h_m3FCNHkinHMass[mode][11]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
        h_m3FCNHkinDR[mode][11]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
        h_m3FCNHkinTopMHc[mode][11]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
        h_DRFCNHkinLepWMass[mode][11]->Fill((lepton+p4met).M(),EventWeight);
        h_DRFCNHkinHadWMass[mode][11]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        h_DRFCNHkinTopMWb[mode][11]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
        h_DRFCNHkinHMass[mode][11]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][11]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][11]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

        //STEP12
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

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][12]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][12]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][12]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][12]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][12]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][12]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][12]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][12]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][12]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][12]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][12]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][12]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][12]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][12]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][12]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][12]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][12]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][12]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][12]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][12]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][12]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][12]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][12]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][12]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][12]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][12]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][12]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][12]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][12]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][12]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][12]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][13]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][13]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][13]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][13]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][13]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][13]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][13]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][13]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][13]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][13]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][13]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][13]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][13]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][13]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][13]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][13]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][13]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][13]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][13]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][13]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][13]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][13]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][13]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][13]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][13]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][13]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][13]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][13]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][13]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][13]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][13]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][14]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][14]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][14]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][14]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][14]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][14]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][14]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][14]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][14]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][14]->Fill(bJetPtHt, EventWeight);
          }
*/
          if( ncjets_m >0 ){
            h_cJetPt[mode][14]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 10000){
            h_kinLepWMass[mode][14]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][14]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][14]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][14]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 10000){
            h_fcnhkinWMass[mode][14]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][14]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][14]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][14]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][14]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][14]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][14]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][14]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][14]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][14]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][14]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][14]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][14]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][14]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][14]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][14]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

          if( nbjets_m >1 ){
            h_bjmDPhi[mode][15]->Fill(bjmDPhi, EventWeight);
            h_bjmDEta[mode][15]->Fill(bjmDEta, EventWeight);
            h_bjmDR[mode][15]->Fill(bjmDR, EventWeight);
            h_HMass_m[mode][15]->Fill(higgsMass_m, EventWeight);
            h_bJetPtHm[mode][15]->Fill(bJetPtHm, EventWeight);
          }
/*
          if( nbjets_t >1 ){
            h_bjtDPhi[mode][15]->Fill(bjtDPhi, EventWeight);
            h_bjtDEta[mode][15]->Fill(bjtDEta, EventWeight);
            h_bjtDR[mode][15]->Fill(bjtDR, EventWeight);
            h_HMass_t[mode][15]->Fill(higgsMass_t, EventWeight);
            h_bJetPtHt[mode][15]->Fill(bJetPtHt, EventWeight);
          }
*/          
          if( ncjets_m >0 ){
            h_cJetPt[mode][15]->Fill(cjetPt, EventWeight);
          }
/*
          //kinfits
          if(*kin_chi2 < 200000){
            h_kinLepWMass[mode][15]->Fill(kinLepW.M(),EventWeight);
            h_kinHadWMass[mode][15]->Fill(kinLepW.M(),EventWeight);
            h_kinTopMWb[mode][15]->Fill(kinLepT.M(),EventWeight);
            h_kinTopMqqb[mode][15]->Fill(kinHadT.M(),EventWeight);
          }
          if(*fcnhkin_chi2 < 200000){
            h_fcnhkinWMass[mode][15]->Fill(fcnhkinLepW.M(),EventWeight);
            h_fcnhkinHMass[mode][15]->Fill(fcnhkinH.M(),EventWeight);
            h_fcnhkinTopMWb[mode][15]->Fill(fcnhkinLepT.M(),EventWeight);
            h_fcnhkinTopMHc[mode][15]->Fill(fcnhkinHcT.M(),EventWeight);
          }

          h_m3FCNHkinLepWMass[mode][15]->Fill((lepton+p4met).M(),EventWeight);
          h_m3FCNHkinHadWMass[mode][15]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_m3FCNHkinTopMWb[mode][15]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_m3FCNHkinHMass[mode][15]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_m3FCNHkinDR[mode][15]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_m3FCNHkinTopMHc[mode][15]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
*/
          h_DRFCNHkinLepWMass[mode][15]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][15]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][15]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][15]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][15]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][15]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
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

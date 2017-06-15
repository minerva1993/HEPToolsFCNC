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

      h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()), "MET", 40,0,400);
      h_MET[ich][i]->SetXTitle("MET (GeV)");
      h_MET[ich][i]->Sumw2();
      fOutput->Add(h_MET[ich][i]);

      h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "WMass", 32 ,0 ,160);
      h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[ich][i]->Sumw2();
      fOutput->Add(h_WMass[ich][i]);

      h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()), "lepton DPhi", 64 ,0 ,3.2);
      h_DPhi[ich][i]->SetXTitle("DPhi");
      h_DPhi[ich][i]->Sumw2();
      fOutput->Add(h_DPhi[ich][i]);

      h_jDPhi[ich][i] = new TH1D(Form("h_jDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "jet Delta Phi", 64 ,0 ,3.2);
      h_jDPhi[ich][i]->SetXTitle("jet Delta Phi");
      h_jDPhi[ich][i]->Sumw2();
      fOutput->Add(h_jDPhi[ich][i]);

      h_jDEta[ich][i] = new TH1D(Form("h_jDEta_Ch%i_S%i_%s",ich,i,option.Data()), "jet Delta Eta", 64 ,0 ,3.2);
      h_jDEta[ich][i]->SetXTitle("jet Delta Eta");
      h_jDEta[ich][i]->Sumw2();
      fOutput->Add(h_jDEta[ich][i]);

      h_jDR[ich][i] = new TH1D(Form("h_jDR_Ch%i_S%i_%s",ich,i,option.Data()), "jet Delta R", 64 ,0 ,3.2);
      h_jDR[ich][i]->SetXTitle("jet Delta R");
      h_jDR[ich][i]->Sumw2();
      fOutput->Add(h_jDR[ich][i]);

      h_jDPhiDEta[ich][i] = new TH2D(Form("h_jDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()),"jet Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_jDPhiDEta[ich][i]->SetYTitle("Delta Eta (jj)");
      h_jDPhiDEta[ich][i]->SetXTitle("Delta Phi (jj)");
      h_jDPhiDEta[ich][i]->Sumw2();
      fOutput->Add(h_jDPhiDEta[ich][i]);

      h_bjmDPhi[ich][i] = new TH1D(Form("h_bjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Phi", 64 ,0 ,3.2);
      h_bjmDPhi[ich][i]->SetXTitle("bjet_m Delta Phi");
      h_bjmDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjmDPhi[ich][i]);

      h_bjmDEta[ich][i] = new TH1D(Form("h_bjmDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Eta", 64 ,0 ,3.2);
      h_bjmDEta[ich][i]->SetXTitle("bjet_m Delta Eta");
      h_bjmDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjmDEta[ich][i]);

      h_bjmDR[ich][i] = new TH1D(Form("h_bjmDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta R", 64 ,0 ,3.2);
      h_bjmDR[ich][i]->SetXTitle("bjet_m Delta R");
      h_bjmDR[ich][i]->Sumw2();
      fOutput->Add(h_bjmDR[ich][i]);

      h_bjmDPhiDEta[ich][i] = new TH2D(Form("h_bjmDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()),"bjet_m Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_bjmDPhiDEta[ich][i]->SetYTitle("Delta Eta (bb)");
      h_bjmDPhiDEta[ich][i]->SetXTitle("Delta Phi (bb)");
      h_bjmDPhiDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjmDPhiDEta[ich][i]);

      h_bjtDPhi[ich][i] = new TH1D(Form("h_bjtDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta Phi", 64 ,0 ,3.2);
      h_bjtDPhi[ich][i]->SetXTitle("bjet_t Delta Phi");
      h_bjtDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjtDPhi[ich][i]);

      h_bjtDEta[ich][i] = new TH1D(Form("h_bjtDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta Eta", 64 ,0 ,3.2);
      h_bjtDEta[ich][i]->SetXTitle("bjet_t Delta Eta");
      h_bjtDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjtDEta[ich][i]);

      h_bjtDR[ich][i] = new TH1D(Form("h_bjtDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_t Delta R", 64 ,0 ,3.2);
      h_bjtDR[ich][i]->SetXTitle("bjet_t Delta R");
      h_bjtDR[ich][i]->Sumw2();
      fOutput->Add(h_bjtDR[ich][i]);

      h_bjtDPhiDEta[ich][i] = new TH2D(Form("h_bjtDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()),"bjet_t Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_bjtDPhiDEta[ich][i]->SetYTitle("Delta Eta (bb)");
      h_bjtDPhiDEta[ich][i]->SetXTitle("Delta Phi (bb)");
      h_bjtDPhiDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjtDPhiDEta[ich][i]);

      h_cjmDPhi[ich][i] = new TH1D(Form("h_cjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "cjet_m Delta Phi", 64 ,0 ,3.2);
      h_cjmDPhi[ich][i]->SetXTitle("cjet_m Delta Phi");
      h_cjmDPhi[ich][i]->Sumw2();
      fOutput->Add(h_cjmDPhi[ich][i]);

      h_cjmDEta[ich][i] = new TH1D(Form("h_cjmDEta_Ch%i_S%i_%s",ich,i,option.Data()), "cjet_m Delta Eta", 64 ,0 ,3.2);
      h_cjmDEta[ich][i]->SetXTitle("cjet_m Delta Eta");
      h_cjmDEta[ich][i]->Sumw2();
      fOutput->Add(h_cjmDEta[ich][i]);

      h_cjmDR[ich][i] = new TH1D(Form("h_cjmDR_Ch%i_S%i_%s",ich,i,option.Data()), "cjet_m Delta R", 64 ,0 ,3.2);
      h_cjmDR[ich][i]->SetXTitle("cjet_m Delta R");
      h_cjmDR[ich][i]->Sumw2();
      fOutput->Add(h_cjmDR[ich][i]);

      h_cjmDPhiDEta[ich][i] = new TH2D(Form("h_cjmDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()),"cjet_m Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_cjmDPhiDEta[ich][i]->SetYTitle("Delta Eta (cc)");
      h_cjmDPhiDEta[ich][i]->SetXTitle("Delta Phi (cc)");
      h_cjmDPhiDEta[ich][i]->Sumw2();
      fOutput->Add(h_cjmDPhiDEta[ich][i]);

      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 100 ,0 ,1);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

      h_LepIsoQCD[ich][i] = new TH1D(Form("h_LepIsoQCD_Ch%i_S%i_%s",ich,i,option.Data()), "LepIsoQCD", 100 ,0 ,1);
      h_LepIsoQCD[ich][i]->SetXTitle("Relative Isolation");
      h_LepIsoQCD[ich][i]->Sumw2();
      fOutput->Add(h_LepIsoQCD[ich][i]);

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

    TLorentzVector jet1, jet2, bjet_m1, bjet_m2, bjet_t1, bjet_t2, cjet_m1, cjet_m2;

    double jDPhi = 0;
    double jDEta = 0;
    double jDR = 0;
    double bjmDPhi = 0;
    double bjmDEta = 0;
    double bjmDR = 0;
    double bjtDPhi = 0;
    double bjtDEta = 0;
    double bjtDR = 0;
    double cjmDPhi = 0;
    double cjmDEta = 0;
    double cjmDR = 0;

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

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        if( njets > 1){
          jet1.SetPtEtaPhiE(jet_pT[1], jet_eta[1], jet_phi[1], jet_E[1]);
          jet2.SetPtEtaPhiE(jet_pT[2], jet_eta[2], jet_phi[2], jet_E[2]);
          jDPhi = jet1.DeltaPhi(jet2);
          jDEta = jet1.Eta()-jet2.Eta();
          jDR = jet1.DeltaR(jet2);
        }
        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          if( nbjets_m > 1){
            bjet_m1.SetPtEtaPhiE(jet_pT[1], jet_eta[1], jet_phi[1], jet_E[1]);            
            bjet_m2.SetPtEtaPhiE(jet_pT[2], jet_eta[2], jet_phi[2], jet_E[2]);
            bjmDPhi = bjet_m1.DeltaPhi(bjet_m2);
            bjmDEta = bjet_m1.Eta()-bjet_m2.Eta();
            bjmDR = bjet_m1.DeltaR(bjet_m2);
          }
        }
        if( jet_CSV[iJet] > 0.9535 ){
          nbjets_t++;
          if( nbjets_t > 1){
            bjet_t1.SetPtEtaPhiE(jet_pT[1], jet_eta[1], jet_phi[1], jet_E[1]);            
            bjet_t2.SetPtEtaPhiE(jet_pT[2], jet_eta[2], jet_phi[2], jet_E[2]);
            bjtDPhi = bjet_t1.DeltaPhi(bjet_t2);
            bjtDEta = bjet_t1.Eta()-bjet_t2.Eta();
            bjtDR = bjet_t1.DeltaR(bjet_t2);
          }
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsL[iJet] > 0.08 ){
          ncjets_m++;
          if( ncjets_m > 1){
            cjet_m1.SetPtEtaPhiE(jet_pT[1], jet_eta[1], jet_phi[1], jet_E[1]);
            cjet_m2.SetPtEtaPhiE(jet_pT[2], jet_eta[2], jet_phi[2], jet_E[2]);
            cjmDPhi = cjet_m1.DeltaPhi(cjet_m2);
            cjmDEta = cjet_m1.Eta()-cjet_m2.Eta();
            cjmDR = cjet_m1.DeltaR(cjet_m2);
          }
        }
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

        h_jDPhi[mode][1]->Fill(jDPhi, EventWeight);
        h_jDEta[mode][1]->Fill(jDEta, EventWeight);
        h_jDPhiDEta[mode][1]->Fill(jDPhi, jDEta, EventWeight);
        h_jDR[mode][1]->Fill(jDR, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][1]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][1]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][1]->Fill(bjmDR, EventWeight);
          h_bjmDPhiDEta[mode][1]->Fill(bjmDPhi, bjmDEta, EventWeight);       
        }
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][1]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][1]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][1]->Fill(bjtDR, EventWeight);
          h_bjtDPhiDEta[mode][1]->Fill(bjtDPhi, bjtDEta, EventWeight);
        }
        if( ncjets_m >1 ){
          h_cjmDPhi[mode][1]->Fill(cjmDPhi, EventWeight);
          h_cjmDEta[mode][1]->Fill(cjmDEta, EventWeight);
          h_cjmDR[mode][1]->Fill(cjmDR, EventWeight);
          h_cjmDPhiDEta[mode][1]->Fill(cjmDPhi, cjmDEta, EventWeight);
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

        h_jDPhi[mode][6]->Fill(jDPhi, EventWeight);
        h_jDEta[mode][6]->Fill(jDEta, EventWeight);
        h_jDPhiDEta[mode][6]->Fill(jDPhi, jDEta, EventWeight);
        h_jDR[mode][6]->Fill(jDR, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][6]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][6]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][6]->Fill(bjmDR, EventWeight);
          h_bjmDPhiDEta[mode][6]->Fill(bjmDPhi, bjmDEta, EventWeight);
        }
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][6]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][6]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][6]->Fill(bjtDR, EventWeight);
          h_bjtDPhiDEta[mode][6]->Fill(bjtDPhi, bjtDEta, EventWeight);
        }
        if( ncjets_m >1 ){
          h_cjmDPhi[mode][6]->Fill(cjmDPhi, EventWeight);
          h_cjmDEta[mode][6]->Fill(cjmDEta, EventWeight);
          h_cjmDR[mode][6]->Fill(cjmDR, EventWeight);
          h_cjmDPhiDEta[mode][6]->Fill(cjmDPhi, cjmDEta, EventWeight);
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

        h_jDPhi[mode][11]->Fill(jDPhi, EventWeight);
        h_jDEta[mode][11]->Fill(jDEta, EventWeight);
        h_jDPhiDEta[mode][11]->Fill(jDPhi, jDEta, EventWeight);
        h_jDR[mode][11]->Fill(jDR, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][11]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][11]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][11]->Fill(bjmDR, EventWeight);
          h_bjmDPhiDEta[mode][11]->Fill(bjmDPhi, bjmDEta, EventWeight);
        }
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][11]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][11]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][11]->Fill(bjtDR, EventWeight);
          h_bjtDPhiDEta[mode][11]->Fill(bjtDPhi, bjtDEta, EventWeight);
        }
        if( ncjets_m >1 ){
          h_cjmDPhi[mode][11]->Fill(cjmDPhi, EventWeight);
          h_cjmDEta[mode][11]->Fill(cjmDEta, EventWeight);
          h_cjmDR[mode][11]->Fill(cjmDR, EventWeight);
          h_cjmDPhiDEta[mode][11]->Fill(cjmDPhi, cjmDEta, EventWeight);
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

        h_jDPhi[mode][16]->Fill(jDPhi, EventWeight);
        h_jDEta[mode][16]->Fill(jDEta, EventWeight);
        h_jDPhiDEta[mode][16]->Fill(jDPhi, jDEta, EventWeight);
        h_jDR[mode][16]->Fill(jDR, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[mode][16]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[mode][16]->Fill(bjmDEta, EventWeight);
          h_bjmDR[mode][16]->Fill(bjmDR, EventWeight);
          h_bjmDPhiDEta[mode][16]->Fill(bjmDPhi, bjmDEta, EventWeight);
        }
        if( nbjets_t >1 ){
          h_bjtDPhi[mode][16]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[mode][16]->Fill(bjtDEta, EventWeight);
          h_bjtDR[mode][16]->Fill(bjtDR, EventWeight);
          h_bjtDPhiDEta[mode][16]->Fill(bjtDPhi, bjtDEta, EventWeight);
        }
        if( ncjets_m >1 ){
          h_cjmDPhi[mode][16]->Fill(cjmDPhi, EventWeight);
          h_cjmDEta[mode][16]->Fill(cjmDEta, EventWeight);
          h_cjmDR[mode][16]->Fill(cjmDR, EventWeight);
          h_cjmDPhiDEta[mode][16]->Fill(cjmDPhi, cjmDEta, EventWeight);
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

        if( nbjets_m >= 2 ){
          h_NJet[mode][20]->Fill(njets, EventWeight);
          h_NBJetCSVv2M[mode][20]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[mode][20]->Fill(nbjets_t, EventWeight);
          h_NCJetM[mode][20]->Fill(ncjets_m, EventWeight);
          h_MET[mode][20]->Fill(*MET, EventWeight);
          h_WMass[mode][20]->Fill(transverseM, EventWeight);
          h_DPhi[mode][20]->Fill(lepDphi, EventWeight);
          h_LepIso[mode][20]->Fill(relIso, EventWeight);
          if( isQCD ) h_LepIsoQCD[mode][20]->Fill(relIso, EventWeight);
        }
*/
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

  for(int ich = 0; ich < 2; ich++){
    for(int i = 0; i < 21; i++){
      fOutput->FindObject(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_NBJetCSVv2T_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_NCJetM_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_LepIsoQCD_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_jDPhi_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_jDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_jDR_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_jDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjmDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjmDR_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjmDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjtDPhi_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjtDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjtDR_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_bjtDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_cjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_cjmDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_cjmDR_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
      fOutput->FindObject(Form("h_cjmDPhiDEta_Ch%i_S%i_%s",ich,i,option.Data()))->Write();
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

#define FCNCAnalysis_cxx

#include "FCNCAnalysis.h"
#include <TH2.h>
#include <TStyle.h>

void FCNCAnalysis::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void FCNCAnalysis::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();

   for(int ich=0; ich < 2; ich++){
     for(int i=0; i < 5; i++){

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

      h_kinWMass[ich][i] = new TH1D(Form("h_kinWMass_Ch%i_S%i_%s",ich,i,option.Data()), "WMass kinfit", 80 ,0 ,160);
      h_kinWMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_kinWMass[ich][i]->Sumw2();
      fOutput->Add(h_kinWMass[ich][i]);

      h_HMass_m[ich][i] = new TH1D(Form("h_HMass_m_Ch%i_S%i_%s",ich,i,option.Data()), "HMass (medium)", 75 ,50 ,200);
      h_HMass_m[ich][i]->SetXTitle("Higg Mass (2 medium b jets) (GeV)");
      h_HMass_m[ich][i]->Sumw2();
      fOutput->Add(h_HMass_m[ich][i]);

      h_kinHMass[ich][i] = new TH1D(Form("h_kinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "HMass kinfit", 75 ,50 ,200);
      h_kinHMass[ich][i]->SetXTitle("Higg Mass (2 b jets) (GeV)");
      h_kinHMass[ich][i]->Sumw2();
      fOutput->Add(h_kinHMass[ich][i]);

      h_dRbb[ich][i] = new TH1D(Form("h_dRbb_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R of bb", 100 ,0 ,5);
      h_dRbb[ich][i]->SetXTitle("Delta R of bb");
      h_dRbb[ich][i]->Sumw2();
      fOutput->Add(h_dRbb[ich][i]);

      h_bJetPtHm[ich][i] = new TH1D(Form("h_bJetPtHm_Ch%i_S%i_%s",ich,i,option.Data()), "b jet (medium) pT from H", 56 ,20 ,300);
      h_bJetPtHm[ich][i]->SetXTitle("b Jet (medium) pT from Higgs (GeV)");
      h_bJetPtHm[ich][i]->Sumw2();
      fOutput->Add(h_bJetPtHm[ich][i]);

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

      h_kinTopMHc[ich][i] = new TH1D(Form("h_kinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc kinfit", 75 ,120 , 270);
      h_kinTopMHc[ich][i]->SetXTitle("Top Mass (Hc) (GeV)");
      h_kinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_kinTopMHc[ich][i]);

      h_kinTopMWb[ich][i] = new TH1D(Form("h_kinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from Wb kinfit", 75 , 120 , 270);
      h_kinTopMWb[ich][i]->SetXTitle("Top Mass (Wb) (GeV)");
      h_kinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_kinTopMWb[ich][i]);

      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 30 ,0 ,0.3);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

    }
  }
}

Bool_t FCNCAnalysis::Process(Long64_t entry)
{
    TString option = GetOption();

    fReader.SetEntry(entry);

    int mode = 0;

    if( mode > 2) return kTRUE; 

    float lep_SF = 1.0;
    float genweight = *GenWeight;
    float puweight = *PUWeight;

    EventWeight = puweight*lep_SF*genweight;///////SHOULD CHANGE!!!!!

    //if( !option.Contains("Data") ) lep_SF = lepton_SF[0];

    float relIso = 0;

    int njets = 0;
    int nbjets_m = 0;
    int nbjets_t = 0;
    int ncjets_m = 0;

  //MET
    TLorentzVector p4met_muon;
    double met_muon = *MT_MuonMET;
    double met_phi_muon = *Phi_MuonMET;
    double apt_muon = TMath::Abs(met_muon);
    double met_x_muon =  apt_muon*TMath::Cos(met_phi_muon);
    double met_y_muon =  apt_muon*TMath::Sin(met_phi_muon);
    p4met_muon.SetPxPyPzE( met_x_muon, met_y_muon, 0, met_muon);

    TLorentzVector p4met_elec;
    double met_elec = *MT_ElectronMET;
    double met_phi_elec = *Phi_ElectronMET;
    double apt_elec = TMath::Abs(met_elec);
    double met_x_elec =  apt_elec*TMath::Cos(met_phi_elec);
    double met_y_elec =  apt_elec*TMath::Sin(met_phi_elec);
    p4met_elec.SetPxPyPzE( met_x_elec, met_y_elec, 0, met_elec);

    double met = 0;

  //Muon, Election 4-vector
    TLorentzVector muon;
    muon.SetPtEtaPhiE(*Muon_Pt, *Muon_Eta, *Muon_Phi, *Muon_E);
    TLorentzVector elec;
    elec.SetPtEtaPhiE(*Electron_Pt, *Electron_Eta, *Electron_Phi, *Electron_E);

    double transverseM = 0;
    double lepDphi = 0;

    double bjmDPhi = 999;
    double bjmDEta = 999;
    double bjmDR = 999;
    double higgsMass = 9999;
    double bJetPtH = 9999;
    double cjetPt = 9999;

    vector<float> jet_csv;
    vector<float> jet_cvsl;
    vector<float> bjm_csv;
    vector<float> cjet_cvsl;


   //Event selection 
    bool passmuon = (*NMuon == 1)  && (muon.Pt() > 27) && (abs(muon.Eta()) <= 2.1); // && (*NLooseMuon + *NLooseElectron) == 0;

    bool passelectron = (*NElectron == 1) && (elec.Pt() > 35) && (abs(elec.Eta()) <= 2.1); // && (*NLooseMuon + *NLooseElectron) == 0; 

  //analysis
//if ( *TTBB == 1 ){
  if ( passmuon || passelectron ){

    vector<float> v_cjet_m;
    vector<TLorentzVector> v_bjet_m;
    vector<TLorentzVector> v_jet;

    if ( passmuon ){
      mode = 0;
      transverseM = transverseMass(muon, p4met_muon);
      lepDphi = muon.DeltaPhi(p4met_muon);
      relIso  = Muon_Iso03[0];
      met = met_muon;
    }

    if ( passelectron ){
      mode = 1;
      transverseM = transverseMass(elec, p4met_elec);
      lepDphi = elec.DeltaPhi(p4met_elec);
      relIso  = Electron_Iso03[0];
      met = met_elec;
    }

    for (unsigned int iJet = 0; iJet < Jet_Pt.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(Jet_Pt[iJet], Jet_Eta[iJet], Jet_Phi[iJet], Jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        v_jet.push_back(jet);
        jet_csv.push_back(Jet_bDiscriminator[iJet]);

        if( Jet_bDiscriminator[iJet] > 0.8484 ){
          nbjets_m++;
          v_bjet_m.push_back(jet);
          bjm_csv.push_back(Jet_bDiscriminator[iJet]);
        }

        if( Jet_bDiscriminator[iJet] > 0.9535 ) nbjets_t++;

        if( Jet_pfCombinedCvsLJetTags[iJet] > -0.1 && Jet_pfCombinedCvsLJetTags[iJet] > 0.08 ){
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
              higgsMass = tmp_higgsMass_m;
              bjmDEta = tmp_bjmDEta;
              bjmDPhi = tmp_bjmDPhi;

              if( tmp_bjmPt1 > tmp_bjmPt2) bJetPtH = tmp_bjmPt1;
              else                         bJetPtH = tmp_bjmPt2;

            }
          }
        }
      }
    }

  //histograms!

    h_NJet[mode][0]->Fill(njets, EventWeight);
    h_NBJetCSVv2M[mode][0]->Fill(nbjets_m, EventWeight);
    h_NBJetCSVv2T[mode][0]->Fill(nbjets_t, EventWeight);
    h_NCJetM[mode][0]->Fill(ncjets_m, EventWeight);
    h_MET[mode][0]->Fill(met, EventWeight);
    h_WMass[mode][0]->Fill(transverseM, EventWeight);
    h_kinWMass[mode][0]->Fill(*Kin_Wmass, EventWeight);
    h_kinTopMHc[mode][0]->Fill(*Kin_TopMHc, EventWeight);
    h_kinTopMWb[mode][0]->Fill(*Kin_TopMWb, EventWeight);
    h_DPhi[mode][0]->Fill(lepDphi, EventWeight);
    h_LepIso[mode][0]->Fill(relIso, EventWeight);

    if( nbjets_m >1 ){
      h_bjmDPhi[mode][0]->Fill(bjmDPhi, EventWeight);
      h_bjmDEta[mode][0]->Fill(bjmDEta, EventWeight);
      h_bjmDR[mode][0]->Fill(bjmDR, EventWeight);
      h_HMass_m[mode][0]->Fill(higgsMass, EventWeight);
      h_kinHMass[mode][0]->Fill(*Kin_Hmass, EventWeight);
      h_dRbb[mode][0]->Fill(*Kin_HdRbb, EventWeight);
      h_bJetPtHm[mode][0]->Fill(bJetPtH, EventWeight);
    }

    if( ncjets_m >0 ){
      h_cJetPt[mode][0]->Fill(cjetPt, EventWeight);
    }

  //step1
    if( njets > 2){
      h_NJet[mode][1]->Fill(njets, EventWeight);
      h_NBJetCSVv2M[mode][1]->Fill(nbjets_m, EventWeight);
      h_NBJetCSVv2T[mode][1]->Fill(nbjets_t, EventWeight);
      h_NCJetM[mode][1]->Fill(ncjets_m, EventWeight);
      h_MET[mode][1]->Fill(met, EventWeight);
      h_WMass[mode][1]->Fill(transverseM, EventWeight);
      h_kinWMass[mode][1]->Fill(*Kin_Wmass, EventWeight);
      h_kinTopMHc[mode][1]->Fill(*Kin_TopMHc, EventWeight);
      h_kinTopMWb[mode][1]->Fill(*Kin_TopMWb, EventWeight);
      h_DPhi[mode][1]->Fill(lepDphi, EventWeight);
      h_LepIso[mode][1]->Fill(relIso, EventWeight);

      if( nbjets_m >1 ){
        h_bjmDPhi[mode][1]->Fill(bjmDPhi, EventWeight);
        h_bjmDEta[mode][1]->Fill(bjmDEta, EventWeight);
        h_bjmDR[mode][1]->Fill(bjmDR, EventWeight);
        h_HMass_m[mode][1]->Fill(higgsMass, EventWeight);
        h_kinHMass[mode][1]->Fill(*Kin_Hmass, EventWeight);
        h_dRbb[mode][1]->Fill(*Kin_HdRbb, EventWeight);
        h_bJetPtHm[mode][1]->Fill(bJetPtH, EventWeight);
      }

      if( ncjets_m >0 ){
        h_cJetPt[mode][1]->Fill(cjetPt, EventWeight);
      }
    }

  }
//}//ttbb
   return kTRUE;
}

void FCNCAnalysis::SlaveTerminate()
{
   TString option = GetOption();
}

void FCNCAnalysis::Terminate()
{
   TString option = GetOption();

   //write out the histogram at the end automatically if the name has "h_". 
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
/*
void FCNCAnalysis::make1DHistogram(  TH1D * h, int & step, TString hname, TString title, double nbins, double low, double max){

   h = new TH1D(Form("%s_S%i",hname.Data(), step ), Form("%s",hname.Data()), nbins, low, max);
   h->SetXTitle(Form("%s",title.Data()));
   h->Sumw2();
   fOutput->Add(h);

}
*/
double FCNCAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}


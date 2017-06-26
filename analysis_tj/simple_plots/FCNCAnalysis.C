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

      h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()), "MET", 40,0,400);
      h_MET[ich][i]->SetXTitle("MET (GeV)");
      h_MET[ich][i]->Sumw2();
      fOutput->Add(h_MET[ich][i]);

      h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "WMass", 32 ,0 ,160);
      h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[ich][i]->Sumw2();
      fOutput->Add(h_WMass[ich][i]);

      h_HMass_m[ich][i] = new TH1D(Form("h_HMass_m_Ch%i_S%i_%s",ich,i,option.Data()), "HMass (medium)", 100 ,50 ,150);
      h_HMass_m[ich][i]->SetXTitle("Higg Mass (2 medium b jets) (GeV)");
      h_HMass_m[ich][i]->Sumw2();
      fOutput->Add(h_HMass_m[ich][i])

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

      h_bjmDR[ich][i] = new TH1D(Form("h_bjmDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta R", 64 ,0 ,3.2);
      h_bjmDR[ich][i]->SetXTitle("bjet_m Delta R");
      h_bjmDR[ich][i]->Sumw2();
      fOutput->Add(h_bjmDR[ich][i]);

      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 100 ,0 ,1);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);

      //make1DHistogram( h_kinhmass[i], i, "h_kinhmass", "Higgs mass (GeV)", 50, 100, 150); 
      //make1DHistogram( h_kinhdRbb[i], i, "h_kinhdRbb", "DR betweeen two b jets", 100, 0, 3.0); 
      //make1DHistogram( h_kinwmass[i], i, "h_kinwmass", "W boson mass (GeV)", 100, 50, 150); 

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

    int njets = 0;
    int nbjets_m = 0;
    int nbjets_t = 0;
    int ncjets_m = 0;
    int jm = 0;
    int jn = 0;

    TLorentzVector p4met;
    double met = *MET;
    double met_x = *MET_Px;
    double met_y = *MET_Py;
    p4met.SetPxPyPzE( met_x, met_y, 0, met);

    TLorentzVector jet1, jet2, bjet_m1, bjet_m2, bjet_t1, bjet_t2, cjet_m1, cjet_m2;

    double jDPhi = 0;
    double jDEta = 0;
    double jDR = 0;
    double bjmDPhi = 0;
    double bjmDEta = 0;
    double bjmDR = 0;
    double higgsMass_m = 0;
    double bJetPtHm = 0;
    double cjetPt = 0;

    vector<double> bjmdr;
    vector<double> bjmdeta;
    vector<double> bjmdphi;
    vector<double> hm;
    vector<float> bjm_pt1;
    vector<float> bjm_pt2;
    vector<float> jet_pt;
    vector<float> jet_csv;
    vector<float> jet_cvsl;
    vector<float> bjm_csv;
    vector<float> cjet_cvsl;
    vector<float> cjet_pt;


   //Event selection 
    TLorentzVector lepton;

    bool passmuon = (mode = 0) && (*NMuon == 1) && (lepton.Pt() > 27) && (abs(lepton.Eta()) <= 2.1) && lepton.SetPtEtaPhiE(*Muon_Pt, *Muon_Eta, *Muon_Phi, *Muon_E); // && (*NLooseMuon + *NLooseElectron) == 0;

    bool passelectron = (mode = 1) && (*NElectron == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1) && lepton.SetPtEtaPhiE(*Electron_Pt, *Electron_Eta, *Electron_Phi, *Electron_E); // && (*NLooseMuon + *NLooseElectron) == 0; 

    double transverseM = transverseMass(lepton, p4met);
    double lepDphi = lepton.DeltaPhi(p4met);

if ( *TTBB == 1 ){
  if ( passmuon || passelectron ){

    for (unsigned int iJet = 0; iJet < Jet_Pt.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(Jet_Pt[iJet], Jet_Eta[iJet], Jet_Phi[iJet], Jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        if( njets > 1){
          jet1.SetPtEtaPhiE(Jet_Pt[1], Jet_Eta[1], Jet_Phi[1], Jet_E[1]);
          jet2.SetPtEtaPhiE(Jet_Pt[2], Jet_Eta[2], Jet_Phi[2], Jet_E[2]);
          jDPhi = jet1.DeltaPhi(jet2);
          jDEta = jet1.Eta()-jet2.Eta();
          jDR = jet1.DeltaR(jet2);
        }
        if( Jet_bDiscriminator[iJet] > 0.8484 ){
          nbjets_m++;
          if( nbjets_m > 1){
            bjet_m1.SetPtEtaPhiE(Jet_Pt[1], Jet_Eta[1], Jet_Phi[1], Jet_E[1]);   
            bjet_m2.SetPtEtaPhiE(Jet_Pt[2], Jet_Eta[2], Jet_Phi[2], Jet_E[2]);
            bjmDPhi = bjet_m1.DeltaPhi(bjet_m2);
            bjmDEta = bjet_m1.Eta()-bjet_m2.Eta();
            bjmDR = bjet_m1.DeltaR(bjet_m2);
          }
        }
        if( Jet_bDiscriminator[iJet] > 0.9535 ){
          nbjets_t++;
          if( nbjets_t > 1){
            bjet_t1.SetPtEtaPhiE(Jet_Pt[1], Jet_Eta[1], Jet_Phi[1], Jet_E[1]);   
            bjet_t2.SetPtEtaPhiE(Jet_Pt[2], Jet_Eta[2], Jet_Phi[2], Jet_E[2]);
            bjtDPhi = bjet_t1.DeltaPhi(bjet_t2);
            bjtDEta = bjet_t1.Eta()-bjet_t2.Eta();
            bjtDR = bjet_t1.DeltaR(bjet_t2);
          }
        }
        if( Jet_pfCombinedCvsLJetTags[iJet] > -0.1 && Jet_pfCombinedCvsLJetTags[iJet] > 0.08 ) ncjets_m++;
      }
    }


    h_NJet[0]->Fill( njets, EventWeight);
    h_NBJetCSVv2M[0]->Fill( nbjets_m, EventWeight);
    h_NBJetCSVv2T[0]->Fill( nbjets_t, EventWeight);
    h_NCJetM[0]->Fill( ncjets_m, EventWeight);
    h_MET[0]->Fill( met, EventWeight);
    h_WMass[0]->Fill(MT_MuonMET[0], EventWeight);
    h_DPhi[0]->Fill(Phi_MuonMET[0], EventWeight);
    h_LepIso[0]->Fill(Muon_Iso03[0], EventWeight);
    //h_kinhmass[0]->Fill(*Kin_Hmass, EventWeight);
    //h_kinhdRbb[0]->Fill(*Kin_HdRbb, EventWeight);
    //h_kinwmass[0]->Fill(*Kin_Wmass, EventWeight);
/*
      if( njets >= 3 ) {
        h_NJet[1]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[1]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[1]->Fill(nbjets_t, EventWeight);
        h_NCJetM[1]->Fill(ncjets_m, EventWeight);
        h_MET[1]->Fill(*MET, EventWeight);
        h_WMass[1]->Fill(transverseM, EventWeight);
        h_DPhi[1]->Fill(lepDphi, EventWeight);
        h_LepIso[11]->Fill(Muon_Iso03[0], EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[1]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[1]->Fill(bjmDEta, EventWeight);
          h_bjmDR[1]->Fill(bjmDR, EventWeight);
          h_bjmDPhiDEta[1]->Fill(bjmDPhi, bjmDEta, EventWeight);
        }
        if( nbjets_t >1 ){
          h_bjtDPhi[1]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[1]->Fill(bjtDEta, EventWeight);
          h_bjtDR[1]->Fill(bjtDR, EventWeight);
          h_bjtDPhiDEta[1]->Fill(bjtDPhi, bjtDEta, EventWeight);
        }
      }
*/
  }
}//ttbb
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


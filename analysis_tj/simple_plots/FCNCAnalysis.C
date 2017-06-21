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

   for( int i=0; i < 21; i++){

      h_NJet[i] = new TH1D(Form("h_NJet_S%i_%s",i,option.Data()), "Number of jets", 12, 0, 12);
      h_NJet[i]->SetXTitle("Jet Multiplicity");
      h_NJet[i]->Sumw2();
      fOutput->Add(h_NJet[i]);

      h_NBJetCSVv2M[i] = new TH1D(Form("h_NBJetCSVv2M_S%i_%s",i,option.Data()), "Number of b tagged jets (medium)", 5, 0, 5);
      h_NBJetCSVv2M[i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
      h_NBJetCSVv2M[i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2M[i]);

      h_NBJetCSVv2T[i] = new TH1D(Form("h_NBJetCSVv2T_S%i_%s",i,option.Data()), "Number of b tagged jets (tight)", 5, 0, 5);
      h_NBJetCSVv2T[i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2T)");
      h_NBJetCSVv2T[i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2T[i]);

      h_NCJetM[i] = new TH1D(Form("h_NCJetM_S%i_%s",i,option.Data()), "Number of c tagged jets", 6, 0, 6);
      h_NCJetM[i]->SetXTitle("c-tagged Jet Multiplicity (M)");
      h_NCJetM[i]->Sumw2();
      fOutput->Add(h_NCJetM[i]);

      h_MET[i] = new TH1D(Form("h_MET_S%i_%s",i,option.Data()), "MET", 40,0,400);
      h_MET[i]->SetXTitle("MET (GeV)");
      h_MET[i]->Sumw2();
      fOutput->Add(h_MET[i]);

      h_WMass[i] = new TH1D(Form("h_WMass_S%i_%s",i,option.Data()), "WMass", 32 ,0 ,160);
      h_WMass[i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[i]->Sumw2();
      fOutput->Add(h_WMass[i]);

      h_DPhi[i] = new TH1D(Form("h_DPhi_S%i_%s",i,option.Data()), "DPhi", 64 ,0 ,3.2);
      h_DPhi[i]->SetXTitle("Angle between MET and lepton");
      h_DPhi[i]->Sumw2();
      fOutput->Add(h_DPhi[i]);

      h_jDPhi[i] = new TH1D(Form("h_jDPhi_S%i_%s",i,option.Data()), "jet Delta Phi", 64 ,0 ,3.2);
      h_jDPhi[i]->SetXTitle("jet Delta Phi");
      h_jDPhi[i]->Sumw2();
      fOutput->Add(h_jDPhi[i]);

      h_jDEta[i] = new TH1D(Form("h_jDEta_S%i_%s",i,option.Data()), "jet Delta Eta", 64 ,0 ,3.2);
      h_jDEta[i]->SetXTitle("jet Delta Eta");
      h_jDEta[i]->Sumw2();
      fOutput->Add(h_jDEta[i]);

      h_jDR[i] = new TH1D(Form("h_jDR_S%i_%s",i,option.Data()), "jet Delta R", 64 ,0 ,3.2);
      h_jDR[i]->SetXTitle("jet Delta R");
      h_jDR[i]->Sumw2();
      fOutput->Add(h_jDR[i]);

      h_jDPhiDEta[i] = new TH2D(Form("h_jDPhiDEta_S%i_%s",i,option.Data()),"jet Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_jDPhiDEta[i]->SetYTitle("Delta Eta (jj)");
      h_jDPhiDEta[i]->SetXTitle("Delta Phi (jj)");
      h_jDPhiDEta[i]->Sumw2();
      fOutput->Add(h_jDPhiDEta[i]);

      h_bjmDPhi[i] = new TH1D(Form("h_bjmDPhi_S%i_%s",i,option.Data()), "bjet_m Delta Phi", 64 ,0 ,3.2);
      h_bjmDPhi[i]->SetXTitle("bjet_m Delta Phi");
      h_bjmDPhi[i]->Sumw2();
      fOutput->Add(h_bjmDPhi[i]);

      h_bjmDEta[i] = new TH1D(Form("h_bjmDEta_S%i_%s",i,option.Data()), "bjet_m Delta Eta", 64 ,0 ,3.2);
      h_bjmDEta[i]->SetXTitle("bjet_m Delta Eta");
      h_bjmDEta[i]->Sumw2();
      fOutput->Add(h_bjmDEta[i]);

      h_bjmDR[i] = new TH1D(Form("h_bjmDR_S%i_%s",i,option.Data()), "bjet_m Delta R", 64 ,0 ,3.2);
      h_bjmDR[i]->SetXTitle("bjet_m Delta R");
      h_bjmDR[i]->Sumw2();
      fOutput->Add(h_bjmDR[i]);

      h_bjmDPhiDEta[i] = new TH2D(Form("h_bjmDPhiDEta_S%i_%s",i,option.Data()),"bjet_m Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_bjmDPhiDEta[i]->SetYTitle("Delta Eta (bb)");
      h_bjmDPhiDEta[i]->SetXTitle("Delta Phi (bb)");
      h_bjmDPhiDEta[i]->Sumw2();
      fOutput->Add(h_bjmDPhiDEta[i]);

      h_bjtDPhi[i] = new TH1D(Form("h_bjtDPhi_S%i_%s",i,option.Data()), "bjet_t Delta Phi", 64 ,0 ,3.2);
      h_bjtDPhi[i]->SetXTitle("bjet_t Delta Phi");
      h_bjtDPhi[i]->Sumw2();
      fOutput->Add(h_bjtDPhi[i]);

      h_bjtDEta[i] = new TH1D(Form("h_bjtDEta_S%i_%s",i,option.Data()), "bjet_t Delta Eta", 64 ,0 ,3.2);
      h_bjtDEta[i]->SetXTitle("bjet_t Delta Eta");
      h_bjtDEta[i]->Sumw2();
      fOutput->Add(h_bjtDEta[i]);

      h_bjtDR[i] = new TH1D(Form("h_bjtDR_S%i_%s",i,option.Data()), "bjet_t Delta R", 64 ,0 ,3.2);
      h_bjtDR[i]->SetXTitle("bjet_t Delta R");
      h_bjtDR[i]->Sumw2();
      fOutput->Add(h_bjtDR[i]);

      h_bjtDPhiDEta[i] = new TH2D(Form("h_bjtDPhiDEta_S%i_%s",i,option.Data()),"bjet_t Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_bjtDPhiDEta[i]->SetYTitle("Delta Eta (bb)");
      h_bjtDPhiDEta[i]->SetXTitle("Delta Phi (bb)");
      h_bjtDPhiDEta[i]->Sumw2();
      fOutput->Add(h_bjtDPhiDEta[i]);

      h_cjmDPhi[i] = new TH1D(Form("h_cjmDPhi_S%i_%s",i,option.Data()), "cjet_m Delta Phi", 64 ,0 ,3.2);
      h_cjmDPhi[i]->SetXTitle("cjet_m Delta Phi");
      h_cjmDPhi[i]->Sumw2();
      fOutput->Add(h_cjmDPhi[i]);

      h_cjmDEta[i] = new TH1D(Form("h_cjmDEta_S%i_%s",i,option.Data()), "cjet_m Delta Eta", 64 ,0 ,3.2);
      h_cjmDEta[i]->SetXTitle("cjet_m Delta Eta");
      h_cjmDEta[i]->Sumw2();
      fOutput->Add(h_cjmDEta[i]);

      h_cjmDR[i] = new TH1D(Form("h_cjmDR_S%i_%s",i,option.Data()), "cjet_m Delta R", 64 ,0 ,3.2);
      h_cjmDR[i]->SetXTitle("cjet_m Delta R");
      h_cjmDR[i]->Sumw2();
      fOutput->Add(h_cjmDR[i]);

      h_cjmDPhiDEta[i] = new TH2D(Form("h_cjmDPhiDEta_S%i_%s",i,option.Data()),"cjet_m Delta Phi Delta Eta", 30, -3, 3, 30, -3, 3);
      h_cjmDPhiDEta[i]->SetYTitle("Delta Eta (cc)");
      h_cjmDPhiDEta[i]->SetXTitle("Delta Phi (cc)");
      h_cjmDPhiDEta[i]->Sumw2();
      fOutput->Add(h_cjmDPhiDEta[i]);
/*
      h_LepIso[i] = new TH1D(Form("h_LepIso_S%i_%s",i,option.Data()), "LepIso", 100 ,0 ,1);
      h_LepIso[i]->SetXTitle("Relative Isolation");
      h_LepIso[i]->Sumw2();
      fOutput->Add(h_LepIso[i]);
*/

      //make1DHistogram( h_kinhmass[i], i, "h_kinhmass", "Higgs mass (GeV)", 50, 100, 150); 
      //make1DHistogram( h_kinhdRbb[i], i, "h_kinhdRbb", "DR betweeen two b jets", 100, 0, 3.0); 
      //make1DHistogram( h_kinwmass[i], i, "h_kinwmass", "W boson mass (GeV)", 100, 50, 150); 

   }

}

Bool_t FCNCAnalysis::Process(Long64_t entry)
{
   TString option = GetOption();

   fReader.SetEntry(entry);

   EventWeight = 1.0;///////SHOULD CHANGE!!!!!

/*
    float lep_SF = 1.0;
    if( !option.Contains("Data") ) lep_SF = lepton_SF[0];
    float Genweight = *GenWeight;
    float puweight = PUWeight[0];
    float EventWeight = puweight*genweight*lep_SF;
*/

    int njets = 0;
    int nbjets_m = 0;
    int nbjets_t = 0;
    int ncjets_m = 0;

    TLorentzVector p4met;
    double met = *MET;
    double met_x = *MET_Px;
    double met_y = *MET_Py;
    p4met.SetPxPyPzE( met_x, met_y, 0, met);

///////FOR MUON!!!!!!
    TLorentzVector lepton;
    lepton.SetPtEtaPhiE(*Muon_Pt, *Muon_Eta, *Muon_Phi, *Muon_E);

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



   //Event selection 
   bool passmuon = (*NMuon == 1) && (lepton.Pt() > 27) && (abs(lepton.Eta()) <= 2.1); // && (*NLooseMuon + *NLooseElectron) == 0;
   bool passelectron = (*NElectron == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1); // && (*NLooseMuon + *NLooseElectron) == 0;

if ( *TTBB == 1 ){
  if ( passmuon ){

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
        if( Jet_pfCombinedCvsLJetTags[iJet] > -0.1 &&  Jet_pfCombinedCvsLJetTags[iJet] > 0.08 ){
          ncjets_m++;
          if( ncjets_m > 1){
            cjet_m1.SetPtEtaPhiE(Jet_Pt[1], Jet_Eta[1], Jet_Phi[1], Jet_E[1]);
            cjet_m2.SetPtEtaPhiE(Jet_Pt[2], Jet_Eta[2], Jet_Phi[2], Jet_E[2]);
            cjmDPhi = cjet_m1.DeltaPhi(cjet_m2);
            cjmDEta = cjet_m1.Eta()-cjet_m2.Eta();
            cjmDR = cjet_m1.DeltaR(cjet_m2);
          }
        }
      }
    }


    h_NJet[0]->Fill( *NJet, EventWeight);
    h_NBJetCSVv2M[0]->Fill( *NBJet, EventWeight);
    h_NBJetCSVv2T[0]->Fill( *NBJet, EventWeight);
    h_NCJetM[0]->Fill(*NBJet, EventWeight);
    h_MET[0]->Fill(*MET, EventWeight);
    h_WMass[0]->Fill(MT_MuonMET[0], EventWeight);
    h_DPhi[0]->Fill(Phi_MuonMET[0], EventWeight);
    //h_LepIso[0]->Fill(Muon_Iso03[0], EventWeight);
    //h_kinhmass[0]->Fill(*Kin_Hmass, EventWeight);
    //h_kinhdRbb[0]->Fill(*Kin_HdRbb, EventWeight);
    //h_kinwmass[0]->Fill(*Kin_Wmass, EventWeight);

      if( njets >= 3 ) {
        h_NJet[11]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[11]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[11]->Fill(nbjets_t, EventWeight);
        h_NCJetM[11]->Fill(ncjets_m, EventWeight);
        h_MET[11]->Fill(*MET, EventWeight);
        h_WMass[11]->Fill(transverseM, EventWeight);
        h_DPhi[11]->Fill(lepDphi, EventWeight);
        //h_LepIso[11]->Fill(relIso, EventWeight);

        h_jDPhi[11]->Fill(jDPhi, EventWeight);
        h_jDEta[11]->Fill(jDEta, EventWeight);
        h_jDPhiDEta[11]->Fill(jDPhi, jDEta, EventWeight);
        h_jDR[11]->Fill(jDR, EventWeight);

        if( nbjets_m >1 ){
          h_bjmDPhi[11]->Fill(bjmDPhi, EventWeight);
          h_bjmDEta[11]->Fill(bjmDEta, EventWeight);
          h_bjmDR[11]->Fill(bjmDR, EventWeight);
          h_bjmDPhiDEta[11]->Fill(bjmDPhi, bjmDEta, EventWeight);
        }
        if( nbjets_t >1 ){
          h_bjtDPhi[11]->Fill(bjtDPhi, EventWeight);
          h_bjtDEta[11]->Fill(bjtDEta, EventWeight);
          h_bjtDR[11]->Fill(bjtDR, EventWeight);
          h_bjtDPhiDEta[11]->Fill(bjtDPhi, bjtDEta, EventWeight);
        }
        if( ncjets_m >1 ){
          h_cjmDPhi[11]->Fill(cjmDPhi, EventWeight);
          h_cjmDEta[11]->Fill(cjmDEta, EventWeight);
          h_cjmDR[11]->Fill(cjmDR, EventWeight);
          h_cjmDPhiDEta[11]->Fill(cjmDPhi, cjmDEta, EventWeight);
        }
      }

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


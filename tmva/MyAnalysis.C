#define MyAnalysis_cxx
#include "MyAnalysis.h"
#include <TMath.h>
void MyAnalysis::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void MyAnalysis::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();

    Tree = new TNtuple(Form("tmva_%s",option.Data()), "tree for tmva", "NJets:NBJets_M:NBJets_T:NCJets_M:JDPhi:JDEta:JDR:BJMDPhi:BJMDEta:BJMDR:BJTDPhi:BJTDEta:BJTDR:CJMDPhi:CJMDEta");
    fOutput->Add(Tree);
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

    double jDPhi = 100;
    double jDEta = 100;
    double jDR = 100;
    double bjmDPhi = 100;
    double bjmDEta = 100;
    double bjmDR = 100;
    double bjtDPhi = 100;
    double bjtDEta = 100;
    double bjtDR = 100;
    double cjmDPhi = 100;
    double cjmDEta = 100;
    double cjmDR = 100;

    Double_t tmp[15];

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
      if( njets == 3 ) {
        tmp[0] = njets;
        tmp[1] = nbjets_m;
        tmp[2] = nbjets_t;
        tmp[3] = ncjets_m;
        tmp[4] = jDPhi;
        tmp[5] = jDEta;
        tmp[6] = jDR;

        if( nbjets_m >1 ){
          tmp[7] = bjmDPhi;
          tmp[8] = bjmDEta;
          tmp[9] = bjmDR;
        }
        if( nbjets_t >1 ){
          tmp[10] = bjtDPhi;
          tmp[11] = bjmDEta;
          tmp[12] = bjmDR;
        }
        if( ncjets_m >1 ){
          tmp[13] = cjmDPhi;
          tmp[14] = cjmDEta;
        }
      }
    Tree->Fill(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9],tmp[10],tmp[11],tmp[12],tmp[13],tmp[14]);
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

    TFile hfile(Form("tmva_%s.root",option.Data()), "RECREATE", "Tree for tmva run");
    fOutput->FindObject(Form("tmva_%s",option.Data()))->Write();
    hfile.Write();
    hfile.Close();
}


double MyAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

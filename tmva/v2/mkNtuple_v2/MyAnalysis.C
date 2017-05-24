#define MyAnalysis_cxx
#include "MyAnalysis.h"
#include <TMath.h>
#include <iostream>
#include <vector>

using namespace std;

void MyAnalysis::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void MyAnalysis::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();

    Tree = new TNtuple(Form("tmva_%s",option.Data()), "tree for tmva", "NJets:NBJets_M:BJet_M_delta_R:H_Mass:CSVv2_M:CvsL_M");
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
    int ncjets_m = 0; 
    int firstb = 0;
    int secondb = 0;
    int jm = 0;
    int jn = 0;

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

    TLorentzVector bjet_m1, bjet_m2, cjet_m;

    Double_t tmp[8];
    vector<double> bjmdr;
    vector<double> hm;
    vector<float> mcsv;
    vector<float> ncsv;
    vector<float> cvsl;
    vector<float> cpt;

    //Event selection 
    bool passmuon = (mode == 0) && (lepton.Pt() > 27) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( passmuon || passelectron ){

    multimap<float /*jet CSV*/, TLorentzVector /*jet_4 vector*/> m_bjets;
      multimap<float, TLorentzVector>::iterator m_itr;
      multimap<float, TLorentzVector>::iterator n_itr;
    multimap<float /*jet CSV*/, TLorentzVector /*jet_4 vector*/> m_cjets;
      multimap<float, TLorentzVector>::iterator c_itr;


    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          m_bjets.insert(pair<float, TLorentzVector>(jet_CSV[iJet],jet));
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsL[iJet] > 0.08 ) ncjets_m++;
          m_cjets.insert(pair<float, TLorentzVector>(jet_CvsL[iJet],jet));
      }
    } 

    if( njets >= 4 ) {

      tmp[0] = njets;
      tmp[1] = nbjets_m;
      if( nbjets_m >1 ){
        for(c_itr = m_cjets.begin(); c_itr != m_cjets.end(); ++c_itr){
          cvsl.push_back(c_itr->first);
          cjet_m = c_itr->second;
          cpt.push_back(cjet_m.Pt());
        }

        for(m_itr = m_bjets.begin(); m_itr != m_bjets.end(); ++m_itr){
          bjet_m1 = m_itr->second;
          jm = distance(m_bjets.begin(),m_itr);          

          for(n_itr = m_bjets.begin(); n_itr != m_bjets.end(); ++n_itr){
            bjet_m2 = n_itr->second;
            jn = distance(m_bjets.begin(),n_itr);

            if(jm < jn){
              bjmdr.push_back(bjet_m1.DeltaR(bjet_m2));
              hm.push_back((bjet_m1 + bjet_m2).M());
              mcsv.push_back(m_itr->first);
              ncsv.push_back(n_itr->first);
            }
          }
        }
        tmp[2] = *min_element(bjmdr.begin(), bjmdr.end());
        int a = distance(begin(bjmdr),min_element(bjmdr.begin(), bjmdr.end()));
        tmp[3] = hm.at(a);
        if(mcsv.at(a) > ncsv.at(a)) tmp[4] = mcsv.at(a);
        else tmp[4] = ncsv.at(a);

        tmp[5] = *max_element(cvsl.begin(), cvsl.end());
      }
    }
    Tree->Fill(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7]);
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

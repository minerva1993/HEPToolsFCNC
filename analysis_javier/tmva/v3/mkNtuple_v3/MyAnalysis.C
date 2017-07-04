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

    tree = new TTree(Form("tmva_%s",option.Data()), "tree for tmva");

    tree->Branch ("NJets", &b_njets, "b_njets/I");
    tree->Branch ("NBJets_M", &b_nbjets_m, "b_nbjets_m/I");
    tree->Branch ("NCJets_M", &b_ncjets_m, "b_ncjets_m/I");
    tree->Branch ("BJet_M_delta_R", &b_bjmdr, "b_bjmdr/F");
    tree->Branch ("H_Mass", &b_hmass, "b_hmass/F");
    tree->Branch ("BJet_Pt", &b_bjmpt, "b_bjmpt/F");
    tree->Branch ("CJet_Pt", &b_cjmpt, "b_cjmpt/F");
    tree->Branch ("Jet1_Pt", &b_jet1pt, "b_jet1pt/F");
    tree->Branch ("Jet2_Pt", &b_jet2pt, "b_jet2pt/F");
    tree->Branch ("Jet3_Pt", &b_jet3pt, "b_jet3pt/F");
    tree->Branch ("Jet4_Pt", &b_jet4pt, "b_jet4pt/F");
    tree->Branch ("Jet1_CSV", &b_jet1csv, "b_jet1csv/F");
    tree->Branch ("Jet2_CSV", &b_jet2csv, "b_jet2csv/F");
    tree->Branch ("Jet3_CSV", &b_jet3csv, "b_jet3csv/F");
    tree->Branch ("Jet4_CSV", &b_jet4csv, "b_jet4csv/F");

    fOutput->Add(tree);


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

    TLorentzVector jet, jet0, jet1, bjet_m1, bjet_m2, cjet_m;

    Double_t tmp[16];
    vector<double> bjmdr;
    vector<double> hm;
    vector<float> bjet_pt1;
    vector<float> bjet_pt2;
    vector<float> jet_pt;
    vector<float> jet_csv;
    vector<float> jet_cvsl;
    vector<float> bjet_csv;
    vector<float> cjet_cvsl;
    vector<float> cjet_pt;

    //Event selection 
    bool passmuon = (mode == 0) && (lepton.Pt() > 27) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( passmuon || passelectron ){

    multimap<float /*jet CSV*/, TLorentzVector /*jet_4 vector*/> m_jets;
      multimap<float, TLorentzVector>::iterator j_itr;
    multimap<float , TLorentzVector > m_bjets;
      multimap<float, TLorentzVector>::iterator m_itr;
      multimap<float, TLorentzVector>::iterator n_itr;
    multimap<float , TLorentzVector > m_cjets;
      multimap<float, TLorentzVector>::iterator c_itr;


    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        m_jets.insert(pair<float, TLorentzVector>(jet_CSV[iJet],jet));
        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          m_bjets.insert(pair<float, TLorentzVector>(jet_pT[iJet],jet));
          bjet_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsL[iJet] > 0.08 ){
          ncjets_m++;
          m_cjets.insert(pair<float, TLorentzVector>(jet_pT[iJet],jet));
          cjet_cvsl.push_back(jet_CvsL[iJet]);
        }
      }
    } 

    if( njets >= 4 && nbjets_m >= 3 ) {
      b_njets = njets;
      b_nbjets_m = nbjets_m;
      b_ncjets_m = ncjets_m;
      
      for(j_itr = m_jets.begin(); j_itr != m_jets.end(); ++j_itr){
        jet0 = j_itr->second;
        jet_pt.push_back(jet0.Pt());
        jet_csv.push_back(j_itr->first);
      }

      if( nbjets_m >1 ){

        if( ncjets_m >= 1 ){
         for(c_itr = m_cjets.begin(); c_itr != m_cjets.end(); ++c_itr){
            cjet_m = c_itr->second;
            cjet_pt.push_back(c_itr->first);
          }
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
              bjet_pt1.push_back(m_itr->first);
              bjet_pt2.push_back(n_itr->first);
            }
          }
        }
        b_bjmdr = *min_element(bjmdr.begin(), bjmdr.end());
        int a = distance(begin(bjmdr),min_element(bjmdr.begin(), bjmdr.end()));
        b_hmass = hm.at(a);

        if(bjet_pt1.at(a) > bjet_pt2.at(a)) b_bjmpt = bjet_pt1.at(a);
        else b_bjmpt = bjet_pt2.at(a);

        if(cjet_pt.size() != 0) b_cjmpt = *max_element(cjet_pt.begin(), cjet_pt.end());

        b_jet1pt = *max_element(jet_pt.begin(), jet_pt.end());
        int disjetpt1 = distance(begin(jet_pt), max_element(jet_pt.begin(), jet_pt.end()));
        jet_pt.erase(jet_pt.begin() + disjetpt1);
        b_jet1csv = jet_csv.at(disjetpt1);
        jet_csv.erase(jet_csv.begin() + disjetpt1);

        b_jet2pt = *max_element(jet_pt.begin(), jet_pt.end());
        int disjetpt2 = distance(begin(jet_pt), max_element(jet_pt.begin(), jet_pt.end()));
        jet_pt.erase(jet_pt.begin() + disjetpt2);
        b_jet2csv = jet_csv.at(disjetpt2);
        jet_csv.erase(jet_csv.begin() + disjetpt2);

        b_jet3pt = *max_element(jet_pt.begin(), jet_pt.end());
        int disjetpt3 = distance(begin(jet_pt), max_element(jet_pt.begin(), jet_pt.end()));
        jet_pt.erase(jet_pt.begin() + disjetpt3);
        b_jet3csv = jet_csv.at(disjetpt3);
        jet_csv.erase(jet_csv.begin() + disjetpt3);

        b_jet4pt = *max_element(jet_pt.begin(), jet_pt.end());
        int disjetpt4 = distance(begin(jet_pt), max_element(jet_pt.begin(), jet_pt.end()));
        b_jet4csv = jet_csv.at(disjetpt4);

      }

    }
    tree->Fill();
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

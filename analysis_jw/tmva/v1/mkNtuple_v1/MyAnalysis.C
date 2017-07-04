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

    tree = new TTree(Form("tmva_%s",option.Data()), "tree for tmva");

    tree->Branch("NJets", &b_njets, "b_njets/I");
    tree->Branch("NBJets_M", &b_nbjets_m, "b_nbjets_m/I");
    tree->Branch("NCJets_M", &b_ncjets_m, "b_ncjets_m/I");
    tree->Branch("Transvers_Mass", &b_transversem, "b_transversem/F");
    tree->Branch("Lepton_DPhi", &b_lepdphi, "b_lepdphi/F");
    tree->Branch("Missing_ET", &b_met, "b_met/F");
    tree->Branch("BJet_M_delta_R", &b_bjmdr, "b_bjmdr/F");
    tree->Branch("BJet_M_delta_Eta", &b_bjmdeta, "b_bjmdeta/F");
    tree->Branch("BJet_M_delta_Phi", &b_bjmdphi, "b_bjmdphi/F");
    tree->Branch("H_Mass", &b_hmass, "b_hmass/F");
    tree->Branch("BJet_Pt_H", &b_bjmpth, "b_bjmpth/F");
    tree->Branch("CJet_Pt", &b_cjmpt, "b_cjmpt/F");
    tree->Branch("Jet1_Pt", &b_jet1pt, "b_jet1pt/F");
    tree->Branch("Jet2_Pt", &b_jet2pt, "b_jet2pt/F");
    tree->Branch("Jet3_Pt", &b_jet3pt, "b_jet3pt/F");
    tree->Branch("Jet4_Pt", &b_jet4pt, "b_jet4pt/F");
    tree->Branch("Jet1_CSV", &b_jet1csv, "b_jet1csv/F");
    tree->Branch("Jet2_CSV", &b_jet2csv, "b_jet2csv/F");
    tree->Branch("Jet3_CSV", &b_jet3csv, "b_jet3csv/F"); tree->Branch("Jet4_CSV", &b_jet4csv, "b_jet4csv/F");
    //kinfit
    tree->Branch("Kin_Lep_WMass", &b_kinLepW, "b_kinLepW/F");
    tree->Branch("Kin_Lep_TopMass", &b_kinLepT, "b_kinLepT/F");
    tree->Branch("Kin_Had_WMass", &b_kinHadW, "b_kinHadW/F");
    tree->Branch("Kin_Had_TopMass", &b_kinHadT, "b_kinHadT/F");
    tree->Branch("fcnhKin_Lep_WMass", &b_fcnhkinLepW, "b_fcnhkinLepW/F");
    tree->Branch("fcnhKin_Lep_TopMass", &b_fcnhkinLepT, "b_fcnhkinLepT/F");
    tree->Branch("fcnhKin_HMass", &b_fcnhkinH, "b_fcnhkinH/F");
    tree->Branch("fcnhKin_Huc_TopMass", &b_fcnhkinHcT, "b_fcnhkinHcT/F");

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
    int nbjets_t = 0; 
    int ncjets_m = 0; 
    int jm = 0;
    int jn = 0;
    int jt = 0;
    int js = 0;



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

    TLorentzVector jet, jet0, jet1, bjet_m1, bjet_m2, cjet_m;

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

    multimap<float /*jet CSV*/, TLorentzVector /*jet_4 vector*/> m_jets;
      multimap<float, TLorentzVector>::iterator j_itr;
    multimap<float , TLorentzVector > m_bjets_m;
      multimap<float, TLorentzVector>::iterator m_itr;
      multimap<float, TLorentzVector>::iterator n_itr;
    multimap<float , TLorentzVector > m_cjets;
      multimap<float, TLorentzVector>::iterator c_itr;

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        m_jets.insert(pair<float, TLorentzVector>(jet_CSV[iJet],jet));
//        n_jets.insert(pair<float, TLorentzVector>(jet_CvsL[iJet],jet));
        if( jet_CSV[iJet] > 0.8484 ){
          nbjets_m++;
          m_bjets_m.insert(pair<float, TLorentzVector>(jet_pT[iJet],jet));
          bjm_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CvsL[iJet] > -0.1 && jet_CvsL[iJet] > 0.08 ){
          ncjets_m++;
          m_cjets.insert(pair<float, TLorentzVector>(jet_pT[iJet],jet));
          cjet_cvsl.push_back(jet_CvsL[iJet]);
        }
      }
    } 
    
    if( njets >= 4 && nbjets_m >= 3){
      b_njets = njets;
      b_nbjets_m = nbjets_m;
      b_ncjets_m = ncjets_m;
      b_met = met;
      b_lepdphi = lepDphi;
      b_transversem = transverseMass(lepton, p4met);

      for(j_itr = m_jets.begin(); j_itr != m_jets.end(); ++j_itr){
        jet0 = j_itr->second;
        jet_pt.push_back(jet0.Pt());
        jet_csv.push_back(j_itr->first);
      }

      if( ncjets_m >= 1 ){
        for(c_itr = m_cjets.begin(); c_itr != m_cjets.end(); ++c_itr){
          cjet_m = c_itr->second;
          cjet_pt.push_back(c_itr->first);
        }
      }

      if(cjet_pt.size() != 0) b_cjmpt = *max_element(cjet_pt.begin(), cjet_pt.end());

      if( nbjets_m >1 ){

        for(m_itr = m_bjets_m.begin(); m_itr != m_bjets_m.end(); ++m_itr){
          bjet_m1 = m_itr->second;
          jm = distance(m_bjets_m.begin(),m_itr);

          for(n_itr = m_bjets_m.begin(); n_itr != m_bjets_m.end(); ++n_itr){
            bjet_m2 = n_itr->second;
            jn = distance(m_bjets_m.begin(),n_itr);

            if(jm < jn){
              bjmdr.push_back(bjet_m1.DeltaR(bjet_m2));
              hm.push_back((bjet_m1 + bjet_m2).M());
              bjmdeta.push_back(abs(bjet_m1.Eta()-bjet_m2.Eta()));
              bjmdphi.push_back(abs(bjet_m1.Phi()-bjet_m2.Phi()));
              bjm_pt1.push_back(m_itr->first);
              bjm_pt2.push_back(n_itr->first);
            }
          }
        }
        b_bjmdr = *min_element(bjmdr.begin(), bjmdr.end());
        int b = distance(begin(bjmdr),min_element(bjmdr.begin(), bjmdr.end()));
        b_hmass = hm.at(b);
        b_bjmdeta = bjmdeta.at(b);
        b_bjmdphi = bjmdphi.at(b);

        if(bjm_pt1.at(b) > bjm_pt2.at(b)) b_bjmpth = bjm_pt1.at(b);
        else b_bjmpth = bjm_pt2.at(b);
      }

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
      fcnhkinH = fcnhkinHb1 + fcnhkinHb2;
      fcnhkinHcT = fcnhkinH + fcnhkinuc;

      if(*kin_chi2 < 10000){
        b_kinLepW = kinLepW.M();
        b_kinLepT = kinLepT.M();
        b_kinHadW = kinHadW.M();
        b_kinHadT = kinHadT.M();
      }
      if(*fcnhkin_chi2 < 10000){
        b_fcnhkinLepW = fcnhkinLepW.M();
        b_fcnhkinLepT = fcnhkinLepT.M();
        b_fcnhkinH = fcnhkinH.M();
        b_fcnhkinHcT = fcnhkinHcT.M();
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


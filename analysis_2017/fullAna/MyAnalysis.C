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
 
      h_PV[ich][i] = new TH1D(Form("h_PV_Ch%i_S%i_%s",ich,i,option.Data()), "Number of primary vertices", 50, 0, 150);
      h_PV[ich][i]->SetXTitle("Good PV");
      h_PV[ich][i]->Sumw2();
      fOutput->Add(h_PV[ich][i]);
    
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

      h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()), "MET", 40,0,200);
      h_MET[ich][i]->SetXTitle("MET (GeV)");
      h_MET[ich][i]->Sumw2();
      fOutput->Add(h_MET[ich][i]);

      h_LepPt[ich][i] = new TH1D(Form("h_LepPt_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton p_{T}", 30,0,300);
      h_LepPt[ich][i]->SetXTitle("Lepton p_{T} (GeV)");
      h_LepPt[ich][i]->Sumw2();
      fOutput->Add(h_LepPt[ich][i]);

      h_LepPhi[ich][i] = new TH1D(Form("h_LepPhi_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton #phi", 30 ,0 ,3.2);
      h_LepPhi[ich][i]->SetXTitle("Lepton |#phi|");
      h_LepPhi[ich][i]->Sumw2();
      fOutput->Add(h_LepPhi[ich][i]);

      h_LepEta[ich][i] = new TH1D(Form("h_LepEta_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton #eta", 30 ,0 ,2.5);
      h_LepEta[ich][i]->SetXTitle("Lepton |#eta|");
      h_LepEta[ich][i]->Sumw2();
      fOutput->Add(h_LepEta[ich][i]);

      h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass", 20 ,0 ,200);
      h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
      h_WMass[ich][i]->Sumw2();
      fOutput->Add(h_WMass[ich][i]);
/*
      h_HMass_m[ich][i] = new TH1D(Form("h_HMassM_Ch%i_S%i_%s",ich,i,option.Data()), "Di-bjet Mass (medium) wrt min DR", 30 ,0 ,300);
      h_HMass_m[ich][i]->SetXTitle("Di-bjet (medium) Mass (GeV)");
      h_HMass_m[ich][i]->Sumw2();
      fOutput->Add(h_HMass_m[ich][i]);

      h_bJetPtHm[ich][i] = new TH1D(Form("h_bJetPtHm_Ch%i_S%i_%s",ich,i,option.Data()), "Leading b jet (medium) pT from di-bjets", 30 , 0, 300);
      h_bJetPtHm[ich][i]->SetXTitle("b Jet (medium) pT from di-bjets (GeV)");
      h_bJetPtHm[ich][i]->Sumw2();
      fOutput->Add(h_bJetPtHm[ich][i]);
*/
      h_cJetPt[ich][i] = new TH1D(Form("h_cJetPt_Ch%i_S%i_%s",ich,i,option.Data()), "leading c jet (medium) pT", 30 , 0 ,300);
      h_cJetPt[ich][i]->SetXTitle("leading c Jet (medium) pT (GeV)");
      h_cJetPt[ich][i]->Sumw2();
      fOutput->Add(h_cJetPt[ich][i]);

      h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton MET #Delta#phi", 30 ,0 ,3.2);
      h_DPhi[ich][i]->SetXTitle("|#Delta#phi_{l,MET}|");
      h_DPhi[ich][i]->Sumw2();
      fOutput->Add(h_DPhi[ich][i]);
/*
      h_bjmDPhi[ich][i] = new TH1D(Form("h_bjmDPhi_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Phi", 30 ,0 ,3.2);
      h_bjmDPhi[ich][i]->SetXTitle("bjet_m |#Delta#phi|");
      h_bjmDPhi[ich][i]->Sumw2();
      fOutput->Add(h_bjmDPhi[ich][i]);

      h_bjmDEta[ich][i] = new TH1D(Form("h_bjmDEta_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta Eta", 30 ,0 ,3.2);
      h_bjmDEta[ich][i]->SetXTitle("bjet_m |#Delta#eta|");
      h_bjmDEta[ich][i]->Sumw2();
      fOutput->Add(h_bjmDEta[ich][i]);

      h_bjmDR[ich][i] = new TH1D(Form("h_bjmDR_Ch%i_S%i_%s",ich,i,option.Data()), "bjet_m Delta R", 30 ,0 ,4);
      h_bjmDR[ich][i]->SetXTitle("bjet_m #Delta R");
      h_bjmDR[ich][i]->Sumw2();
      fOutput->Add(h_bjmDR[ich][i]);
*/
      h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 20 ,0 ,0.15);
      h_LepIso[ich][i]->SetXTitle("Relative Isolation");
      h_LepIso[ich][i]->Sumw2();
      fOutput->Add(h_LepIso[ich][i]);
/*
      h_csvv2[ich][i] = new TH1D(Form("h_csvv2_Ch%i_S%i_%s",ich,i,option.Data()), "CSVv2", 20 ,0 ,1);
      h_csvv2[ich][i]->SetXTitle("CSVv2");
      h_csvv2[ich][i]->Sumw2();
      fOutput->Add(h_csvv2[ich][i]);

      h_cvsl[ich][i] = new TH1D(Form("h_cvsl_Ch%i_S%i_%s",ich,i,option.Data()), "CvsL", 20 , -0.1 ,1);
      h_cvsl[ich][i]->SetXTitle("CvsL");
      h_cvsl[ich][i]->Sumw2();
      fOutput->Add(h_cvsl[ich][i]);

      h_cvsb[ich][i] = new TH1D(Form("h_cvsb_Ch%i_S%i_%s",ich,i,option.Data()), "CvsB", 20 , 0.08 ,1);
      h_cvsb[ich][i]->SetXTitle("CvsB");
      h_cvsb[ich][i]->Sumw2();
      fOutput->Add(h_cvsb[ich][i]);

      h_DRFCNHkinLepWMass[ich][i] = new TH1D(Form("h_DRFCNHkinLepWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass (Lep)", 30 , 0, 300);
      h_DRFCNHkinLepWMass[ich][i]->SetXTitle("W Mass (Lep) (GeV)");
      h_DRFCNHkinLepWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinLepWMass[ich][i]);

      h_DRFCNHkinHadWMass[ich][i] = new TH1D(Form("h_DRFCNHkinHadWMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass (Had)", 30, 0, 300);
      h_DRFCNHkinHadWMass[ich][i]->SetXTitle("W Mass (Had) (GeV)");
      h_DRFCNHkinHadWMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHadWMass[ich][i]);

      h_DRFCNHkinHMass[ich][i] = new TH1D(Form("h_DRFCNHkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs Mass (bb)", 30, 0,300);
      h_DRFCNHkinHMass[ich][i]->SetXTitle("Higgs Mass (GeV)");
      h_DRFCNHkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHMass[ich][i]);

      h_DRFCNHkinDR[ich][i] = new TH1D(Form("h_DRFCNHkinDR_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta R of bb", 30, 0,4);
      h_DRFCNHkinDR[ich][i]->SetXTitle("#Delta R of b jets from Higgs");
      h_DRFCNHkinDR[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinDR[ich][i]);

      h_DRFCNHkinTopMWb[ich][i] = new TH1D(Form("h_DRFCNHkinTopMWb_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass (Lep)", 30 , 0, 400);
      h_DRFCNHkinTopMWb[ich][i]->SetXTitle("Top Mass (Lep) (GeV)");
      h_DRFCNHkinTopMWb[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMWb[ich][i]);

      h_DRFCNHkinTopMHc[ich][i] = new TH1D(Form("h_DRFCNHkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top Mass from Hc/u", 30, 0, 400);
      h_DRFCNHkinTopMHc[ich][i]->SetXTitle("Top Mass (Had) (GeV)");
      h_DRFCNHkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMHc[ich][i]);

      h_DRFCNHkinHPt[ich][i] = new TH1D(Form("h_DRFCNHkinHPt_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs p_{T}", 30, 0,300);
      h_DRFCNHkinHPt[ich][i]->SetXTitle("Higgs p_{T} (GeV)");
      h_DRFCNHkinHPt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHPt[ich][i]);

      h_DRFCNHkinHdPhi[ich][i] = new TH1D(Form("h_DRFCNHkinHdPhi_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta#phi of bb from Higgs", 30, 0, 3.2);
      h_DRFCNHkinHdPhi[ich][i]->SetXTitle("|#Delta#phi_{bb}|");
      h_DRFCNHkinHdPhi[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHdPhi[ich][i]);

      h_DRFCNHkinHdEta[ich][i] = new TH1D(Form("h_DRFCNHkinHdEta_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta#eta of bb from Higgs", 30, 0, 2.5);
      h_DRFCNHkinHdEta[ich][i]->SetXTitle("|#Delta#eta_{bb}|");
      h_DRFCNHkinHdEta[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHdEta[ich][i]);

      h_DRFCNHkinHb1Pt[ich][i] = new TH1D(Form("h_DRFCNHkinHb1Pt_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs b_{1} p_{T}", 30, 0,300);
      h_DRFCNHkinHb1Pt[ich][i]->SetXTitle("Higgs b_{1} p_{T} (GeV)");
      h_DRFCNHkinHb1Pt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHb1Pt[ich][i]);

      h_DRFCNHkinHb2Pt[ich][i] = new TH1D(Form("h_DRFCNHkinHb2Pt_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs b_{2} p_{T}", 30, 0,300);
      h_DRFCNHkinHb2Pt[ich][i]->SetXTitle("Higgs b_{2} p_{T} (GeV)");
      h_DRFCNHkinHb2Pt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHb2Pt[ich][i]);

      h_DRFCNHkinHb1CSV[ich][i] = new TH1D(Form("h_DRFCNHkinHb1CSV_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs b_{1} CSVv2", 20, 0.8 ,1);
      h_DRFCNHkinHb1CSV[ich][i]->SetXTitle("Higgs b_{1} CSVv2 (GeV)");
      h_DRFCNHkinHb1CSV[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHb1CSV[ich][i]);

      h_DRFCNHkinHb2CSV[ich][i] = new TH1D(Form("h_DRFCNHkinHb2CSV_Ch%i_S%i_%s",ich,i,option.Data()), "Higgs b_{2} CSVv2", 20, 0.8 ,1);
      h_DRFCNHkinHb2CSV[ich][i]->SetXTitle("Higgs b_{2} CSVv2 (GeV)");
      h_DRFCNHkinHb2CSV[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHb2CSV[ich][i]);

      h_DRFCNHkinLepTopPt[ich][i] = new TH1D(Form("h_DRFCNHkinLepTopPt_Ch%i_S%i_%s",ich,i,option.Data()), "Top p_{T} (Lep)", 30 , 0, 400);
      h_DRFCNHkinLepTopPt[ich][i]->SetXTitle("Top p_{T} (Lep) (GeV)");
      h_DRFCNHkinLepTopPt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinLepTopPt[ich][i]);

      h_DRFCNHkinHadTopPt[ich][i] = new TH1D(Form("h_DRFCNHkinHadTopPt_Ch%i_S%i_%s",ich,i,option.Data()), "Top p_{T} (Had)", 30 , 0, 400);
      h_DRFCNHkinHadTopPt[ich][i]->SetXTitle("Top p_{T} (Had) (GeV)");
      h_DRFCNHkinHadTopPt[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHadTopPt[ich][i]);

      //GenInfo
      h_genDR[ich][i] = new TH1D(Form("h_genDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen b jets from Higgs", 30, 0, 4);
      h_genDR[ich][i]->SetXTitle("gen #Delta R");
      h_genDR[ich][i]->Sumw2();
      fOutput->Add(h_genDR[ich][i]);

      h_matchDR[ich][i] = new TH1D(Form("h_matchDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen matched b jets from Higgs", 30, 1, 4);
      h_matchDR[ich][i]->SetXTitle("gen matched #Delta R");
      h_matchDR[ich][i]->Sumw2();
      fOutput->Add(h_matchDR[ich][i]);

      h_genHm[ich][i] = new TH1D(Form("h_genHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen Higgs mass", 30, 0, 250);
      h_genHm[ich][i]->SetXTitle("gen Higgs Mass (GeV)");
      h_genHm[ich][i]->Sumw2();
      fOutput->Add(h_genHm[ich][i]);

      h_matchHm[ich][i] = new TH1D(Form("h_matchHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen matched Higgs mass", 30, 0, 250);
      h_matchHm[ich][i]->SetXTitle("gen matched Higgs Mass (GeV)");
      h_matchHm[ich][i]->Sumw2();
      fOutput->Add(h_matchHm[ich][i]);
*/
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
    //if( !option.Contains("Data") ) lep_SF = lepton_SF[0];
    float genweight = *genWeight;
    float puweight = PUWeight[0];
    //float jetsf = jet_SF_CSV_30[0];
    float EventWeight = puweight*genweight;//*lep_SF*jetsf;

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
    double higgsMass_m = 9999;
    double bJetPtHm = 9999;
    double cjetPt = 0;

    vector<float> bjm_csv;
    vector<float> bjt_csv;
    vector<float> cjm_cvsl;
    vector<float> cjm_cvsb;

    //for Goh's Kin fit
    vector<size_t> jetIdxs;
    int b_kin_bjetcode;
    bool match1 = false;
    bool match2 = false;
    double gendR = -1.0;
    double matchdR = -1.0;
    double genHm = 0;
    double matchHm = 0;

    //Selection Option
    bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
    bool makeIso = true;
    bool isIso = *lepton_isIso; 
    if( makeIso && !isIso ) return kTRUE;
    if( !makeIso && isIso ) return kTRUE;

    //Event selection 
    bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 38) && (abs(lepton.Eta()) <= 2.1);

  if( passmuon || passelectron ){

    vector<float> v_cjet_m;
    vector<TLorentzVector> v_bjet_m;
    vector<TLorentzVector> v_bjet_t;
    vector<TLorentzVector> v_jet;

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
      //if( !option.Contains("Data") ) jet = jet * jet_JER_Nom[iJet];

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        jetIdxs.push_back(iJet);//Goh's kinfit
        if( jet_CSV[iJet] > 0.8838 ){ //new WP
          nbjets_m++;
          v_bjet_m.push_back(jet);
          //bjm_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CSV[iJet] > 0.9693 ){ //new WP
          nbjets_t++;
          v_bjet_t.push_back(jet);
          //bjt_csv.push_back(jet_CSV[iJet]);
        }
        if( jet_CvsL[iJet] > 0.07 && jet_CvsB[iJet] > -0.10 ){ //new WP
          ncjets_m++;
          v_cjet_m.push_back(jet.Pt());
          //cjm_cvsl.push_back(jet_CvsL[iJet]);
          //cjm_cvsb.push_back(jet_CvsB[iJet]);
        }
      }
    }

    //if( (option.Contains("Hct") || option.Contains("Hut")) && (*addHbjet1_pt)*(*addHbjet2_pt) == 0) return kTRUE;

/*
    TLorentzVector hbjet1, hbjet2, genH;
    if(*addHbjet1_pt > 20 && *addHbjet2_pt > 20 && abs(*addHbjet1_eta) < 2.4 && abs(*addHbjet2_eta) < 2.4){
      hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
      hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

      genH = hbjet1 + hbjet2;
      gendR = hbjet1.DeltaR(hbjet2);
      genHm = genH.M();
    }
*/
    if( ncjets_m != 0 ) cjetPt = *max_element(v_cjet_m.begin(), v_cjet_m.end());
/*
    //DR kin
    std::vector<size_t> bestIdxsDR;
    TLorentzVector jetP4sDR[4];
    size_t lepidx = 0;

    if ( njets >= 4 ){
      double minDRlep = 1e9;
      for ( auto ii0 = jetIdxs.begin(); ii0 != jetIdxs.end(); ++ii0 ) {
        jetP4sDR[0].SetPtEtaPhiE(jet_pT[*ii0], jet_eta[*ii0], jet_phi[*ii0], jet_E[*ii0]);
        if ( jet_CSV[*ii0] < 0.8484 ) continue;
        const double dRlep = jetP4sDR[0].DeltaR(lepton);
        if ( dRlep < minDRlep ) {
          lepidx = *ii0;
          minDRlep = dRlep;
        }
      }

      double minDR = 1e9;
      for ( auto ii1 = jetIdxs.begin(); ii1 != jetIdxs.end(); ++ii1 ) {
        if ( *ii1 == lepidx ) continue;
        jetP4sDR[1].SetPtEtaPhiE(jet_pT[*ii1], jet_eta[*ii1], jet_phi[*ii1], jet_E[*ii1]);

        for ( auto ii2 = ii1+1; ii2 != jetIdxs.end(); ++ii2 ) {
          if ( *ii2 == lepidx ) continue;
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
            bestIdxsDR = { lepidx, *ii1, *ii2, size_t(njets)};
            minDR = dR;
          }
        }
      }
      if ( !bestIdxsDR.empty() ) {
        const auto ilep = bestIdxsDR[0], i1 = bestIdxsDR[1], i2 = bestIdxsDR[2];
        jetP4sDR[1].SetPtEtaPhiE(jet_pT[i1], jet_eta[i1], jet_phi[i1], jet_E[i1]);
        jetP4sDR[2].SetPtEtaPhiE(jet_pT[i2], jet_eta[i2], jet_phi[i2], jet_E[i2]);
        if( !option.Contains("Data") ){
          jetP4sDR[1] = jetP4sDR[1] *  jet_JER_Nom[i1];
          jetP4sDR[2] = jetP4sDR[2] *  jet_JER_Nom[i2];
        }
        const auto wP4 = jetP4sDR[1]+jetP4sDR[2];
        double minmassdiff = 1e9;
        for ( auto i3 : jetIdxs ) {
          if ( i3 == i1 or i3 == i2 or i3 == ilep) continue;
          //if ( jet_CSV[i3] > 0.8484 ) continue;
          jetP4sDR[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);
          if( !option.Contains("Data") ) jetP4sDR[3] = jetP4sDR[3] * jet_JER_Nom[i3];

          double mass = (jetP4sDR[3] + wP4).M();
          double massdiff = (mass - 172.5)*(mass - 172.5);
          if ( massdiff < minmassdiff) {
            bestIdxsDR[3] = i3;
            minmassdiff = massdiff;
          }
        }
        if ( minmassdiff == 1e9 ) bestIdxsDR.clear();
      }
      if (bestIdxsDR.empty()) return kTRUE;
      stable_sort(next(bestIdxsDR.begin()), bestIdxsDR.end(),
                       [&](size_t a, size_t b){ return jet_CSV[a] > jet_CSV[b]; });

      for ( size_t i=0; i<4; ++i ) {
        const size_t j = bestIdxsDR[i];
        jetP4sDR[i].SetPtEtaPhiE(jet_pT[j], jet_eta[j], jet_phi[j], jet_E[j]);
        if( !option.Contains("Data") ) jetP4sDR[i] = jetP4sDR[i] * jet_JER_Nom[j];
      }

      //Gen vs reco higgs->bjet matching
      if( option.Contains("Hct") || option.Contains("Hut") ){
        if(hbjet1.DeltaR(jetP4sDR[1]) < 0.4 or hbjet1.DeltaR(jetP4sDR[2]) < 0.4) match1 = true;
        if(hbjet2.DeltaR(jetP4sDR[1]) < 0.4 or hbjet2.DeltaR(jetP4sDR[2]) < 0.4) match2 = true;
      }
    }
*/

    /////Fill histograms

    int Ncuts = 16;
    bool eventSelection[Ncuts];
    for(unsigned int bcut=0; bcut < Ncuts; bcut++) eventSelection[bcut] = false;

    eventSelection[0] = true;
    eventSelection[1] = ( njets >= 3 );
    eventSelection[2] = ( njets >= 3 ) && ( nbjets_m == 2 );
    eventSelection[3] = ( njets >= 3 ) && ( nbjets_m == 3 );
    eventSelection[4] = ( njets >= 3 ) && ( nbjets_m == 2 || nbjets_m == 3 );
    eventSelection[5] = ( njets >= 3 ) && ( nbjets_m >= 3 );
    eventSelection[6] = ( njets >= 4 );
    eventSelection[7] = ( njets >= 4 ) && ( nbjets_m == 2 );
    eventSelection[8] = ( njets >= 4 ) && ( nbjets_m == 3 );
    eventSelection[9] = ( njets >= 4 ) && ( nbjets_m == 4 );
    eventSelection[10] = ( njets >= 4 ) && ( nbjets_m >= 3 );
    eventSelection[11] = ( njets >= 4 ) && ( nbjets_m >= 4 );
    eventSelection[12] = ( njets >= 6 ); 
    eventSelection[13] = ( njets >= 6 ) && ( nbjets_m == 3 );
    eventSelection[14] = ( njets >= 6 ) && ( nbjets_m == 2 || nbjets_m == 3 );
    eventSelection[15] = ( njets >= 6 ) && ( nbjets_m >= 3 );

    for( int cut = 0; cut < 16; cut++){
      if(eventSelection[cut]){
        h_PV[mode][cut]->Fill(*GoodPV,EventWeight);
        h_NJet[mode][cut]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][cut]->Fill(nbjets_m, EventWeight);
        h_NBJetCSVv2T[mode][cut]->Fill(nbjets_t, EventWeight);
        h_NCJetM[mode][cut]->Fill(ncjets_m, EventWeight);
        h_LepPt[mode][cut]->Fill(lepton.Pt(), EventWeight);
        h_LepPhi[mode][cut]->Fill(lepton.Phi(), EventWeight);
        h_LepEta[mode][cut]->Fill(lepton.Eta(), EventWeight);
        h_MET[mode][cut]->Fill(*MET, EventWeight);
        h_WMass[mode][cut]->Fill(transverseM, EventWeight);
        h_DPhi[mode][cut]->Fill(lepDphi, EventWeight);
        h_LepIso[mode][cut]->Fill(relIso, EventWeight);
/*
        if( !bestIdxsDR.empty() ){
          for( int i=0; i<bestIdxsDR.size(); ++i ){
            const size_t j = bestIdxsDR[i];
            h_csvv2[mode][cut]->Fill(jet_CSV[j],EventWeight);
            h_cvsl[mode][cut]->Fill(jet_CvsL[j],EventWeight);
            h_cvsb[mode][cut]->Fill(jet_CvsB[j],EventWeight);
          }
          h_DRFCNHkinLepWMass[mode][cut]->Fill((lepton+p4met).M(),EventWeight);
          h_DRFCNHkinHadWMass[mode][cut]->Fill((jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinTopMWb[mode][cut]->Fill((lepton+p4met+jetP4sDR[0]).M(),EventWeight);
          h_DRFCNHkinHMass[mode][cut]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          h_DRFCNHkinDR[mode][cut]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
          h_DRFCNHkinTopMHc[mode][cut]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
          h_DRFCNHkinHPt[mode][cut]->Fill((jetP4sDR[1]+jetP4sDR[2]).Pt(),EventWeight);
          h_DRFCNHkinHdPhi[mode][cut]->Fill(abs(jetP4sDR[1].DeltaPhi(jetP4sDR[2])),EventWeight);
          h_DRFCNHkinHdEta[mode][cut]->Fill(abs((jetP4sDR[1]-jetP4sDR[2]).Eta()),EventWeight);
          h_DRFCNHkinHb1CSV[mode][cut]->Fill(jet_CSV[bestIdxsDR[1]],EventWeight);
          h_DRFCNHkinHb2CSV[mode][cut]->Fill(jet_CSV[bestIdxsDR[2]],EventWeight);
          h_DRFCNHkinLepTopPt[mode][cut]->Fill((lepton+p4met+jetP4sDR[0]).Pt(),EventWeight);
          h_DRFCNHkinHadTopPt[mode][cut]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).Pt(),EventWeight);
        }

        if( ncjets_m >0 ){
          h_cJetPt[mode][cut]->Fill(cjetPt, EventWeight);
        }

        if(genH.Pt() > 0){
          h_genDR[mode][cut]->Fill(gendR, EventWeight);
          h_genHm[mode][cut]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][cut]->Fill(gendR, EventWeight);
            h_matchHm[mode][cut]->Fill(genHm, EventWeight);
          }
        }
*/
      }
    }//selection loopa
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

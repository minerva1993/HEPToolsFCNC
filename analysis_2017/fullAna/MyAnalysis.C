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
  string sample = option.Data();

  if( option.Contains("_") ) reco = true;
  else reco = false;

  //Delete ntuple number and data era so that we can merge histos w.r.t. dataset, prepare assign ntuple
  const char* assign_file = "";
  if( reco ){
    if     (option.Contains("Run2017")) sample.erase(sample.find_first_of("_")-1,string::npos);
    else if(option.Contains("part")) sample.erase(sample.find_last_of("p"),string::npos);
    else   sample.erase(sample.find_first_of("_"),string::npos);
    assign_file = Form("/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/classifier/2017/assignSTFCNC01/assign_deepReco_%s.root", option.Data());

    bool file_exist = true;
    string file_tmp_path = assign_file;
    ifstream file_tmp(file_tmp_path);
    if( file_tmp.is_open() ){
      assignF = TFile::Open(assign_file, "READ");
      assignT = (TTree*) assignF->Get("tree");
      int nevt = assignT->GetEntries();
      if( nevt > 0){
        for(int i = 0; i < nevt; i++){
          assignT->GetEntry(i);
          double pt = assignT->GetLeaf("leptonPt")->GetValue(0);
          double met = assignT->GetLeaf("missingET")->GetValue(0);
          lepPt.push_back(pt);
          missET.push_back(met);
        }
      }
    }
    else cout << option.Data() << endl;
  }

  //cout << "SlaveBegin" << endl;
  for(int ich=0; ich < 2; ich++){
    for(int i=0; i < 18; i++){

    h_PV[ich][i] = new TH1D(Form("h_PV_Ch%i_S%i_%s",ich,i,sample.c_str()), "Number of primary vertices", 60, 0, 60);
    h_PV[ich][i]->SetXTitle("Good PV");
    h_PV[ich][i]->Sumw2();
    fOutput->Add(h_PV[ich][i]);

    h_EventWeight[ich][i] = new TH1D(Form("h_EventWeight_Ch%i_S%i_%s",ich,i,sample.c_str()), "Event Weights", 100, 0, 2);
    h_EventWeight[ich][i]->SetXTitle("Event Weight");
    h_EventWeight[ich][i]->Sumw2();
    fOutput->Add(h_EventWeight[ich][i]);
  
    h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,sample.c_str()), "Number of jets", 12, 0, 12);
    h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
    h_NJet[ich][i]->Sumw2();
    fOutput->Add(h_NJet[ich][i]);

    h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,sample.c_str()), "Number of b tagged jets (medium)", 6, 0, 6);
    h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (DeepCSVM)");
    h_NBJetCSVv2M[ich][i]->Sumw2();
    fOutput->Add(h_NBJetCSVv2M[ich][i]);

    h_NBJetCSVv2T[ich][i] = new TH1D(Form("h_NBJetCSVv2T_Ch%i_S%i_%s",ich,i,sample.c_str()), "Number of b tagged jets (tight)", 6, 0, 6);
    h_NBJetCSVv2T[ich][i]->SetXTitle("b-tagged Jet Multiplicity (DeepCSVT)");
    h_NBJetCSVv2T[ich][i]->Sumw2();
    fOutput->Add(h_NBJetCSVv2T[ich][i]);

    h_NCJetM[ich][i] = new TH1D(Form("h_NCJetM_Ch%i_S%i_%s",ich,i,sample.c_str()), "Number of c tagged jets", 6, 0, 6);
    h_NCJetM[ich][i]->SetXTitle("c-tagged Jet Multiplicity (MediumWP)");
    h_NCJetM[ich][i]->Sumw2();
    fOutput->Add(h_NCJetM[ich][i]);  

    h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,sample.c_str()), "MET", 30,0,200);
    h_MET[ich][i]->SetXTitle("MET (GeV)");
    h_MET[ich][i]->Sumw2();
    fOutput->Add(h_MET[ich][i]);

    h_LepPt[ich][i] = new TH1D(Form("h_LepPt_Ch%i_S%i_%s",ich,i,sample.c_str()), "Lepton p_{T}", 30,0,200);
    h_LepPt[ich][i]->SetXTitle("Lepton p_{T} (GeV)");
    h_LepPt[ich][i]->Sumw2();
    fOutput->Add(h_LepPt[ich][i]);

    h_LepPhi[ich][i] = new TH1D(Form("h_LepPhi_Ch%i_S%i_%s",ich,i,sample.c_str()), "Lepton #phi", 30 ,0 ,3.2);
    h_LepPhi[ich][i]->SetXTitle("Lepton |#phi|");
    h_LepPhi[ich][i]->Sumw2();
    fOutput->Add(h_LepPhi[ich][i]);

    h_LepEta[ich][i] = new TH1D(Form("h_LepEta_Ch%i_S%i_%s",ich,i,sample.c_str()), "Lepton #eta", 30 ,0 ,2.5);
    h_LepEta[ich][i]->SetXTitle("Lepton |#eta|");
    h_LepEta[ich][i]->Sumw2();
    fOutput->Add(h_LepEta[ich][i]);

    h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,sample.c_str()), "W Mass", 30 ,0 ,200);
    h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
    h_WMass[ich][i]->Sumw2();
    fOutput->Add(h_WMass[ich][i]);

    h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,sample.c_str()), "Lepton MET #Delta#phi", 30 ,0 ,3.2);
    h_DPhi[ich][i]->SetXTitle("|#Delta#phi_{l,MET}|");
    h_DPhi[ich][i]->Sumw2();
    fOutput->Add(h_DPhi[ich][i]);

    h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,sample.c_str()), "LepIso", 20 ,0 ,0.15);
    h_LepIso[ich][i]->SetXTitle("Relative Isolation");
    h_LepIso[ich][i]->Sumw2();
    fOutput->Add(h_LepIso[ich][i]);

    h_JetCSV[ich][i] = new TH1D(Form("h_JetCSV_Ch%i_S%i_%s",ich,i,sample.c_str()), "DeepCSV", 20 ,0 ,1);
    h_JetCSV[ich][i]->SetXTitle("DeepCSV (all jets)");
    h_JetCSV[ich][i]->Sumw2();
    fOutput->Add(h_JetCSV[ich][i]);

    h_csv[ich][i] = new TH1D(Form("h_csv_Ch%i_S%i_%s",ich,i,sample.c_str()), "DeepCSV", 20 ,0 ,1);
    h_csv[ich][i]->SetXTitle("DeepCSV");
    h_csv[ich][i]->Sumw2();
    fOutput->Add(h_csv[ich][i]);

    h_cvsl[ich][i] = new TH1D(Form("h_cvsl_Ch%i_S%i_%s",ich,i,sample.c_str()), "CvsL", 20 , 0 ,1);
    h_cvsl[ich][i]->SetXTitle("DeepCvsL");
    h_cvsl[ich][i]->Sumw2();
    fOutput->Add(h_cvsl[ich][i]);

    h_cvsb[ich][i] = new TH1D(Form("h_cvsb_Ch%i_S%i_%s",ich,i,sample.c_str()), "CvsB", 20 , 0 ,1);
    h_cvsb[ich][i]->SetXTitle("DeepCvsB");
    h_cvsb[ich][i]->Sumw2();
    fOutput->Add(h_cvsb[ich][i]);

    h_FCNHkinLepWMass[ich][i] = new TH1D(Form("h_FCNHkinLepWMass_Ch%i_S%i_%s",ich,i,sample.c_str()), "W Mass (Lep)", 30 , 0, 300);
    h_FCNHkinLepWMass[ich][i]->SetXTitle("W Mass (Lep) (GeV)");
    h_FCNHkinLepWMass[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinLepWMass[ich][i]);

    h_FCNHkinHadWMass[ich][i] = new TH1D(Form("h_FCNHkinHadWMass_Ch%i_S%i_%s",ich,i,sample.c_str()), "W Mass (Had)", 30, 0, 300);
    h_FCNHkinHadWMass[ich][i]->SetXTitle("W Mass (Had) (GeV)");
    h_FCNHkinHadWMass[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHadWMass[ich][i]);

    h_FCNHkinHMass[ich][i] = new TH1D(Form("h_FCNHkinHMass_Ch%i_S%i_%s",ich,i,sample.c_str()), "Higgs Mass (bb)", 30, 0,250);
    h_FCNHkinHMass[ich][i]->SetXTitle("Higgs Mass (GeV)");
    h_FCNHkinHMass[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHMass[ich][i]);

    h_FCNHkinDR[ich][i] = new TH1D(Form("h_FCNHkinDR_Ch%i_S%i_%s",ich,i,sample.c_str()), "#Delta R of bb", 30, 0,4);
    h_FCNHkinDR[ich][i]->SetXTitle("#Delta R of b jets from Higgs");
    h_FCNHkinDR[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinDR[ich][i]);

    h_FCNHkinLepTopM[ich][i] = new TH1D(Form("h_FCNHkinLepTopM_Ch%i_S%i_%s",ich,i,sample.c_str()), "Top Mass (Lep)", 30 , 0, 400);
    h_FCNHkinLepTopM[ich][i]->SetXTitle("Top Mass (Lep) (GeV)");
    h_FCNHkinLepTopM[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinLepTopM[ich][i]);

    h_FCNHkinHadTopM[ich][i] = new TH1D(Form("h_FCNHkinHadTopM_Ch%i_S%i_%s",ich,i,sample.c_str()), "Top Mass from Hc/u", 30, 0, 400);
    h_FCNHkinHadTopM[ich][i]->SetXTitle("Top Mass (Had) (GeV)");
    h_FCNHkinHadTopM[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHadTopM[ich][i]);

    h_FCNHkinHPt[ich][i] = new TH1D(Form("h_FCNHkinHPt_Ch%i_S%i_%s",ich,i,sample.c_str()), "Higgs p_{T}", 30, 0,300);
    h_FCNHkinHPt[ich][i]->SetXTitle("Higgs p_{T} (GeV)");
    h_FCNHkinHPt[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHPt[ich][i]);

    h_FCNHkinHdPhi[ich][i] = new TH1D(Form("h_FCNHkinHdPhi_Ch%i_S%i_%s",ich,i,sample.c_str()), "#Delta#phi of bb from Higgs", 30, 0, 3.2);
    h_FCNHkinHdPhi[ich][i]->SetXTitle("|#Delta#phi_{bb}|");
    h_FCNHkinHdPhi[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHdPhi[ich][i]);

    h_FCNHkinHdEta[ich][i] = new TH1D(Form("h_FCNHkinHdEta_Ch%i_S%i_%s",ich,i,sample.c_str()), "#Delta#eta of bb from Higgs", 30, 0, 2.5);
    h_FCNHkinHdEta[ich][i]->SetXTitle("|#Delta#eta_{bb}|");
    h_FCNHkinHdEta[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHdEta[ich][i]);

    h_FCNHkinHb1CSV[ich][i] = new TH1D(Form("h_FCNHkinHb1CSV_Ch%i_S%i_%s",ich,i,sample.c_str()), "Higgs b_{1} DeepCSV", 20, 0.5 ,1);
    h_FCNHkinHb1CSV[ich][i]->SetXTitle("Higgs b_{1} DeepCSV (GeV)");
    h_FCNHkinHb1CSV[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHb1CSV[ich][i]);

    h_FCNHkinHb2CSV[ich][i] = new TH1D(Form("h_FCNHkinHb2CSV_Ch%i_S%i_%s",ich,i,sample.c_str()), "Higgs b_{2} DeepCSV", 20, 0.5 ,1);
    h_FCNHkinHb2CSV[ich][i]->SetXTitle("Higgs b_{2} DeepCSV (GeV)");
    h_FCNHkinHb2CSV[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHb2CSV[ich][i]);

    h_FCNHkinLepTopPt[ich][i] = new TH1D(Form("h_FCNHkinLepTopPt_Ch%i_S%i_%s",ich,i,sample.c_str()), "Top p_{T} (Lep)", 30 , 0, 400);
    h_FCNHkinLepTopPt[ich][i]->SetXTitle("Top p_{T} (Lep) (GeV)");
    h_FCNHkinLepTopPt[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinLepTopPt[ich][i]);

    h_FCNHkinHadTopPt[ich][i] = new TH1D(Form("h_FCNHkinHadTopPt_Ch%i_S%i_%s",ich,i,sample.c_str()), "Top p_{T} (Had)", 30 , 0, 400);
    h_FCNHkinHadTopPt[ich][i]->SetXTitle("Top p_{T} (Had) (GeV)");
    h_FCNHkinHadTopPt[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHadTopPt[ich][i]);

    //GenInfo
    h_genDR[ich][i] = new TH1D(Form("h_genDR_Ch%i_S%i_%s",ich,i,sample.c_str()), "Delta R between gen b jets from Higgs", 30, 0, 4);
    h_genDR[ich][i]->SetXTitle("gen #Delta R");
    h_genDR[ich][i]->Sumw2();
    fOutput->Add(h_genDR[ich][i]);

    h_matchDR[ich][i] = new TH1D(Form("h_matchDR_Ch%i_S%i_%s",ich,i,sample.c_str()), "Delta R between gen matched b jets from Higgs", 30, 0, 4);
    h_matchDR[ich][i]->SetXTitle("gen matched #Delta R");
    h_matchDR[ich][i]->Sumw2();
    fOutput->Add(h_matchDR[ich][i]);

    h_genHm[ich][i] = new TH1D(Form("h_genHm_Ch%i_S%i_%s",ich,i,sample.c_str()), "Gen Higgs mass", 30, 0, 250);
    h_genHm[ich][i]->SetXTitle("gen Higgs Mass (GeV)");
    h_genHm[ich][i]->Sumw2();
    fOutput->Add(h_genHm[ich][i]);

    h_matchHm[ich][i] = new TH1D(Form("h_matchHm_Ch%i_S%i_%s",ich,i,sample.c_str()), "Gen matched Higgs mass", 30, 0, 250);
    h_matchHm[ich][i]->SetXTitle("gen matched Higgs Mass (GeV)");
    h_matchHm[ich][i]->Sumw2();
    fOutput->Add(h_matchHm[ich][i]);
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
  if( !option.Contains("Run2017") && (*TruePV < 10 || *TruePV > 75) ) return kTRUE;

  float lep_SF = 1.0;
  float genweight = 1.0;
  float puweight = 1.0;
  float jetsf = 1.0;
  float wrongPVrate = 1;
  if( !option.Contains("Run2017") ){
    lep_SF = lepton_SF[0];
    genweight = *genWeight;
    puweight = PUWeight[0];

    //180703, 10-75
    if      ( option.Contains("DYJets4to50HT100to200v2") ) wrongPVrate = 1.04044583396;
    else if ( option.Contains("DYJets4to50HT100to200_") or option.Contains("DYJets4to50HT100to200part2") ) wrongPVrate = 1.04044583396;
    else if ( option.Contains("DYJets4to50HT200to400") ) wrongPVrate = 1.03548601524;
    else if ( option.Contains("DYJets4to50HT400to600") ) wrongPVrate = 1.04007039394;
    else if ( option.Contains("DYJets4to50HT600toinf") ) wrongPVrate = 1.00150187987;
    else if ( option.Contains("DYJets4to50HT70to100") ) wrongPVrate = 1.04510609321;
    else if ( option.Contains("DYJetsv2") ) wrongPVrate = 1.04239354791;
    else if ( option.Contains("DYJets_") or option.Contains("DYJetspart2") ) wrongPVrate = 1.04239354791;
    else if ( option.Contains("STTH1L3BHct") ) wrongPVrate = 1.04386426774;
    else if ( option.Contains("STTH1L3BHut") ) wrongPVrate = 1.03094683784;
    else if ( option.Contains("SingleTbart") ) wrongPVrate = 1.04162486525;
    else if ( option.Contains("SingleTbartW") ) wrongPVrate = 1.04264471988;
    else if ( option.Contains("SingleTops") ) wrongPVrate = 1.03967280483;
    else if ( option.Contains("SingleTopt") ) wrongPVrate = 1.04020417817;
    else if ( option.Contains("SingleToptW") ) wrongPVrate = 1.04474130408;
    else if ( option.Contains("TTWJetsToLNuPSweight") ) wrongPVrate = 1.04021612264;
    else if ( option.Contains("TTWJetsToQQ") ) wrongPVrate = 1.02450278629;
    else if ( option.Contains("TTZToLLNuNu") ) wrongPVrate = 1.04205962372;
    else if ( option.Contains("TTZToQQ") ) wrongPVrate = 1.04739932325;
    else if ( option.Contains("TTpowhegttbb") ) wrongPVrate = 1.0484454797;
    else if ( option.Contains("TTpowhegttbj") ) wrongPVrate = 1.04795518207;
    else if ( option.Contains("TTpowhegttcc") ) wrongPVrate = 1.04737048218;
    else if ( option.Contains("TTpowhegttlf") ) wrongPVrate = 1.047466385;
    else if ( option.Contains("TTpowhegttother") ) wrongPVrate = 1.04766363512;
    else if ( option.Contains("W1JetsToLNu150-250") ) wrongPVrate = 1.03420702039;
    else if ( option.Contains("W1JetsToLNu250-400") ) wrongPVrate = 1.03580555626;
    else if ( option.Contains("W1JetsToLNu400-inf") ) wrongPVrate = 1.03252405725;
    else if ( option.Contains("W1JetsToLNu50-150") ) wrongPVrate = 1.03266766201;
    else if ( option.Contains("W2JetsToLNu250-400") ) wrongPVrate = 1.03481315504;
    else if ( option.Contains("W2JetsToLNu400-inf") ) wrongPVrate = 1.0317626647;
    else if ( option.Contains("W2JetsToLNu50-150") ) wrongPVrate = 1.00150425214;
    else if ( option.Contains("W3JetsToLNu") ) wrongPVrate = 1.04141687195;
    else if ( option.Contains("W4JetsToLNu") ) wrongPVrate = 1.04339321182;
    else if ( option.Contains("WW") ) wrongPVrate = 1.04679128652;
    else if ( option.Contains("WZ") ) wrongPVrate = 1.04266907923;
    else if ( option.Contains("ZZ") ) wrongPVrate = 1.02940993982;
    else if ( option.Contains("ttHToNonbb") ) wrongPVrate = 1.02082052772;
    else if ( option.Contains("ttHTobb") ) wrongPVrate = 1.03782992744;
    else    wrongPVrate = 1.0;
  }
  float EventWeight = puweight*genweight*lep_SF*wrongPVrate;

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

  //for Kin 
  vector<size_t> jetIdxs;
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
  bool passelectron = (mode == 1) && (lepton.Pt() > 33) && (abs(lepton.Eta()) <= 2.1);

  //Single Lepton only
  if( option.Contains("SingleMuon") ){
    if( !passmuon ) return kTRUE;//RDMu
    if( passelectron) return kTRUE;//RDMu
  }
  else if( option.Contains("SingleElectron") ){
    if( !passelectron ) return kTRUE;//RDelec
    if( passmuon ) return kTRUE;//RDelec
  }
  else{
    if( !passmuon && !passelectron ) return kTRUE;
  }
  //if( passmuon || passelectron ){

  if( passelectron )  EventWeight *= jet_SF_deepCSV_38[0];
  else if ( passmuon) EventWeight *= jet_SF_deepCSV_30[0];

  int jetIdx[4];
  TLorentzVector jetP4s[4];

  for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
    //if( !option.Contains("Run2017") ) jet = jet * jet_JER_Nom[iJet];

    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      if( passelectron and  iJet==0 and jet_pT[iJet] < 38 ) continue;
      njets++;
      jetIdxs.push_back(iJet);

      if( jet_deepCSV[iJet] > 0.4941){
        nbjets_m++;
      }
      if( jet_deepCSV[iJet] > 0.8001 ){
        nbjets_t++;
      }
      if( jet_deepCvsL[iJet] > 0.15 && jet_deepCvsB[iJet] > 0.28 ){
        ncjets_m++;
      }
    }
  }

  TLorentzVector hbjet1, hbjet2, genH;
  if( option.Contains("Hct") || option.Contains("Hut") ){
    if(*addHbjet1_pt > 20 && *addHbjet2_pt > 20 && abs(*addHbjet1_eta) < 2.4 && abs(*addHbjet2_eta) < 2.4){
      hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
      hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

      genH = hbjet1 + hbjet2;
      gendR = hbjet1.DeltaR(hbjet2);
      genHm = genH.M();
    }
  }


  if( reco ){
    //Jet Assignment
    vector<double>::iterator iter;
    int evtIdx = 0;
    //if( njets >= 4 && nbjets_m >= 2 && !lepPt.empty() ){
    if( njets >= 3 && nbjets_m >= 2 && !lepPt.empty() ){
      for( iter = lepPt.begin(); iter != lepPt.end(); iter++){
        if( *iter == static_cast<float>(lepton.Pt()) ){
          int tmpIdx = distance(lepPt.begin(), iter);
          if( missET[tmpIdx] == met ) evtIdx = tmpIdx;
          else continue;
        }
      }
      dupCheck.push_back(evtIdx);

      assignT->GetEntry(evtIdx);
      int i0 = assignT->GetLeaf("idx0")->GetValue(0);
      int i1 = assignT->GetLeaf("idx1")->GetValue(0);
      int i2 = assignT->GetLeaf("idx2")->GetValue(0);
      int i3 = assignT->GetLeaf("idx3")->GetValue(0);
      jetIdx[0] = i0; jetIdx[1] = i1; jetIdx[2] = i2; jetIdx[3] = i3;
      //cout << i0 << endl;

      jetP4s[0].SetPtEtaPhiE(jet_pT[i0], jet_eta[i0], jet_phi[i0], jet_E[i0]);
      jetP4s[1].SetPtEtaPhiE(jet_pT[i1], jet_eta[i1], jet_phi[i1], jet_E[i1]);
      jetP4s[2].SetPtEtaPhiE(jet_pT[i2], jet_eta[i2], jet_phi[i2], jet_E[i2]);
      jetP4s[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);

      /*
      if( !option.Contains("Run2017") ){
        jetP4s[0] = jetP4s[0] * jet_JER_Nom[i0];
        jetP4s[1] = jetP4s[1] * jet_JER_Nom[i1];
        jetP4s[2] = jetP4s[2] * jet_JER_Nom[i2];
        jetP4s[3] = jetP4s[3] * jet_JER_Nom[i3];
      }
      */

      if( option.Contains("Hct") || option.Contains("Hut") ){
        if(hbjet1.DeltaR(jetP4s[1]) < 0.4 or hbjet1.DeltaR(jetP4s[2]) < 0.4) match1 = true;
        if(hbjet2.DeltaR(jetP4s[1]) < 0.4 or hbjet2.DeltaR(jetP4s[2]) < 0.4) match2 = true;
      }
    }
  }//reco option

  /////Fill histograms
  int Ncuts = 18;
  bool eventSelection[Ncuts];
  for(int bcut=0; bcut < Ncuts; bcut++) eventSelection[bcut] = false;

  eventSelection[0] = true;
  eventSelection[1] = ( njets == 3 );
  eventSelection[2] = ( njets == 3 ) && ( nbjets_m == 2 );
  eventSelection[3] = ( njets == 3 ) && ( nbjets_m == 3 );
  eventSelection[4] = ( njets == 3 ) && ( nbjets_m >= 2 );
  eventSelection[5] = ( njets == 3 ) && ( nbjets_m >= 3 );
  eventSelection[6] = ( njets >= 3 ) && ( nbjets_m >= 2 );
  eventSelection[7] = ( njets >= 3 ) && ( nbjets_m >= 3 );
  eventSelection[8] = ( njets == 4 );
  eventSelection[9] = ( njets == 4 ) && ( nbjets_m == 2 );
  eventSelection[10] = ( njets == 4 ) && ( nbjets_m == 3 );
  eventSelection[11] = ( njets == 4 ) && ( nbjets_m == 4 );
  eventSelection[12] = ( njets == 4 ) && ( nbjets_m >= 2 ); 
  eventSelection[13] = ( njets == 4 ) && ( nbjets_m >= 3 );
  eventSelection[14] = ( njets == 4 ) && ( nbjets_m >= 4 );//it's dummy LoL
  eventSelection[15] = ( njets >= 4 ) && ( nbjets_m >= 2 );
  eventSelection[16] = ( njets >= 4 ) && ( nbjets_m >= 3 );
  eventSelection[17] = ( njets >= 4 ) && ( nbjets_m >= 4 );

  for( int cut = 0; cut < 18; cut++){
    if(eventSelection[cut]){
      h_PV[mode][cut]->Fill(*GoodPV,EventWeight);
      h_EventWeight[mode][cut]->Fill(EventWeight);
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
      for( int ii = 0; ii < njets; ii++){
        const size_t ii1 = jetIdxs[ii];
        h_JetCSV[mode][cut]->Fill(jet_deepCSV[ii1],EventWeight);
      }

      if( reco ){
        if( njets >=3 && nbjets_m >=2 ){
          for( int i = 0; i < 3; ++i ){
            const size_t j = jetIdx[i];
            h_csv[mode][cut]->Fill(jet_deepCSV[j],EventWeight);
            h_cvsl[mode][cut]->Fill(jet_deepCvsL[j],EventWeight);
            h_cvsb[mode][cut]->Fill(jet_deepCvsB[j],EventWeight);
          }
          h_FCNHkinLepWMass[mode][cut]->Fill((lepton+p4met).M(),EventWeight);
          h_FCNHkinHadWMass[mode][cut]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_FCNHkinLepTopM[mode][cut]->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
          h_FCNHkinHMass[mode][cut]->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
          h_FCNHkinDR[mode][cut]->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
          h_FCNHkinHadTopM[mode][cut]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
          h_FCNHkinHPt[mode][cut]->Fill((jetP4s[1]+jetP4s[2]).Pt(),EventWeight);
          h_FCNHkinHdPhi[mode][cut]->Fill(abs(jetP4s[1].DeltaPhi(jetP4s[2])),EventWeight);
          h_FCNHkinHdEta[mode][cut]->Fill(abs((jetP4s[1]-jetP4s[2]).Eta()),EventWeight);
          h_FCNHkinHb1CSV[mode][cut]->Fill(jet_deepCSV[jetIdx[2]],EventWeight);
          h_FCNHkinHb2CSV[mode][cut]->Fill(jet_deepCSV[jetIdx[1]],EventWeight);
          h_FCNHkinLepTopPt[mode][cut]->Fill((lepton+p4met+jetP4s[0]).Pt(),EventWeight);
          h_FCNHkinHadTopPt[mode][cut]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).Pt(),EventWeight);
        }
        if(genH.Pt() > 0){
          h_genDR[mode][cut]->Fill(gendR, EventWeight);
          h_genHm[mode][cut]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][cut]->Fill(jetP4s[1].DeltaR(jetP4s[2]), EventWeight);
            h_matchHm[mode][cut]->Fill((jetP4s[1]+jetP4s[2]).M(), EventWeight);
          }
        }
      }//reco
    }//selection loopa
  }
  evtNum++;
  //cout << evtNum << '\r';
  return kTRUE;
}

void MyAnalysis::SlaveTerminate()
{
  TString option = GetOption();
}
   

void MyAnalysis::Terminate()
{
  TString option = GetOption();

  if( option.Contains("_") ) reco = true;
  else reco = false;

  if(reco) out = TFile::Open(Form("doReco/hist_%s.root",option.Data()),"RECREATE");
  else out = TFile::Open(Form("doNoReco/hist_%s.root",option.Data()),"RECREATE");

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

  const char* assign_file = "";
  if( reco ){
    assign_file = Form("/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/classifier/2017/assignSTFCNC01/assign_deepReco_%s.root", option.Data());
    string file_tmp_path = assign_file;
    ifstream file_tmp(file_tmp_path);

    if(file_tmp.is_open()){
      //cout << lepcount << endl;
      auto it = unique( dupCheck.begin(), dupCheck.end() );
      cout << ((it == dupCheck.end()) ? "Unique\n" : "Duplicate(s)\n");
    }
  }
}

double MyAnalysis::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

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

  if( option.Contains("_") ){
    reco = true;
    syst = true;
  }
  else{//for no reco
    reco = false;
    syst = false;
  }

  //Delete ntuple number and data era so that we can merge histos w.r.t. dataset, prepare assign ntuple
  const char* assign_file = "";
  if( reco ){
    //if     ( option.Contains("Run2017") ) sample.erase(sample.find_first_of("_")-1, string::npos);
    //else if( option.Contains("part") )    sample.erase(sample.find_last_of("p"),    string::npos);
    //else                                  sample.erase(sample.find_first_of("_"),   string::npos);
    string reco_scheme = sample.substr(sample.find_first_of("-")+1,string::npos);
    if     ( reco_scheme.find("STFCNC") != string::npos ) reco_id = 0;
    else if( reco_scheme.find("TTFCNC") != string::npos ) reco_id = 1;
    else if( reco_scheme.find("TTBKG") != string::npos )  reco_id = 2;
    else                                                  reco_id = -1;

    if( reco_scheme.find("jec") != string::npos or reco_scheme.find("jer") != string::npos ){
      syst = false;//make another hist file
      if     ( reco_scheme.find("jecup") != string::npos )   syst_ext = "jecup";
      else if( reco_scheme.find("jecdown") != string::npos ) syst_ext = "jecdown";
      else if( reco_scheme.find("jerup") != string::npos )   syst_ext = "jerup";
      else if( reco_scheme.find("jerdown") != string::npos ) syst_ext = "jerdown";
    }

    sample.erase( sample.find_first_of("-"),string::npos );
    assign_file = Form("/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/assign%s/assign_deepReco_%s.root",
                       reco_scheme.c_str(), sample.c_str());

    string file_tmp_path = assign_file;
    ifstream file_tmp(file_tmp_path);
    if( file_tmp.is_open() ){
      assignF = TFile::Open(assign_file, "READ");
      assignT = (TTree*) assignF->Get("tree");
      int nevt = assignT->GetEntries();
      if( nevt > 0){
        for(int i = 0; i < nevt; i++){
          assignT->GetEntry(i);
          float pt = assignT->GetLeaf("leptonPt")->GetValue(0);
          float met = assignT->GetLeaf("missingET")->GetValue(0);
          lepPt.push_back(pt);
          missET.push_back(met);
        }
      }
    }
    else cout << option.Data() << endl;

    if( !syst and syst_ext.length() < 5 ) cout << sample.c_str() << " " << "No jec/jer systematic option!" << endl;
  }


  //cout << "SlaveBegin" << endl;
  for( int ich=0; ich < 3; ich++ ){
    for( int i=0; i < 12; i++ ){
      for( int syst = 0; syst != syst_num; ++syst ){

        h_PV[ich][i][syst] = new TH1D(Form("h_PV_Ch%i_S%i%s",ich,i,syst_name[syst]), "Number of primary vertices", 60, 0, 60);
        h_PV[ich][i][syst]->SetXTitle("Good PV");
        h_PV[ich][i][syst]->Sumw2();
        fOutput->Add(h_PV[ich][i][syst]);

        h_EventWeight[ich][i][syst] = new TH1D(Form("h_EventWeight_Ch%i_S%i%s",ich,i,syst_name[syst]), "Event Weights", 100, 0, 2);
        h_EventWeight[ich][i][syst]->SetXTitle("Event Weight");
        h_EventWeight[ich][i][syst]->Sumw2();
        fOutput->Add(h_EventWeight[ich][i][syst]);
      
        h_NJet[ich][i][syst] = new TH1D(Form("h_NJet_Ch%i_S%i%s",ich,i,syst_name[syst]), "Number of jets", 12, 0, 12);
        h_NJet[ich][i][syst]->SetXTitle("Jet Multiplicity");
        h_NJet[ich][i][syst]->Sumw2();
        fOutput->Add(h_NJet[ich][i][syst]);

        h_NBJetCSVv2M[ich][i][syst] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i%s",ich,i,syst_name[syst]), "Number of b tagged jets (medium)", 6, 0, 6);
        h_NBJetCSVv2M[ich][i][syst]->SetXTitle("b-tagged Jet Multiplicity (DeepCSVM)");
        h_NBJetCSVv2M[ich][i][syst]->Sumw2();
        fOutput->Add(h_NBJetCSVv2M[ich][i][syst]);

        h_NBJetCSVv2T[ich][i][syst] = new TH1D(Form("h_NBJetCSVv2T_Ch%i_S%i%s",ich,i,syst_name[syst]), "Number of b tagged jets (tight)", 6, 0, 6);
        h_NBJetCSVv2T[ich][i][syst]->SetXTitle("b-tagged Jet Multiplicity (DeepCSVT)");
        h_NBJetCSVv2T[ich][i][syst]->Sumw2();
        fOutput->Add(h_NBJetCSVv2T[ich][i][syst]);

        h_NCJetM[ich][i][syst] = new TH1D(Form("h_NCJetM_Ch%i_S%i%s",ich,i,syst_name[syst]), "Number of c tagged jets", 6, 0, 6);
        h_NCJetM[ich][i][syst]->SetXTitle("c-tagged Jet Multiplicity (MediumWP)");
        h_NCJetM[ich][i][syst]->Sumw2();
        fOutput->Add(h_NCJetM[ich][i][syst]);  

        h_MET[ich][i][syst] = new TH1D(Form("h_MET_Ch%i_S%i%s",ich,i,syst_name[syst]), "MET", 30,0,200);
        h_MET[ich][i][syst]->SetXTitle("MET (GeV)");
        h_MET[ich][i][syst]->Sumw2();
        fOutput->Add(h_MET[ich][i][syst]);

        h_LepPt[ich][i][syst] = new TH1D(Form("h_LepPt_Ch%i_S%i%s",ich,i,syst_name[syst]), "Lepton p_{T}", 30,0,200);
        h_LepPt[ich][i][syst]->SetXTitle("Lepton p_{T} (GeV)");
        h_LepPt[ich][i][syst]->Sumw2();
        fOutput->Add(h_LepPt[ich][i][syst]);

        h_LepPhi[ich][i][syst] = new TH1D(Form("h_LepPhi_Ch%i_S%i%s",ich,i,syst_name[syst]), "Lepton #phi", 30 ,0 ,3.2);
        h_LepPhi[ich][i][syst]->SetXTitle("Lepton |#phi|");
        h_LepPhi[ich][i][syst]->Sumw2();
        fOutput->Add(h_LepPhi[ich][i][syst]);

        h_LepEta[ich][i][syst] = new TH1D(Form("h_LepEta_Ch%i_S%i%s",ich,i,syst_name[syst]), "Lepton #eta", 30 ,-2.5 ,2.5);
        h_LepEta[ich][i][syst]->SetXTitle("Lepton #eta");
        h_LepEta[ich][i][syst]->Sumw2();
        fOutput->Add(h_LepEta[ich][i][syst]);

        h_WMass[ich][i][syst] = new TH1D(Form("h_WMass_Ch%i_S%i%s",ich,i,syst_name[syst]), "W Mass", 30 ,0 ,200);
        h_WMass[ich][i][syst]->SetXTitle("Transverse Mass (GeV)");
        h_WMass[ich][i][syst]->Sumw2();
        fOutput->Add(h_WMass[ich][i][syst]);

        h_DPhi[ich][i][syst] = new TH1D(Form("h_DPhi_Ch%i_S%i%s",ich,i,syst_name[syst]), "Lepton MET #Delta#phi", 30 ,0 ,3.2);
        h_DPhi[ich][i][syst]->SetXTitle("|#Delta#phi_{l,MET}|");
        h_DPhi[ich][i][syst]->Sumw2();
        fOutput->Add(h_DPhi[ich][i][syst]);

        h_LepIso[ich][i][syst] = new TH1D(Form("h_LepIso_Ch%i_S%i%s",ich,i,syst_name[syst]), "LepIso", 20 ,0 ,0.15);
        h_LepIso[ich][i][syst]->SetXTitle("Relative Isolation");
        h_LepIso[ich][i][syst]->Sumw2();
        fOutput->Add(h_LepIso[ich][i][syst]);

        h_JetCSV[ich][i][syst] = new TH1D(Form("h_JetCSV_Ch%i_S%i%s",ich,i,syst_name[syst]), "DeepCSV", 20 ,0 ,1);
        h_JetCSV[ich][i][syst]->SetXTitle("DeepCSV (all jets)");
        h_JetCSV[ich][i][syst]->Sumw2();
        fOutput->Add(h_JetCSV[ich][i][syst]);

        h_csv[ich][i][syst] = new TH1D(Form("h_csv_Ch%i_S%i%s",ich,i,syst_name[syst]), "DeepCSV", 20 ,0 ,1);
        h_csv[ich][i][syst]->SetXTitle("DeepCSV");
        h_csv[ich][i][syst]->Sumw2();
        fOutput->Add(h_csv[ich][i][syst]);

        h_cvsl[ich][i][syst] = new TH1D(Form("h_cvsl_Ch%i_S%i%s",ich,i,syst_name[syst]), "CvsL", 20 , 0 ,1);
        h_cvsl[ich][i][syst]->SetXTitle("DeepCvsL");
        h_cvsl[ich][i][syst]->Sumw2();
        fOutput->Add(h_cvsl[ich][i][syst]);

        h_cvsb[ich][i][syst] = new TH1D(Form("h_cvsb_Ch%i_S%i%s",ich,i,syst_name[syst]), "CvsB", 20 , 0 ,1);
        h_cvsb[ich][i][syst]->SetXTitle("DeepCvsB");
        h_cvsb[ich][i][syst]->Sumw2();
        fOutput->Add(h_cvsb[ich][i][syst]);

        h_FCNHkinLepWMass[ich][i][syst] = new TH1D(Form("h_FCNHkinLepWMass_Ch%i_S%i%s",ich,i,syst_name[syst]), "W Mass (Lep)", 30 , 0, 300);
        h_FCNHkinLepWMass[ich][i][syst]->SetXTitle("W Mass (Lep) (GeV)");
        h_FCNHkinLepWMass[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinLepWMass[ich][i][syst]);

        h_FCNHkinHadWMass[ich][i][syst] = new TH1D(Form("h_FCNHkinHadWMass_Ch%i_S%i%s",ich,i,syst_name[syst]), "W Mass (Had)", 30, 0, 300);
        h_FCNHkinHadWMass[ich][i][syst]->SetXTitle("W Mass (Had) (GeV)");
        h_FCNHkinHadWMass[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHadWMass[ich][i][syst]);

        h_FCNHkinHMass[ich][i][syst] = new TH1D(Form("h_FCNHkinHMass_Ch%i_S%i%s",ich,i,syst_name[syst]), "Higgs Mass (bb)", 30, 0,250);
        h_FCNHkinHMass[ich][i][syst]->SetXTitle("Higgs Mass (GeV)");
        h_FCNHkinHMass[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHMass[ich][i][syst]);

        h_FCNHkinDR[ich][i][syst] = new TH1D(Form("h_FCNHkinDR_Ch%i_S%i%s",ich,i,syst_name[syst]), "#Delta R of bb", 30, 0,4);
        h_FCNHkinDR[ich][i][syst]->SetXTitle("#Delta R of b jets from Higgs");
        h_FCNHkinDR[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinDR[ich][i][syst]);

        h_FCNHkinLepTopM[ich][i][syst] = new TH1D(Form("h_FCNHkinLepTopM_Ch%i_S%i%s",ich,i,syst_name[syst]), "Top Mass (Lep)", 30 , 0, 400);
        h_FCNHkinLepTopM[ich][i][syst]->SetXTitle("Top Mass (Lep) (GeV)");
        h_FCNHkinLepTopM[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinLepTopM[ich][i][syst]);

        h_FCNHkinHadTopM[ich][i][syst] = new TH1D(Form("h_FCNHkinHadTopM_Ch%i_S%i%s",ich,i,syst_name[syst]), "Top Mass from Hc/u", 30, 0, 400);
        h_FCNHkinHadTopM[ich][i][syst]->SetXTitle("Top Mass (Had) (GeV)");
        h_FCNHkinHadTopM[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHadTopM[ich][i][syst]);

        h_FCNHkinHPt[ich][i][syst] = new TH1D(Form("h_FCNHkinHPt_Ch%i_S%i%s",ich,i,syst_name[syst]), "Higgs p_{T}", 30, 0,300);
        h_FCNHkinHPt[ich][i][syst]->SetXTitle("Higgs p_{T} (GeV)");
        h_FCNHkinHPt[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHPt[ich][i][syst]);

        h_FCNHkinHdPhi[ich][i][syst] = new TH1D(Form("h_FCNHkinHdPhi_Ch%i_S%i%s",ich,i,syst_name[syst]), "#Delta#phi of bb from Higgs", 30, 0, 3.2);
        h_FCNHkinHdPhi[ich][i][syst]->SetXTitle("|#Delta#phi_{bb}|");
        h_FCNHkinHdPhi[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHdPhi[ich][i][syst]);

        h_FCNHkinHdEta[ich][i][syst] = new TH1D(Form("h_FCNHkinHdEta_Ch%i_S%i%s",ich,i,syst_name[syst]), "#Delta#eta of bb from Higgs", 30, 0, 2.5);
        h_FCNHkinHdEta[ich][i][syst]->SetXTitle("|#Delta#eta_{bb}|");
        h_FCNHkinHdEta[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHdEta[ich][i][syst]);

        h_FCNHkinHb1CSV[ich][i][syst] = new TH1D(Form("h_FCNHkinHb1CSV_Ch%i_S%i%s",ich,i,syst_name[syst]), "Higgs b_{1} DeepCSV", 20, 0.5 ,1);
        h_FCNHkinHb1CSV[ich][i][syst]->SetXTitle("Higgs b_{1} DeepCSV (GeV)");
        h_FCNHkinHb1CSV[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHb1CSV[ich][i][syst]);

        h_FCNHkinHb2CSV[ich][i][syst] = new TH1D(Form("h_FCNHkinHb2CSV_Ch%i_S%i%s",ich,i,syst_name[syst]), "Higgs b_{2} DeepCSV", 20, 0.5 ,1);
        h_FCNHkinHb2CSV[ich][i][syst]->SetXTitle("Higgs b_{2} DeepCSV (GeV)");
        h_FCNHkinHb2CSV[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHb2CSV[ich][i][syst]);

        h_FCNHkinLepTopPt[ich][i][syst] = new TH1D(Form("h_FCNHkinLepTopPt_Ch%i_S%i%s",ich,i,syst_name[syst]), "Top p_{T} (Lep)", 30 , 0, 400);
        h_FCNHkinLepTopPt[ich][i][syst]->SetXTitle("Top p_{T} (Lep) (GeV)");
        h_FCNHkinLepTopPt[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinLepTopPt[ich][i][syst]);

        h_FCNHkinHadTopPt[ich][i][syst] = new TH1D(Form("h_FCNHkinHadTopPt_Ch%i_S%i%s",ich,i,syst_name[syst]), "Top p_{T} (Had)", 30 , 0, 400);
        h_FCNHkinHadTopPt[ich][i][syst]->SetXTitle("Top p_{T} (Had) (GeV)");
        h_FCNHkinHadTopPt[ich][i][syst]->Sumw2();
        fOutput->Add(h_FCNHkinHadTopPt[ich][i][syst]);

        //GenInfo
        h_genDR[ich][i][syst] = new TH1D(Form("h_genDR_Ch%i_S%i%s",ich,i,syst_name[syst]), "Delta R between gen b jets from Higgs", 30, 0, 4);
        h_genDR[ich][i][syst]->SetXTitle("gen #Delta R");
        h_genDR[ich][i][syst]->Sumw2();
        fOutput->Add(h_genDR[ich][i][syst]);

        h_matchDR[ich][i][syst] = new TH1D(Form("h_matchDR_Ch%i_S%i%s",ich,i,syst_name[syst]), "Delta R between gen matched b jets from Higgs", 30, 0, 4);
        h_matchDR[ich][i][syst]->SetXTitle("gen matched #Delta R");
        h_matchDR[ich][i][syst]->Sumw2();
        fOutput->Add(h_matchDR[ich][i][syst]);

        h_genHm[ich][i][syst] = new TH1D(Form("h_genHm_Ch%i_S%i%s",ich,i,syst_name[syst]), "Gen Higgs mass", 30, 0, 250);
        h_genHm[ich][i][syst]->SetXTitle("gen Higgs Mass (GeV)");
        h_genHm[ich][i][syst]->Sumw2();
        fOutput->Add(h_genHm[ich][i][syst]);

        h_matchHm[ich][i][syst] = new TH1D(Form("h_matchHm_Ch%i_S%i%s",ich,i,syst_name[syst]), "Gen matched Higgs mass", 30, 0, 250);
        h_matchHm[ich][i][syst]->SetXTitle("gen matched Higgs Mass (GeV)");
        h_matchHm[ich][i][syst]->Sumw2();
        fOutput->Add(h_matchHm[ich][i][syst]);
      }
    }
  }
} 

Bool_t MyAnalysis::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  int mode = 999; 
  mode = *channel;

  if( mode > 2 ) return kTRUE;

  float wrongPVrate = 1;
  if( !option.Contains("Run2017") ){
    //180922, 10-75
    if      ( option.Contains("DYJets4to50HT100to200v2") ) wrongPVrate = 1.04010725093;
    else if ( option.Contains("DYJets4to50HT100to200_") or option.Contains("DYJets4to50HT100to200part2") ) wrongPVrate = 1.04010725093;
    else if ( option.Contains("DYJets4to50HT200to400") ) wrongPVrate = 1.0354286358;
    else if ( option.Contains("DYJets4to50HT400to600") ) wrongPVrate = 1.0395588053;
    else if ( option.Contains("DYJets4to50HT70to100") ) wrongPVrate = 1.04495754249;
    else if ( option.Contains("DYJetsv2") ) wrongPVrate = 1.04239771686;
    else if ( option.Contains("DYJets_") or option.Contains("DYJetspart2") ) wrongPVrate = 1.04239771686;
    else if ( option.Contains("STTH1L3BHct") ) wrongPVrate = 1.04397325747;
    else if ( option.Contains("STTH1L3BHut") ) wrongPVrate = 1.03100541427;
    else if ( option.Contains("SingleTbart") ) wrongPVrate = 1.04171033265;
    else if ( option.Contains("SingleTbartW") ) wrongPVrate = 1.04261195626;
    else if ( option.Contains("SingleTops") ) wrongPVrate = 1.03972787611;
    else if ( option.Contains("SingleTopt") ) wrongPVrate = 1.04028244802;
    else if ( option.Contains("SingleToptW") ) wrongPVrate = 1.04478300674;
    else if ( option.Contains("TTHadpowheg") ) wrongPVrate = 1.02911490683;
    else if ( option.Contains("TTLLpowheg") ) wrongPVrate = 1.04707001305;
    else if ( option.Contains("TTWJetsToLNuPSweight") ) wrongPVrate = 1.04084725366;
    else if ( option.Contains("TTWJetsToQQ") ) wrongPVrate = 1.02412815424;
    else if ( option.Contains("TTZToLLNuNu") ) wrongPVrate = 1.04242525748;
    else if ( option.Contains("TTZToQQ") ) wrongPVrate = 1.04903621775;
    else if ( option.Contains("TTpowhegttbb") ) wrongPVrate = 1.04855161159;
    else if ( option.Contains("TTpowhegttbj") ) wrongPVrate = 1.0481276857;
    else if ( option.Contains("TTpowhegttcc") ) wrongPVrate = 1.04745462065;
    else if ( option.Contains("TTpowhegttlf") ) wrongPVrate = 1.04754742837;
    else if ( option.Contains("TTpowhegttother") ) wrongPVrate = 1.04775924349;
    else if ( option.Contains("W3JetsToLNu") ) wrongPVrate = 1.04151096637;
    else if ( option.Contains("W4JetsToLNu") ) wrongPVrate = 1.04351715127;
    else if ( option.Contains("WW") ) wrongPVrate = 1.04691597274;
    else if ( option.Contains("WZ") ) wrongPVrate = 1.04251110292;
    else if ( option.Contains("ZZ") ) wrongPVrate = 1.02960069789;
    else if ( option.Contains("ttHToNonbb") ) wrongPVrate = 1.03169876149;
    else if ( option.Contains("ttHTobb") ) wrongPVrate = 1.03747398495;
    else    wrongPVrate = 1.0;
  }
  if( wrongPVrate > 1.01 ){
    if( *TruePV < 10 || *TruePV > 75 ) return kTRUE;
  }
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
  lepton.SetPtEtaPhiE(*lepton_pt, *lepton_eta, *lepton_phi, *lepton_e);

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
  bool passelectron;
  if ( *elec_trg == 10 ) passelectron = (mode == 1) && (lepton.Pt() > 33) && (abs(lepton.Eta()) <= 2.1);
  else                   passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

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

  int jetIdx[4];
  TLorentzVector jetP4s[4];

  for (unsigned int iJet = 0; iJet < jet_pt.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pt[iJet], jet_eta[iJet], jet_phi[iJet], jet_e[iJet]);
/*
    if( !option.Contains("Run2017") ){
      if     ( syst_ext == "jecup" )   jet = jet * jet_JER_Nom[iJet] * jet_JES_Up[iJet];
      else if( syst_ext == "jecdown" ) jet = jet * jet_JER_Nom[iJet] * jet_JES_Down[iJet];
      else if( syst_ext == "jerup" )   jet = jet * jet_JER_Up[iJet];
      else if( syst_ext == "jerdown" ) jet = jet * jet_JER_Down[iJet];
      else                             jet = jet * jet_JER_Nom[iJet];
    }
*/
    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      if( passelectron and  *elec_trg == 10 and njets == 0 and jet_pt[iJet] < 38 ) continue;
      njets++;
      jetIdxs.push_back(iJet);

      if( jet_deepCSV[iJet] > 0.4941 ) nbjets_m++;
      if( jet_deepCSV[iJet] > 0.8001 ) nbjets_t++;
      if( jet_deepCvsL[iJet] > 0.15 && jet_deepCvsB[iJet] > 0.28 ) ncjets_m++;
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

  int njet_cut = 0;
  if( reco ){
    //Jet Assignment
    vector<double>::iterator iter;
    int evtIdx = 0;
    if     ( reco_id == 0 )                 njet_cut = 3;
    else if( reco_id == 1 or reco_id == 2 ) njet_cut = 4;
    //find combination
    if( njet_cut != 0 && njets >= njet_cut && nbjets_m >= 2 && !lepPt.empty() ){
      for( iter = lepPt.begin(); iter != lepPt.end(); iter++){
        if( *iter == static_cast<float>(lepton.Pt()) ){
          int tmpIdx = distance(lepPt.begin(), iter);
          if( missET[tmpIdx] == met ) evtIdx = tmpIdx;
          else continue;
        }
      }
      dupCheck.push_back(evtIdx);
      cout << evtIdx << endl;

      assignT->GetEntry(evtIdx);
      int i0 = assignT->GetLeaf("idx0")->GetValue(0);
      int i1 = assignT->GetLeaf("idx1")->GetValue(0);
      int i2 = assignT->GetLeaf("idx2")->GetValue(0);
      int i3 = assignT->GetLeaf("idx3")->GetValue(0);
      jetIdx[0] = i0; jetIdx[1] = i1; jetIdx[2] = i2; jetIdx[3] = i3;
      //cout << i0 << endl;

      jetP4s[0].SetPtEtaPhiE(jet_pt[i0], jet_eta[i0], jet_phi[i0], jet_e[i0]);
      jetP4s[1].SetPtEtaPhiE(jet_pt[i1], jet_eta[i1], jet_phi[i1], jet_e[i1]);
      jetP4s[2].SetPtEtaPhiE(jet_pt[i2], jet_eta[i2], jet_phi[i2], jet_e[i2]);
      jetP4s[3].SetPtEtaPhiE(jet_pt[i3], jet_eta[i3], jet_phi[i3], jet_e[i3]);

      if( !option.Contains("Run2017") ){
        if     ( syst_ext == "jecup" )   for( int i=0; i < 4; i++) jetP4s[i] = jetP4s[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Up[jetIdx[i]];
        else if( syst_ext == "jecdown" ) for( int i=0; i < 4; i++) jetP4s[i] = jetP4s[i] * jet_JER_Nom[jetIdx[i]] * jet_JES_Down[jetIdx[i]];
        else if( syst_ext == "jerup" )   for( int i=0; i < 4; i++) jetP4s[i] = jetP4s[i] * jet_JER_Up[jetIdx[i]];
        else if( syst_ext == "jerdown" ) for( int i=0; i < 4; i++) jetP4s[i] = jetP4s[i] * jet_JER_Down[jetIdx[i]];
        else                             for( int i=0; i < 4; i++) jetP4s[i] = jetP4s[i] * jet_JER_Nom[jetIdx[i]];
      }

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

  eventSelection[0]  = true;
  eventSelection[1]  = ( njets == 3 );
  eventSelection[2]  = ( njets == 3 ) && ( nbjets_m == 2 );
  eventSelection[3]  = ( njets == 3 ) && ( nbjets_m == 3 );
  eventSelection[4]  = ( njets >= 3 ) && ( nbjets_m >= 2 );
  eventSelection[5]  = ( njets == 4 );
  eventSelection[6]  = ( njets >= 4 ) && ( nbjets_m == 2 ); 
  eventSelection[7]  = ( njets >= 4 ) && ( nbjets_m == 3 );
  eventSelection[8]  = ( njets >= 4 ) && ( nbjets_m == 4 );
  eventSelection[9]  = ( njets >= 4 ) && ( nbjets_m >= 2 );
  eventSelection[10] = ( njets >= 4 ) && ( nbjets_m >= 3 );
  eventSelection[11] = ( njets >= 4 ) && ( nbjets_m >= 4 );

  int modeArray[2] = {mode, 2};

  for( int MODE : modeArray ){
    for( int cut = 0; cut < 12; cut++){
      if( eventSelection[cut] ){
        for( int syst = 0; syst != syst_num; ++syst ){
          if( syst > 0 and !syst ) continue;
          float EventWeight = 1;
          //Multiply syst. to event weight
          if( !option.Contains("Run2017") ){
            EventWeight *= wrongPVrate;
            EventWeight *= *genweight;
            //PUWight
            if      ( isPartOf("puup", std::string(syst_name[syst])) )   EventWeight *= PUWeight[1];
            else if ( isPartOf("pudown", std::string(syst_name[syst])) ) EventWeight *= PUWeight[2];
            else    EventWeight *= PUWeight[0];
            //leptonSF
            if      ( isPartOf("lepSFup", std::string(syst_name[syst])) )   EventWeight *= lepton_SF[1];
            else if ( isPartOf("lepSFdown", std::string(syst_name[syst])) ) EventWeight *= lepton_SF[2];
            else    EventWeight *= lepton_SF[0];
            //scaleWeight
            //[0] = muF up , [1] = muF down, [2] = muR up, [3] = muR up && muF up, [4] = muR down, [5] = muF down && muF down
            if( option.Contains("TTpowheg") or option.Contains("TTLL") or option.Contains("TTHad") ){
              if      ( isPartOf("scale0", std::string(syst_name[syst])) ) EventWeight *= scaleweight[0];
              else if ( isPartOf("scale1", std::string(syst_name[syst])) ) EventWeight *= scaleweight[1];
              else if ( isPartOf("scale2", std::string(syst_name[syst])) ) EventWeight *= scaleweight[2];
              else if ( isPartOf("scale3", std::string(syst_name[syst])) ) EventWeight *= scaleweight[3];
              else if ( isPartOf("scale4", std::string(syst_name[syst])) ) EventWeight *= scaleweight[4];
              else if ( isPartOf("scale5", std::string(syst_name[syst])) ) EventWeight *= scaleweight[5];
              else    EventWeight *= 1;
            }
            else EventWeight *= 1;
            //Deep CSV shape 
            if( passelectron && *elec_trg == 10 ){
              if      ( isPartOf("lfup",        std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[3];
              else if ( isPartOf("lfdown",      std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[4];
              else if ( isPartOf("hfup",        std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[5];
              else if ( isPartOf("hfdown",      std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[6];
              else if ( isPartOf("hfstat1up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[7];
              else if ( isPartOf("hfstat1down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[8];
              else if ( isPartOf("hfstat2up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[9];
              else if ( isPartOf("hfstat2down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[10];
              else if ( isPartOf("lfstat1up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[11];
              else if ( isPartOf("lfstat1down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[12];
              else if ( isPartOf("lfstat2up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[13];
              else if ( isPartOf("lfstat2down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[14];
              else if ( isPartOf("cferr1up",    std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[15];
              else if ( isPartOf("cferr1down",  std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[16];
              else if ( isPartOf("cferr2up",    std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]+jet_SF_deepCSV_38[17];
              else if ( isPartOf("cferr2down",  std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_38[0]-jet_SF_deepCSV_38[18];
              else                                                              EventWeight *= jet_SF_deepCSV_38[0];
            }
            else{
              if      ( isPartOf("lfup",        std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[3];
              else if ( isPartOf("lfdown",      std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[4];
              else if ( isPartOf("hfup",        std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[5];
              else if ( isPartOf("hfdown",      std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[6];
              else if ( isPartOf("hfstat1up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[7];
              else if ( isPartOf("hfstat1down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[8];
              else if ( isPartOf("hfstat2up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[9];
              else if ( isPartOf("hfstat2down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[10];
              else if ( isPartOf("lfstat1up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[11];
              else if ( isPartOf("lfstat1down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[12];
              else if ( isPartOf("lfstat2up",   std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[13];
              else if ( isPartOf("lfstat2down", std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[14];
              else if ( isPartOf("cferr1up",    std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[15];
              else if ( isPartOf("cferr1down",  std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[16];
              else if ( isPartOf("cferr2up",    std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]+jet_SF_deepCSV_30[17];
              else if ( isPartOf("cferr2down",  std::string(syst_name[syst])) ) EventWeight *= jet_SF_deepCSV_30[0]-jet_SF_deepCSV_30[18];
              else                                                              EventWeight *= jet_SF_deepCSV_30[0];
            }
          }

          h_PV[MODE][cut][syst]         ->Fill(*GoodPV,EventWeight);
          h_EventWeight[MODE][cut][syst]->Fill(EventWeight);
          h_NJet[MODE][cut][syst]       ->Fill(njets, EventWeight);
          h_NBJetCSVv2M[MODE][cut][syst]->Fill(nbjets_m, EventWeight);
          h_NBJetCSVv2T[MODE][cut][syst]->Fill(nbjets_t, EventWeight);
          h_NCJetM[MODE][cut][syst]     ->Fill(ncjets_m, EventWeight);
          h_LepPt[MODE][cut][syst]      ->Fill(lepton.Pt(), EventWeight);
          h_LepPhi[MODE][cut][syst]     ->Fill(lepton.Phi(), EventWeight);
          h_LepEta[MODE][cut][syst]     ->Fill(lepton.Eta(), EventWeight);
          h_MET[MODE][cut][syst]        ->Fill(*MET, EventWeight);
          h_WMass[MODE][cut][syst]      ->Fill(transverseM, EventWeight);
          h_DPhi[MODE][cut][syst]       ->Fill(lepDphi, EventWeight);
          h_LepIso[MODE][cut][syst]     ->Fill(relIso, EventWeight);
          for( int ii = 0; ii < njets; ii++){
            const size_t ii1 = jetIdxs[ii];
            h_JetCSV[MODE][cut][syst]->Fill(jet_deepCSV[ii1],EventWeight);
          }

          if( reco ){
            if( njets >= njet_cut && nbjets_m >= 2 ){
              for( int i = 0; i < 3; ++i ){
                const size_t j = jetIdx[i];
                h_csv[MODE][cut][syst] ->Fill(jet_deepCSV[j],EventWeight);
                h_cvsl[MODE][cut][syst]->Fill(jet_deepCvsL[j],EventWeight);
                h_cvsb[MODE][cut][syst]->Fill(jet_deepCvsB[j],EventWeight);
              }
              h_FCNHkinLepWMass[MODE][cut][syst]->Fill((lepton+p4met).M(),EventWeight);
              h_FCNHkinHadWMass[MODE][cut][syst]->Fill((jetP4s[2]+jetP4s[3]).M(),EventWeight);
              h_FCNHkinLepTopM[MODE][cut][syst] ->Fill((lepton+p4met+jetP4s[0]).M(),EventWeight);
              h_FCNHkinHMass[MODE][cut][syst]   ->Fill((jetP4s[1]+jetP4s[2]).M(),EventWeight);
              h_FCNHkinDR[MODE][cut][syst]      ->Fill(jetP4s[1].DeltaR(jetP4s[2]),EventWeight);
              h_FCNHkinHadTopM[MODE][cut][syst] ->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).M(),EventWeight);
              h_FCNHkinHPt[MODE][cut][syst]     ->Fill((jetP4s[1]+jetP4s[2]).Pt(),EventWeight);
              h_FCNHkinHdPhi[MODE][cut][syst]   ->Fill(abs(jetP4s[1].DeltaPhi(jetP4s[2])),EventWeight);
              h_FCNHkinHdEta[MODE][cut][syst]   ->Fill(abs((jetP4s[1]-jetP4s[2]).Eta()),EventWeight);
              h_FCNHkinHb1CSV[MODE][cut][syst]  ->Fill(jet_deepCSV[jetIdx[1]],EventWeight);
              h_FCNHkinHb2CSV[MODE][cut][syst]  ->Fill(jet_deepCSV[jetIdx[2]],EventWeight);
              h_FCNHkinLepTopPt[MODE][cut][syst]->Fill((lepton+p4met+jetP4s[0]).Pt(),EventWeight);
              h_FCNHkinHadTopPt[MODE][cut][syst]->Fill((jetP4s[1]+jetP4s[2]+jetP4s[3]).Pt(),EventWeight);
            }
            if(genH.Pt() > 0){
              h_genDR[MODE][cut][syst]->Fill(gendR, EventWeight);
              h_genHm[MODE][cut][syst]->Fill(genHm, EventWeight);
              if(match1 && match2){
                h_matchDR[MODE][cut][syst]->Fill(jetP4s[1].DeltaR(jetP4s[2]), EventWeight);
                h_matchHm[MODE][cut][syst]->Fill((jetP4s[1]+jetP4s[2]).M(), EventWeight);
              }
            }
          }//reco
        }//syst loop
      }//selection loopa
    }//mode loop : e or mu + emu
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
  string sample = option.Data();

  if( option.Contains("_") ){
    reco = true;
    syst = true;
  }
  else{//for no reco
    reco = false;
    syst = false;
  }

  const char* assign_file = "";
  if( reco ){
    string reco_scheme = sample.substr(sample.find_first_of("-")+1,string::npos);
    if( reco_scheme.find("jec") != string::npos or reco_scheme.find("jer") != string::npos ){
      syst_ext = "";
      if     ( reco_scheme.find("jecup") != string::npos )   syst_ext = "jecup";
      else if( reco_scheme.find("jecdown") != string::npos ) syst_ext = "jecdown";
      else if( reco_scheme.find("jerup") != string::npos )   syst_ext = "jerup";
      else if( reco_scheme.find("jerdown") != string::npos ) syst_ext = "jerdown";
    }
    sample.erase( sample.find_first_of("-"),string::npos );
    assign_file = Form("/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/assign%s/assign_deepReco_%s.root",
                       reco_scheme.c_str(), sample.c_str());
    string file_tmp_path = assign_file;
    ifstream file_tmp(file_tmp_path);

    if(file_tmp.is_open()){
      //cout << lepcount << endl;
      auto it = unique( dupCheck.begin(), dupCheck.end() );
      if( it != dupCheck.end() ) cout << string(option.Data()) + string(": Duplicate(s)") << endl;
    }
    else cout << "file " + file_tmp_path + " not opened!" << endl;
    out = TFile::Open(Form("doReco/temp/hist_%s%s.root", sample.c_str(), syst_ext.c_str()),"RECREATE");
  }
  else out = TFile::Open(Form("doNoReco/hist_%s.root", option.Data()),"RECREATE");

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

bool MyAnalysis::isPartOf(const std::string& word, const std::string& sentence) {
    return sentence.find(word)    // this returns the index of the first instance
                                  // word
           != std::string::npos;  // which will take this value if it's not found
}

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

    h_PV[ich][i] = new TH1D(Form("h_PV_Ch%i_S%i_%s",ich,i,option.Data()), "Number of primary vertices", 80, 0, 80);
    h_PV[ich][i]->SetXTitle("Good PV");
    h_PV[ich][i]->Sumw2();
    fOutput->Add(h_PV[ich][i]);
  
    h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()), "Number of jets", 12, 0, 12);
    h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
    h_NJet[ich][i]->Sumw2();
    fOutput->Add(h_NJet[ich][i]);

    h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (medium)", 6, 0, 6);
    h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (DeepCSVM)");
    h_NBJetCSVv2M[ich][i]->Sumw2();
    fOutput->Add(h_NBJetCSVv2M[ich][i]);

    h_NBJetCSVv2T[ich][i] = new TH1D(Form("h_NBJetCSVv2T_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (tight)", 6, 0, 6);
    h_NBJetCSVv2T[ich][i]->SetXTitle("b-tagged Jet Multiplicity (DeepCSVT)");
    h_NBJetCSVv2T[ich][i]->Sumw2();
    fOutput->Add(h_NBJetCSVv2T[ich][i]);

    h_NCJetM[ich][i] = new TH1D(Form("h_NCJetM_Ch%i_S%i_%s",ich,i,option.Data()), "Number of c tagged jets", 6, 0, 6);
    h_NCJetM[ich][i]->SetXTitle("c-tagged Jet Multiplicity (MediumWP)");
    h_NCJetM[ich][i]->Sumw2();
    fOutput->Add(h_NCJetM[ich][i]);  

    h_MET[ich][i] = new TH1D(Form("h_MET_Ch%i_S%i_%s",ich,i,option.Data()), "MET", 30,0,200);
    h_MET[ich][i]->SetXTitle("MET (GeV)");
    h_MET[ich][i]->Sumw2();
    fOutput->Add(h_MET[ich][i]);

    h_LepPt[ich][i] = new TH1D(Form("h_LepPt_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton p_{T}", 30,0,200);
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

    h_WMass[ich][i] = new TH1D(Form("h_WMass_Ch%i_S%i_%s",ich,i,option.Data()), "W Mass", 30 ,0 ,200);
    h_WMass[ich][i]->SetXTitle("Transverse Mass (GeV)");
    h_WMass[ich][i]->Sumw2();
    fOutput->Add(h_WMass[ich][i]);

    h_DPhi[ich][i] = new TH1D(Form("h_DPhi_Ch%i_S%i_%s",ich,i,option.Data()), "Lepton MET #Delta#phi", 30 ,0 ,3.2);
    h_DPhi[ich][i]->SetXTitle("|#Delta#phi_{l,MET}|");
    h_DPhi[ich][i]->Sumw2();
    fOutput->Add(h_DPhi[ich][i]);

    h_LepIso[ich][i] = new TH1D(Form("h_LepIso_Ch%i_S%i_%s",ich,i,option.Data()), "LepIso", 20 ,0 ,0.15);
    h_LepIso[ich][i]->SetXTitle("Relative Isolation");
    h_LepIso[ich][i]->Sumw2();
    fOutput->Add(h_LepIso[ich][i]);
/*
    h_csvv2[ich][i] = new TH1D(Form("h_csvv2_Ch%i_S%i_%s",ich,i,option.Data()), "DeepCSV", 20 ,0 ,1);
    h_csvv2[ich][i]->SetXTitle("DeepCSV");
    h_csvv2[ich][i]->Sumw2();
    fOutput->Add(h_csvv2[ich][i]);

    h_cvsl[ich][i] = new TH1D(Form("h_cvsl_Ch%i_S%i_%s",ich,i,option.Data()), "CvsL", 20 , 0 ,1);
    h_cvsl[ich][i]->SetXTitle("CvsL");
    h_cvsl[ich][i]->Sumw2();
    fOutput->Add(h_cvsl[ich][i]);

    h_cvsb[ich][i] = new TH1D(Form("h_cvsb_Ch%i_S%i_%s",ich,i,option.Data()), "CvsB", 20 , 0 ,1);
    h_cvsb[ich][i]->SetXTitle("CvsB");
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

    h_FCNHkinTopMWb[ich][i] = new TH1D(Form("h_FCNHkinTopMWb_Ch%i_S%i_%s",ich,i,sample.c_str()), "Top Mass (Lep)", 30 , 0, 400);
    h_FCNHkinTopMWb[ich][i]->SetXTitle("Top Mass (Lep) (GeV)");
    h_FCNHkinTopMWb[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinTopMWb[ich][i]);

    h_FCNHkinTopMHc[ich][i] = new TH1D(Form("h_FCNHkinTopMHc_Ch%i_S%i_%s",ich,i,sample.c_str()), "Top Mass from Hc/u", 30, 0, 400);
    h_FCNHkinTopMHc[ich][i]->SetXTitle("Top Mass (Had) (GeV)");
    h_FCNHkinTopMHc[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinTopMHc[ich][i]);

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

    h_FCNHkinHb1CSV[ich][i] = new TH1D(Form("h_FCNHkinHb1CSV_Ch%i_S%i_%s",ich,i,sample.c_str()), "Higgs b_{1} CSVv2", 20, 0.8 ,1);
    h_FCNHkinHb1CSV[ich][i]->SetXTitle("Higgs b_{1} CSVv2 (GeV)");
    h_FCNHkinHb1CSV[ich][i]->Sumw2();
    fOutput->Add(h_FCNHkinHb1CSV[ich][i]);

    h_FCNHkinHb2CSV[ich][i] = new TH1D(Form("h_FCNHkinHb2CSV_Ch%i_S%i_%s",ich,i,sample.c_str()), "Higgs b_{2} CSVv2", 20, 0.8 ,1);
    h_FCNHkinHb2CSV[ich][i]->SetXTitle("Higgs b_{2} CSVv2 (GeV)");
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
    h_genDR[ich][i] = new TH1D(Form("h_genDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen b jets from Higgs", 30, 0, 4);
    h_genDR[ich][i]->SetXTitle("gen #Delta R");
    h_genDR[ich][i]->Sumw2();
    fOutput->Add(h_genDR[ich][i]);

    h_matchDR[ich][i] = new TH1D(Form("h_matchDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R between gen matched b jets from Higgs", 30, 0, 4);
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
    float genweight = 1.0;
    float puweight = 1.0;
    float jetsf = 1.0;
    if( !option.Contains("Data") ){
      lep_SF = lepton_SF[0];
      genweight = *genWeight;
      puweight = PUWeight[0];
      jetsf = jet_SF_deepCSV_30[0];
    }
    float EventWeight = puweight*genweight*lep_SF*jetsf;

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

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
      //if( !option.Contains("Data") ) jet = jet * jet_JER_Nom[iJet];

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        jetIdxs.push_back(iJet);//Goh's kinfit
        if( jet_deepCSV[iJet] > 0.4941){
          nbjets_m++;
        }
        if( jet_deepCSV[iJet] > 0.8001 ){
          nbjets_t++;
        }
        if( jet_CvsL[iJet] > 0.15 && jet_CvsB[iJet] > 0.28 ){//Data=?
          ncjets_m++;
        }
      }
    }

    /////Fill histograms

    int Ncuts = 16;
    bool eventSelection[Ncuts];
    for(unsigned int bcut=0; bcut < Ncuts; bcut++) eventSelection[bcut] = false;

    eventSelection[0] = true;
    eventSelection[1] = ( njets == 3 );
    eventSelection[2] = ( njets == 3 ) && ( nbjets_m == 2 );
    eventSelection[3] = ( njets == 3 ) && ( nbjets_m == 3 );
    eventSelection[4] = ( njets == 3 ) && ( nbjets_m == 4 );
    eventSelection[5] = ( njets == 4 );
    eventSelection[6] = ( njets == 4 ) && ( nbjets_m == 2 );
    eventSelection[7] = ( njets == 4 ) && ( nbjets_m == 3 );
    eventSelection[8] = ( njets == 4 ) && ( nbjets_m == 4 );
    eventSelection[9] = ( njets >= 3 ) && ( nbjets_m >= 2 );
    eventSelection[10] = ( njets >= 3 ) && ( nbjets_m >= 3 );
    eventSelection[11] = ( njets >= 4 ) && ( nbjets_m >= 2 );
    eventSelection[12] = ( njets >= 4 ) && ( nbjets_m >= 3 ); 
    eventSelection[13] = ( njets >= 4 ) && ( nbjets_m >= 4 );
    eventSelection[14] = ( njets >= 6 ) && ( nbjets_m >= 2 );
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

        if(genH.Pt() > 0){
          h_genDR[mode][cut]->Fill(gendR, EventWeight);
          h_genHm[mode][cut]->Fill(genHm, EventWeight);
          if(match1 && match2){
            h_matchDR[mode][cut]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]), EventWeight);
            h_matchHm[mode][cut]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(), EventWeight);
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

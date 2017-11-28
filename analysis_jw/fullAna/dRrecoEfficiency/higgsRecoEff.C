#define higgsRecoEff_cxx

#include "higgsRecoEff.h"
#include <TH2.h>
#include <TStyle.h>

void higgsRecoEff::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void higgsRecoEff::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
   
   cout << "SlaveBegin" << endl;
   for(int ich=0; ich < 2; ich++){
     for(int i=0; i < 2; i++){
     
      h_NJet[ich][i] = new TH1D(Form("h_NJet_Ch%i_S%i_%s",ich,i,option.Data()), "Number of jets", 12, 0, 12);
      h_NJet[ich][i]->SetXTitle("Jet Multiplicity");
      h_NJet[ich][i]->Sumw2();
      fOutput->Add(h_NJet[ich][i]);

      h_NBJetCSVv2M[ich][i] = new TH1D(Form("h_NBJetCSVv2M_Ch%i_S%i_%s",ich,i,option.Data()), "Number of b tagged jets (medium)", 6, 0, 6);
      h_NBJetCSVv2M[ich][i]->SetXTitle("b-tagged Jet Multiplicity (CSVv2M)");
      h_NBJetCSVv2M[ich][i]->Sumw2();
      fOutput->Add(h_NBJetCSVv2M[ich][i]);

      h_DRFCNHkinHMass[ich][i] = new TH1D(Form("h_DRFCNHkinHMass_Ch%i_S%i_%s",ich,i,option.Data()), "H Mass from DRFCNHkinfit (bb)", 100 ,0 ,1000);
      h_DRFCNHkinHMass[ich][i]->SetXTitle("Mass (GeV)");
      h_DRFCNHkinHMass[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinHMass[ich][i]);

      h_DRFCNHkinDR[ich][i] = new TH1D(Form("h_DRFCNHkinDR_Ch%i_S%i_%s",ich,i,option.Data()), "Delta R from DRFCNHkinfit", 40, 0 ,4);
      h_DRFCNHkinDR[ich][i]->SetXTitle("#Delta R");
      h_DRFCNHkinDR[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinDR[ich][i]);

      h_DRFCNHkinTopMHc[ich][i] = new TH1D(Form("h_DRFCNHkinTopMHc_Ch%i_S%i_%s",ich,i,option.Data()), "Top mass from Hc/u DRFCNHkinfit", 35 , 50 , 400);
      h_DRFCNHkinTopMHc[ich][i]->SetXTitle("Mass (GeV)");
      h_DRFCNHkinTopMHc[ich][i]->Sumw2();
      fOutput->Add(h_DRFCNHkinTopMHc[ich][i]);

      //GenMatching
      h_HbjetsDR[ich][i] = new TH1D(Form("h_HbjetsDR_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta R between gen b jets from Higgs", 40 , 0 , 4);
      h_HbjetsDR[ich][i]->SetXTitle("#Delta R");
      h_HbjetsDR[ich][i]->Sumw2();
      fOutput->Add(h_HbjetsDR[ich][i]);

      h_Hpt[ich][i] = new TH1D(Form("h_Hpt_Ch%i_S%i_%s",ich,i,option.Data()), "Gen Higgs pt", 60 , 0 , 300);
      h_Hpt[ich][i]->SetXTitle("pt (GeV)");
      h_Hpt[ich][i]->Sumw2();
      fOutput->Add(h_Hpt[ich][i]);

      h_genHm[ich][i] = new TH1D(Form("h_genHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen Higgs mass", 100 , 0 , 1000);
      h_genHm[ich][i]->SetXTitle("Mass (GeV)");
      h_genHm[ich][i]->Sumw2();
      fOutput->Add(h_genHm[ich][i]);

      h_matchHm[ich][i] = new TH1D(Form("h_matchHm_Ch%i_S%i_%s",ich,i,option.Data()), "Gen matched Higgs mass", 100 , 0 , 1000);
      h_matchHm[ich][i]->SetXTitle("Mass (GeV)");
      h_matchHm[ich][i]->Sumw2();
      fOutput->Add(h_matchHm[ich][i]);

      h_matchDR[ich][i] = new TH1D(Form("h_matchDR_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta R between gen and matched b", 40 , 0 , 4);
      h_matchDR[ich][i]->SetXTitle("matched #Delta R");
      h_matchDR[ich][i]->Sumw2();
      fOutput->Add(h_matchDR[ich][i]);

      h_HptDR[ich][i] = new TH2D(Form("h_HptDR_Ch%i_S%i_%s",ich,i,option.Data()),"Gen #Delta R vs Higgs pt", 60, 0, 300, 40, 0, 4);
      h_HptDR[ich][i]->SetXTitle("pt (GeV)");
      h_HptDR[ich][i]->SetYTitle("Higgs->bb #Delta R");
      h_HptDR[ich][i]->Sumw2();
      fOutput->Add(h_HptDR[ich][i]);

      h_recoHptDR[ich][i] = new TH2D(Form("h_recoHptDR_Ch%i_S%i_%s",ich,i,option.Data()),"reco #Delta R vs reco Higgs pt", 60, 0, 300, 40, 0, 4);
      h_recoHptDR[ich][i]->SetXTitle("pt (GeV)");
      h_recoHptDR[ich][i]->SetYTitle("reco Higgs->bb #Delta R");
      h_recoHptDR[ich][i]->Sumw2();
      fOutput->Add(h_recoHptDR[ich][i]);

      h_dRdiff[ich][i] = new TH1D(Form("h_dRdiff_Ch%i_S%i_%s",ich,i,option.Data()),"#Delta R (reco-gen)", 40, -4, 20);
      h_dRdiff[ich][i]->SetXTitle("#DeltaR reco-gen");
      h_dRdiff[ich][i]->Sumw2();
      fOutput->Add(h_dRdiff[ich][i]);

      h_hbjet1csv[ich][i] = new TH1D(Form("h_hbjet1csv_Ch%i_S%i_%s",ich,i,option.Data()),"hbjet1 CSVv2", 40, 0, 1);
      h_hbjet1csv[ich][i]->SetXTitle("CSVv2 (b1)");
      h_hbjet1csv[ich][i]->Sumw2();
      fOutput->Add(h_hbjet1csv[ich][i]);

      h_hbjet2csv[ich][i] = new TH1D(Form("h_hbjet2csv_Ch%i_S%i_%s",ich,i,option.Data()),"hbjet2 CSVv2", 40, 0, 1);
      h_hbjet2csv[ich][i]->SetXTitle("CSVv2 (b2)");
      h_hbjet2csv[ich][i]->Sumw2();
      fOutput->Add(h_hbjet2csv[ich][i]);

      h_HbjetsDphi[ich][i] = new TH1D(Form("h_HbjetsDphi_Ch%i_S%i_%s",ich,i,option.Data()), "#Delta #phi between gen b jets from Higgs", 40 , 0 , 4);
      h_HbjetsDphi[ich][i]->SetXTitle("#Delta #phi");
      h_HbjetsDphi[ich][i]->Sumw2();
      fOutput->Add(h_HbjetsDphi[ich][i]);

      h_HptDphi[ich][i] = new TH2D(Form("h_HptDphi_Ch%i_S%i_%s",ich,i,option.Data()),"Gen #Delta #phi vs Higgs pt", 60, 0, 300, 40, 0, 4);
      h_HptDphi[ich][i]->SetXTitle("pt (GeV)");
      h_HptDphi[ich][i]->SetYTitle("Higgs->bb #Delta #phi");
      h_HptDphi[ich][i]->Sumw2();
      fOutput->Add(h_HptDphi[ich][i]);

      }
    }
} 

Bool_t higgsRecoEff::Process(Long64_t entry)
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
    float jetsf = jet_SF_CSV_30[0];
    float EventWeight = puweight*genweight*lep_SF*jetsf;

    float relIso = *lepton_relIso; 

    //Object selection
    int njets = 0;
    int nbjets_m = 0; 

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

    TLorentzVector hbjet1, hbjet2, genH;
    if(*addHbjet1_pt > 25 && *addHbjet2_pt > 25 && abs(*addHbjet1_eta) < 2.5 && abs(*addHbjet2_eta) < 2.5){
      hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
      hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

      genH = hbjet1 + hbjet2;
    }
    else return kTRUE;

    bool match1 = false;
    bool match2 = false;

    //for Goh's Kin fit
    vector<size_t> jetIdxs;
    int matchbIdx[2] = {-1};
    int b_kin_bjetcode;

    //Selection Option
    bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
    bool makeIso = true;
    bool isIso = *lepton_isIso; 
    if( makeIso && !isIso ) return kTRUE;
    if( !makeIso && isIso ) return kTRUE;

    //Event selection 
    bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
    bool passelectron = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);

  if( passmuon || passelectron ){

    for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

      TLorentzVector jet;
      jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);

      if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
        njets++;
        jetIdxs.push_back(iJet);//Goh's kinfit
        if( jet_CSV[iJet] > 0.8484 ) nbjets_m++;
 
       if( jet.DeltaR(hbjet1) < 0.4 ) matchbIdx[0] = iJet;
       if( jet.DeltaR(hbjet2) < 0.4 ) matchbIdx[1] = iJet;
      }
    } 

    //DR kin
    std::vector<size_t> bestIdxsDR;
    TLorentzVector jetP4sDR[4];
    size_t lepidx = 0;

    if ( njets >= 3 ){
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
          //if ( dR < ( -0.01*(jetP4sDR[1]+jetP4sDR[2]).Pt() + 2.4) ||  dR > ( -0.01*(jetP4sDR[1]+jetP4sDR[2]).Pt() + 3.5) ) continue;
          double tmpPt = (jetP4sDR[1]+jetP4sDR[2]).Pt();
          if (dR < (2.87463 - 0.0198239 * tmpPt + 0.0000435288 * tmpPt * tmpPt) || dR > (3.68692 - 0.0144628 * tmpPt + 0.0000177433 * tmpPt * tmpPt)) continue;

          if ( dR < minDR ) {
            //bestIdxsDR = { size_t(njets), *ii1, *ii2, size_t(njets)};
            bestIdxsDR = { lepidx, *ii1, *ii2, size_t(njets)};
            minDR = dR;
          }
        }
      }
      if ( !bestIdxsDR.empty() ) {
        const auto i1 = bestIdxsDR[1], i2 = bestIdxsDR[2];
        jetP4sDR[1].SetPtEtaPhiE(jet_pT[i1], jet_eta[i1], jet_phi[i1], jet_E[i1]);
        jetP4sDR[2].SetPtEtaPhiE(jet_pT[i2], jet_eta[i2], jet_phi[i2], jet_E[i2]);
        const auto wP4 = jetP4sDR[1]+jetP4sDR[2];
        //double minDR2 = 1e9;
        double minmassdiff = 1e9;
        for ( auto i3 : jetIdxs ) {
          if ( i3 == i1 or i3 == i2 ) continue;
          if ( jet_CSV[i3] > 0.8484 ) continue;//
          jetP4sDR[3].SetPtEtaPhiE(jet_pT[i3], jet_eta[i3], jet_phi[i3], jet_E[i3]);

/*
          const double dR = jetP4sDR[3].DeltaR(wP4);
          if ( dR < minDR2 ) {
            bestIdxsDR[3] = i3;
            minDR2 = dR;
          }
        }
        if ( minDR2 == 1e9 ) bestIdxsDR.clear();
*/
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

/*
      for ( auto i : jetIdxs ) {
        if ( i == bestIdxsDR[1] or i == bestIdxsDR[2] or i == bestIdxsDR[3] ) continue;
        if ( bestIdxsDR[0] == size_t(njets) or jet_pT[bestIdxsDR[0]] < jet_pT[i] ) {
          bestIdxsDR[0] = i;
        }
      }
*/
      for ( size_t i=0; i<4; ++i ) {
        const size_t j = bestIdxsDR[i];
        jetP4sDR[i].SetPtEtaPhiE(jet_pT[j], jet_eta[j], jet_phi[j], jet_E[j]);
      }
    }

    /////Fill histograms

    h_NJet[mode][0]->Fill(njets, EventWeight);
    h_NBJetCSVv2M[mode][0]->Fill(nbjets_m, EventWeight);

    h_DRFCNHkinHMass[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
    h_DRFCNHkinDR[mode][0]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
    h_DRFCNHkinTopMHc[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);

    if(hbjet1.DeltaR(hbjet2) > 0){
      h_HbjetsDR[mode][0]->Fill(hbjet1.DeltaR(hbjet2),EventWeight);
      h_Hpt[mode][0]->Fill(genH.Pt(),EventWeight);
      h_HptDR[mode][0]->Fill(genH.Pt(),hbjet1.DeltaR(hbjet2),EventWeight);
      h_genHm[mode][0]->Fill(genH.M(),EventWeight);
    }

    if(match2) h_matchHm[mode][0]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);

    if( njets >= 4) {
      //STEP10
      if( nbjets_m >= 3 ){

        double gendR = -10;
        double recodR = 10;
        double matchdR = 10;

        //Gen vs reco higgs->bjet matching
        if(hbjet1.DeltaR(jetP4sDR[1]) < 0.4 or hbjet1.DeltaR(jetP4sDR[2]) < 0.4) match1 = true;
        if(hbjet2.DeltaR(jetP4sDR[1]) < 0.4 or hbjet2.DeltaR(jetP4sDR[2]) < 0.4) match2 = true;

        h_NJet[mode][1]->Fill(njets, EventWeight);
        h_NBJetCSVv2M[mode][1]->Fill(nbjets_m, EventWeight);

        h_DRFCNHkinHMass[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
        h_DRFCNHkinDR[mode][1]->Fill(jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);
        h_DRFCNHkinTopMHc[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]+jetP4sDR[3]).M(),EventWeight);
        recodR = jetP4sDR[1].DeltaR(jetP4sDR[2]);

        h_HbjetsDR[mode][1]->Fill(hbjet1.DeltaR(hbjet2),EventWeight);
        h_Hpt[mode][1]->Fill(genH.Pt(),EventWeight);
        h_HptDR[mode][1]->Fill(genH.Pt(),hbjet1.DeltaR(hbjet2),EventWeight);
        h_genHm[mode][1]->Fill(genH.M(),EventWeight);
        gendR = hbjet1.DeltaR(hbjet2);

        h_HbjetsDphi[mode][1]->Fill(hbjet1.DeltaPhi(hbjet2), EventWeight);
        h_HptDphi[mode][1]->Fill(genH.Pt(),hbjet1.DeltaPhi(hbjet2),EventWeight);

        h_recoHptDR[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]).Pt(),jetP4sDR[1].DeltaR(jetP4sDR[2]),EventWeight);

        if(match1 && match2){
          h_matchHm[mode][1]->Fill((jetP4sDR[1]+jetP4sDR[2]).M(),EventWeight);
          matchdR = jetP4sDR[1].DeltaR(jetP4sDR[2]);
          h_matchDR[mode][1]->Fill(matchdR, EventWeight);
        }

        h_dRdiff[mode][1]->Fill(recodR-gendR, EventWeight);
        if(matchbIdx[0] != -1) h_hbjet1csv[mode][1]->Fill(jet_CSV[matchbIdx[0]], EventWeight);
        if(matchbIdx[1] != -1) h_hbjet2csv[mode][1]->Fill(jet_CSV[matchbIdx[1]], EventWeight);
      }
    }
  }
  return kTRUE;
}

void higgsRecoEff::SlaveTerminate()
{
  TString option = GetOption();
}
   

void higgsRecoEff::Terminate()
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

double higgsRecoEff::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

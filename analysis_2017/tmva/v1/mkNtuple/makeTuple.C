#define makeTuple_cxx

#include "makeTuple.h"
#include <TH2.h>
#include <TStyle.h>

void makeTuple::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void makeTuple::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();

  treeTMVA = new TTree("tmva_tree","tree for tmva");

  treeTMVA->Branch("nevt"         , &b_nevt         , "nevt/I");
  treeTMVA->Branch("GoodPV"       , &b_GoodPV       , "GoodPV/I");
  treeTMVA->Branch("EventCategory", &b_EventCategory, "EventCategory/I");
  treeTMVA->Branch("GenMatch"     , &b_GenMatch     , "GenMatch/I");
  treeTMVA->Branch("EventWeight"  , &b_EventWeight  , "EventWeight/F");

  treeTMVA->Branch("njets"         , &b_njets      , "njets/I");
  treeTMVA->Branch("nbjets_m"      , &b_nbjets_m   , "nbjets_m/I");
  treeTMVA->Branch("ncjets_m"      , &b_ncjets_m   , "ncjets_m/I");
  treeTMVA->Branch("transverseMass", &b_transversem, "transversem/F");
  treeTMVA->Branch("lepDPhi"       , &b_lepdphi    , "lepdphi/F");
  treeTMVA->Branch("missingET"     , &b_met        , "met/F");
  treeTMVA->Branch("cjetPt"        , &b_cjmpt      , "cjmpt/F");

  //jets
  treeTMVA->Branch("lepWpt"  , &b_lepWpt  , "lepWpt/F");
  treeTMVA->Branch("lepWeta" , &b_lepWeta , "lepWeta/F");
  treeTMVA->Branch("lepWdeta", &b_lepWdeta, "lepWdeta/F");
  treeTMVA->Branch("lepWphi" , &b_lepWphi , "lepWphi/F");
  treeTMVA->Branch("lepWdphi", &b_lepWdphi, "lepWdphi/F");
  treeTMVA->Branch("lepWm"   , &b_lepWm   , "lepWm/F");

  treeTMVA->Branch("jet0pt"  , &b_jet0pt  , "jet0pt/F");
  treeTMVA->Branch("jet0eta" , &b_jet0eta , "jet0eta/F");
  treeTMVA->Branch("jet0phi" , &b_jet0phi , "jet0phi/F");
  treeTMVA->Branch("jet0m"   , &b_jet0m   , "jet0m/F");
  treeTMVA->Branch("jet0csv" , &b_jet0csv , "jet0csv/F");
  treeTMVA->Branch("jet0cvsl", &b_jet0cvsl, "jet0cvsl/F");
  treeTMVA->Branch("jet0cvsb", &b_jet0cvsb, "jet0cvsb/F");

  treeTMVA->Branch("jet1pt"  , &b_jet1pt  , "jet1pt/F");
  treeTMVA->Branch("jet1eta" , &b_jet1eta , "jet1eta/F");
  treeTMVA->Branch("jet1phi" , &b_jet1phi , "jet1phi/F");
  treeTMVA->Branch("jet1m"   , &b_jet1m   , "jet1m/F");
  treeTMVA->Branch("jet1csv" , &b_jet1csv , "jet1csv/F");
  treeTMVA->Branch("jet1cvsl", &b_jet1cvsl, "jet1cvsl/F");
  treeTMVA->Branch("jet1cvsb", &b_jet1cvsb, "jet1cvsb/F");

  treeTMVA->Branch("jet2pt"  , &b_jet2pt  , "jet2pt/F");
  treeTMVA->Branch("jet2eta" , &b_jet2eta , "jet2eta/F");
  treeTMVA->Branch("jet2phi" , &b_jet2phi , "jet2phi/F");
  treeTMVA->Branch("jet2m"   , &b_jet2m   , "jet2m/F");
  treeTMVA->Branch("jet2csv" , &b_jet2csv , "jet2csv/F");
  treeTMVA->Branch("jet2cvsl", &b_jet2cvsl, "jet2cvsl/F");
  treeTMVA->Branch("jet2cvsb", &b_jet2cvsb, "jet2cvsb/F");

  treeTMVA->Branch("jet3pt"  , &b_jet3pt  , "jet3pt/F");
  treeTMVA->Branch("jet3eta" , &b_jet3eta , "jet3eta/F");
  treeTMVA->Branch("jet3phi" , &b_jet3phi , "jet3phi/F");
  treeTMVA->Branch("jet3m"   , &b_jet3m   , "jet3m/F");
  treeTMVA->Branch("jet3csv" , &b_jet3csv , "jet3csv/F");
  treeTMVA->Branch("jet3cvsl", &b_jet3cvsl, "jet3cvsl/F");
  treeTMVA->Branch("jet3cvsb", &b_jet3cvsb, "jet3cvsb/F");

  treeTMVA->Branch("jet12pt"  , &b_jet12pt  , "jet12pt/F");
  treeTMVA->Branch("jet12eta" , &b_jet12eta , "jet12eta/F");
  treeTMVA->Branch("jet12deta", &b_jet12deta, "jet12deta/F");
  treeTMVA->Branch("jet12phi" , &b_jet12phi , "jet12phi/F");//mass->higgs
  treeTMVA->Branch("jet12dphi", &b_jet12dphi, "jet12dphi/F");
  treeTMVA->Branch("jet12m"   , &b_jet12m   , "jet12m/F");//higgs candidate
  treeTMVA->Branch("jet12DR"  , &b_jet12DR  , "jet12DR/F");

  treeTMVA->Branch("jet23pt"  , &b_jet23pt  , "jet23pt/F");
  treeTMVA->Branch("jet23eta" , &b_jet23eta , "jet23eta/F");
  treeTMVA->Branch("jet23deta", &b_jet23deta, "jet23deta/F");
  treeTMVA->Branch("jet23phi" , &b_jet23phi , "jet23phi/F");
  treeTMVA->Branch("jet23dphi", &b_jet23dphi, "jet23dphi/F");
  treeTMVA->Branch("jet23m"   , &b_jet23m   , "jet23m/F");

  treeTMVA->Branch("jet31pt"  , &b_jet31pt  , "jet31pt/F");
  treeTMVA->Branch("jet31eta" , &b_jet31eta , "jet31eta/F");
  treeTMVA->Branch("jet31deta", &b_jet31deta, "jet31deta/F");
  treeTMVA->Branch("jet31phi" , &b_jet31phi , "jet31phi/F");
  treeTMVA->Branch("jet31dphi", &b_jet31dphi, "jet31dphi/F");
  treeTMVA->Branch("jet31m"   , &b_jet31m   , "jet31m/F");

  treeTMVA->Branch("lepTpt"  , &b_lepTpt  , "lepTpt/F");
  treeTMVA->Branch("lepTeta" , &b_lepTeta , "lepTeta/F");
  treeTMVA->Branch("lepTdeta", &b_lepTdeta, "lepTdeta/F");
  treeTMVA->Branch("lepTphi" , &b_lepTphi , "lepTphi/F");
  treeTMVA->Branch("lepTdphi", &b_lepTdphi, "lepTdphi/F");//W and b
  treeTMVA->Branch("lepTm"   , &b_lepTm   , "lepTm/F");

  treeTMVA->Branch("hadTpt"    , &b_hadTpt    , "hadTpt/F");
  treeTMVA->Branch("hadTeta"   , &b_hadTeta   , "hadTeta/F");
  treeTMVA->Branch("hadTHbdeta", &b_hadTHbdeta, "hadTHbdeta/F");
  treeTMVA->Branch("hadTWbdeta", &b_hadTWbdeta, "hadTWbdeta/F");
  treeTMVA->Branch("hadTphi"   , &b_hadTphi   , "hadTphi/F");
  treeTMVA->Branch("hadTHbdphi", &b_hadTHbdphi, "hadTHbdphi/F");//H and b
  treeTMVA->Branch("hadTWbdphi", &b_hadTWbdphi, "hadTWbdphi/F");//W and b
  treeTMVA->Branch("hadTm"     , &b_hadTm     , "hadTm/F");

  fOutput->Add(treeTMVA);

  assignF = TFile::Open(Form("/home/minerva1993/fcnc/analysis_2017/reco/classifier/2017/assignST01/assign_deepReco_%s.root", option.Data()), "READ");
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

Bool_t makeTuple::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();

  int mode = 999; 
  mode = *channel;

  if( mode > 2) return kTRUE; 
  if( !option.Contains("Data") && *TruePV == 0 ) return kTRUE;

  float lep_SF = 1.0;
  float genweight = 1.0;
  float puweight = 1.0;
  float jetsf = 1.0;
  float wrongPVrate = 1.0;
  if( !option.Contains("Data") ){
    lep_SF = lepton_SF[0];
    genweight = *genWeight;
    puweight = PUWeight[0];
    jetsf = jet_SF_deepCSV_30[0];
    wrongPVrate = 1.0293;
  }
  float EventWeight = puweight*genweight*lep_SF*jetsf;

  float relIso = *lepton_relIso; 

  //Object selection
  int njets = 0;
  int nbjets_m = 0; 
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

  TLorentzVector hbjet1, hbjet2, genH;
  if(*addHbjet1_pt > 20 && *addHbjet2_pt > 20 && abs(*addHbjet1_eta) < 2.4 && abs(*addHbjet2_eta) < 2.4){
    hbjet1.SetPtEtaPhiE(*addHbjet1_pt, *addHbjet1_eta, *addHbjet1_phi, *addHbjet1_e);
    hbjet2.SetPtEtaPhiE(*addHbjet2_pt, *addHbjet2_eta, *addHbjet2_phi, *addHbjet2_e);

    genH = hbjet1 + hbjet2;
  }

  //for Goh's Kin fit
  bool match1 = false;
  bool match2 = false;

  //Selection Option
  bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
  bool makeIso = true;
  bool isIso = *lepton_isIso; 
  if( makeIso && !isIso ) return kTRUE;
  if( !makeIso && isIso ) return kTRUE;

  //Event selection 
  bool passmuon = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
  bool passelectron = (mode == 1) && (lepton.Pt() > 38) && (abs(lepton.Eta()) <= 2.1);

  if( option.Contains("DataSingleMu") ){
    if( !passmuon ) return kTRUE;//RDMu
    if( passelectron) return kTRUE;//RDMu
  }
  else if( option.Contains("DataSingleEG") ){
    if( !passelectron ) return kTRUE;//RDelec
    if( passmuon ) return kTRUE;//RDelec
  }
  else{
    if( !passmuon && !passelectron ) return kTRUE;
  }

  vector<float> v_cjet_m;
  int jetIdx[4];
  TLorentzVector jetP4s[4];

  for (unsigned int iJet = 0; iJet < jet_pT.GetSize() ; ++iJet) {

    TLorentzVector jet;
    jet.SetPtEtaPhiE(jet_pT[iJet], jet_eta[iJet], jet_phi[iJet], jet_E[iJet]);
    //if( !option.Contains("Data") ) jet = jet * jet_JER_Nom[iJet];

    if( jet.Pt() > 30 && abs(jet.Eta())<=2.4){
      njets++;

      if( jet_deepCSV[iJet] > 0.4941 ){
        nbjets_m++;
      }
  
      if( jet_deepCvsL[iJet] > 0.15 && jet_CvsB[iJet] > 0.28 ){
        ncjets_m++;
        v_cjet_m.push_back(jet.Pt());
      }
    }
  } 

  //////////////////////
  //EVENT SELECTION!!!!!
  //////////////////////
  if( njets <  3 || nbjets_m < 2) return kTRUE; 
  
  //cout << nevt << endl;
  if( option.Contains("Data") ) b_EventCategory = -1;
  else if( option.Contains("Hct") || option.Contains("Hut") ) b_EventCategory = 0;
  else if( option.Contains("ttbb") ) b_EventCategory = 1;
  else if( option.Contains("ttbj") ) b_EventCategory = 2;
  else if( option.Contains("ttcc") ) b_EventCategory = 3;
  else if( option.Contains("ttLF") || option.Contains("ttother") ) b_EventCategory = 4;
  else if( option.Contains("channel") ) b_EventCategory = 5; //singletop
  else if( option.Contains("zjets") ) b_EventCategory = 6; //DY
  else b_EventCategory = 7;

  b_EventWeight = EventWeight;
  b_GoodPV = *GoodPV;
  b_nevt = nevt;

  b_njets = njets;
  b_nbjets_m = nbjets_m;
  b_ncjets_m = ncjets_m;

  b_met = met;
  b_lepdphi = lepDphi;
  b_transversem = transverseMass(lepton, p4met);

  if( ncjets_m != 0 ) b_cjmpt  = *max_element(v_cjet_m.begin(), v_cjet_m.end());

  //Jet Assignment
  /////////////////
  //NJET!!!!
  ////////////////
  vector<double>::iterator iter;
  int evtIdx = 0;
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
    if( !option.Contains("Data") ){
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

  if(match1 && match2){
    if(b_EventCategory == 0) b_GenMatch = 2;
    else if(b_EventCategory != 0 && b_EventCategory > -1 ) b_GenMatch = 1;   
  }
  else b_GenMatch = -1;

  //fill trees
  b_lepWpt = (lepton+p4met).Pt();
  b_lepWeta = (lepton+p4met).Eta();
  b_lepWdeta = (lepton-p4met).Eta();
  b_lepWphi = (lepton+p4met).Phi();
  b_lepWdphi = lepton.DeltaPhi(p4met);
  b_lepWm = (lepton+p4met).M();

  b_jet0pt = jetP4s[0].Pt();
  b_jet0eta = jetP4s[0].Eta();
  b_jet0phi = jetP4s[0].Phi();
  b_jet0m = jetP4s[0].M();
  b_jet0csv = jet_deepCSV[jetIdx[0]];
  b_jet0cvsl = jet_CvsL[jetIdx[0]];
  b_jet0cvsb = jet_CvsB[jetIdx[0]];

  b_jet1pt = jetP4s[1].Pt();
  b_jet1eta = jetP4s[1].Eta();
  b_jet1phi = jetP4s[1].Phi();
  b_jet1m = jetP4s[1].M();
  b_jet1csv = jet_deepCSV[jetIdx[1]];
  b_jet1cvsl = jet_CvsL[jetIdx[1]];
  b_jet1cvsb = jet_CvsB[jetIdx[1]];

  b_jet2pt = jetP4s[2].Pt();
  b_jet2eta = jetP4s[2].Eta();
  b_jet2phi = jetP4s[2].Phi();
  b_jet2m = jetP4s[2].M();
  b_jet2csv = jet_deepCSV[jetIdx[2]];
  b_jet2cvsl = jet_CvsL[jetIdx[2]];
  b_jet2cvsb = jet_CvsB[jetIdx[2]];

  b_jet3pt = jetP4s[3].Pt();
  b_jet3eta = jetP4s[3].Eta();
  b_jet3phi = jetP4s[3].Phi();
  b_jet3m = jetP4s[3].M();
  b_jet3csv = jet_deepCSV[jetIdx[3]];
  b_jet3cvsl = jet_CvsL[jetIdx[3]];
  b_jet3cvsb = jet_CvsB[jetIdx[3]];

  b_jet12pt = (jetP4s[1]+jetP4s[2]).Pt();
  b_jet12eta = (jetP4s[1]+jetP4s[2]).Eta();
  b_jet12deta = (jetP4s[1]-jetP4s[2]).Eta();
  b_jet12phi = (jetP4s[1]+jetP4s[2]).Phi();
  b_jet12dphi = jetP4s[1].DeltaPhi(jetP4s[2]);
  b_jet12m = (jetP4s[1]+jetP4s[2]).M();
  b_jet12DR = jetP4s[1].DeltaR(jetP4s[2]);

  b_jet23pt = (jetP4s[2]+jetP4s[3]).Pt();
  b_jet23eta = (jetP4s[2]+jetP4s[3]).Eta();
  b_jet23deta = (jetP4s[2]-jetP4s[3]).Eta();
  b_jet23phi = (jetP4s[2]+jetP4s[3]).Phi();
  b_jet23dphi = jetP4s[2].DeltaPhi(jetP4s[3]);
  b_jet23m = (jetP4s[2]+jetP4s[3]).M();

  b_jet31pt = (jetP4s[1]+jetP4s[3]).Pt();
  b_jet31eta = (jetP4s[1]+jetP4s[3]).Eta();
  b_jet31deta = (jetP4s[1]-jetP4s[3]).Eta();
  b_jet31phi = (jetP4s[1]+jetP4s[3]).Phi();
  b_jet31dphi = jetP4s[1].DeltaPhi(jetP4s[3]);
  b_jet31m = (jetP4s[1]+jetP4s[3]).M();

  b_lepTpt = (lepton+p4met+jetP4s[0]).Pt();
  b_lepTeta = (lepton+p4met+jetP4s[0]).Eta();
  b_lepTdeta = (lepton+p4met-jetP4s[0]).Eta();
  b_lepTphi = (lepton+p4met+jetP4s[0]).Phi();
  b_lepTdphi = (lepton+p4met).DeltaPhi(jetP4s[0]);
  b_lepTm = (lepton+p4met+jetP4s[0]).M();

  b_hadTpt = (jetP4s[1]+jetP4s[2]+jetP4s[3]).Pt();
  b_hadTeta = (jetP4s[1]+jetP4s[2]+jetP4s[3]).Eta();
  b_hadTHbdeta = (jetP4s[1]+jetP4s[2]-jetP4s[3]).Eta();
  b_hadTWbdeta = (-jetP4s[1]+jetP4s[2]+jetP4s[3]).Eta();
  b_hadTphi = (jetP4s[1]+jetP4s[2]+jetP4s[3]).Phi();
  b_hadTHbdphi = (jetP4s[1]+jetP4s[2]).DeltaPhi(jetP4s[3]);
  b_hadTWbdphi = (jetP4s[2]+jetP4s[3]).DeltaPhi(jetP4s[1]);
  b_hadTm = (jetP4s[1]+jetP4s[2]+jetP4s[3]).M();

  treeTMVA->Fill();

  nevt++;

  return kTRUE;
}

void makeTuple::SlaveTerminate()
{
  TString option = GetOption();
}
   

void makeTuple::Terminate()
{
  TString option = GetOption();

  TFile *hfile = TFile::Open(Form("/home/minerva1993/fcnc/analysis_2017/tmva/v1/mkNtuple/temp/tmva_%s.root",option.Data()),"RECREATE");

  fOutput->FindObject("tmva_tree")->Write();

  hfile->Write();
  hfile->Close();

}

double makeTuple::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

#define selection_cxx

#include "selection.h"
#include <TH2.h>
#include <TStyle.h>

void selection::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void selection::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
   
   cout << "SlaveBegin" << endl;
   for(int ich=0; ich < 2; ich++){
     for(int i=0; i < 4; i++){

      h_muPt[ich][i] = new TH1D(Form("h_muPt_Ch%i_S%i_%s",ich,i,option.Data()), "Muon p_{T}", 100, 0, 200);
      h_muPt[ich][i]->SetXTitle("Muon p_{T} (GeV)");
      h_muPt[ich][i]->Sumw2();
      fOutput->Add(h_muPt[ich][i]);

      h_elPt[ich][i] = new TH1D(Form("h_elPt_Ch%i_S%i_%s",ich,i,option.Data()), "Electron p_{T}", 100, 0, 200);
      h_elPt[ich][i]->SetXTitle("Eletron p_{T} (GeV)");
      h_elPt[ich][i]->Sumw2();
      fOutput->Add(h_elPt[ich][i]);

      }
    }
} 

Bool_t selection::Process(Long64_t entry)
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

    //Selection Option
    bool isQCD = transverseM < 10 && met < 10 && lepDphi < 1;
    bool makeIso = true;
    bool isIso = *lepton_isIso; 
    if( makeIso && !isIso ) return kTRUE;
    if( !makeIso && isIso ) return kTRUE;

    //Event selection 
    bool muS0 = (mode == 0) && (lepton.Pt() > 24) && (abs(lepton.Eta()) <= 2.1);//Mu24
    bool muS1 = (mode == 0) && (lepton.Pt() > 27) && (abs(lepton.Eta()) <= 2.1);
    bool muS2 = (mode == 0) && (lepton.Pt() > 30) && (abs(lepton.Eta()) <= 2.1);
    bool elS0 = (mode == 1) && (lepton.Pt() > 32) && (abs(lepton.Eta()) <= 2.1);//El32
    bool elS1 = (mode == 1) && (lepton.Pt() > 35) && (abs(lepton.Eta()) <= 2.1);
    bool elS2 = (mode == 1) && (lepton.Pt() > 38) && (abs(lepton.Eta()) <= 2.1);

  if( muS0 ) h_muPt[mode][0]->Fill(lepton.Pt(),EventWeight);
  if( muS1 ) h_muPt[mode][1]->Fill(lepton.Pt(),EventWeight);
  if( muS2 ) h_muPt[mode][2]->Fill(lepton.Pt(),EventWeight);
  if( elS0 ) h_elPt[mode][0]->Fill(lepton.Pt(),EventWeight);
  if( elS1 ) h_elPt[mode][1]->Fill(lepton.Pt(),EventWeight);
  if( elS2 ) h_elPt[mode][2]->Fill(lepton.Pt(),EventWeight);

  return kTRUE;
}

void selection::SlaveTerminate()
{
  TString option = GetOption();
}
   

void selection::Terminate()
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

double selection::transverseMass( const TLorentzVector & lepton, const TLorentzVector & met){

  TLorentzVector leptonT(lepton.Px(),lepton.Py(),0.,lepton.E()*TMath::Sin(lepton.Theta()));
  TLorentzVector metT(met.Px(), met.Py(), 0, met.E());

  TLorentzVector sumT=leptonT+metT;
  double out = TMath::Sqrt( sumT.M2() );

  return out;

}

//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jun 13 13:36:07 2017 by ROOT version 6.06/01
// from TTree events/Tree for Top quark study
// found on file: /xrootd/store/user/tjkim/ntuples/hep2017/V1/vallot_DYJets_10to50.root
//////////////////////////////////////////////////////////

#ifndef FCNCAnalysis_h
#define FCNCAnalysis_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TH1D.h>
#include <vector>
#include <TLorentzVector.h>
#include <TH2D.h>
// Headers needed by this particular selector


class FCNCAnalysis : public TSelector {
public :
   TTreeReader     fReader;  //!the tree reader
   TTree          *fChain = 0;   //!pointer to the analyzed TTree or TChain

   // Readers to access the data (delete the ones you do not need).
   //TTreeReaderValue<Int_t> EVENT = {fReader, "EVENT"};
   //TTreeReaderValue<Int_t> RUN = {fReader, "RUN"};
   //TTreeReaderValue<Int_t> LUMI = {fReader, "LUMI"};
   TTreeReaderValue<Float_t> PUWeight = {fReader, "PUWeight"};
   TTreeReaderValue<Float_t> GenWeight = {fReader, "GenWeight"};
   TTreeReaderArray<Float_t> CSVWeight_ = {fReader, "CSVWeight"};
   TTreeReaderValue<Int_t> NVertex = {fReader, "NVertex"};
   TTreeReaderValue<Float_t> MET = {fReader, "MET"};
   TTreeReaderValue<Float_t> MET_Px = {fReader, "MET_Px"};
   TTreeReaderValue<Float_t> MET_Py = {fReader, "MET_Py"};
   TTreeReaderValue<Int_t> NMuon = {fReader, "NMuon"};
   TTreeReaderValue<Float_t> Muon_Pt = {fReader, "Muon_Pt"}; //arry->value
   TTreeReaderValue<Float_t> Muon_Eta = {fReader, "Muon_Eta"};//arry->value
   TTreeReaderValue<Float_t> Muon_Phi = {fReader, "Muon_Phi"};//arry->value
   TTreeReaderValue<Float_t> Muon_E = {fReader, "Muon_E"};//arry->value
   TTreeReaderArray<Float_t> Muon_Iso03 = {fReader, "Muon_Iso03"};
   TTreeReaderArray<Float_t> Muon_Iso04 = {fReader, "Muon_Iso04"};
   TTreeReaderArray<Float_t> Muon_Charge = {fReader, "Muon_Charge"};
   TTreeReaderValue<Int_t> NLooseMuon = {fReader, "NLooseMuon"};
   TTreeReaderArray<Float_t> LooseMuon_Pt = {fReader, "LooseMuon_Pt"};
   TTreeReaderArray<Float_t> LooseMuon_Eta = {fReader, "LooseMuon_Eta"};
   TTreeReaderArray<Float_t> LooseMuon_Phi = {fReader, "LooseMuon_Phi"};
   TTreeReaderArray<Float_t> LooseMuon_E = {fReader, "LooseMuon_E"};
   TTreeReaderArray<Float_t> LooseMuon_Iso03 = {fReader, "LooseMuon_Iso03"};
   TTreeReaderArray<Float_t> LooseMuon_Iso04 = {fReader, "LooseMuon_Iso04"};
   TTreeReaderArray<Float_t> LooseMuon_Charge = {fReader, "LooseMuon_Charge"};
   TTreeReaderValue<Int_t> NElectron = {fReader, "NElectron"};
   TTreeReaderArray<Float_t> Electron_Pt = {fReader, "Electron_Pt"};
   TTreeReaderArray<Float_t> Electron_Eta = {fReader, "Electron_Eta"};
   TTreeReaderArray<Float_t> Electron_Phi = {fReader, "Electron_Phi"};
   TTreeReaderArray<Float_t> Electron_E = {fReader, "Electron_E"};
   TTreeReaderArray<Float_t> Electron_Iso03 = {fReader, "Electron_Iso03"};
   TTreeReaderArray<Float_t> Electron_Iso04 = {fReader, "Electron_Iso04"};
   TTreeReaderArray<Float_t> Electron_Charge = {fReader, "Electron_Charge"};
   TTreeReaderValue<Int_t> NLooseElectron = {fReader, "NLooseElectron"};
   TTreeReaderArray<Float_t> LooseElectron_Pt = {fReader, "LooseElectron_Pt"};
   TTreeReaderArray<Float_t> LooseElectron_Eta = {fReader, "LooseElectron_Eta"};
   TTreeReaderArray<Float_t> LooseElectron_Phi = {fReader, "LooseElectron_Phi"};
   TTreeReaderArray<Float_t> LooseElectron_E = {fReader, "LooseElectron_E"};
   TTreeReaderArray<Float_t> LooseElectron_Iso03 = {fReader, "LooseElectron_Iso03"};
   TTreeReaderArray<Float_t> LooseElectron_Iso04 = {fReader, "LooseElectron_Iso04"};
   TTreeReaderArray<Float_t> LooseElectron_Charge = {fReader, "LooseElectron_Charge"};
   TTreeReaderValue<Int_t> NJet = {fReader, "NJet"};
   TTreeReaderArray<Float_t> Jet_Pt = {fReader, "Jet_Pt"};
   TTreeReaderArray<Float_t> Jet_Eta = {fReader, "Jet_Eta"};
   TTreeReaderArray<Float_t> Jet_Phi = {fReader, "Jet_Phi"};
   TTreeReaderArray<Float_t> Jet_E = {fReader, "Jet_E"};
   TTreeReaderArray<Float_t> Jet_partonFlavour = {fReader, "Jet_partonFlavour"};
   TTreeReaderArray<Float_t> Jet_hadronFlavour = {fReader, "Jet_hadronFlavour"};
   TTreeReaderArray<Float_t> Jet_BTag = {fReader, "Jet_BTag"};
   TTreeReaderArray<Float_t> Jet_bDiscriminator = {fReader, "Jet_bDiscriminator"};
   TTreeReaderArray<Float_t> Jet_pfCombinedCvsLJetTags = {fReader, "Jet_pfCombinedCvsLJetTags"};
   TTreeReaderArray<Float_t> Jet_pfCombinedCvsBJetTags = {fReader, "Jet_pfCombinedCvsBJetTags"};
   TTreeReaderArray<Float_t> Jet_JES_Up = {fReader, "Jet_JES_Up"};
   TTreeReaderArray<Float_t> Jet_JES_Dw = {fReader, "Jet_JES_Dw"};
   TTreeReaderArray<Int_t> csvid = {fReader, "csvid"};
   TTreeReaderValue<Int_t> NBJet = {fReader, "NBJet"};
   TTreeReaderValue<Int_t> DiLeptonic = {fReader, "DiLeptonic"};
   TTreeReaderValue<Int_t> SemiLeptonic = {fReader, "SemiLeptonic"};
   TTreeReaderValue<Int_t> TTBJ = {fReader, "TTBJ"};
   TTreeReaderValue<Int_t> TTBB = {fReader, "TTBB"};
   TTreeReaderValue<Int_t> TTCC = {fReader, "TTCC"};
   TTreeReaderValue<Int_t> TTJJ = {fReader, "TTJJ"};
   TTreeReaderValue<Int_t> GenNJet20 = {fReader, "GenNJet20"};
   TTreeReaderValue<Int_t> GenNBJet20 = {fReader, "GenNBJet20"};
   TTreeReaderValue<Int_t> GenNCJet20 = {fReader, "GenNCJet20"};
   TTreeReaderValue<Int_t> GenNAddJet20 = {fReader, "GenNAddJet20"};
   TTreeReaderValue<Int_t> GenNAddBJet20 = {fReader, "GenNAddBJet20"};
   TTreeReaderValue<Int_t> GenNAddCJet20 = {fReader, "GenNAddCJet20"};
   TTreeReaderValue<Float_t> GenLepton1_Pt = {fReader, "GenLepton1_Pt"};
   TTreeReaderValue<Float_t> GenLepton1_Eta = {fReader, "GenLepton1_Eta"};
   TTreeReaderValue<Float_t> GenLepton2_Pt = {fReader, "GenLepton2_Pt"};
   TTreeReaderValue<Float_t> GenLepton2_Eta = {fReader, "GenLepton2_Eta"};
   TTreeReaderArray<Float_t> MT_MuonMET = {fReader, "MT_MuonMET"};
   TTreeReaderArray<Float_t> Phi_MuonMET = {fReader, "Phi_MuonMET"};
   TTreeReaderArray<Float_t> MT_ElectronMET = {fReader, "MT_ElectronMET"};
   TTreeReaderArray<Float_t> Phi_ElectronMET = {fReader, "Phi_ElectronMET"};
   TTreeReaderValue<Float_t> Kin_Hmass = {fReader, "Kin_Hmass"};
   TTreeReaderValue<Float_t> Kin_HdRbb = {fReader, "Kin_HdRbb"};
   TTreeReaderValue<Float_t> Kin_Chi2 = {fReader, "Kin_Chi2"};
   TTreeReaderValue<Float_t> Kin_TopMHc = {fReader, "Kin_TopMHc"};
   TTreeReaderValue<Float_t> Kin_TopMWb = {fReader, "Kin_TopMWb"};
   TTreeReaderValue<Float_t> Kin_Wmass = {fReader, "Kin_Wmass"};
   TTreeReaderValue<Float_t> Gen_Hmass = {fReader, "Gen_Hmass"};
   TTreeReaderValue<Float_t> Gen_HdRbb = {fReader, "Gen_HdRbb"};
   TTreeReaderValue<Float_t> upJet_dR = {fReader, "upJet_dR"};
   TTreeReaderValue<Float_t> HbJets1_dR = {fReader, "HbJets1_dR"};
   TTreeReaderValue<Float_t> HbJets2_dR = {fReader, "HbJets2_dR"};
   TTreeReaderValue<Int_t> IsMuonTrig = {fReader, "IsMuonTrig"};
   TTreeReaderValue<Int_t> IsElectronTrig = {fReader, "IsElectronTrig"};


   FCNCAnalysis(TTree * /*tree*/ =0) { }
   virtual ~FCNCAnalysis() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   double transverseMass(const TLorentzVector & l, const TLorentzVector & nu);


   void make1DHistogram( TH1D * h, int & step, TString hname, TString title, double nbins, double low, double max); 

   ClassDef(FCNCAnalysis,0);

   TH1D *h_NJet[21];
   TH1D *h_NBJetCSVv2M[21];
   TH1D *h_NBJetCSVv2T[21];
   TH1D *h_NCJetM[21];
   TH1D *h_MET[21];
   TH1D *h_WMass[21];
   TH1D *h_LepIso[21];
   TH1D *h_kinhmass[21];
   TH1D *h_kinhdRbb[21];
   TH1D *h_kinwmass[21];

   TH1D *h_DPhi[21];
   TH1D *h_jDPhi[21];
   TH1D *h_jDEta[21];
   TH1D *h_jDR[21];
   TH2D *h_jDPhiDEta[21];
   TH1D *h_bjmDPhi[21];
   TH1D *h_bjmDEta[21];
   TH1D *h_bjmDR[21];
   TH2D *h_bjmDPhiDEta[21];
   TH1D *h_bjtDPhi[21];
   TH1D *h_bjtDEta[21];
   TH1D *h_bjtDR[21];
   TH2D *h_bjtDPhiDEta[21];
   TH1D *h_cjmDPhi[21];
   TH1D *h_cjmDEta[21];
   TH1D *h_cjmDR[21];
   TH2D *h_cjmDPhiDEta[21];


   float EventWeight;

};

#endif

#ifdef FCNCAnalysis_cxx
void FCNCAnalysis::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the reader is initialized.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   fReader.SetTree(tree);
}

Bool_t FCNCAnalysis::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}


#endif // #ifdef FCNCAnalysis_cxx

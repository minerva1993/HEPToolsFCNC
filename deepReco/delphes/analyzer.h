//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Feb  6 14:04:11 2018 by ROOT version 6.10/04
// from TTree tree/tree
// found on file: flat_ctautau.root
//////////////////////////////////////////////////////////

#ifndef analyzer_h
#define analyzer_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TLorentzVector.h>

// Header file for the classes stored in the TTree if any.

class analyzer {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   UShort_t        run;
   UInt_t          event;
   Float_t         weight;
   Float_t         MET_pt;
   Float_t         MET_phi;
   UShort_t        nMuon;
   Float_t         Muon_pt[4];   //[nMuon]
   Float_t         Muon_eta[4];   //[nMuon]
   Float_t         Muon_phi[4];   //[nMuon]
   Float_t         Muon_m[4];   //[nMuon]
   Short_t         Muon_q[4];   //[nMuon]
   Float_t         Muon_relIso[4];   //[nMuon]
   UShort_t        nElectron;
   Float_t         Electron_pt[4];   //[nElectron]
   Float_t         Electron_eta[4];   //[nElectron]
   Float_t         Electron_phi[4];   //[nElectron]
   Float_t         Electron_m[4];   //[nElectron]
   Short_t         Electron_q[4];   //[nElectron]
   Float_t         Electron_relIso[4];   //[nElectron]
   UShort_t        nJet;
   Float_t         Jet_pt[14];   //[nJet]
   Float_t         Jet_eta[14];   //[nJet]
   Float_t         Jet_phi[14];   //[nJet]
   Float_t         Jet_m[14];   //[nJet]
   Short_t         Jet_flav[14];   //[nJet]
   Float_t         Jet_bTag[14];   //[nJet]
   Float_t         Jet_TauTag[14];   //[nJet]
   UShort_t        nGenParticle;
   Float_t         GenParticle_pt[9];   //[nGenParticle]
   Float_t         GenParticle_eta[9];   //[nGenParticle]
   Float_t         GenParticle_phi[9];   //[nGenParticle]
   Float_t         GenParticle_m[9];   //[nGenParticle]
   Short_t         GenParticle_pdgId[9];   //[nGenParticle]
   Short_t         GenParticle_q3[9];   //[nGenParticle]
   Short_t         GenParticle_mother[9];   //[nGenParticle]
   Short_t         GenParticle_dau1[9];   //[nGenParticle]
   Short_t         GenParticle_dau2[9];   //[nGenParticle]
   UShort_t        nSubJet;
   Float_t         SubJet_pt[255];   //[nSubJet]
   Float_t         SubJet_eta[255];   //[nSubJet]
   Float_t         SubJet_phi[255];   //[nSubJet]
   Short_t         SubJet_q[255];   //[nSubJet]
   Short_t         SubJet_pdgId[255];   //[nSubJet]
   Short_t         SubJet_jetIdx[255];   //[nSubJet]

   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_event;   //!
   TBranch        *b_weight;   //!
   TBranch        *b_MET_pt;   //!
   TBranch        *b_MET_phi;   //!
   TBranch        *b_nMuon;   //!
   TBranch        *b_Muon_pt;   //!
   TBranch        *b_Muon_eta;   //!
   TBranch        *b_Muon_phi;   //!
   TBranch        *b_Muon_m;   //!
   TBranch        *b_Muon_q;   //!
   TBranch        *b_Muon_relIso;   //!
   TBranch        *b_nElectron;   //!
   TBranch        *b_Electron_pt;   //!
   TBranch        *b_Electron_eta;   //!
   TBranch        *b_Electron_phi;   //!
   TBranch        *b_Electron_m;   //!
   TBranch        *b_Electron_q;   //!
   TBranch        *b_Electron_relIso;   //!
   TBranch        *b_nJet;   //!
   TBranch        *b_Jet_pt;   //!
   TBranch        *b_Jet_eta;   //!
   TBranch        *b_Jet_phi;   //!
   TBranch        *b_Jet_m;   //!
   TBranch        *b_Jet_flav;   //!
   TBranch        *b_Jet_bTag;   //!
   TBranch        *b_Jet_TauTag;   //!
   TBranch        *b_nGenParticle;   //!
   TBranch        *b_GenParticle_pt;   //!
   TBranch        *b_GenParticle_eta;   //!
   TBranch        *b_GenParticle_phi;   //!
   TBranch        *b_GenParticle_m;   //!
   TBranch        *b_GenParticle_pdgId;   //!
   TBranch        *b_GenParticle_q3;   //!
   TBranch        *b_GenParticle_mother;   //!
   TBranch        *b_GenParticle_dau1;   //!
   TBranch        *b_GenParticle_dau2;   //!
   TBranch        *b_nSubJet;   //!
   TBranch        *b_SubJet_pt;   //!
   TBranch        *b_SubJet_eta;   //!
   TBranch        *b_SubJet_phi;   //!
   TBranch        *b_SubJet_q;   //!
   TBranch        *b_SubJet_pdgId;   //!
   TBranch        *b_SubJet_jetIdx;   //!

   analyzer(TTree *tree=0);
   virtual ~analyzer();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(const std::string outFileName);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);

};

#endif

#ifdef analyzer_cxx
analyzer::analyzer(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("flat_ctautau.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("flat_ctautau.root");
      }
      f->GetObject("tree",tree);

   }
   Init(tree);
}

analyzer::~analyzer()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t analyzer::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t analyzer::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void analyzer::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("event", &event, &b_event);
   fChain->SetBranchAddress("weight", &weight, &b_weight);
   fChain->SetBranchAddress("MET_pt", &MET_pt, &b_MET_pt);
   fChain->SetBranchAddress("MET_phi", &MET_phi, &b_MET_phi);
   fChain->SetBranchAddress("nMuon", &nMuon, &b_nMuon);
   fChain->SetBranchAddress("Muon_pt", Muon_pt, &b_Muon_pt);
   fChain->SetBranchAddress("Muon_eta", Muon_eta, &b_Muon_eta);
   fChain->SetBranchAddress("Muon_phi", Muon_phi, &b_Muon_phi);
   fChain->SetBranchAddress("Muon_m", Muon_m, &b_Muon_m);
   fChain->SetBranchAddress("Muon_q", Muon_q, &b_Muon_q);
   fChain->SetBranchAddress("Muon_relIso", Muon_relIso, &b_Muon_relIso);
   fChain->SetBranchAddress("nElectron", &nElectron, &b_nElectron);
   fChain->SetBranchAddress("Electron_pt", Electron_pt, &b_Electron_pt);
   fChain->SetBranchAddress("Electron_eta", Electron_eta, &b_Electron_eta);
   fChain->SetBranchAddress("Electron_phi", Electron_phi, &b_Electron_phi);
   fChain->SetBranchAddress("Electron_m", Electron_m, &b_Electron_m);
   fChain->SetBranchAddress("Electron_q", Electron_q, &b_Electron_q);
   fChain->SetBranchAddress("Electron_relIso", Electron_relIso, &b_Electron_relIso);
   fChain->SetBranchAddress("nJet", &nJet, &b_nJet);
   fChain->SetBranchAddress("Jet_pt", Jet_pt, &b_Jet_pt);
   fChain->SetBranchAddress("Jet_eta", Jet_eta, &b_Jet_eta);
   fChain->SetBranchAddress("Jet_phi", Jet_phi, &b_Jet_phi);
   fChain->SetBranchAddress("Jet_m", Jet_m, &b_Jet_m);
   fChain->SetBranchAddress("Jet_flav", Jet_flav, &b_Jet_flav);
   fChain->SetBranchAddress("Jet_bTag", Jet_bTag, &b_Jet_bTag);
   fChain->SetBranchAddress("nGenParticle", &nGenParticle, &b_nGenParticle);
   fChain->SetBranchAddress("GenParticle_pt", GenParticle_pt, &b_GenParticle_pt);
   fChain->SetBranchAddress("GenParticle_eta", GenParticle_eta, &b_GenParticle_eta);
   fChain->SetBranchAddress("GenParticle_phi", GenParticle_phi, &b_GenParticle_phi);
   fChain->SetBranchAddress("GenParticle_m", GenParticle_m, &b_GenParticle_m);
   fChain->SetBranchAddress("GenParticle_pdgId", GenParticle_pdgId, &b_GenParticle_pdgId);
   fChain->SetBranchAddress("GenParticle_q3", GenParticle_q3, &b_GenParticle_q3);
   fChain->SetBranchAddress("GenParticle_mother", GenParticle_mother, &b_GenParticle_mother);
   fChain->SetBranchAddress("GenParticle_dau1", GenParticle_dau1, &b_GenParticle_dau1);
   fChain->SetBranchAddress("GenParticle_dau2", GenParticle_dau2, &b_GenParticle_dau2);
   fChain->SetBranchAddress("nSubJet", &nSubJet, &b_nSubJet);
   fChain->SetBranchAddress("SubJet_pt", SubJet_pt, &b_SubJet_pt);
   fChain->SetBranchAddress("SubJet_eta", SubJet_eta, &b_SubJet_eta);
   fChain->SetBranchAddress("SubJet_phi", SubJet_phi, &b_SubJet_phi);
   fChain->SetBranchAddress("SubJet_q", SubJet_q, &b_SubJet_q);
   fChain->SetBranchAddress("SubJet_pdgId", SubJet_pdgId, &b_SubJet_pdgId);
   fChain->SetBranchAddress("SubJet_jetIdx", SubJet_jetIdx, &b_SubJet_jetIdx);
   Notify();
}

Bool_t analyzer::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void analyzer::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t analyzer::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef analyzer_cxx

//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jan 31 23:15:22 2017 by ROOT version 6.06/01
// from TTree tree/TopTree
// found on file: /xrootd/store/user/brochero/v8-0-4/Tree_LepJets_NewCSVSF_v8-0-4_Spring16-80X_36814pb-1_ttbar_PowhegPythia.root
//////////////////////////////////////////////////////////

#ifndef makeTuple_h
#define makeTuple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TH1D.h>
#include <TH2D.h>
// Headers needed by this particular selector
#include <vector>
#include <TLorentzVector.h>

#include <iostream>

class makeTuple : public TSelector {
public :
   TTreeReader     fReader;  //!the tree reader
   TTree          *fChain = 0;   //!pointer to the analyzed TTree or TChain

   // Readers to access the data (delete the ones you do not need).
   //TTreeReaderValue<Int_t> event = {fReader, "event"};
   //TTreeReaderValue<Int_t> run = {fReader, "run"};
   //TTreeReaderValue<Int_t> luminumber = {fReader, "luminumber"};
   TTreeReaderValue<Float_t> genWeight = {fReader, "genweight"};
   TTreeReaderValue<Int_t> GoodPV = {fReader, "GoodPV"};
   TTreeReaderValue<Int_t> channel = {fReader, "channel"};
   TTreeReaderArray<float> PUWeight = {fReader, "PUWeight"};
   TTreeReaderValue<Float_t> MET = {fReader, "MET"};
   TTreeReaderValue<Float_t> MET_phi = {fReader, "MET_phi"};
   TTreeReaderValue<Float_t> lepton_pT = {fReader, "lepton_pT"};
   TTreeReaderValue<Float_t> lepton_eta = {fReader, "lepton_eta"};
   TTreeReaderValue<Float_t> lepton_phi = {fReader, "lepton_phi"};
   TTreeReaderValue<Float_t> lepton_E = {fReader, "lepton_E"};
   TTreeReaderValue<Float_t> lepton_LES = {fReader, "lepton_LES"};
   TTreeReaderValue<Float_t> lepton_relIso = {fReader, "lepton_relIso"};
   TTreeReaderValue<bool> lepton_isIso = {fReader, "lepton_isIso"};
   TTreeReaderArray<float> lepton_SF = {fReader, "lepton_SF"};
   TTreeReaderArray<float> jet_pT = {fReader, "jet_pT"};
   TTreeReaderArray<float> jet_eta = {fReader, "jet_eta"};
   TTreeReaderArray<float> jet_phi = {fReader, "jet_phi"};
   TTreeReaderArray<float> jet_E = {fReader, "jet_E"};
   TTreeReaderArray<int> jet_index = {fReader, "jet_index"};
   TTreeReaderArray<int> jet_gencone_mom = {fReader, "jet_gencone_mom"};
   TTreeReaderArray<float> jet_CSV = {fReader, "jet_CSV"};
   TTreeReaderArray<float> jet_SF_CSV_25 = {fReader, "jet_SF_CSV_25"};
   TTreeReaderArray<float> jet_SF_CSV_30 = {fReader, "jet_SF_CSV_30"};
   TTreeReaderArray<float> jet_SF_CSV_35 = {fReader, "jet_SF_CSV_35"};
   TTreeReaderArray<float> jet_SF_CSV_40 = {fReader, "jet_SF_CSV_40"};
   TTreeReaderArray<float> jet_SF_CSV = {fReader, "jet_SF_CSV"};
   TTreeReaderArray<float> jet_CvsL = {fReader, "jet_CvsL"};
   TTreeReaderArray<float> jet_CvsB = {fReader, "jet_CvsB"};
   //TTreeReaderValue<Int_t> jet_number = {fReader, "jet_number"};

   TTreeReaderArray<int> jet_partonFlavour = {fReader, "jet_partonFlavour"};
   TTreeReaderArray<int> jet_hadronFlavour = {fReader, "jet_hadronFlavour"};
   TTreeReaderArray<float> jet_JES_Up = {fReader, "jet_JES_Up"};
   TTreeReaderArray<float> jet_JES_Down = {fReader, "jet_JES_Down"};
   TTreeReaderArray<float> jet_JER_Up = {fReader, "jet_JER_Up"};
   TTreeReaderArray<float> jet_JER_Nom = {fReader, "jet_JER_Nom"};
   TTreeReaderArray<float> jet_JER_Down = {fReader, "jet_JER_Down"};
/*
   TTreeReaderValue<Float_t> kin_chi2 = {fReader, "kin_chi2"};
   TTreeReaderValue<Float_t> kinnu_pT = {fReader, "kinnu_pT"};
   TTreeReaderValue<Float_t> kinnu_eta = {fReader, "kinnu_eta"};
   TTreeReaderValue<Float_t> kinnu_phi = {fReader, "kinnu_phi"};
   TTreeReaderValue<Float_t> kinnu_E = {fReader, "kinnu_E"};
   TTreeReaderArray<float> kinjet_pT = {fReader, "kinjet_pT"};
   TTreeReaderArray<float> kinjet_eta = {fReader, "kinjet_eta"};
   TTreeReaderArray<float> kinjet_phi = {fReader, "kinjet_phi"};
   TTreeReaderArray<float> kinjet_E = {fReader, "kinjet_E"};
   TTreeReaderArray<int> kinjet_index = {fReader, "kinjet_index"};
   TTreeReaderValue<Float_t> fcnhkin_chi2 = {fReader, "fcnhkin_chi2"};
   TTreeReaderValue<Float_t> fcnhkinnu_pT = {fReader, "fcnhkinnu_pT"};
   TTreeReaderValue<Float_t> fcnhkinnu_eta = {fReader, "fcnhkinnu_eta"};
   TTreeReaderValue<Float_t> fcnhkinnu_phi = {fReader, "fcnhkinnu_phi"};
   TTreeReaderValue<Float_t> fcnhkinnu_E = {fReader, "fcnhkinnu_E"};
   TTreeReaderArray<float> fcnhkinjet_pT = {fReader, "fcnhkinjet_pT"};
   TTreeReaderArray<float> fcnhkinjet_eta = {fReader, "fcnhkinjet_eta"};
   TTreeReaderArray<float> fcnhkinjet_phi = {fReader, "fcnhkinjet_phi"};
   TTreeReaderArray<float> fcnhkinjet_E = {fReader, "fcnhkinjet_E"};
   TTreeReaderArray<int> fcnhkinjet_index = {fReader, "fcnhkinjet_index"};
*/
   TTreeReaderArray<float> pdfweight = {fReader, "pdfweight"};
   TTreeReaderArray<float> scaleweight = {fReader, "scaleweight"};
   TTreeReaderArray<int> jet_MatchedGenJetIndex = {fReader, "jet_MatchedGenJetIndex"};
   TTreeReaderArray<int> genconecatid = {fReader, "genconecatid"};
   TTreeReaderArray<float> gencone_gjet_pT = {fReader, "gencone_gjet_pT"};
   TTreeReaderArray<float> gencone_gjet_eta = {fReader, "gencone_gjet_eta"};
   TTreeReaderArray<float> gencone_gjet_phi = {fReader, "gencone_gjet_phi"};
   TTreeReaderArray<float> gencone_gjet_E = {fReader, "gencone_gjet_E"};
   TTreeReaderArray<int> gencone_gjetIndex = {fReader, "gencone_gjetIndex"};
   TTreeReaderArray<int> gencone_gJetFlavW = {fReader, "gencone_gJetFlavW"};
   TTreeReaderValue<Int_t> b_GenCone_NgJetsW = {fReader, "gencone_NgjetsW"};
   TTreeReaderValue<Float_t> DRAddJets = {fReader, "draddjets"};
   TTreeReaderValue<Int_t> genhiggscatid = {fReader, "genhiggscatid"};
   TTreeReaderValue<Int_t> genchannel = {fReader, "genchannel"};
   TTreeReaderValue<Float_t> genlepton_pT = {fReader, "genlepton_pT"};
   TTreeReaderValue<Float_t> genlepton_eta = {fReader, "genlepton_eta"};
   TTreeReaderValue<Float_t> genlepton_phi = {fReader, "genlepton_phi"};
   TTreeReaderValue<Float_t> genlepton_E = {fReader, "genlepton_E"};
   TTreeReaderValue<Float_t> gennu_pT = {fReader, "gennu_pT"};
   TTreeReaderValue<Float_t> gennu_eta = {fReader, "gennu_eta"};
   TTreeReaderValue<Float_t> gennu_phi = {fReader, "gennu_phi"};
   TTreeReaderValue<Float_t> gennu_E = {fReader, "gennu_E"};
   TTreeReaderArray<float> genjet_pT = {fReader, "genjet_pT"};
   TTreeReaderArray<float> genjet_eta = {fReader, "genjet_eta"};
   TTreeReaderArray<float> genjet_phi = {fReader, "genjet_phi"};
   TTreeReaderArray<float> genjet_E = {fReader, "genjet_E"};
   TTreeReaderArray<int> genjet_mom = {fReader, "genjet_mom"};
   TTreeReaderArray<int> genjet_gencone_mom = {fReader, "genjet_gencone_mom"};
   TTreeReaderValue<Float_t> addHbjet1_pt = {fReader, "addHbjet1_pt"};
   TTreeReaderValue<Float_t> addHbjet1_eta = {fReader, "addHbjet1_eta"};
   TTreeReaderValue<Float_t> addHbjet1_phi = {fReader, "addHbjet1_phi"};
   TTreeReaderValue<Float_t> addHbjet1_e = {fReader, "addHbjet1_e"};
   TTreeReaderValue<Float_t> addHbjet2_pt = {fReader, "addHbjet2_pt"};
   TTreeReaderValue<Float_t> addHbjet2_eta = {fReader, "addHbjet2_eta"};
   TTreeReaderValue<Float_t> addHbjet2_phi = {fReader, "addHbjet2_phi"};
   TTreeReaderValue<Float_t> addHbjet2_e = {fReader, "addHbjet2_e"};
   TTreeReaderValue<Float_t> dRHbb = {fReader, "dRHbb"};

   makeTuple(TTree * /*tree*/ =0) { }
   virtual ~makeTuple() { }
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
   //int totalevt = fChain->GetTree()->GetEntries();

    TTree *sigTree = 0;
    TTree *bkgTree = 0;

    int nevt = 0;

    //objects for ntuple
    int b_nevt = 0;
    int b_file = -1;
    int b_GoodPV = 0;
    int b_EventCategory = -1;
    float b_EventWeight = 1.0;
    int b_genMatch = -1;

    int b_njets = 0;
    int b_nbjets_m = 0;
    float b_met = 0;
    float b_met_phi = -10;
    float b_lepdphi = 0;
    float b_transversem = 0;

    float b_lepWpt = 0;
    float b_lepWeta = 10;
    float b_lepWphi = 10;
    float b_lepWdphi = 10;
    float b_lepWm = 0;

    float b_jet0pt = 0;
    float b_jet0eta = 10;
    float b_jet0phi = 10;
    float b_jet0m = 0;
    float b_jet0csv = 5;
    float b_jet0cvsl = 5;
    float b_jet0cvsb = 5;

    float b_jet1pt = 0;
    float b_jet1eta = 10;
    float b_jet1phi = 10;
    float b_jet1m = 0;
    float b_jet1csv = 5;
    float b_jet1cvsl = 5;
    float b_jet1cvsb = 5;

    float b_jet2pt = 0;
    float b_jet2eta = 10;
    float b_jet2phi = 10;
    float b_jet2m = 0;
    float b_jet2csv = 5;
    float b_jet2cvsl = 5;
    float b_jet2cvsb = 5;

    float b_jet3pt = 0;
    float b_jet3eta = 10;
    float b_jet3phi = 10;
    float b_jet3m = 0;
    float b_jet3csv = 5;
    float b_jet3cvsl = 5;
    float b_jet3cvsb = 5;

    float b_jet12pt = 0;
    float b_jet12eta = 10;
    float b_jet12deta = 10;
    float b_jet12phi = 10;
    float b_jet12dphi = 10;
    float b_jet12m = 0;
    float b_jet12dR = 0;
    float b_jet12DR = 0;

    float b_jet23pt = 0;
    float b_jet23eta = 10;
    float b_jet23deta = 10;
    float b_jet23phi = 10;
    float b_jet23dphi = 10;
    float b_jet23dR = 0;
    float b_jet23m = 0;

    float b_jet31pt = 0;
    float b_jet31eta = 10;
    float b_jet31deta = 10;
    float b_jet31phi = 10;
    float b_jet31dphi = 10;
    float b_jet31dR = 0;
    float b_jet31m = 0;

    float b_lepTpt = 0;
    float b_lepTeta = 10;
    float b_lepTdeta = 10;
    float b_lepTphi = 10;
    float b_lepTdphi = 10;
    float b_lepTdR = 0;
    float b_lepTm = 0;

    float b_hadTpt = 0;
    float b_hadTeta = 10;
    float b_hadT12_3deta = 10;
    float b_hadT23_1deta = 10;
    float b_hadT31_2deta = 10;
    float b_hadTphi = 10;
    float b_hadT12_3dphi = 10;
    float b_hadT23_1dphi = 10;
    float b_hadT31_2dphi = 10;
    float b_hadT12_3dR = 0;
    float b_hadT23_1dR = 0;
    float b_hadT31_2dR = 0;
    float b_hadTm = 0;
    float b_genHadW = 0;


   ClassDef(makeTuple,0);

};

#endif

#ifdef makeTuple_cxx
void makeTuple::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the reader is initialized.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).
  
   fReader.SetTree(tree);

}

Bool_t makeTuple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}


#endif // #ifdef makeTuple_cxx

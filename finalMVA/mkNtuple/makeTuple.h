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
#include <TLeaf.h>
#include <string>
#include <iostream>
using namespace std;

class makeTuple : public TSelector {
public :
   TTreeReader     fReader;  //!the tree reader
   TTree          *fChain = 0;   //!pointer to the analyzed TTree or TChain

   // Readers to access the data (delete the ones you do not need).
   TTreeReaderValue<Int_t> event = {fReader, "event"};
   TTreeReaderValue<Int_t> run = {fReader, "run"};
   TTreeReaderValue<Int_t> luminumber = {fReader, "luminumber"};
   TTreeReaderValue<Float_t> genweight = {fReader, "genweight"};
   TTreeReaderValue<Int_t> TruePV = {fReader, "TruePV"};
   TTreeReaderValue<Int_t> GoodPV = {fReader, "GoodPV"};
   TTreeReaderValue<Int_t> channel = {fReader, "channel"};
   TTreeReaderArray<float> PUWeight = {fReader, "PUWeight"};
   TTreeReaderArray<double> prefireweight = {fReader, "prefireweight"};
   TTreeReaderArray<float> pdfweight = {fReader, "pdfweight"};
   TTreeReaderArray<float> scaleweight = {fReader, "scaleweight"};
   TTreeReaderArray<float> psweight = {fReader, "psweight"};
   TTreeReaderValue<Float_t> MET = {fReader, "MET"};
   TTreeReaderValue<Float_t> MET_phi = {fReader, "MET_phi"};
   TTreeReaderArray<float> MET_unc_x = {fReader, "MET_unc_x"};
   TTreeReaderArray<float> MET_unc_y = {fReader, "MET_unc_y"};
   TTreeReaderValue<Float_t> lepton_pt = {fReader, "lepton_pt"};
   TTreeReaderValue<Float_t> lepton_eta = {fReader, "lepton_eta"};
   TTreeReaderValue<Float_t> lepton_phi = {fReader, "lepton_phi"};
   TTreeReaderValue<Float_t> lepton_e = {fReader, "lepton_e"};
   TTreeReaderArray<float> lepton_SF = {fReader, "lepton_SF"};
   TTreeReaderArray<float> lepton_scale = {fReader, "lepton_scale"};
   TTreeReaderValue<Float_t> lepton_relIso = {fReader, "lepton_relIso"};
   TTreeReaderValue<Bool_t> lepton_isIso = {fReader, "lepton_isIso"};
   TTreeReaderArray<float> jet_pt = {fReader, "jet_pt"};
   TTreeReaderArray<float> jet_eta = {fReader, "jet_eta"};
   TTreeReaderArray<float> jet_phi = {fReader, "jet_phi"};
   TTreeReaderArray<float> jet_e = {fReader, "jet_e"};
   TTreeReaderArray<int> jet_index = {fReader, "jet_index"};
   TTreeReaderArray<float> jet_deepCSV = {fReader, "jet_deepCSV"};
   TTreeReaderArray<float> jet_SF_deepCSV_30 = {fReader, "jet_SF_deepCSV_30"};
   TTreeReaderArray<float> jet_deepCvsL = {fReader, "jet_deepCvsL"};
   TTreeReaderArray<float> jet_deepCvsB = {fReader, "jet_deepCvsB"};
   TTreeReaderValue<Int_t> jet_njet = {fReader, "jet_njet"};
   TTreeReaderValue<Int_t> jet_nbjetm = {fReader, "jet_nbjetm"};
   TTreeReaderArray<int> jet_partonFlavour = {fReader, "jet_partonFlavour"};
   TTreeReaderArray<int> jet_hadronFlavour = {fReader, "jet_hadronFlavour"};
   TTreeReaderArray<float> jet_JES_Up = {fReader, "jet_JES_Up"};
   TTreeReaderArray<float> jet_JES_Down = {fReader, "jet_JES_Down"};
   TTreeReaderArray<float> jet_JER_Up = {fReader, "jet_JER_Up"};
   TTreeReaderArray<float> jet_JER_Nom = {fReader, "jet_JER_Nom"};
   TTreeReaderArray<float> jet_JER_Down = {fReader, "jet_JER_Down"};
   TTreeReaderValue<Float_t> Hbjet1_pt = {fReader, "Hbjet1_pt"};
   TTreeReaderValue<Float_t> Hbjet1_eta = {fReader, "Hbjet1_eta"};
   TTreeReaderValue<Float_t> Hbjet1_phi = {fReader, "Hbjet1_phi"};
   TTreeReaderValue<Float_t> Hbjet1_e = {fReader, "Hbjet1_e"};
   TTreeReaderValue<Float_t> Hbjet2_pt = {fReader, "Hbjet2_pt"};
   TTreeReaderValue<Float_t> Hbjet2_eta = {fReader, "Hbjet2_eta"};
   TTreeReaderValue<Float_t> Hbjet2_phi = {fReader, "Hbjet2_phi"};
   TTreeReaderValue<Float_t> Hbjet2_e = {fReader, "Hbjet2_e"};
   TTreeReaderValue<Float_t> dRHbb = {fReader, "dRHbb"};
   TTreeReaderValue<Float_t> Hbquarkjet1_pt = {fReader, "Hbquarkjet1_pt"};
   TTreeReaderValue<Float_t> Hbquarkjet1_eta = {fReader, "Hbquarkjet1_eta"};
   TTreeReaderValue<Float_t> Hbquarkjet1_phi = {fReader, "Hbquarkjet1_phi"};
   TTreeReaderValue<Float_t> Hbquarkjet1_e = {fReader, "Hbquarkjet1_e"};
   TTreeReaderValue<Float_t> Hbquarkjet2_pt = {fReader, "Hbquarkjet2_pt"};
   TTreeReaderValue<Float_t> Hbquarkjet2_eta = {fReader, "Hbquarkjet2_eta"};
   TTreeReaderValue<Float_t> Hbquarkjet2_phi = {fReader, "Hbquarkjet2_phi"};
   TTreeReaderValue<Float_t> Hbquarkjet2_e = {fReader, "Hbquarkjet2_e"};

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
   float transverseMass(const TLorentzVector & l, const TLorentzVector & nu);
   //int totalevt = fChain->GetTree()->GetEntries();

    TTree *tree = 0;

    int nevt = 0;

    //objects for ntuple
    int b_nevt = 0; int b_GoodPV = 0; int b_EventCategory = -1; int b_TruePV = 0;
    int b_trigger = 0; int b_njets = 0; int b_nbjets_m = 0; int b_channel = -1; float b_EventWeight = 1.0;

    float b_lepton_pt = 0; float b_lepton_phi = 10; float b_lepton_eta = 10; float b_lepton_m = 0;
    float b_met = 0; float b_met_phi = 10; float b_lepdphi = 0;
    float b_lepWpt = 0; float b_lepWeta = 10; float b_lepWphi = 10; float b_lepWdphi = 10; float b_lepWmt = 0;

    //ST FCNC
    int b_stfcnc_genMatch = -1; float b_stfcnc_score = -1;
    float b_stfcnc_jet0pt = 0; float b_stfcnc_jet0eta = 10; float b_stfcnc_jet0phi = 10; float b_stfcnc_jet0m = 0;
    float b_stfcnc_jet0csv = 5; float b_stfcnc_jet0cvsl = 5; float b_stfcnc_jet0cvsb = 5; int b_stfcnc_jet0Idx = -1;
    float b_stfcnc_jet1pt = 0; float b_stfcnc_jet1eta = 10; float b_stfcnc_jet1phi = 10; float b_stfcnc_jet1m = 0;
    float b_stfcnc_jet1csv = 5; float b_stfcnc_jet1cvsl = 5; float b_stfcnc_jet1cvsb = 5; int b_stfcnc_jet1Idx = -1;
    float b_stfcnc_jet2pt = 0; float b_stfcnc_jet2eta = 10; float b_stfcnc_jet2phi = 10; float b_stfcnc_jet2m = 0;
    float b_stfcnc_jet2csv = 5; float b_stfcnc_jet2cvsl = 5; float b_stfcnc_jet2cvsb = 5; int b_stfcnc_jet2Idx = -1;
    float b_stfcnc_jet3pt = 0; float b_stfcnc_jet3eta = 10; float b_stfcnc_jet3phi = 10; float b_stfcnc_jet3m = 0;
    float b_stfcnc_jet3csv = 5; float b_stfcnc_jet3cvsl = 5; float b_stfcnc_jet3cvsb = 5; int b_stfcnc_jet3Idx = -1;
    float b_stfcnc_jet12pt = 0; float b_stfcnc_jet12eta = 10; float b_stfcnc_jet12deta = 10; float b_stfcnc_jet12phi = 10;
    float b_stfcnc_jet12dphi = 10; float b_stfcnc_jet12dR = 0; float b_stfcnc_jet12m = 0;
    float b_stfcnc_jet23pt = 0; float b_stfcnc_jet23eta = 10; float b_stfcnc_jet23deta = 10; float b_stfcnc_jet23phi = 10;
    float b_stfcnc_jet23dphi = 10; float b_stfcnc_jet23dR = 0; float b_stfcnc_jet23m = 0;
    float b_stfcnc_jet31pt = 0; float b_stfcnc_jet31eta = 10; float b_stfcnc_jet31deta = 10; float b_stfcnc_jet31phi = 10;
    float b_stfcnc_jet31dphi = 10; float b_stfcnc_jet31dR = 0; float b_stfcnc_jet31m = 0;
    float b_stfcnc_lepTpt = 0; float b_stfcnc_lepTeta = 10; float b_stfcnc_lepTdeta = 10;
    float b_stfcnc_lepTphi = 10; float b_stfcnc_lepTdphi = 10; float b_stfcnc_lepTdR = 0; float b_stfcnc_lepTmt = 0;
    float b_stfcnc_hadTpt = 0; float b_stfcnc_hadTeta = 10; float b_stfcnc_hadTphi = 10;
    float b_stfcnc_hadT12_3deta = 10; float b_stfcnc_hadT23_1deta = 10; float b_stfcnc_hadT31_2deta = 10;
    float b_stfcnc_hadT12_3dphi = 10; float b_stfcnc_hadT23_1dphi = 10; float b_stfcnc_hadT31_2dphi = 10;
    float b_stfcnc_hadT12_3dR = 0; float b_stfcnc_hadT23_1dR = 0; float b_stfcnc_hadT31_2dR = 0; float b_stfcnc_hadTm = 0;
    float b_stfcnc_jet0lepdR = 0; float b_stfcnc_jet1lepdR = 0; float b_stfcnc_jet2lepdR = 0; float b_stfcnc_jet3lepdR = 0;
    float b_stfcnc_jet01dR = 0; float b_stfcnc_jet02dR = 0; float b_stfcnc_jet03dR = 0;
    float b_stfcnc_jet12_lepdR = 0; float b_stfcnc_jet23_lepdR = 0; float b_stfcnc_jet31_lepdR = 0;
    float b_stfcnc_jet12_0dR = 0; float b_stfcnc_jet23_0dR = 0; float b_stfcnc_jet31_0dR = 0;
    float b_stfcnc_lepTjet12dphi = 10; float b_stfcnc_lepTjet23dphi = 10; float b_stfcnc_lepTjet31dphi = 10;
    float b_stfcnc_hadT_jet0dR = 0;


    //TT FCNC
    int b_ttfcnc_genMatch = -1; float b_ttfcnc_score = -1;
    float b_ttfcnc_jet0pt = 0; float b_ttfcnc_jet0eta = 10; float b_ttfcnc_jet0phi = 10; float b_ttfcnc_jet0m = 0;
    float b_ttfcnc_jet0csv = 5; float b_ttfcnc_jet0cvsl = 5; float b_ttfcnc_jet0cvsb = 5; int b_ttfcnc_jet0Idx = -1;
    float b_ttfcnc_jet1pt = 0; float b_ttfcnc_jet1eta = 10; float b_ttfcnc_jet1phi = 10; float b_ttfcnc_jet1m = 0;
    float b_ttfcnc_jet1csv = 5; float b_ttfcnc_jet1cvsl = 5; float b_ttfcnc_jet1cvsb = 5; int b_ttfcnc_jet1Idx = -1;
    float b_ttfcnc_jet2pt = 0; float b_ttfcnc_jet2eta = 10; float b_ttfcnc_jet2phi = 10; float b_ttfcnc_jet2m = 0;
    float b_ttfcnc_jet2csv = 5; float b_ttfcnc_jet2cvsl = 5; float b_ttfcnc_jet2cvsb = 5; int b_ttfcnc_jet2Idx = -1;
    float b_ttfcnc_jet3pt = 0; float b_ttfcnc_jet3eta = 10; float b_ttfcnc_jet3phi = 10; float b_ttfcnc_jet3m = 0;
    float b_ttfcnc_jet3csv = 5; float b_ttfcnc_jet3cvsl = 5; float b_ttfcnc_jet3cvsb = 5; int b_ttfcnc_jet3Idx = -1;
    float b_ttfcnc_jet12pt = 0; float b_ttfcnc_jet12eta = 10; float b_ttfcnc_jet12deta = 10; float b_ttfcnc_jet12phi = 10;
    float b_ttfcnc_jet12dphi = 10; float b_ttfcnc_jet12dR = 0; float b_ttfcnc_jet12m = 0;
    float b_ttfcnc_jet23pt = 0; float b_ttfcnc_jet23eta = 10; float b_ttfcnc_jet23deta = 10; float b_ttfcnc_jet23phi = 10;
    float b_ttfcnc_jet23dphi = 10; float b_ttfcnc_jet23dR = 0; float b_ttfcnc_jet23m = 0;
    float b_ttfcnc_jet31pt = 0; float b_ttfcnc_jet31eta = 10; float b_ttfcnc_jet31deta = 10; float b_ttfcnc_jet31phi = 10;
    float b_ttfcnc_jet31dphi = 10; float b_ttfcnc_jet31dR = 0; float b_ttfcnc_jet31m = 0;
    float b_ttfcnc_lepTpt = 0; float b_ttfcnc_lepTeta = 10; float b_ttfcnc_lepTdeta = 10;
    float b_ttfcnc_lepTphi = 10; float b_ttfcnc_lepTdphi = 10; float b_ttfcnc_lepTdR = 0; float b_ttfcnc_lepTmt = 0;
    float b_ttfcnc_hadTpt = 0; float b_ttfcnc_hadTeta = 10; float b_ttfcnc_hadTphi = 10;
    float b_ttfcnc_hadT12_3deta = 10; float b_ttfcnc_hadT23_1deta = 10; float b_ttfcnc_hadT31_2deta = 10;
    float b_ttfcnc_hadT12_3dphi = 10; float b_ttfcnc_hadT23_1dphi = 10; float b_ttfcnc_hadT31_2dphi = 10;
    float b_ttfcnc_hadT12_3dR = 0; float b_ttfcnc_hadT23_1dR = 0; float b_ttfcnc_hadT31_2dR = 0; float b_ttfcnc_hadTm = 0;
    float b_ttfcnc_jet0lepdR = 0; float b_ttfcnc_jet1lepdR = 0; float b_ttfcnc_jet2lepdR = 0; float b_ttfcnc_jet3lepdR = 0;
    float b_ttfcnc_jet01dR = 0; float b_ttfcnc_jet02dR = 0; float b_ttfcnc_jet03dR = 0;
    float b_ttfcnc_jet12_lepdR = 0; float b_ttfcnc_jet23_lepdR = 0; float b_ttfcnc_jet31_lepdR = 0;
    float b_ttfcnc_jet12_0dR = 0; float b_ttfcnc_jet23_0dR = 0; float b_ttfcnc_jet31_0dR = 0;
    float b_ttfcnc_lepTjet12dphi = 10; float b_ttfcnc_lepTjet23dphi = 10; float b_ttfcnc_lepTjet31dphi = 10;
    float b_ttfcnc_hadT_jet0dR = 0;

    //TT BKG
    int b_ttbkg_genMatch = -1; float b_ttbkg_score = -1;
    float b_ttbkg_jet0pt = 0; float b_ttbkg_jet0eta = 10; float b_ttbkg_jet0phi = 10; float b_ttbkg_jet0m = 0;
    float b_ttbkg_jet0csv = 5; float b_ttbkg_jet0cvsl = 5; float b_ttbkg_jet0cvsb = 5; int b_ttbkg_jet0Idx = -1;
    float b_ttbkg_jet1pt = 0; float b_ttbkg_jet1eta = 10; float b_ttbkg_jet1phi = 10; float b_ttbkg_jet1m = 0;
    float b_ttbkg_jet1csv = 5; float b_ttbkg_jet1cvsl = 5; float b_ttbkg_jet1cvsb = 5; int b_ttbkg_jet1Idx = -1;
    float b_ttbkg_jet2pt = 0; float b_ttbkg_jet2eta = 10; float b_ttbkg_jet2phi = 10; float b_ttbkg_jet2m = 0;
    float b_ttbkg_jet2csv = 5; float b_ttbkg_jet2cvsl = 5; float b_ttbkg_jet2cvsb = 5; int b_ttbkg_jet2Idx = -1;
    float b_ttbkg_jet3pt = 0; float b_ttbkg_jet3eta = 10; float b_ttbkg_jet3phi = 10; float b_ttbkg_jet3m = 0;
    float b_ttbkg_jet3csv = 5; float b_ttbkg_jet3cvsl = 5; float b_ttbkg_jet3cvsb = 5; int b_ttbkg_jet3Idx = -1;
    float b_ttbkg_jet12pt = 0; float b_ttbkg_jet12eta = 10; float b_ttbkg_jet12deta = 10; float b_ttbkg_jet12phi = 10;
    float b_ttbkg_jet12dphi = 10; float b_ttbkg_jet12dR = 0; float b_ttbkg_jet12m = 0;
    float b_ttbkg_jet23pt = 0; float b_ttbkg_jet23eta = 10; float b_ttbkg_jet23deta = 10; float b_ttbkg_jet23phi = 10;
    float b_ttbkg_jet23dphi = 10; float b_ttbkg_jet23dR = 0; float b_ttbkg_jet23m = 0;
    float b_ttbkg_jet31pt = 0; float b_ttbkg_jet31eta = 10; float b_ttbkg_jet31deta = 10; float b_ttbkg_jet31phi = 10;
    float b_ttbkg_jet31dphi = 10; float b_ttbkg_jet31dR = 0; float b_ttbkg_jet31m = 0;
    float b_ttbkg_lepTpt = 0; float b_ttbkg_lepTeta = 10; float b_ttbkg_lepTdeta = 10;
    float b_ttbkg_lepTphi = 10; float b_ttbkg_lepTdphi = 10; float b_ttbkg_lepTdR = 0; float b_ttbkg_lepTmt = 0;
    float b_ttbkg_hadTpt = 0; float b_ttbkg_hadTeta = 10; float b_ttbkg_hadTphi = 10;
    float b_ttbkg_hadT12_3deta = 10; float b_ttbkg_hadT23_1deta = 10; float b_ttbkg_hadT31_2deta = 10;
    float b_ttbkg_hadT12_3dphi = 10; float b_ttbkg_hadT23_1dphi = 10; float b_ttbkg_hadT31_2dphi = 10;
    float b_ttbkg_hadT12_3dR = 0; float b_ttbkg_hadT23_1dR = 0; float b_ttbkg_hadT31_2dR = 0; float b_ttbkg_hadTm = 0;
    float b_ttbkg_jet0lepdR = 0; float b_ttbkg_jet1lepdR = 0; float b_ttbkg_jet2lepdR = 0; float b_ttbkg_jet3lepdR = 0;
    float b_ttbkg_jet01dR = 0; float b_ttbkg_jet02dR = 0; float b_ttbkg_jet03dR = 0;
    float b_ttbkg_jet12_lepdR = 0; float b_ttbkg_jet23_lepdR = 0; float b_ttbkg_jet31_lepdR = 0;
    float b_ttbkg_jet12_0dR = 0; float b_ttbkg_jet23_0dR = 0; float b_ttbkg_jet31_0dR = 0;
    float b_ttbkg_lepTjet12dphi = 10; float b_ttbkg_lepTjet23dphi = 10; float b_ttbkg_lepTjet31dphi = 10;
    float b_ttbkg_hadT_jet0dR = 0;

    ////RECO
    TFile *stfcnc_File, *ttfcnc_File, *ttbkg_File;
    TTree *stfcnc_Tree, *ttfcnc_Tree, *ttbkg_Tree;
    string syst_ext;

    vector<double> stfcnc_lepPt, ttfcnc_lepPt, ttbkg_lepPt;
    vector<double> stfcnc_missET, ttfcnc_missET, ttbkg_missET;
    vector<double> stfcnc_score, ttfcnc_score, ttbkg_score;

    vector<int> stfcnc_dupCheck, ttfcnc_dupCheck, ttbkg_dupCheck;

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

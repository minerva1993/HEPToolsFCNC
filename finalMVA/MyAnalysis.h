#ifndef MyAnalysis_h
#define MyAnalysis_h

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

class MyAnalysis : public TSelector {
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
   TTreeReaderValue<Float_t> topptweight = {fReader, "topptweight"};
   TTreeReaderValue<Float_t> MET = {fReader, "MET"};
   TTreeReaderValue<Float_t> MET_phi = {fReader, "MET_phi"};
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

   MyAnalysis(TTree * /*tree*/ =0) { }
   virtual ~MyAnalysis() { }
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
   bool isPartOf(const std::string& word, const std::string& sentence);

   ClassDef(MyAnalysis,0);

    //Declare systematics here FIXME
    const std::vector<const char*> syst_name = {"",
          "__puup", "__pudown", "__prefireup", "__prefiredown",
          "__muidup", "__muiddown", "__muisoup", "__muisodown", "__mutrgup", "__mutrgdown",
          "__elidup", "__eliddown", "__elrecoup", "__elrecodown",
          "__elzvtxup", "__elzvtxdown", "__eltrgup", "__eltrgdown",
          "__lfup", "__lfdown", "__hfup", "__hfdown",
          "__hfstat1up", "__hfstat1down", "__hfstat2up", "__hfstat2down",
          "__lfstat1up", "__lfstat1down", "__lfstat2up", "__lfstat2down",
          "__cferr1up", "__cferr1down", "__cferr2up", "__cferr2down",
          "__scale0", "__scale1", "__scale2", "__scale3", "__scale4", "__scale5",
          "__ps0", "__ps1", "__ps2", "__ps3",
          "__pdf0", "__pdf1", "__pdf2", "__pdf3", "__pdf4",
          "__pdf5", "__pdf6", "__pdf7", "__pdf8", "__pdf9",
          "__pdf10", "__pdf11", "__pdf12", "__pdf13", "__pdf14",
          "__pdf15", "__pdf16", "__pdf17", "__pdf18", "__pdf19",
          "__pdf20", "__pdf21", "__pdf22", "__pdf23", "__pdf24",
          "__pdf25", "__pdf26", "__pdf27", "__pdf28", "__pdf29",
          "__pdf30", "__pdf31", "__pdf32", "__pdf33", "__pdf34",
          "__pdf35", "__pdf36", "__pdf37", "__pdf38", "__pdf39",
          "__pdf40", "__pdf41", "__pdf42", "__pdf43", "__pdf44",
          "__pdf45", "__pdf46", "__pdf47", "__pdf48", "__pdf49",
          "__pdf50", "__pdf51", "__pdf52", "__pdf53", "__pdf54",
          "__pdf55", "__pdf56", "__pdf57", "__pdf58", "__pdf59",
          "__pdf60", "__pdf61", "__pdf62", "__pdf63", "__pdf64",
          "__pdf65", "__pdf66", "__pdf67", "__pdf68", "__pdf69",
          "__pdf70", "__pdf71", "__pdf72", "__pdf73", "__pdf74",
          "__pdf75", "__pdf76", "__pdf77", "__pdf78", "__pdf79",
          "__pdf80", "__pdf81", "__pdf82", "__pdf83", "__pdf84",
          "__pdf85", "__pdf86", "__pdf87", "__pdf88", "__pdf89",
          "__pdf90", "__pdf91", "__pdf92", "__pdf93", "__pdf94",
          "__pdf95", "__pdf96", "__pdf97", "__pdf98", "__pdf99",
          "__pdf100", "__pdf101", "__pdf102",};
    int syst_num = syst_name.size();

//    TH1D *h_MVA_b2[3][148];
//    TH1D *h_MVA_b3[3][148];
//    TH1D *h_MVA_b4[3][148];
    TH1D *h_MVA_j3b2[3][148];
    TH1D *h_MVA_j3b3[3][148];
    TH1D *h_MVA_j4b2[3][148];
    TH1D *h_MVA_j4b3[3][148];
    TH1D *h_MVA_j4b4[3][148];

//    TFile *scoreF;
//    TTree *scoreT;
    TFile *scoreF_j3b2, *scoreF_j3b3, *scoreF_j4b2, *scoreF_j4b3, *scoreF_j4b4;
    TTree *scoreT_j3b2, *scoreT_j3b3, *scoreT_j4b2, *scoreT_j4b3, *scoreT_j4b4;

//    vector<double> lepPt;
//    vector<double> missET;
    vector<double> lepPt_j3b2, lepPt_j3b3, lepPt_j4b2, lepPt_j4b3, lepPt_j4b4;
    vector<double> missET_j3b2, missET_j3b3, missET_j4b2, missET_j4b3, missET_j4b4;

//    vector<int> dupCheck;
    vector<int> dupCheck_j3b2, dupCheck_j3b3, dupCheck_j4b2, dupCheck_j4b3, dupCheck_j4b4;
 
    int evtNum = 0;

    bool doreco;
    bool dosyst;
    string syst_ext;

    TFile *out;
};

#endif

#ifdef MyAnalysis_cxx
void MyAnalysis::Init(TTree *tree)
{
   fReader.SetTree(tree);

}

Bool_t MyAnalysis::Notify()
{
   return kTRUE;
}

#endif // #ifdef MyAnalysis_cxx
/*CSV Unc order
    CENTRAL, JES_UP, JES_DN,
    LF_UP, LF_DN, HF_UP, HF_DN,
    HFSTAT1_UP, HFSTAT1_DN, HFSTAT2_UP, HFSTAT2_DN,
    LFSTAT1_UP, LFSTAT1_DN, LFSTAT2_UP, LFSTAT2_DN,
    CFERR1_UP, CFERR1_DN, CFERR2_UP, CFERR2_DN
https://github.com/vallot/CATTools/blob/cat90x/CatAnalyzer/interface/BTagWeightEvaluator.h
*/

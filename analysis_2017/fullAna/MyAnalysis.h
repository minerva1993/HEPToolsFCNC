//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jan 31 23:15:22 2017 by ROOT version 6.06/01
// from TTree tree/TopTree
// found on file: /xrootd/store/user/brochero/v8-0-4/Tree_LepJets_NewCSVSF_v8-0-4_Spring16-80X_36814pb-1_ttbar_PowhegPythia.root
//////////////////////////////////////////////////////////

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
   //TTreeReaderValue<Int_t> event = {fReader, "event"};
   //TTreeReaderValue<Int_t> run = {fReader, "run"};
   //TTreeReaderValue<Int_t> luminumber = {fReader, "luminumber"};
   TTreeReaderValue<Float_t> genWeight = {fReader, "genweight"};
   TTreeReaderValue<Int_t> GoodPV = {fReader, "GoodPV"};
   TTreeReaderValue<Int_t> TruePV = {fReader, "TruePV"};
   TTreeReaderValue<Int_t> channel = {fReader, "channel"};
   TTreeReaderArray<float> PUWeight = {fReader, "PUWeight"};
   TTreeReaderValue<Float_t> MET = {fReader, "MET"};
   TTreeReaderValue<Float_t> MET_phi = {fReader, "MET_phi"};
   TTreeReaderValue<Float_t> lepton_pT = {fReader, "lepton_pt"};
   TTreeReaderValue<Float_t> lepton_eta = {fReader, "lepton_eta"};
   TTreeReaderValue<Float_t> lepton_phi = {fReader, "lepton_phi"};
   TTreeReaderValue<Float_t> lepton_E = {fReader, "lepton_e"};
   TTreeReaderValue<Float_t> lepton_LES = {fReader, "lepton_LES"};
   TTreeReaderValue<Float_t> lepton_relIso = {fReader, "lepton_relIso"};
   TTreeReaderValue<bool> lepton_isIso = {fReader, "lepton_isIso"};
   TTreeReaderArray<float> lepton_SF = {fReader, "lepton_SF"};
   TTreeReaderArray<float> jet_pT = {fReader, "jet_pt"};
   TTreeReaderArray<float> jet_eta = {fReader, "jet_eta"};
   TTreeReaderArray<float> jet_phi = {fReader, "jet_phi"};
   TTreeReaderArray<float> jet_E = {fReader, "jet_e"};
   TTreeReaderArray<int> jet_index = {fReader, "jet_index"};
   //TTreeReaderArray<int> jet_gencone_mom = {fReader, "jet_gencone_mom"};//not in new 2017 ntuple
   TTreeReaderArray<float> jet_CSV = {fReader, "jet_CSV"};
   TTreeReaderArray<float> jet_deepCSV = {fReader, "jet_deepCSV"};
   TTreeReaderArray<float> jet_SF_deepCSV_25 = {fReader, "jet_SF_deepCSV_25"};
   TTreeReaderArray<float> jet_SF_deepCSV_30 = {fReader, "jet_SF_deepCSV_30"};
   TTreeReaderArray<float> jet_SF_deepCSV_35 = {fReader, "jet_SF_deepCSV_35"};
   TTreeReaderArray<float> jet_SF_deepCSV_40 = {fReader, "jet_SF_deepCSV_40"};
   TTreeReaderArray<float> jet_SF_deepCSV_38 = {fReader, "jet_SF_deepCSV_38"};
   TTreeReaderArray<float> jet_SF_deepCSV = {fReader, "jet_SF_deepCSV"};
   TTreeReaderArray<float> jet_CvsL = {fReader, "jet_CvsL"};
   TTreeReaderArray<float> jet_CvsB = {fReader, "jet_CvsB"};
   TTreeReaderArray<float> jet_deepCvsL = {fReader, "jet_deepCvsL"};
   TTreeReaderArray<float> jet_deepCvsB = {fReader, "jet_deepCvsB"};
   //TTreeReaderValue<Int_t> jet_number = {fReader, "jet_number"};
   TTreeReaderArray<int> jet_partonFlavour = {fReader, "jet_partonFlavour"};
   TTreeReaderArray<int> jet_hadronFlavour = {fReader, "jet_hadronFlavour"};
   TTreeReaderArray<float> jet_JES_Up = {fReader, "jet_JES_Up"};
   TTreeReaderArray<float> jet_JES_Down = {fReader, "jet_JES_Down"};
   TTreeReaderArray<float> jet_JER_Up = {fReader, "jet_JER_Up"};
   TTreeReaderArray<float> jet_JER_Nom = {fReader, "jet_JER_Nom"};
   TTreeReaderArray<float> jet_JER_Down = {fReader, "jet_JER_Down"};
   //TTreeReaderArray<float> pdfweight = {fReader, "pdfweight"};
   //TTreeReaderArray<float> scaleweight = {fReader, "scaleweight"};
   //TTreeReaderArray<int> jet_MatchedGenJetIndex = {fReader, "jet_MatchedGenJetIndex"};
   //TTreeReaderArray<int> genconecatid = {fReader, "genconecatid"};
   //TTreeReaderArray<float> gencone_gjet_pT = {fReader, "gencone_gjet_pT"};
   //TTreeReaderArray<float> gencone_gjet_eta = {fReader, "gencone_gjet_eta"};
   //TTreeReaderArray<float> gencone_gjet_phi = {fReader, "gencone_gjet_phi"};
   //TTreeReaderArray<float> gencone_gjet_E = {fReader, "gencone_gjet_E"};
   //TTreeReaderArray<int> gencone_gjetIndex = {fReader, "gencone_gjetIndex"};
   //TTreeReaderArray<int> gencone_gJetFlavW = {fReader, "gencone_gJetFlavW"};
   //TTreeReaderValue<Int_t> b_GenCone_NgJetsW = {fReader, "gencone_NgjetsW"};
   //TTreeReaderValue<Float_t> DRAddJets = {fReader, "draddjets"};
   //TTreeReaderValue<Int_t> genhiggscatid = {fReader, "genhiggscatid"};
   //TTreeReaderValue<Int_t> genchannel = {fReader, "genchannel"};
   //TTreeReaderValue<Float_t> genlepton_pT = {fReader, "genlepton_pT"};
   //TTreeReaderValue<Float_t> genlepton_eta = {fReader, "genlepton_eta"};
   //TTreeReaderValue<Float_t> genlepton_phi = {fReader, "genlepton_phi"};
   //TTreeReaderValue<Float_t> genlepton_E = {fReader, "genlepton_E"};
   //TTreeReaderValue<Float_t> gennu_pT = {fReader, "gennu_pT"};
   //TTreeReaderValue<Float_t> gennu_eta = {fReader, "gennu_eta"};
   //TTreeReaderValue<Float_t> gennu_phi = {fReader, "gennu_phi"};
   //TTreeReaderValue<Float_t> gennu_E = {fReader, "gennu_E"};
   TTreeReaderValue<Float_t> addHbjet1_pt = {fReader, "addHbjet1_pt"};
   TTreeReaderValue<Float_t> addHbjet1_eta = {fReader, "addHbjet1_eta"};
   TTreeReaderValue<Float_t> addHbjet1_phi = {fReader, "addHbjet1_phi"};
   TTreeReaderValue<Float_t> addHbjet1_e = {fReader, "addHbjet1_e"};
   TTreeReaderValue<Float_t> addHbjet2_pt = {fReader, "addHbjet2_pt"};
   TTreeReaderValue<Float_t> addHbjet2_eta = {fReader, "addHbjet2_eta"};
   TTreeReaderValue<Float_t> addHbjet2_phi = {fReader, "addHbjet2_phi"};
   TTreeReaderValue<Float_t> addHbjet2_e = {fReader, "addHbjet2_e"};
   TTreeReaderValue<Float_t> dRHbb = {fReader, "dRHbb"};

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
   double transverseMass(const TLorentzVector & l, const TLorentzVector & nu); 
   bool isPartOf(const std::string& word, const std::string& sentence);

   ClassDef(MyAnalysis,0);

    //Declare systematics here FIXME
    const std::vector<const char*> syst_name = {"", "_puup", "_pudown", "_lepSFup", "_lepSFdown",
                                                "_lfup", "_lfdown", "_hfup", "_hfdown",
                                                "_hfstat1up", "_hfstat1down", "_hfstat2up", "_hfstat2down",
                                                "_lfstat1up", "_lfstat1down", "_lfstat2up", "_lfstat2down",
                                                "_cferr1up", "_cferr1down", "_cferr2up", "_cferr2down"};
    int syst_num = syst_name.size();

    TH1D *h_PV[3][12][21];
    TH1D *h_EventWeight[3][12][21];
    TH1D *h_NJet[3][12][21];
    TH1D *h_NBJetCSVv2M[3][12][21];
    TH1D *h_NBJetCSVv2T[3][12][21];
    TH1D *h_NCJetM[3][12][21];
    TH1D *h_LepPt[3][12][21];
    TH1D *h_LepPhi[3][12][21];
    TH1D *h_LepEta[3][12][21];
    TH1D *h_MET[3][12][21];

    TH1D *h_WMass[3][12][21];
    TH1D *h_LepIso[3][12][21];
    TH1D *h_LepIsoQCD[3][12][21];
    TH1D *h_DPhi[3][12][21];
    TH1D *h_JetCSV[3][12][21];

    //tagging variables
    TH1D *h_csv[3][12][21];
    TH1D *h_cvsl[3][12][21];
    TH1D *h_cvsb[3][12][21];

    //DR
    TH1D *h_FCNHkinLepWMass[3][12][21];
    TH1D *h_FCNHkinHadWMass[3][12][21];
    TH1D *h_FCNHkinHMass[3][12][21];
    TH1D *h_FCNHkinDR[3][12][21];
    TH1D *h_FCNHkinLepTopM[3][12][21];
    TH1D *h_FCNHkinHadTopM[3][12][21];
    TH1D *h_FCNHkinHPt[3][12][21];
    TH1D *h_FCNHkinHdPhi[3][12][21];
    TH1D *h_FCNHkinHdEta[3][12][21];
    TH1D *h_FCNHkinHb1Pt[3][12][21];
    TH1D *h_FCNHkinHb2Pt[3][12][21];
    TH1D *h_FCNHkinHb1CSV[3][12][21];
    TH1D *h_FCNHkinHb2CSV[3][12][21];
    TH1D *h_FCNHkinLepTopPt[3][12][21];
    TH1D *h_FCNHkinHadTopPt[3][12][21];

    TH1D *h_genDR[3][12][21];
    TH1D *h_matchDR[3][12][21];
    TH1D *h_genHm[3][12][21];
    TH1D *h_matchHm[3][12][21];

    ////RECO
    TFile *assignF;// = new TFile("assign/ref_ttbb.root", "READ");
    TTree *assignT;// = (TTree*) assignF->Get("tree");
    //int nevt = assignT->GetEntries();

    vector<double> lepPt;
    vector<double> missET;
    vector<double> BDTScore;

    vector<int> dupCheck;
    int lepcount = 0;
    int evtNum = 0;

    bool reco;

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

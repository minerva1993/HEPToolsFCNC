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
//   TTreeReaderArray<float> jet_CSV = {fReader, "jet_CSV"};
   TTreeReaderArray<float> jet_deepCSV = {fReader, "jet_deepCSV"};
   TTreeReaderArray<float> jet_SF_deepCSV_30 = {fReader, "jet_SF_deepCSV_30"};
//   TTreeReaderArray<float> jet_CvsL = {fReader, "jet_CvsL"};
//   TTreeReaderArray<float> jet_CvsB = {fReader, "jet_CvsB"};
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
   double transverseMass(const TLorentzVector & l, const TLorentzVector & nu); 
   bool isPartOf(const std::string& word, const std::string& sentence);

   ClassDef(MyAnalysis,0);

    //Declare systematics here FIXME
    const std::vector<const char*> syst_name = {"",
          "__puup", "__pudown", "__prefireup", "_prefiredown",
          "__muidup", "__muiddown", "__muisoup", "__muisodown", "__mutrgup", "__mutrgdown",
          "__elidup", "__eliddown", "__elrecoup", "__elrecodown",
          "__elzvtxup", "__elzvtxdown", "__eltrgup", "__eltrgdown",
          "__lfup", "__lfdown", "__hfup", "__hfdown",
          "__hfstat1up", "__hfstat1down", "__hfstat2up", "__hfstat2down",
          "__lfstat1up", "__lfstat1down", "__lfstat2up", "__lfstat2down",
          "__cferr1up", "__cferr1down", "__cferr2up", "__cferr2down",
          "__scale0", "__scale1", "__scale2", "__scale3", "__scale4", "__scale5",
          "__ps0", "__ps1", "__ps2", "__ps3"};
    int syst_num = syst_name.size();

    TH1D *h_PV[3][12][45];
    TH1D *h_EventWeight[3][12][45];
    TH1D *h_NJet[3][12][45];
    TH1D *h_NBJetCSVv2M[3][12][45];
    TH1D *h_NBJetCSVv2T[3][12][45];
    TH1D *h_NCJetM[3][12][45];
    TH1D *h_LepPt[3][12][45];
    TH1D *h_LepPhi[3][12][45];
    TH1D *h_LepEta[3][12][45];
    TH1D *h_MET[3][12][45];

    TH1D *h_WMass[3][12][45];
    TH1D *h_LepIso[3][12][45];
    TH1D *h_LepIsoQCD[3][12][45];
    TH1D *h_DPhi[3][12][45];
    TH1D *h_JetCSV[3][12][45];

    //leading and subleading jets
    TH1D *h_LeadJetPt[3][12][45];
    TH1D *h_LeadJetEta[3][12][45];
    TH1D *h_SubleadJetPt[3][12][45];
    TH1D *h_SubleadJetEta[3][12][45];

    //Reco
    TH1D *h_csv[3][12][45];
    TH1D *h_cvsl[3][12][45];
    TH1D *h_cvsb[3][12][45];
    TH1D *h_FCNHkinLepWMass[3][12][45];
    TH1D *h_FCNHkinHadWMass[3][12][45];
    TH1D *h_FCNHkinHMass[3][12][45];
    TH1D *h_FCNHkinDR[3][12][45];
    TH1D *h_FCNHkinLepTopM[3][12][45];
    TH1D *h_FCNHkinHadTopM[3][12][45];
    TH1D *h_FCNHkinHPt[3][12][45];
    TH1D *h_FCNHkinHdPhi[3][12][45];
    TH1D *h_FCNHkinHdEta[3][12][45];
    TH1D *h_FCNHkinHb1Pt[3][12][45];
    TH1D *h_FCNHkinHb2Pt[3][12][45];
    TH1D *h_FCNHkinHb1CSV[3][12][45];
    TH1D *h_FCNHkinHb2CSV[3][12][45];
    TH1D *h_FCNHkinHb1CSVfull[3][12][45];
    TH1D *h_FCNHkinHb2CSVfull[3][12][45];
    TH1D *h_FCNHkinLepTopPt[3][12][45];
    TH1D *h_FCNHkinHadTopPt[3][12][45];
    TH1D *h_genDR[3][12][45];
    TH1D *h_matchDR[3][12][45];
    TH1D *h_genHm[3][12][45];
    TH1D *h_matchHm[3][12][45];

    //SF specific histos
    TH1D *h_PVnoSF[3][12][45];
    TH1D *h_JetCSVnoSF[3][12][45];

    ////RECO
    TFile *assignF;// = new TFile("assign/ref_ttbb.root", "READ");
    TTree *assignT;// = (TTree*) assignF->Get("tree");
    int reco_id = -1;//STFCNC==0, TTFCNC==1, TTBKG==2

    vector<double> lepPt;
    vector<double> missET;

    vector<int> dupCheck;
 
    int lepcount = 0;
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

#define tmva_score_cxx
// The class definition in tmva_score.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.


// The following methods are defined in this file:
//    Begin():        called every time a loop on the tree starts,
//                    a convenient place to create your histograms.
//    SlaveBegin():   called after Begin(), when on PROOF called only on the
//                    slave servers.
//    Process():      called for each event, in this function you decide what
//                    to read and fill your histograms.
//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//                    called only on the slave servers.
//    Terminate():    called at the end of the loop on the tree,
//                    a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// root> T->Process("tmva_score.C")
// root> T->Process("tmva_score.C","some options")
// root> T->Process("tmva_score.C+")
//


#include "tmva_score.h"
#include <TH2.h>
#include <TStyle.h>

void tmva_score::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

  TString option = GetOption();
}

void tmva_score::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

  TString option = GetOption();

  h_scoreBDT = new TH1D(Form("h_scoreBDT_%s",option.Data()), "BDT score", 40, -0.33, 0.34);
  h_scoreBDT->SetXTitle("BDT score");
  h_scoreBDT->Sumw2();
  fOutput->Add(h_scoreBDT);

  h_scoreKeras = new TH1D(Form("h_scoreKeras_%s",option.Data()), "Keras score", 20, 0, 1.0);
  h_scoreKeras->SetXTitle("Keras score");
  h_scoreKeras->Sumw2();
  fOutput->Add(h_scoreKeras);
}

Bool_t tmva_score::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // When processing keyed objects with PROOF, the object is already loaded
   // and is available via the fObject pointer.
   //
   // This function should contain the \"body\" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

  fReader.SetEntry(entry);
  TString option = GetOption();

  h_scoreBDT->Fill(*BDTScore, *Event_Weight);
  h_scoreKeras->Fill(*KerasScore, *Event_Weight);

  return kTRUE;
}

void tmva_score::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

  TString option = GetOption();

}

void tmva_score::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

  TString option = GetOption();

  TFile * out = TFile::Open(Form("shape_%s.root",option.Data()),"RECREATE");

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


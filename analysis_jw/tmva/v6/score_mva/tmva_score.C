#define tmva_score_cxx

#include "tmva_score.h"
#include <TH2.h>
#include <TStyle.h>

void tmva_score::Begin(TTree * /*tree*/)
{
  TString option = GetOption();
}

void tmva_score::SlaveBegin(TTree * /*tree*/)
{
  TString option = GetOption();

  h_scoreBDT = new TH1D(Form("h_scoreBDT_%s",option.Data()), "BDT score", 20, -0.8, 0.8);
  h_scoreBDT->SetXTitle("BDT score");
  h_scoreBDT->Sumw2();
  fOutput->Add(h_scoreBDT);

  h_scoreKeras = new TH1D(Form("h_scoreKeras_%s",option.Data()), "Keras score", 20, 0, 1);//0-1 20
  h_scoreKeras->SetXTitle("Keras score");
  h_scoreKeras->Sumw2();
  fOutput->Add(h_scoreKeras);
}

Bool_t tmva_score::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();
  if(*BDTScore == 2 && *Event_Weight == 10) return kTRUE;

  h_scoreBDT->Fill(*BDTScore, *Event_Weight);
  h_scoreKeras->Fill(*KerasScore, *Event_Weight);

  return kTRUE;
}

void tmva_score::SlaveTerminate()
{
  TString option = GetOption();

}

void tmva_score::Terminate()
{
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


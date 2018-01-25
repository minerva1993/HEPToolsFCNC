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

  h_scoreBDT = new TH1D(Form("h_scoreBDT_%s",option.Data()), "BDT score", 20, -0.6, 0.6);
  h_scoreBDT->SetXTitle("BDT score");
  h_scoreBDT->Sumw2();
  fOutput->Add(h_scoreBDT);

  h_scoreKeras = new TH1D(Form("h_scoreKeras_%s",option.Data()), "Keras score", 20, 0, 0.5);
  h_scoreKeras->SetXTitle("Keras score");
  h_scoreKeras->Sumw2();
  fOutput->Add(h_scoreKeras);

  h_scoreBDTGen = new TH1D(Form("h_scoreBDTGen_%s",option.Data()), "BDT score", 20, -0.6, 0.6);
  h_scoreBDTGen->SetXTitle("BDT score (gen Matched)");
  h_scoreBDTGen->Sumw2();
  fOutput->Add(h_scoreBDTGen);

  h_scoreKerasGen = new TH1D(Form("h_scoreKerasGen_%s",option.Data()), "Keras score", 20, 0, 0.5);
  h_scoreKerasGen->SetXTitle("Keras score (gen Matched)");
  h_scoreKerasGen->Sumw2();
  fOutput->Add(h_scoreKerasGen);
}

Bool_t tmva_score::Process(Long64_t entry)
{
  fReader.SetEntry(entry);
  TString option = GetOption();
  if(*BDTScore == 2 && *Event_Weight == 10) return kTRUE;

  h_scoreBDT->Fill(*BDTScore, *Event_Weight);
  h_scoreKeras->Fill(*KerasScore, *Event_Weight);

  if(*genMatch == 2){
    h_scoreBDTGen->Fill(*BDTScore, *Event_Weight);
    h_scoreKerasGen->Fill(*KerasScore, *Event_Weight);
  }

  return kTRUE;
}

void tmva_score::SlaveTerminate()
{
  TString option = GetOption();

}

void tmva_score::Terminate()
{
  TString option = GetOption();

  h_scoreBDT->AddBinContent(20, h_scoreBDT->GetBinContent(21));
  h_scoreKeras->AddBinContent(20, h_scoreKeras->GetBinContent(21));
  h_scoreBDTGen->AddBinContent(20, h_scoreBDTGen->GetBinContent(21));
  h_scoreKerasGen->AddBinContent(20, h_scoreKerasGen->GetBinContent(21));
  h_scoreBDT->AddBinContent(1, h_scoreBDT->GetBinContent(0));
  h_scoreKeras->AddBinContent(1, h_scoreKeras->GetBinContent(0));
  h_scoreBDTGen->AddBinContent(1, h_scoreBDTGen->GetBinContent(0));
  h_scoreKerasGen->AddBinContent(1, h_scoreKerasGen->GetBinContent(0));

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


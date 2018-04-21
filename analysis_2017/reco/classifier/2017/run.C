#include "idxToNtuple.h"
#include <TSystem.h>
#include <string>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( TString name )
{

  TFile *f = TFile::Open("/home/minerva1993/fcnc/analysis_2017/reco/classifier/2017/scoreST01/"+name, "READ");
  if (f->IsOpen()==kFALSE){
    f->Close();
    gSystem->Exit(0);
  }
  else f->Close();


  TChain assign("tree");
  assign.Add("/home/minerva1993/fcnc/analysis_2017/reco/classifier/2017/scoreST01/"+name);

  idxToNtuple t(&assign);

  t.Loop();

  gSystem->Exit(0);
}

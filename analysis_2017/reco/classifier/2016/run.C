#include "idxToNtuple.h"
#include <TSystem.h>
#include <string>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( TString name )
{

  //TFile *f = TFile::Open("root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/score04/"+name, "READ");
  TFile *f = TFile::Open("/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/scoreST02/"+name, "READ");
  if (f->IsOpen()==kFALSE){
    f->Close();
    gSystem->Exit(0);
  }
  else f->Close();


  TChain assign("tree");
  //assign.Add("root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/reco/score04/"+name);
  assign.Add("/home/minerva1993/fcnc/analysis_2017/reco/classifier/2016/scoreST02/"+name);

  idxToNtuple t(&assign);

  t.Loop();

  gSystem->Exit(0);
}

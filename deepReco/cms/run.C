#include "idxToNtuple.h"
#include <TSystem.h>

R__LOAD_LIBRARY(idxToNtuple.C+)

void run( TString name )
{
  TChain assign("tree");
  assign.Add("score04/"+name);

  idxToNtuple t(&assign);

  t.Loop();

  gSystem->Exit(0);
}

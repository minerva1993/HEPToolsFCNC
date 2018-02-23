#include "idxToNtuple.h"
R__LOAD_LIBRARY(idxToNtuple.C+)

void run()
{
  TChain assign("tree");
  assign.Add("score/score_deepReco_ttbb_44.root");

  idxToNtuple t(&assign);

  t.Loop();
}

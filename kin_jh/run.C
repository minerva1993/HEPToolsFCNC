#include "Analyzer2.h"
R__LOAD_LIBRARY(Analyzer2.C+)

void run()
{
  gSystem->CompileMacro("Analyzer2.C", "k");

  // mode:  1(ttbar) or 2(fcnc)
  const int mode = 2;

/*
  TChain chain("tree");
  if ( mode == 1 ) chain.Add("../ntuple_ttbb.root");
  else if ( mode == 2 ) chain.Add("../ntuple_tch.root");
*/

  TChain chain("ttbbLepJets/tree");
  if ( mode == 1 ) chain.Add("/home/minerva1993/fcnc/ntuple_jw/v2/TT_powheg_ttbb.root");
  else if ( mode == 2 ) chain.Add("/home/minerva1993/fcnc/ntuple_jw/v2/TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root");

  Analyzer* t = new Analyzer(&chain);
  t->Loop(mode);
}

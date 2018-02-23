#include "analyzer.h"
R__LOAD_LIBRARY(analyzer.C+)

void run()
{
   gSystem->CompileMacro("analyzer.C");
   TChain chain("tree");
   //chain.Add("flat_ttLJ.root");
   chain.Add("flat_tch.root");

   analyzer t(&chain);
   //t.Loop("ntuple_ttLJ.root");
   t.Loop("ntuple_tch.root");
}
